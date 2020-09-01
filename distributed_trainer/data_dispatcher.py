from dataclasses import dataclass,field,asdict
import random
from typing import List
from threading import Thread
from .dataset import DistributedDataset
from .utils import save_json_to_file,load_json_from_file,dir_exists
import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

ALLOWED_DISPATCHING_METHODS = [
    'random',
    'round_robin'
]
@dataclass
class DispatcherControlParams:
    '''
    block_size: that is the number of data items (e.g. CSV rows, images) in a block.
    approach: dispatching approach which can be random or round robin.
    num_workers: in a cluster, n, which you can use n=5 as default.
    '''
    # for Quick sampling 
    sample:int=None

    # For dispatching
    block_size:int=2
    num_workers:int=5
    approach:str='random'

    # Before Dispatch:
    shuffle_before:bool=False

    
    def validate(self):
        if self.approach not in ALLOWED_DISPATCHING_METHODS:
            raise ValueError('Allowed Values %s'%','.join(ALLOWED_DISPATCHING_METHODS))
    
    def __post_init__(self):
        self.validate()

@dataclass
class DataBlock:
    data_item_indexes:List = field(default_factory=lambda:[])

    def update(self,data_item_indexes):
        self.data_item_indexes.extend(data_item_indexes)
    
    def __repr__(self) -> str:
        return (
            'DataBlock('
            f'data_item_indexes={len(self.data_item_indexes)!r}, '
            ')'
        )
    def __len__(self):
        return len(self.data_item_indexes)

class BlockDistributedDataset(DistributedDataset):
    """BlockDistributedDataset [summary]
    This is the dataset given to allworkers. \n
    It contains the complete data and indexes that they need to to create a `TensorDataset`.\n  
    ### Methods of `get_train_dataset` and `get_test_dataset` need to be overridden leveraging while leveraging the `List[DataBlock]`s
    
    """
    def __init__(self,train_dataset,blocks:List[DataBlock],test_dataset=None,metadata=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.blocks = blocks
        self.metadata_dict = metadata

    def get_metadata(self):
        return self.metadata_dict


@dataclass
class DataStore:
    '''
    This will be created by the dispatcher 
    to ensure emperical a common structure among all 
    experiments with one dispatcher. 
    '''
    dataset:List = field(default_factory=lambda:[])
    blocks:List[DataBlock] = field(default_factory=lambda:[])
    control_params:DispatcherControlParams=None

    def __repr__(self) -> str:
        return (
            'DataStore('
            f'dataset={len(self.dataset)!r}, '
            f'blocks={str(self.blocks)!r}, '
            f'control_params={str(self.control_params)}'
            )

    def get_distributed_dataset(self,\
                        UsedClass=BlockDistributedDataset,\
                        test_set=None,\
                        **kwargs) -> DistributedDataset:
        '''
        Create a DistributedDataset of `UsedClass` to create a BlockDistributedDataset for training 
        '''
        return UsedClass(self.dataset,self.blocks,test_dataset=test_set,metadata=asdict(self.control_params),**kwargs)
    
    @classmethod
    def from_json(cls,json_object):
        blocks = []
        for blk in json_object['blocks']:
            blocks.append(DataBlock(**blk))
        json_object['blocks'] = blocks
        if json_object['control_params'] is not None:
            json_object['control_params'] = DispatcherControlParams(**json_object['control_params'])
        return cls(**json_object)

class SamplerSession:
    def __init__(self,list_length,num_workers,block_size,datastore=None):
        self.list_length = list_length
        self.num_workers = num_workers
        self.block_size = block_size
        self.datastore = None
        if datastore is None:
            self.shuffle_datastore()
        else:
            self.datastore= datastore
        
    def shuffle_datastore(self):
        # self.datastore = 
        disp = Dispatcher(
            DispatcherControlParams(
                block_size=self.block_size,
                num_workers=self.num_workers,
                approach='round_robin',
                shuffle_before=True,
            ),
            [i for i in range(self.list_length)],
            None
        )
        self.datastore = disp.run()
    
    def to_json(self):
        return asdict(self.datastore)

class DistributedIndexSamplerServer(rpyc.Service):
    """DistributedIndexSamplerServer 
    - Every worker will connect with this singular distributed sampler server. 
    - Running Strategy
        1. Rank 0 : Connects to create an index list and call shuffle. 
        2. `connection_id` will be used to do reference the sampler being used. 
        3. if multiple training sessions can use the same sampling server. 
    """
    def __init__(self,session_storage_path='./sampler_session.json'):
        rpyc.Service.__init__(self)
        self.storage_path = session_storage_path
        self.session_map = self.load_map() # {conn_id : SamplerSession}
        
    def load_map(self): # Call on init of Service.  # Load Datastore and give it to the sampler session
        if not dir_exists(self.storage_path): # If no File in FS. return {}
            return {}
        session_map = load_json_from_file(self.storage_path)
        for conn_id in session_map: # Create a new Sampler Session From the Datastore stored on file. 
            datastore = DataStore.from_json(session_map[conn_id])
            session_map[conn_id] = SamplerSession(
                len(datastore.dataset),
                len(datastore.blocks),
                datastore.control_params.block_size,
                datastore=datastore
            )
        print("Loaded Sesssion MAP!! : ",session_map.keys())
        return session_map
    
    def save_map(self): # Save Datastore
        sess_json = {}
        for k in self.session_map:
            sess_json[k] = self.session_map[k].to_json() # Save Datastore of each session
        save_json_to_file(sess_json,self.storage_path)


    def exposed_init(self,list_length,num_workers,block_size) -> int: # Returns a ConnID that will be used by other workers to connect
        """exposed_init 
        This will create a sampler sessionId which will be used to create a 
        DataStore. 
        This ID Will be loaded on new Workers too 
        :param list_length: [description]
        :param num_workers: [description]
        :param block_size: [description]
        """
        return self.create_datastore(list_length,num_workers,block_size)

    def create_datastore(self,list_length,num_workers,block_size):
        conn_id = str(random.randint(0,10000))
        print(f"Creating Datastore For Connection ID : {conn_id}")
        session = SamplerSession(
            list_length,num_workers,block_size
        )
        self.session_map[conn_id] = session
        self.save_map()
        return conn_id
       

    def exposed_shuffle(self,connection_id):
        print(f"Called Shuffle For Connection ID : {connection_id}")
        # self.create_datastore(self.list_length,self.num_workers,self.block_size)
        self.session_map[connection_id].shuffle_datastore()

    def exposed_get_indexes(self,connection_id):
        print(f"Getting Indexes Connection ID : {connection_id}")
        print(self.session_map.keys())
        if str(connection_id) not in self.session_map:
            return []
        return [
            block.data_item_indexes for block in self.session_map[connection_id].datastore.blocks
        ]

    def exposed_delete_session(self,connection_id):
        del self.session_map[connection_id]
        self.save_map()


@dataclass
class SamplerArgs:
    block_size:int=2
    num_workers:int=5
    port:int=5000 # Port to DistributedIndexSampler
    host:str='127.0.0.1' # host of the DistributedIndexSampler
    UsedClass:BlockDistributedDataset=BlockDistributedDataset
    connection_id:str=None # Id of the Remote Connection
        
        

class DistributedSampler:
    '''

    '''
    def __init__(self,
                sampler_args:SamplerArgs,
                train_set=[],
                test_set=None): 
        self.train_set = train_set
        self.num_workers = sampler_args.num_workers
        self.test_set = test_set
        self.UsedClass = sampler_args.UsedClass
        self.port= sampler_args.port
        self.host= sampler_args.host

        # Do init with service to create the remote sampler session
        if sampler_args.connection_id is None:
            self.connection_id = self.create_session(len(train_set),sampler_args.num_workers,sampler_args.block_size,host=sampler_args.host,port=sampler_args.port)
        else:
            self.connection_id = sampler_args.connection_id


    def shuffle(self):
        '''
        Call Remote Shuffle here. 
        '''
        conn = rpyc.connect(port=self.port,host=self.host,config = rpyc.core.protocol.DEFAULT_CONFIG)
        conn.root.shuffle(self.connection_id)
        conn.close()

    def get_distributed_dataset(self):
        '''
        Get Indexes from the `DistributedDataStore` and then transform that into a 
        `BlockDistributedDataset`
        '''
        conn = rpyc.connect(port=self.port,host=self.host)
        index_lists = conn.root.get_indexes(self.connection_id)
        # THERE IS A MASSIVE SERIALIISATION ISSUE WITH RPYC. 
        # ALL data Needs to Be serialised Properly!. 
        data_blocks = [DataBlock(data_item_indexes=list(idx_list)) for idx_list in index_lists]
        conn.close()
        return self.UsedClass(self.train_set,data_blocks,test_dataset=self.test_set,metadata=dict(distributed_sampler=True))
    
    def close_session(self):
        conn = rpyc.connect(port=self.port,host=self.host)
        index_lists = conn.root.delete_session(self.connection_id)
        conn.close()
        print("Session On Sampler Deleted. ")

    @staticmethod
    def create_session(list_length,workers,block_size,host='localhost',port=5003):
        conn = rpyc.connect(port=port,host=host,config = rpyc.core.protocol.DEFAULT_CONFIG)
        conn_id  = conn.root.init(list_length,workers,block_size)
        conn.close()
        return conn_id
    

class Dispatcher:
    '''
    Purpose : From a datasource : Extract `block_size` items from source using  
    
    Args: 
        - control_params(`DispatcherControlParams`) : hyper params for data dispatching
        - train_dataset : Training Dataset : Iteratable[] : Keeping it as torch.utils.Dataset
        - test_dataset : Training Dataset : Iteratable[]

    Dispatching Method:
    - Random
        - The random dispatching is for every block, you uniformed sample a random variable in 0, 1, 2, 3, 4, if 0 is sampled, you just send the block to node 0.   random()%n
    - Round Robin
        - The round-robin dispatching is for every block, you maintain a counter K, with a initialization value K=1, and you always send the block to K%n, and increment K after each sending.

    '''
    def __init__(self,\
                control_params:DispatcherControlParams,\
                train_dataset,\
                test_dataset):
        self.control_params = control_params
        # if control_params.sample is not None:
        #     num_samples = control_params.sample
        #     train_dataset = random.sample(train_dataset,num_samples)
        #     test_dataset = random.sample(test_dataset,num_samples)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def create_blocks(worker_count):
        final_blocks = [DataBlock() for i in range(worker_count)]
        return final_blocks
    
    def run(self):
        block_map = self.create_blocks(self.control_params.num_workers)
        flush_items = []
        flush_counter = 0
        item_indexes = [i for i in range(len(self.train_dataset))]
        if self.control_params.shuffle_before:
            random.shuffle(item_indexes)
        for index_val,item in enumerate(item_indexes):
            if index_val % self.control_params.block_size == 0 and index_val > 0 and self.control_params.block_size!=1:
                self.dispatch_block(flush_items,block_map,flush_counter)
                flush_counter+=1
                flush_items = []
            flush_items.append(item)
        
        if len(flush_items)  > 0:
            self.dispatch_block(flush_items,block_map,flush_counter)

        # print("Distribution of Dataset : %s"%','.join([str(len(m)) for m in block_map]))
        datastore = DataStore(
            dataset=self.train_dataset,
            blocks = block_map,
            control_params = self.control_params
        )
        return datastore
        
    def dispatch_block(self,index_list:List,blocks:List[DataBlock],flush_counter:int):
        selected_worker_block = None
        
        if self.control_params.approach == 'random':
            selected_index = random.randint(0,len(blocks)-1)
            selected_worker_block = blocks[selected_index]
        
        elif self.control_params.approach =='round_robin':
            selected_index = flush_counter % self.control_params.num_workers
            # print("dispatching to %d With Len %d"%(selected_index,len(index_list)))
            selected_worker_block = blocks[selected_index]

        selected_worker_block.update(index_list)
        return 
        
