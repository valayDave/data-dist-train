from dataclasses import dataclass,field,asdict
import random
from typing import List
from .dataset import DistributedDataset
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
    shuffle_after:bool=True

    
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
        if self.control_params.shuffle_before:
            random.shuffle(self.train_dataset)
        item_indexes = [i for i in range(len(self.train_dataset))]
        for index in item_indexes:
            if index % self.control_params.block_size == 0 and index > 0 and self.control_params.block_size!=1:
                self.dispatch_block(flush_items,block_map,flush_counter)
                flush_counter+=1
                flush_items = []
            flush_items.append(index)
        
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
        


        
        
        