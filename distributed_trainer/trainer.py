import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from dataclasses import dataclass
import torch.multiprocessing as mp
from typing import List
import os
from datetime import datetime
from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel
import dataclasses
import pickle
from .utils import \
    AverageMeter,\
    ConfusionMatrix,\
    create_logger,\
    ExperimentResultsBundle,\
    ExperimentBundle,\
    TrainerArgs,\
    DistTrainerArgs,\
    CheckpointingArgs,\
    safe_mkdir

class NetworkArgs:
    """ 
    Will contain all Network optimisation paramaters and other relaated classes
    theses will be initailsed during training by trainer. 
    """
    optimizer = optim.Adam
    optimizer_args_dict = dict(
        lr = 1e-3
    )
    model = None
    loss_fn = None
    model_args_dict=None

class Dataset:
    """
    Base class which needs to be implemeted for working with the DistributedTrainer Process. 
    """    
    def get_train_dataset(self)->torch.utils.data.TensorDataset:
        raise NotImplementedError()
    
    def get_test_dataset(self)->torch.utils.data.TensorDataset:
        raise NotImplementedError()

    def get_metadata(self)->dict: # To help get Quick dataset Related Metadata.
        raise NotImplementedError()

    def get_labels(self)->List:
        raise NotImplementedError()

    def model_alterations(self,model): # This is to change the model configurations acc to dataset. 
        return model

class DistributedDataset(Dataset):
    
    def get_train_dataset(self,rank)->torch.utils.data.TensorDataset:
        raise NotImplementedError()
    
    def get_test_dataset(self,rank)->torch.utils.data.TensorDataset:
        raise NotImplementedError()


class BaseTrainer:
    """ 
    """
    optimizer = None
    neural_network:nn.Module
    loss_fn = None
    dataset :Dataset=None
    train_data = None
    # todo assert datastructures
    def __init__(self,network_args=NetworkArgs(),training_args = TrainerArgs(),dataset=None):
        self.network_args = network_args
        self.dataset = dataset
        self.training_args = training_args
        self.gpu_enabled = False
        self.checkpoint_args = training_args.checkpoint_args
        checkpoint_base_path = self.checkpoint_args.path
        exp_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.checkpoint_save_path = os.path.join(checkpoint_base_path,self.__class__.__name__+'__'+exp_date)

    def get_meters(self):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        acc = AverageMeter('Accuracy', ':6.2f')
        conf_matrix = ConfusionMatrix(self.dataset.get_labels())
        return (conf_matrix,acc,losses,batch_time)

    def initialise_network(self):
        """initialise_network 
        Uses the data in `NetworkArgs` initialise the BaseTrainer Experiment. 
        Ensure that Training Initialisations Like  Initialisation from 
        neural netowrk params and LR etc.  
        Initialise : 
        - `Network`
        - `optimizer`
        - `Loss`
        Override if Needed 
        """
        if self.network_args.loss_fn is None : 
            raise Exception("Loss Function Needed")
        if self.network_args.model is None:
            raise Exception("Loss Function Needed")

        if self.network_args.model_args_dict is not None:
            self.neural_network = self.network_args.model(**self.network_args.model_args_dict)
        else:
            self.neural_network = self.network_args.model()
        
        self.optimizer = self.network_args.optimizer(self.neural_network.parameters(),**self.network_args.optimizer_args_dict)
        self.loss_fn = self.network_args.loss_fn()

    # override in DistTrainer  
    def train_loop(self,train_data_loader) -> ExperimentResultsBundle:
        """train_loop
        Run the Main Training Loop
        """
        raise NotImplementedError()

    # override in DistTrainer        
    def get_train_dataloader(self):
        tensor_dataset = self.dataset.get_train_dataset()
        data_loader = torch.utils.data.DataLoader(
            tensor_dataset,\
            batch_size = self.training_args.batch_size,\
            shuffle = self.training_args.shuffle,\
        )
        return dataloader
    
    # override in DistTrainer
    def run(self):
        experiment_results = []
        self.setup_gpu_dp()
        train_data_loader = self.get_train_dataloader()
        for epoch in range(self.training_args.num_epochs):
            results_bundle = self.train_loop(train_data_loader)
            results_bundle.epoch = epoch
            experiment_results.append(results_bundle)
            if self.checkpoint_args.save_experiment:
                if epoch % self.checkpoint_args.checkpoint_frequency:
                    bundle = self.create_experiment_bundle(experiment_results)
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_save_path,epoch),
                        bundle,\
                    )
        
        if self.checkpoint_args.save_experiment:
            self.logger.info("Created Bundle For Checkpoint ")
            bundle = self.create_experiment_bundle(experiment_results)
            self.save_checkpoint(
                os.path.join(self.checkpoint_save_path,epoch),
                bundle,\
            )

    # override in DistTrainer
    def create_experiment_bundle(self,train_experiment_results:List[ExperimentResultsBundle]) -> ExperimentBundle:
        dataset_meta = None
        try:
            dataset_meta = self.dataset.get_metadata()
        except:
            pass
        model = self.neural_network
        best_model_state_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
        model_dict = OrderedDict(best_model_state_dict)
        opt_dict = self.optimizer.state_dict()
        return ExperimentBundle(
            train_epoch_results = [dataclasses.asdict(res) for res in train_experiment_results],
            train_args = dataclasses.asdict(self.training_args),
            dataset_metadata = dataset_meta,
            model = model_dict,
            optimizer = opt_dict,
            model_args = self.network_args.model_args_dict,
            optimizer_args = self.network_args.optimizer_args_dict,
            loss_fn = str(self.loss_fn)
        )

    @staticmethod
    def save_checkpoint(model_save_path,bundle:ExperimentBundle,checkpoint_name='model_checkpoint.pt'):
        safe_mkdir(model_save_path)
        bundle_dict = dataclasses.asdict(bundle)
        torch.save(
            bundle_dict,
            os.path.join(model_save_path,checkpoint_name)
        )
    
    def setup_gpu_dp(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            self.neural_network = nn.DataParallel(self.neural_network, device_ids=[rank])
            self.neural_network.to(device)
            self.gpu_enabled = True
        else : 
            self.neural_network.to(device)
    
    @staticmethod
    def load_bundle_from_disk(path) -> ExperimentBundle:
        with open(os.path.join(model_save_path,checkpoint_name), 'wb') as handle:
            bundle = pickle.load(handle)
        return bundle
        
class DistributedTrainer(BaseTrainer):
    """DistributedTrainer 

    """
    def __init__(self,dist_args=DistTrainerArgs(),**kwargs):
        self.dist_args = dist_args
        super(DistributedTrainer, self).__init__(**kwargs)
        self.rank = None
        self.logger = None

    def setup(self,rank,world_size):
        os.environ['MASTER_ADDR'] = self.dist_args.master_ip
        os.environ['MASTER_PORT'] = self.dist_args.master_port
        proc_name = self.__class__.__name__+'__'+str(rank)
        self.logger = create_logger(proc_name)
        # initialize the process group
        if self.dist_args.backend == 'gloo':
            dist.init_process_group(self.dist_args.backend, rank=rank, world_size=world_size,init_method='env://'+self.dist_args.master_ip+':'+self.dist_args.master_port)
        else: 
            dist.init_process_group(self.dist_args.backend, rank=rank, world_size=world_size,init_method='tcp://'+self.dist_args.master_ip+':'+self.dist_args.master_port)
        torch.manual_seed(1234)

    def setup_gpu_ddp(self,rank):
        if torch.cuda.is_available():
            self.neural_network.to(rank)
            device = rank 
            self.neural_network = DistributedDataParallel(self.neural_network, device_ids=[rank])
            self.gpu_enabled = True
        else : 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.neural_network.to(device)
            self.rank = device

    def run(self,rank,world_size):
        self.setup(rank,world_size)
        self.initialise_network()
        self.setup_gpu_ddp(rank)
        self.sync_params()
        train_data_loader = self.get_train_dataloader(rank)
        self.neural_network = self.dataset.model_alterations(self.neural_network) # This needs Fixes. 
        self.rank = rank
        experiment_results = []
        for epoch in range(self.training_args.num_epochs):
            self.logger.info("Starting Epoch %d With %s"%(epoch,'GPU' if self.gpu_enabled else 'CPU'))
            results_bundle = self.train_loop(train_data_loader)# Rank can be accessed as a property. 
            results_bundle.epoch = epoch
            experiment_results.append(results_bundle) 
            if self.train_loop_checkpoint_validation(epoch):
                bundle = self.create_experiment_bundle(experiment_results)
                self.logger.info("Created Bundle For Checkpoint On Rank %s"%str(self.rank))
                self.save_checkpoint(
                    os.path.join(self.checkpoint_save_path,\
                                'Rank-'+str(self.rank),\
                                str(epoch)),
                    bundle
                )       
        if self.on_completion_checkpoint_validaiton():
            bundle = self.create_experiment_bundle(experiment_results)
            self.logger.info("Created Bundle For Checkpoint On Rank %s For Path %s"%(str(self.rank),self.checkpoint_save_path))
            self.save_checkpoint(
                os.path.join(self.checkpoint_save_path,\
                            'Rank-'+str(self.rank),\
                            str(epoch)),
                bundle
            )

    def on_completion_checkpoint_validaiton(self):
        if self.checkpoint_args.save_experiment:
            if self.checkpoint_args.checkpoint_all_ranks:
                return True 
            if self.rank == self.checkpoint_args.checkpoint_rank:
                return True 
        return False


    def train_loop_checkpoint_validation(self,epoch):
        if not self.checkpoint_args.save_experiment:
             return False
        
        if epoch % self.checkpoint_args.checkpoint_frequency==0:
            if self.checkpoint_args.checkpoint_all_ranks:
                return True 
            if self.rank == self.checkpoint_args.checkpoint_rank:
                return True 
        return False
          
    
    def get_train_dataloader(self,rank):
        tensor_dataset = self.dataset.get_train_dataset(rank)
        data_loader = torch.utils.data.DataLoader(
            tensor_dataset,\
            batch_size = self.training_args.batch_size,\
            shuffle = self.training_args.shuffle,\
        )
        return data_loader


    def create_experiment_bundle(self,train_experiment_results:List[ExperimentResultsBundle]) -> ExperimentBundle:
        dataset_meta = None
        try:
            dataset_meta = self.dataset.get_metadata()
        except:
            pass
        model = self.neural_network
        best_model_state_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
        model_dict = OrderedDict(best_model_state_dict)
        opt_dict = self.optimizer.state_dict()
        return ExperimentBundle(
            train_epoch_results = [dataclasses.asdict(res) for res in train_experiment_results],
            train_args = dataclasses.asdict(self.training_args),
            dataset_metadata = dataset_meta,
            model = model_dict,
            rank = self.rank,
            distributed=True,
            optimizer = opt_dict,
            model_args = self.network_args.model_args_dict,
            optimizer_args = self.network_args.optimizer_args_dict,
            loss_fn = str(self.loss_fn)
        )


    def sync_params(self):
        """ broadcast rank 0 parameter to all ranks """
        for param in self.neural_network.parameters():
            dist.broadcast(param.data, 0)

    def sync_grads(self):
        """ all_reduce grads from all ranks """
        size = float(dist.get_world_size())
        for param in self.neural_network.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size
        


def create_trainer(rank,\
                    world_size,\
                    network_args,\
                    training_args,\
                    dataset,\
                    dist_args,\
                    trainer):
    training_prog = trainer(
        network_args=network_args,
        training_args=training_args,
        dataset=dataset,
        dist_args=dist_args
    )
    training_prog.run(rank,world_size)

def train_distributed(
        world_size,
        network_args:NetworkArgs,\
        training_args:TrainerArgs,\
        dataset:DistributedDataset,\
        dist_args:DistTrainerArgs,
        trainer:DistributedTrainer       
    ):
    
    mp.spawn(create_trainer,\
            args=(world_size,\
                network_args,\
                training_args,\
                dataset,\
                dist_args,\
                trainer,
                ),\
            nprocs=world_size,\
            join=True
            )