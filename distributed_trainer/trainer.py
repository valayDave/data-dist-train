import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from dataclasses import dataclass
import torch.multiprocessing as mp
from typing import List,Tuple
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
    safe_mkdir,\
    DistributionArgs,\
    ModelBundle

MODEL_META_FILENAME='model_checkpoint.pt'
MODEL_FILENAME = 'model_results.pt'
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
        self.checkpoint_save_path = os.path.join(checkpoint_base_path,self.__class__.__name__,exp_date)
        self.logger = None

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
    def test_loop(self,test_data_loader) -> ExperimentResultsBundle:
        """train_loop
        Run the Testing Loop
        """
        return None

    # override in DistTrainer        
    def get_train_dataloader(self):
        tensor_dataset = self.dataset.get_train_dataset()
        data_loader = torch.utils.data.DataLoader(
            tensor_dataset,\
            batch_size = self.training_args.batch_size,\
            shuffle = self.training_args.shuffle,\
        )
        return data_loader

    # override in DistTrainer
    def get_test_dataloader(self):
        tensor_dataset = self.dataset.get_test_dataset()
        data_loader = torch.utils.data.DataLoader(
            tensor_dataset,\
            batch_size = self.training_args.batch_size,\
            shuffle = self.training_args.shuffle,\
        )
        return data_loader

    def setup(self):
        proc_name = self.__class__.__name__
        self.logger = create_logger(proc_name)

    # override in DistTrainer
    def run(self):
        self.setup()
        experiment_results = []
        validation_results = []
        self.initialise_network()
        self.setup_gpu_dp()
        train_data_loader = self.get_train_dataloader()
        test_data_loader = self.get_test_dataloader()
        self.neural_network = self.dataset.model_alterations(self.neural_network) # This needs Fixes. 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = device
        for epoch in range(self.training_args.num_epochs):
            self.logger.info("Starting Epoch %d With %s And Dataset Size %s"%(epoch,'GPU' if self.gpu_enabled else 'CPU',str(len(train_data_loader))))
            results_bundle = self.train_loop(train_data_loader)
            validation_bundle = self.test_loop(test_data_loader)
            results_bundle.epoch = epoch
            experiment_results.append(results_bundle)
            if validation_bundle is not None:
                validation_bundle.epoch = epoch
                validation_results.append(validation_bundle)

            if self.checkpoint_args.save_experiment:
                if epoch % self.checkpoint_args.checkpoint_frequency==0:
                    exp_bundle,model_bundle = self.create_experiment_bundle(experiment_results,validation_results)
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_save_path,str(epoch)),
                        exp_bundle,\
                        model_bundle
                    )

        if self.checkpoint_args.save_experiment:
            self.logger.info("Created Bundle For Checkpoint")
            exp_bundle,model_bundle = self.create_experiment_bundle(experiment_results,validation_results)
            self.save_checkpoint(
                os.path.join(self.checkpoint_save_path,'completion'),
                exp_bundle,
                model_bundle,\
            )

    # override in DistTrainer
    def create_experiment_bundle(self,\
                                train_experiment_results:List[ExperimentResultsBundle],\
                                validation_results:List[ExperimentResultsBundle]) -> Tuple[ExperimentBundle,ModelBundle]:
        dataset_meta = None
        try:
            dataset_meta = self.dataset.get_metadata()
        except:
            pass
        model_dict = self.neural_network.state_dict() if not self.gpu_enabled else self.neural_network.module.state_dict()
        opt_dict = self.optimizer.state_dict()
        model_bundle = ModelBundle(
            model = model_dict,
            model_name = self.neural_network.__class__.__name__,
            optimizer = opt_dict,
            model_args = self.network_args.model_args_dict,
            optimizer_args = self.network_args.optimizer_args_dict,
            loss_fn = str(self.loss_fn),
            train_args = dataclasses.asdict(self.training_args),
        )
        experiment_bundle = ExperimentBundle(
            train_epoch_results = [dataclasses.asdict(res) for res in train_experiment_results],
            validation_epoch_results=[dataclasses.asdict(res) for res in validation_results],
            train_args = dataclasses.asdict(self.training_args),
            dataset_metadata = dataset_meta,
        )
        return (experiment_bundle,model_bundle)

    @staticmethod
    def save_checkpoint(model_save_path,\
                        exp_bundle:ExperimentBundle,\
                        model_bundle:ModelBundle,\
                        model_meta_name=MODEL_META_FILENAME,\
                        model_checkpoint_name=MODEL_FILENAME):
        safe_mkdir(model_save_path)
        exp_bundle_dict = dataclasses.asdict(exp_bundle)
        model_bundle_dict = dataclasses.asdict(model_bundle)
        torch.save(
            exp_bundle_dict,
            os.path.join(model_save_path,model_meta_name)
        )
        torch.save(
            model_bundle_dict,
            os.path.join(model_save_path,model_checkpoint_name)
        )
    
    def setup_gpu_dp(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            self.neural_network = nn.DataParallel(self.neural_network)
            self.neural_network.to(device)
            self.gpu_enabled = True
        else : 
            self.neural_network.to(device)
    
    def get_accuracy(self,output:torch.Tensor, target:torch.Tensor,conf_matrix:ConfusionMatrix):
        raise NotImplementedError()
        
class DistributedTrainer(BaseTrainer):
    """DistributedTrainer 

    """
    def __init__(self,dist_args=DistTrainerArgs(),**kwargs):
        self.dist_args = dist_args
        super(DistributedTrainer, self).__init__(**kwargs)
        self.rank = None
        

    def setup(self,rank,world_size):
        proc_name = self.__class__.__name__+'__'+str(rank)
        self.logger = create_logger(proc_name)
        os.environ['MASTER_ADDR'] = self.dist_args.master_ip
        os.environ['MASTER_PORT'] = self.dist_args.master_port
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
        test_data_loader = self.get_test_dataloader(rank)
        self.neural_network = self.dataset.model_alterations(self.neural_network) # This needs Fixes. 
        self.rank = rank
        experiment_results = []
        validation_results = []
        for epoch in range(self.training_args.num_epochs):
            self.logger.info("Starting Epoch %d With %s And Dataset Size %s"%(epoch,'GPU' if self.gpu_enabled else 'CPU',str(len(train_data_loader))))
            results_bundle = self.train_loop(train_data_loader)# Rank can be accessed as a property. 
            validation_bundle = self.test_loop(test_data_loader)
            results_bundle.epoch = epoch
            experiment_results.append(results_bundle) 
            if validation_bundle is not None:
                validation_bundle.epoch = epoch
                validation_results.append(validation_bundle)

            if self.train_loop_checkpoint_validation(epoch):
                exp_bundle,model_bundle = self.create_experiment_bundle(experiment_results,validation_results)
                self.logger.info("Created Bundle For Checkpoint On Rank %s"%str(self.rank))
                self.save_checkpoint(
                    os.path.join(self.checkpoint_save_path,\
                                'Rank-'+str(self.rank),\
                                'completion'),
                    exp_bundle,
                    model_bundle
                )       
        if self.on_completion_checkpoint_validaiton():
            exp_bundle,model_bundle = self.create_experiment_bundle(experiment_results,validation_results)
            self.logger.info("Created Bundle For Checkpoint On Rank %s For Path %s"%(str(self.rank),self.checkpoint_save_path))
            self.save_checkpoint(
                os.path.join(self.checkpoint_save_path,\
                            'Rank-'+str(self.rank),\
                            str(epoch)),
                exp_bundle,
                model_bundle
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

    def get_test_dataloader(self,rank):
        tensor_dataset = self.dataset.get_test_dataset(rank)
        data_loader = torch.utils.data.DataLoader(
            tensor_dataset,\
            batch_size = self.training_args.batch_size,\
            shuffle = self.training_args.shuffle,\
        )
        return data_loader


    def create_experiment_bundle(self,\
                    train_experiment_results:List[ExperimentResultsBundle],\
                    validation_results:List[ExperimentResultsBundle]\
                    ) -> Tuple[ExperimentBundle,ModelBundle]:
        dataset_meta = None
        try:
            dataset_meta = self.dataset.get_metadata()
        except:
            pass
        # self.neural_network.to('cpu')
        model_dict = self.neural_network.state_dict()
        # self.neural_network.to(self.rank)
        opt_dict = self.optimizer.state_dict()
        model_bundle = ModelBundle(
            model_name = self.neural_network.__class__.__name__,
            model = model_dict,
            optimizer = opt_dict,
            model_args = self.network_args.model_args_dict,
            optimizer_args = self.network_args.optimizer_args_dict,
            loss_fn = str(self.loss_fn),
            train_args = dataclasses.asdict(self.training_args),

        )
        experimental_bundle = ExperimentBundle(
            train_epoch_results = [dataclasses.asdict(res) for res in train_experiment_results],
            validation_epoch_results=[dataclasses.asdict(res) for res in validation_results],
            train_args = dataclasses.asdict(self.training_args),
            dataset_metadata = dataset_meta,
            rank = self.rank,
            distributed=True
        )
        return (experimental_bundle,model_bundle)


    def sync_params(self):
        """ broadcast rank 0 parameter to all ranks """
        for param in self.neural_network.parameters():
            dist.broadcast(param.data, 0)

    def sync_grads(self):
        """ all_reduce grads from all ranks """
        size = float(dist.get_world_size())
        for param in self.neural_network.parameters():
            dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
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


def train_monolith(
        network_args:NetworkArgs,\
        training_args:TrainerArgs,\
        dataset:DistributedDataset,\
        trainer:BaseTrainer
        ):
    training_prog = trainer(
        network_args=network_args,
        training_args=training_args,
        dataset=dataset,
    )
    training_prog.run()


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