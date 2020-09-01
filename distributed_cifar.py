import os
import torch
import pandas
import time
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
import dataclasses 
import json
from sklearn.preprocessing import StandardScaler
from cifar_dataset import CifarBlockDataset,\
        DispatcherControlParams,\
        ALLOWED_DISPATCHING_METHODS,\
        DistributedSampler

import cifar_dataset

from distributed_trainer import \
    NetworkArgs,\
    safe_mkdir,\
    TrainerArgs,\
    DistTrainerArgs,\
    DistributedClassificationTrainer,\
    train_distributed,\
    train_monolith,\
    CheckpointingArgs,\
    DistributionArgs,\
    MonolithClassificationTrainer

from cifar_models import ModelFactory

import click

FACTORY = ModelFactory()
DEFAULT_MODEL,DEFAULT_ARGS = FACTORY.default
BACKEND_CHOICE = ['gloo','nccl']
DEFAULT_CHECKPOINT = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'model_data',
    'cifar_model'
)

class CifarDistributedTrainer(DistributedClassificationTrainer):
    def get_accuracy(self,output:torch.Tensor, target:torch.Tensor,conf_matrix):
        with torch.no_grad():
            _, pred = output.max(1) 
            conf_matrix.update(pred.view(-1,1).long(),target)
            num_correct_preds = pred.eq(target).sum(0) # Compare the pred with the target
            return float(num_correct_preds)/output.shape[0]


class CifarExpNetworkArgs(NetworkArgs):
    model = DEFAULT_MODEL
    loss_fn = nn.CrossEntropyLoss
    model_args_dict = DEFAULT_ARGS
    optimizer = optim.Adam
    optimizer_args_dict = dict(
        lr = 1e-3
    )

@click.group()
def cli():
    pass

@cli.command(help='Train CIFAR-10 Dataset With Distributed Training')
@click.option('--batch_size',default=128,help='Batch size For Training/Validation')
@click.option('--epochs',default=20,help='Number of Epochs For Training')
@click.option('--backend',default='gloo',help='Backend For Training',type=click.Choice(BACKEND_CHOICE))
@click.option('--master_ip',default='127.0.0.1',help='IP Address of the Master node which will synchronise for all reduce')
@click.option('--master_port',default='12355',help='Port of the Master node which will synchronise for all reduce')
@click.option('--learning_rate','--lr',default=0.0001,help='Learning Rate')
@click.option('--checkpoint_dir',default=DEFAULT_CHECKPOINT,help='Directory To publish Experiment Information')
@click.option('--dont_save',default=False,is_flag=True,help='Flag to specify weather to Save Results of The Experiment')
@click.option('--model',default='ResNet18',type=click.Choice(list(FACTORY.models)), help='Neural Network to Run the Experiment')
@click.option('--non_uniform',default=False,is_flag=True,help='Flag to specify To have Uniform Splits or None Uniform Splits of the data')
@click.option('--sample',default=None,type=int,help='Sample N Values from the DistributedDataset')
@click.option('--world_size',default=5,type=int,help='Number of Distributed Processes for Distributed Training')
@click.option('--note',default=None,type=str,help='Some Note to Add while Saving Experiment')
@click.option('--block_size',default=2,type=int,help='Block Size For the Dispatcher')
@click.option('--dispatching_approach','--da',default=ALLOWED_DISPATCHING_METHODS[0],type=click.Choice(ALLOWED_DISPATCHING_METHODS),help='Approach to take for Dispatching data')
def distributed(\
                batch_size=128,
                epochs=128,
                backend='gloo',
                master_ip='127.0.0.1',
                master_port='12355',
                learning_rate=0.0001,
                checkpoint_dir=DEFAULT_CHECKPOINT,
                dont_save=False,
                sample=None,
                non_uniform=False,
                model='ResNet18',
                world_size=5,
                note=None,
                block_size=2,
                dispatching_approach=ALLOWED_DISPATCHING_METHODS[0],
                ):
    run_dist_trainer(
        batch_size = batch_size,
        epochs = epochs,
        backend = backend,
        master_ip = master_ip,
        master_port = master_port,
        learning_rate = learning_rate,
        checkpoint_dir = checkpoint_dir,
        dont_save = dont_save,
        sample = sample,
        non_uniform = non_uniform,
        model = model,
        world_size = world_size,
        note = note,
        block_size = block_size,
        dispatching_approach = dispatching_approach,
    )

@cli.command(help='Train CIFAR-10 Dataset With Distributed Training And Shuffling data across workers at Every Epoch')
@click.option('--batch_size',default=128,help='Batch size For Training/Validation')
@click.option('--epochs',default=20,help='Number of Epochs For Training')
@click.option('--backend',default='gloo',help='Backend For Training',type=click.Choice(BACKEND_CHOICE))
@click.option('--master_ip',default='127.0.0.1',help='IP Address of the Master node which will synchronise for all reduce')
@click.option('--master_port',default='12355',help='Port of the Master node which will synchronise for all reduce')
@click.option('--learning_rate','--lr',default=0.0001,help='Learning Rate')
@click.option('--checkpoint_dir',default=DEFAULT_CHECKPOINT,help='Directory To publish Experiment Information')
@click.option('--dont_save',default=False,is_flag=True,help='Flag to specify weather to Save Results of The Experiment')
@click.option('--model',default='ResNet18',type=click.Choice(list(FACTORY.models)), help='Neural Network to Run the Experiment')
@click.option('--non_uniform',default=False,is_flag=True,help='Flag to specify To have Uniform Splits or None Uniform Splits of the data')
@click.option('--sample',default=None,type=int,help='Sample N Values from the DistributedDataset')
@click.option('--world_size',default=5,type=int,help='Number of Distributed Processes for Distributed Training')
@click.option('--note',default=None,type=str,help='Some Note to Add while Saving Experiment')
@click.option('--block_size',default=2,type=int,help='Block Size For the Dispatcher')
@click.option('--sampler_host',default='127.0.0.1',help='IP address of the Remote Datastore where the Trainer will get the Dataset from at Each epoch')
@click.option('--sampler_port',default=5003,help='Port of the Remote Datastore where the Trainer will get the Dataset from at Each epoch')
def distributed_global_shuffle(\
                batch_size=128,
                epochs=128,
                backend='gloo',
                master_ip='127.0.0.1',
                master_port='12355',
                learning_rate=0.0001,
                checkpoint_dir=DEFAULT_CHECKPOINT,
                dont_save=False,
                sample=None,
                non_uniform=False,
                model='ResNet18',
                world_size=5,
                note=None,
                block_size=2,
                sampler_host='127.0.0.1',
                sampler_port=5003
                ):
    run_dist_trainer_global_suffle(
        batch_size = batch_size,
        epochs = epochs,
        backend = backend,
        master_ip = master_ip,
        master_port = master_port,
        learning_rate = learning_rate,
        checkpoint_dir = checkpoint_dir,
        dont_save = dont_save,
        sample = sample,
        non_uniform = non_uniform,
        model = model,
        world_size = world_size,
        note = note,
        block_size=block_size,
        sampler_host = sampler_host,
        sampler_port = sampler_port
        )


def run_trainer_from_dataset(batch_size=128,
                epochs=128,
                backend='gloo',
                master_ip='127.0.0.1',
                master_port='12355',
                learning_rate=0.0001,
                checkpoint_dir=DEFAULT_CHECKPOINT,
                dont_save=False,
                non_uniform=False,
                model='ResNet18',
                world_size=5,
                note=None,
                distributed_dataset=None):
    
    nnargs = CifarExpNetworkArgs()
    model_class,args = FACTORY.get_model(model)
    nnargs.model = model_class
    nnargs.model_args_dict = args
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    trainer_args = TrainerArgs(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=epochs,
            checkpoint_args = CheckpointingArgs(
                path = checkpoint_dir,
                save_experiment=not dont_save,
            ),
            global_shuffle=True,
    )
    args_str = '''
    Training Stats : 
        Batch Size : {batch_size}\n
        Learning Rate : {learning_rate}\n
        Number of Epochs : {num_epochs}\n
        Dispatcher Args :\n
    '''.format(**dict(
        batch_size=str(batch_size),
        num_epochs=str(epochs),
        learning_rate=str(learning_rate)
    ))

    click.secho('Starting Distributed %s Training With %s Workers'%(model,str(world_size)),fg='green',bold=True)
    click.secho(args_str+'\n\n',fg='magenta')
    print("Master Port : ",master_port)
    return train_distributed(
        world_size,
        nnargs,
        trainer_args,
        distributed_dataset,
        DistTrainerArgs(
            backend=backend,
            master_ip=master_ip,
            master_port=master_port,
            world_size=world_size,
        ),
        CifarDistributedTrainer,
        note
    )


def run_dist_trainer(batch_size=128,
                epochs=128,
                backend='gloo',
                master_ip='127.0.0.1',
                master_port='12355',
                learning_rate=0.0001,
                checkpoint_dir=DEFAULT_CHECKPOINT,
                dont_save=False,
                sample=None,
                non_uniform=False,
                model='ResNet18',
                world_size=5,
                note=None,
                block_size = 2,
                dispatching_approach = ALLOWED_DISPATCHING_METHODS[0]):

    dispatcher_params = DispatcherControlParams(
        num_workers=world_size,
        sample=sample,
        approach=dispatching_approach,
        block_size=block_size
    )
    distributed_dataset,_ = cifar_dataset.get_distributed_dataset(dispatcher_params)

    nnargs = CifarExpNetworkArgs()
    model_class,args = FACTORY.get_model(model)
    nnargs.model = model_class
    nnargs.model_args_dict = args
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    trainer_args = TrainerArgs(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=epochs,
            checkpoint_args = CheckpointingArgs(
                path = checkpoint_dir,
                save_experiment=not dont_save,
        )
    )
    json_str = json.dumps(dataclasses.asdict(dispatcher_params),indent=4)

    args_str = '''
    Training Stats : 
        Batch Size : {batch_size}\n
        Learning Rate : {learning_rate}\n
        Number of Epochs : {num_epochs}\n
        Dispatcher Args :\n
        {dispatcher_params}\n
    '''.format(**dict(
        batch_size=str(batch_size),
        num_epochs=str(epochs),
        learning_rate=str(learning_rate),
        dispatcher_params=json_str.replace('\t','\t\t')
    ))

    click.secho('Starting Distributed %s Training With %s Workers'%(model,str(world_size)),fg='green',bold=True)
    click.secho(args_str+'\n\n',fg='magenta')

    return train_distributed(
        world_size,
        nnargs,
        trainer_args,
        distributed_dataset,
        DistTrainerArgs(
            backend=backend,
            master_ip=master_ip,
            master_port=master_port,
            world_size=world_size,
        ),
        CifarDistributedTrainer,
        note
    )



def run_dist_trainer_global_suffle(
    batch_size=128,
    epochs=128,
    backend='gloo',
    master_ip='127.0.0.1',
    master_port='12355',
    learning_rate=0.0001,
    checkpoint_dir=DEFAULT_CHECKPOINT,
    dont_save=False,
    sample=None,
    non_uniform=False,
    model='ResNet18',
    world_size=5,
    note=None,
    block_size=2,
    sampler_host='127.0.0.1',
    sampler_port=5003
    ):
    train_set,test_set = cifar_dataset.create_dataset()
    sampler_session_id = DistributedSampler.create_session(len(train_set),world_size,block_size,host=sampler_host,port=sampler_port)
    distributed_sampler = DistributedSampler(
        train_set,
        block_size=block_size,
        num_workers=world_size,
        host=sampler_host,
        test_set=test_set,
        port=sampler_port,
        UsedClass=cifar_dataset.CifarBlockDataset,
        connection_id=sampler_session_id
    )
    distributed_dataset = distributed_sampler.get_distributed_dataset()

    nnargs = CifarExpNetworkArgs()
    model_class,args = FACTORY.get_model(model)
    nnargs.model = model_class
    nnargs.model_args_dict = args
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    trainer_args = TrainerArgs(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=epochs,
            checkpoint_args = CheckpointingArgs(
                path = checkpoint_dir,
                save_experiment=not dont_save,
        )
    )

    args_str = '''
    Training Stats : 
        Batch Size : {batch_size}\n
        Learning Rate : {learning_rate}\n
        Number of Epochs : {num_epochs}\n

    '''.format(**dict(
        batch_size=str(batch_size),
        num_epochs=str(epochs),
        learning_rate=str(learning_rate)
    ))

    click.secho('Starting Distributed %s Training With %s Workers'%(model,str(world_size)),fg='green',bold=True)
    click.secho(args_str+'\n\n',fg='magenta')
    return train_distributed(
        world_size,
        nnargs,
        trainer_args,
        distributed_dataset,
        DistTrainerArgs(
            backend=backend,
            master_ip=master_ip,
            master_port=master_port,
            world_size=world_size,
            global_shuffle=True
        ),
        CifarDistributedTrainer,
        note,
        distributed_sampler=distributed_sampler
    )




if __name__=='__main__':
    cli()