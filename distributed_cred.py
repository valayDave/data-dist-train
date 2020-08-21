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
from fraud_dataset import FraudDataset,\
        DispatcherControlParams,\
        ALLOWED_DISPATCHING_METHODS

import fraud_dataset

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

from fraud_network import ModelFactory

import click

FACTORY = ModelFactory()
DEFAULT_MODEL,DEFAULT_ARGS = FACTORY.default
BACKEND_CHOICE = ['gloo','nccl']
DEFAULT_CHECKPOINT = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'model_data',
    'fraud_model'
)

class FraudDistributedTrainer(DistributedClassificationTrainer):
    def get_accuracy(self,output:torch.Tensor, target:torch.Tensor,conf_matrix):
        with torch.no_grad():
            pred = torch.round(output) 
            conf_matrix.update(pred.long(),target)
            pred = pred.t() # Transpose the pred value use in comparison
            num_correct_preds = pred.eq(target.view(1,-1).expand_as(pred)).view(-1).sum(0) # Compare the pred with the target
            return float(num_correct_preds)/output.shape[0]

class FraudTrainer(MonolithClassificationTrainer):
    
    def get_accuracy(self,output:torch.Tensor, target:torch.Tensor,conf_matrix):
        with torch.no_grad():
            pred = torch.round(output) # Convert softmax logits argmax based selection of index to get prediction value
            conf_matrix.update(pred.long(),target)
            pred = pred.t() # Transpose the pred value use in comparison
            num_correct_preds = pred.eq(target.view(1,-1).expand_as(pred)).view(-1).sum(0) # Compare the pred with the target
            return float(num_correct_preds)/output.shape[0]

class FraudTrainerArgs(TrainerArgs):
    pass
class FraudExpNetworkArgs(NetworkArgs):
    model = DEFAULT_MODEL
    loss_fn = nn.BCELoss
    model_args_dict = DEFAULT_ARGS
    optimizer = optim.Adam
    optimizer_args_dict = dict(
        lr = 1e-3
    )

@click.group()
def cli():
    pass

@cli.command(help='Train Credit Card Fraud Dataset With Distributed Training')
@click.option('--batch_size',default=128,help='Batch size For Training/Validation')
@click.option('--epochs',default=20,help='Number of Epochs For Training')
@click.option('--backend',default='gloo',help='Backend For Training',type=click.Choice(BACKEND_CHOICE))
@click.option('--master_ip',default='127.0.0.1',help='IP Address of the Master node which will synchronise for all reduce')
@click.option('--master_port',default='12355',help='Port of the Master node which will synchronise for all reduce')
@click.option('--learning_rate','--lr',default=0.0001,help='Learning Rate')
@click.option('--checkpoint_dir',default=DEFAULT_CHECKPOINT,help='Directory To publish Experiment Information')
@click.option('--dont_save',default=False,is_flag=True,help='Flag to specify weather to Save Results of The Experiment')
@click.option('--model',default='CNN',type=click.Choice(list(FACTORY.models)), help='Neural Network to Run the Experiment')
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
                model='CNN',
                world_size=5,
                note=None,
                block_size=2,
                dispatching_approach=ALLOWED_DISPATCHING_METHODS[0]
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
                model='CNN',
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
    distributed_dataset,_ = fraud_dataset.get_distributed_dataset(dispatcher_params)

    nnargs = FraudExpNetworkArgs()
    model_class,args = FACTORY.get_model(model)
    nnargs.model = model_class
    nnargs.model_args_dict = args
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    trainer_args = FraudTrainerArgs(
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

    click.secho('Starting Distributed Training With %s Workers'%(str(world_size)),fg='green',bold=True)
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
        FraudDistributedTrainer,
        note
    )


@cli.command(help='Train Credit Card Fraud Dataset With Monolith Dataparallel')
@click.option('--batch_size',default=128,help='Batch size For Training/Validation')
@click.option('--epochs',default=20,help='Number of Epochs For Training')
@click.option('--learning_rate','--lr',default=0.0001,help='Learning Rate')
@click.option('--checkpoint_dir',default=DEFAULT_CHECKPOINT,help='Directory To publish Experiment Information')
@click.option('--dont_save',default=False,is_flag=True,help='Flag to specify weather to Save Results of The Experiment')
@click.option('--sample',default=None,type=int,help='Sample N Values from the DistributedDataset')
@click.option('--model',default='CNN',type=click.Choice(list(FACTORY.models)), help='Neural Network to Run the Experiment')
@click.option('--note',default=None,type=str,help='Some Note to Add while Saving Experiment')
def monolith(batch_size=128,
            epochs=128,
            learning_rate=0.0001,
            checkpoint_dir=DEFAULT_CHECKPOINT,
            dont_save=False,
            model='CNN',
            sample=None,
            test_set_split=0.3,
            note=None):

    nnargs = FraudExpNetworkArgs()
    model_class,args = FACTORY.get_model(model)
    nnargs.model = model_class
    nnargs.model_args_dict = args
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    trainer_args = FraudTrainerArgs(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=epochs,
            checkpoint_args = CheckpointingArgs(
                path = checkpoint_dir,
                save_experiment=not dont_save,

        )
    )
    train_monolith(
        nnargs,
        trainer_args,
        FraudDataset(
            sample=sample,
        ),
        FraudTrainer,
        note
    )


if __name__=='__main__':
    cli()