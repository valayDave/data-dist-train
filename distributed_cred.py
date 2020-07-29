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
        FraudDistributedDataset,\
        FraudDataEngine,\
        SplitDataEngine
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
DEFAULT_DISTRIBUTION = 'n_5_b_2'
DATASET_CHOICES = ['n_5_b_2','n_5_b_112']
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
@click.option('--test_set_split',default=0.3,type=float,help='Percentage of the overall Dataset which will be used as a Test Set')
@click.option('--use_split',default=None,type=click.Choice(DATASET_CHOICES),help='Override The chosen Dataset')
@click.option('--note',default=None,type=str,help='Some Note to Add while Saving Experiment')
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
                test_set_split=0.3,
                use_split=None,
                note=None
                ):
    selected_dist = DEFAULT_DISTRIBUTION
    if use_split is not None:
        selected_dist = use_split

    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dataset-repo',
        selected_dist,
        'dispatcher_folder_credit',
    )
    data_engine = None
    if use_split is not None:
        click.secho("Using Split For Training : %s"%selected_dist,fg='green')
        world_size = 5
        data_engine = SplitDataEngine(
            DATASET_PATH,\
            DistributionArgs(\
                sample=sample,\
                uniform_label_distribution = None,\
                test_set_portion=test_set_split,
                selected_split = selected_dist
            ),\
            world_size=world_size
        )
    else:
        data_engine = FraudDataEngine(
            DATASET_PATH,\
            DistributionArgs(\
                sample=sample,\
                uniform_label_distribution = not non_uniform,\
                test_set_portion=test_set_split
            ),\
            world_size=world_size
        )

    nnargs = FraudExpNetworkArgs()
    model_class,args = FACTORY.get_model(model)
    nnargs.model = model_class
    nnargs.model_args_dict = args
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    trainer_args =FraudTrainerArgs(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=epochs,
            checkpoint_args = CheckpointingArgs(
                path = checkpoint_dir,
                save_experiment=not dont_save,

        )
    )
    json_str = json.dumps(dataclasses.asdict(data_engine.dist_args),indent=4)

    args_str = '''
    Training Stats : 
        Batch Size : {batch_size}\n
        Learning Rate : {learning_rate}\n
        Number of Epochs : {num_epochs}\n
        Worker Label Distribution : {distribution}\n
        Data Distribution Args :\n
        {data_dist_args}\n
    '''.format(**dict(
        batch_size=str(batch_size),
        num_epochs=str(epochs),
        learning_rate=str(learning_rate),
        distribution='Uniform' if not non_uniform else 'Non-Uniform',
        data_dist_args=json_str.replace('\t','\t\t')
    ))

    click.secho('Starting Distributed Training With %s Workers'%(str(world_size)),fg='green',bold=True)
    click.secho(args_str+'\n\n',fg='magenta')

    train_distributed(
        world_size,
        nnargs,
        trainer_args,
        data_engine.get_distributed_dataset(),
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
@click.option('--test_set_split',default=0.3,type=float,help='Percentage of the overall Dataset which will be used as a Test Set')
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

    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dataset-repo',
        DEFAULT_DISTRIBUTION,
        'dispatcher_folder_credit',
    )

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
            DATASET_PATH,
            sample=sample,
            test_split=test_set_split
        ),
        FraudTrainer,
        note
    )


if __name__=='__main__':
    cli()