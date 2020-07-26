import os
import torch
import pandas
import time
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from distributed_trainer import \
    NetworkArgs,\
    TrainerArgs,\
    DistTrainerArgs,\
    ClassificationTrainer,\
    DistributedDataset,\
    train_distributed,\
    CheckpointingArgs

import click
from utils import *



class FraudDataset(DistributedDataset):
    
    def __init__(self,data_path,sample=None):
        self.data_path = data_path
        self.sample = sample

    def get_train_dataset(self,rank)->torch.utils.data.TensorDataset:
        final_path = os.path.join(self.data_path,'output'+str(rank+1)+'.csv')
        df = pandas.read_csv(final_path)
        return self.dataset_transform(df,sample=self.sample)
    
    def get_test_dataset(self,rank)->torch.utils.data.TensorDataset:
        raise NotImplementedError()
    
    @staticmethod
    def dataset_transform(df,sample=None):
        if sample is not None:
            df = df.sample(sample)
        X = df.iloc[:, 1:-1].values # extracting features
        y = df.iloc[:, -1].values # extracting labels
        
        sc = StandardScaler()
        X = sc.fit_transform(X)
        partition = torch.from_numpy(X).unsqueeze(1).double()
        labels = torch.from_numpy(y).view(-1,1).double()
        # print('LABELS',labels.shape)
        part = torch.utils.data.TensorDataset(partition,labels)
        return part

    def get_labels(self):
        return [0,1]

    def model_alterations(self,model):
        return model.double()



class FraudModel(nn.Module):
    def __init__(self,input_length=30):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(32,affine=True),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64,affine=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128,affine=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128*input_length,512),
            nn.Dropout(0.5),
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        op = self.model(x)
        return op
        

class FraudExpNetworkArgs(NetworkArgs):
    model = FraudModel
    loss_fn = nn.BCELoss
    model_args_dict = {
        'input_length':30
    }
    optimizer = optim.Adam
    optimizer_args_dict = dict(
        lr = 1e-3
    )

class FraudTrainerArgs(TrainerArgs):
    batch_size=128
    shuffle=True
    num_epochs=10

DEFAULT_DISTRIBUTION = 'n_5_b_2'
DATASET_CHOICES = ['n_5_b_2','n_5_b_112','n_5_b_130']
BACKEND_CHOICE = ['gloo','nccl']
DEFAULT_CHECKPOINT = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'model_data',
    'fraud_model'
)

@click.command(help='Train Credit Card Fraud Dataset With Distributed Training')
@click.option('--selected_dataset',default='n_5_b_2',type=click.Choice(DATASET_CHOICES),help='Dataset To Run the Model On.')
@click.option('--batch_size',default=128,help='Batch size For Training/Validation')
@click.option('--epochs',default=20,help='Number of Epochs For Training')
@click.option('--backend',default='gloo',help='Backend For Training',type=click.Choice(BACKEND_CHOICE))
@click.option('--master_ip',default='127.0.0.1',help='IP Address of the Master node which will synchronise for all reduce')
@click.option('--master_port',default='12355',help='Port of the Master node which will synchronise for all reduce')
@click.option('--learning_rate','--lr',default=0.0001,help='Learning Rate')
@click.option('--checkpoint_dir',default=DEFAULT_CHECKPOINT,help='Directory To publish Experiment Information')
@click.option('--dont_save',default=False,is_flag=True,help='Flag to specify weather to Save Results of The Experiment')
@click.option('--sample',default=None,type=int,help='Sample N Values from the DistributedDataset')
def call_trainer(selected_dataset=DEFAULT_DISTRIBUTION,\
                batch_size=128,
                epochs=128,
                backend='gloo',
                master_ip='127.0.0.1',
                master_port='12355',
                learning_rate=0.0001,
                checkpoint_dir=DEFAULT_CHECKPOINT,
                dont_save=False,
                sample=None
                ):
    
    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dataset-repo',
        selected_dataset,
        'dispatcher_folder_credit',
    )

    nnargs = FraudExpNetworkArgs()
    nnargs.optimizer_args_dict = {
        'lr': learning_rate
    }
    train_distributed(
        5,
        nnargs,
        FraudTrainerArgs(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=epochs,
            checkpoint_args = CheckpointingArgs(
                path = checkpoint_dir,
                save_experiment=not dont_save,

            )
        ),
        FraudDataset(
            DATASET_PATH,
            sample=sample
        ),
        DistTrainerArgs(
            backend=backend,
            master_ip=master_ip,
            master_port=master_port,
            world_size=5,
        ),
        ClassificationTrainer
    )



if __name__=='__main__':
    call_trainer()