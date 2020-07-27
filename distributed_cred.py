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
from distributed_trainer import \
    NetworkArgs,\
    safe_mkdir,\
    TrainerArgs,\
    DistTrainerArgs,\
    DistributedClassificationTrainer,\
    DistributedDataset,\
    Dataset,\
    train_distributed,\
    train_monolith,\
    CheckpointingArgs,\
    DistributionArgs,\
    MonolithClassificationTrainer

import click
from utils import *

class FraudDistributedTrainer(DistributedClassificationTrainer):
    pass

class FraudTrainer(MonolithClassificationTrainer):
    pass

class FraudData:
    column_values = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5',
       'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
       'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
       'V27', 'V28', 'Amount']

    def dataset_transform(self,df,sample=None):
        if sample is not None:
            df = df.sample(sample)
        y = df['Class'].values # extracting labels
        df = df.drop('Class', axis=1)
        X = df[self.column_values].values # extracting features        
        sc = StandardScaler()
        X = sc.fit_transform(X)
        partition = torch.from_numpy(X).unsqueeze(1).double()
        labels = torch.from_numpy(y).view(-1,1).double()
        part = torch.utils.data.TensorDataset(partition,labels)
        return part
    
    @staticmethod
    def create_mask(df,prob=0.3):
        msk = np.random.rand(len(df)) < prob
        return msk

class FraudDistributedDataset(DistributedDataset,FraudData):
    
    def __init__(self,data_path,sample=None,metadata=None):
        self.data_path = data_path
        self.sample = sample
        self.metadata = metadata

    def get_train_dataset(self,rank)->torch.utils.data.TensorDataset:
        final_path = os.path.join(self.data_path,'output'+str(rank+1)+'.csv')
        df = pandas.read_csv(final_path)
        return self.dataset_transform(df,sample=self.sample)
    
    def get_test_dataset(self,rank)->torch.utils.data.TensorDataset:
        final_path = os.path.join(self.data_path,'output-test.csv')
        df = pandas.read_csv(final_path)
        return self.dataset_transform(df,sample=self.sample)
    
    def get_labels(self):
        return [0,1]

    def get_metadata(self):
        return self.metadata

    def model_alterations(self,model):
        return model.double()

class FraudDataEngine(FraudData):
    def __init__(self,data_path,distirbution_args:DistributionArgs,world_size=5,temp_path=None):
        super().__init__()
        self.data_path = data_path
        self.dist_args = distirbution_args
        if temp_path is None:
            self.temp_path = os.path.join(os.path.abspath(data_path),'experiment_data_shards')
        else:
            self.temp_path = temp_path
        self.world_size = world_size
        safe_mkdir(self.root_path)
        self.setup_shards()
    
    def get_full_dataframe(self):
        csv_file_paths = [os.path.join(self.data_path,'output'+str(i+1)+'.csv') for i in range(5)]    
        dfs = [pandas.read_csv(i) for i in csv_file_paths]
        df = pandas.concat(dfs)
        return df

    def write_to_path(self,df,index):
        path = os.path.join(self.root_path,'output'+str(index)+'.csv')
        df.to_csv(path)
    
    @property
    def root_path(self):
        path = os.path.join(self.temp_path,str(self.world_size))
        return path

    def setup_shards(self):
        df = self.get_full_dataframe()
        non_fraud = df[df['Class']==0]
        fraud = df[df['Class']==1]
        test_set_portion = 1 - self.dist_args.test_set_portion
        fraud_mask = self.create_mask(fraud,test_set_portion) 
        non_fraud_mask = self.create_mask(non_fraud,test_set_portion)
        
        fraud_test = fraud[~fraud_mask]
        non_fraud_test = non_fraud[~non_fraud_mask]
        # save Test dataframe
        self.write_to_path(pandas.concat([fraud_test,non_fraud_test]),'-test')
        # Split the main dataframe. 
        fraud = fraud[fraud_mask]
        non_fraud = non_fraud[non_fraud_mask]
        
        if self.dist_args.uniform_label_distribution:
            non_fraud_dfs = np.array_split(non_fraud,self.world_size)
            fraud_dfs = np.array_split(fraud,self.world_size)
            index = 0
            for fr_df,nfr_df in zip(non_fraud_dfs,fraud_dfs):
                write_df = pandas.concat([fr_df,nfr_df])
                self.write_to_path(write_df,index+1)
                index+=1
        else:
            split_vals = self.dist_args.label_split_values
            
            if len(split_vals) == 0 or len(split_vals) > self.world_size or len(split_vals) < self.world_size:
                rand_dist = np.random.randint(0,1000,(self.world_size))
                prob_dist = rand_dist/np.sum(rand_dist)
                prob_dist = prob_dist.tolist()
            else:
                prob_dist = self.dist_args.label_split_values
            self.dist_args.label_split_values = prob_dist
            non_fraud_dfs = np.array_split(non_fraud,self.world_size)
            for index,prob in enumerate(prob_dist):
                fraud_sample_df = fraud.sample(frac=prob,random_state=200)
                fraud.drop(fraud_sample_df.index,axis=0,inplace=True)
                write_df = pandas.concat([fraud_sample_df,non_fraud_dfs.pop()])
                self.write_to_path(write_df,index+1)

    def get_distibuted_dataset(self):
        return FraudDistributedDataset(self.root_path,self.dist_args.sample,dataclasses.asdict(self.dist_args))

class FraudDataset(Dataset,FraudData):
    def __init__(self,data_path,sample=None,world_size=5,test_split=0.3):
        self.data_path = data_path
        self.sample = sample
        self.world_size = world_size
        self.dataset = None
        self.test_split = test_split
        self.train = None
        self.test = None

    def get_train_dataset(self)->torch.utils.data.TensorDataset:
        if self.train is not None:
            return self.train
        self.set_attributes()
        return self.train
    
    def set_attributes(self):
        df = self.get_pandas_frame()
        test_set_portion = 1 - self.test_split
        df_mask = self.create_mask(df,test_set_portion)
        test_df = df[~df_mask]
        train_df = df[df_mask]
        self.train = self.dataset_transform(train_df,sample=self.sample)
        self.test = self.dataset_transform(test_df,sample=self.sample)
    
    def get_pandas_frame(self):
        csv_file_paths = [os.path.join(self.data_path,'output'+str(i+1)+'.csv') for i in range(self.world_size)]    
        dfs = [pandas.read_csv(i) for i in csv_file_paths]
        df = pandas.concat(dfs)
        return df

    def get_test_dataset(self)->torch.utils.data.TensorDataset:
        if self.test is not None:
            return self.test
        self.set_attributes()
        return self.test
    
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
@click.option('--non_uniform',default=False,is_flag=True,help='Flag to specify To have Uniform Splits or None Uniform Splits of the data')
@click.option('--sample',default=None,type=int,help='Sample N Values from the DistributedDataset')
@click.option('--world_size',default=5,type=int,help='Number of Distributed Processes for Distributed Training')
@click.option('--test_set_split',default=0.3,type=float,help='Percentage of the overall Dataset which will be used as a Test Set')
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
                world_size=5,
                test_set_split=0.3
                ):
    
    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dataset-repo',
        DEFAULT_DISTRIBUTION,
        'dispatcher_folder_credit',
    )
    data_engine = FraudDataEngine(
        DATASET_PATH,\
        DistributionArgs(\
            sample=sample,\
            uniform_label_distribution = not non_uniform,\
            test_set_portion=test_set_split
        ),\
        world_size=world_size)

    nnargs = FraudExpNetworkArgs()
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
        data_engine.get_distibuted_dataset(),
        DistTrainerArgs(
            backend=backend,
            master_ip=master_ip,
            master_port=master_port,
            world_size=world_size,
        ),
        FraudDistributedTrainer
    )


@cli.command(help='Train Credit Card Fraud Dataset With Monolith Dataparallel')
@click.option('--selected_dataset',default='n_5_b_2',type=click.Choice(DATASET_CHOICES),help='Dataset To Run the Model On.')
@click.option('--batch_size',default=128,help='Batch size For Training/Validation')
@click.option('--epochs',default=20,help='Number of Epochs For Training')
@click.option('--learning_rate','--lr',default=0.0001,help='Learning Rate')
@click.option('--checkpoint_dir',default=DEFAULT_CHECKPOINT,help='Directory To publish Experiment Information')
@click.option('--dont_save',default=False,is_flag=True,help='Flag to specify weather to Save Results of The Experiment')
@click.option('--sample',default=None,type=int,help='Sample N Values from the DistributedDataset')
@click.option('--test_set_split',default=0.3,type=float,help='Percentage of the overall Dataset which will be used as a Test Set')
def monolith(selected_dataset=DEFAULT_DISTRIBUTION,\
                batch_size=128,
                epochs=128,
                learning_rate=0.0001,
                checkpoint_dir=DEFAULT_CHECKPOINT,
                dont_save=False,
                sample=None,
                test_set_split=0.3):

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
        FraudTrainer
    )



if __name__=='__main__':
    cli()