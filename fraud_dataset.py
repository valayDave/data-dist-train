import os
import torch
from torch.utils.data import TensorDataset
import pandas
import numpy as np
from dataclasses import dataclass
import dataclasses 
import json
import random
from sklearn.preprocessing import StandardScaler
from distributed_trainer import \
    safe_mkdir,\
    DistributedDataset,\
    Dataset,\
    DistributionArgs

from distributed_trainer.data_dispatcher import \
    Dispatcher,\
    DispatcherControlParams,\
    BlockDistributedDataset,\
    DataBlock

BASE_PATH = os.path.join(os.path.dirname(__file__),'dataset-repo')
TRAIN_PATH = os.path.join(BASE_PATH,'Cred_train.csv')
TEST_PATH = os.path.join(BASE_PATH,'Cred_test.csv')

def get_train_dataset_path():
    return TRAIN_PATH

def get_test_dataset_path():
    return TEST_PATH



class FraudBlockDataset(BlockDistributedDataset):
    def __init__(self, train_dataset, blocks, test_dataset=None, metadata=None):
        BlockDistributedDataset.__init__(self,train_dataset, blocks, test_dataset=test_dataset, metadata=metadata)

    def transform_from_block(self,dataset,block:DataBlock):
        selected_items = [dataset[i] for i in block.data_item_indexes]
        return self.transform(selected_items)

    @staticmethod
    def transform(dataset):
        features = []
        labels = []
        for tensor,value in dataset:
            features.append(tensor)
            labels.append(value)
        features = torch.stack(features).double()
        labels = torch.Tensor(labels).view(-1,1).double()
        return TensorDataset(features,labels)

    def get_train_dataset(self, rank):
        selected_block = self.blocks[rank]
        return self.transform_from_block(self.train_dataset,selected_block)
    
    def get_test_dataset(self,rank):
        return self.transform(self.test_dataset)

    def get_labels(self):
        return [0,1]

    def model_alterations(self,model):
        return model.double()

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

class FraudDataset(Dataset,FraudData):
    def __init__(self,sample=None):
        self.train_path = get_train_dataset_path()
        self.test_path = get_test_dataset_path()
        self.sample = sample
        self.train = None
        self.test = None

    def get_train_dataset(self)->torch.utils.data.TensorDataset:
        if self.train is not None:
            return self.train
        self.set_attributes()
        return self.train
    
    def set_attributes(self):
        train_df = self.get_train_df()
        test_df = self.get_test_df()
        self.train = self.dataset_transform(train_df,sample=self.sample)
        self.test = self.dataset_transform(test_df,sample=self.sample)
    
    def get_test_df(self):
        return pandas.read_csv(self.test_path)
    
    def get_train_df(self):
        df = pandas.read_csv(self.train_path)
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


def get_distributed_dataset(dispatcher_args:DispatcherControlParams):
    train_st = FraudData()\
                .dataset_transform(\
                    pandas.read_csv(TRAIN_PATH) if dispatcher_args.sample is None else pandas.read_csv(TRAIN_PATH).sample(dispatcher_args.sample)
                )
    test_st =  FraudData()\
                .dataset_transform(\
                    pandas.read_csv(TEST_PATH) if dispatcher_args.sample is None else pandas.read_csv(TEST_PATH).sample(dispatcher_args.sample)
                )
    disp = Dispatcher(dispatcher_args,train_st,test_st)
    datastore = disp.run()
    fraud_block_dist_dataset = datastore.get_distributed_dataset(
        UsedClass=FraudBlockDataset,
        test_set=test_st
    )
    return fraud_block_dist_dataset,datastore
