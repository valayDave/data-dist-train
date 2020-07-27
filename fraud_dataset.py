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
    safe_mkdir,\
    DistributedDataset,\
    Dataset,\
    DistributionArgs


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
                rand_dist = np.random.randint(10,1000,(self.world_size))
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


