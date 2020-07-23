import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from typing import List


from math import ceil
from random import Random
from torch.multiprocessing import Process

from torchvision import datasets, transforms
import numpy as np
import pandas
from dist_trainer import *

import os
DEFAULT_MODEL_SAVE_PATH =\
        os.path.join(\
            DEFAULT_MODEL_PATH,\
            'credit_card',
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')\
        )

gbatch_size = 128
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'dataset-repo',
    'n_5_b_2',
    'dispatcher_folder_credit',
)
import pickle
from sklearn.preprocessing import StandardScaler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        """
        Be cautious about index shuffle, this is performed on each rank
        The shuffled index must be unique across all ranks
        Theoretically with the same seed Random() generates the same sequence
        This might not be true in rare cases
        You can add an additional synchronization for 'indexes', just for safety
        Anyway, this won't take too much time
        e.g.
            dist.broadcast(indexes, 0)
        """
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 18)
        self.fc3 = nn.Linear(18, 20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def get_dataset(data_path):
    size = dist.get_world_size()
    bsz = int(gbatch_size / float(size))
    final_path = os.path.join(DATASET_PATH,'output'+str(dist.get_rank()+1)+'.csv')
    df = pandas.read_csv(final_path)
    X = df.iloc[:, 1:-1].values # extracting features
    y = df.iloc[:, -1].values # extracting labels
    sc = StandardScaler()
    X = sc.fit_transform(X)
    partition = torch.from_numpy(X)
    labels = torch.from_numpy(y).double()
    part = torch.utils.data.TensorDataset(partition,labels)
    train_set = torch.utils.data.DataLoader(
        part, batch_size=bsz, shuffle=True)
    print("Getting Dataset For %d %d"%(dist.get_rank(),size))
    print(partition.shape,'partition.shape')
    print(labels.shape,'labels.shape')
    return train_set, bsz


def save_data(model_save_path,model,optimizer,meta):
    with open(os.path.join(model_save_path,'epoch_arr'), 'wb') as handle:
        pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save({
        'model': model.state_dict(),
        'optimizer':optimizer.state_dict(),
    },os.path.join(model_save_path,'model.pth'))
    

    
def run(rank, size,model_save_path,checkpoint_every):
    """ Distributed Synchronous SGD Example """
    setup(rank,world_size=size)
    torch.manual_seed(1234)
    train_set, bsz = get_dataset(DATASET_PATH)
    
    model = Net()
    model = model.double()
    safe_mkdir(model_save_path)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    #print("num_batches = ", num_batches)
    epoch_tuples = []
    num_labels = 2
    sync_params(model)
    loss_fn = nn.BCELoss()
    for epoch in range(10):
        # make sure we have the same parameters for all ranks
        conf_matrix = ConfusionMatrix([i for i in range(num_labels)])
        train_loop_resp = class_train_loop(train_set,model,optimizer,conf_matrix,loss_fn=loss_fn)
        loss_meter = train_loop_resp[3]
        acc_meter = train_loop_resp[2]
        # val_loop_resp = class_validation_loop(test_set,model,10)
        epoch_tuples.append((epoch,train_loop_resp))
        
        print('Epoch {} Loss {:.6f} Acc {:.6f} Global batch size {} on {} ranks For Rank : {}'.format(
            epoch, loss_meter.avg,acc_meter.avg, gbatch_size, dist.get_world_size(),dist.get_rank()))
        if rank == 0 and epoch % checkpoint_every==0:
            save_data(model_save_path,model,optimizer,epoch_tuples)
        

def run_demo(demo_fn, world_size,model_save_path=DEFAULT_MODEL_SAVE_PATH,checkpoint_every=1):
    mp.spawn(demo_fn,
             args=(world_size,model_save_path,checkpoint_every),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    # dist.init_process_group(backend='gloo')
    # size = dist.get_world_size()
    # rank = dist.get_rank()
    # init_print(rank, size)

    # run(rank, size)
    run_demo(run,4)
    print("Now I am Done")


