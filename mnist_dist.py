"""
Code PLUCKED and Modified FROM : 
    https://github.com/xhzhao/PyTorch-MPI-DDP-example.git
    TO Support MacOS Running Etc. 

THIS IS ULTRA NON OPTIMAL BUT GOOD TOY BOILER PLATE TO START FROM

Synchronous SGD training on MNIST
Use distributed MPI backend

PyTorch distributed tutorial:
    http://pytorch.org/tutorials/intermediate/dist_tuto.html

This example make following updates upon the tutorial
1. Add params sync at beginning of each epoch
2. Allreduce gradients across ranks, not averaging
3. Sync the shuffled index during data partition
4. Remove torch.multiprocessing in __main__
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

gbatch_size = 128


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
        print(len(data),'data')
        print(len(index),'index')

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
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = int(gbatch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    print("Getting Dataset For %d %d"%(dist.get_rank(),size))
    return train_set, bsz

def sync_params(model):
    """ broadcast rank 0 parameter to all ranks """
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def sync_grads(model):
    """ all_reduce grads from all ranks """
    for param in model.parameters():
        dist.all_reduce(param.grad.data)

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    setup(rank,world_size=size)
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    #print("num_batches = ", num_batches)
    for epoch in range(10):
        epoch_loss = 0.0
        # make sure we have the same parameters for all ranks
        sync_params(model)
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            loss.backward()
            # all_reduce grads
            sync_grads(model)
            optimizer.step()
        print('Epoch {} Loss {:.6f} Global batch size {} on {} ranks For Rank : {}'.format(
            epoch, epoch_loss / num_batches, gbatch_size, dist.get_world_size(),dist.get_rank()))

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    # dist.init_process_group(backend='gloo')
    # size = dist.get_world_size()
    # rank = dist.get_rank()
    # init_print(rank, size)

    # run(rank, size)
    datasets.MNIST('./data',download=True)
    run_demo(run,4)


