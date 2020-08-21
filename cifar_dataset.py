# coding: utf-8
from distributed_trainer.data_dispatcher import \
    Dispatcher,\
    DispatcherControlParams,\
    BlockDistributedDataset,\
    DataBlock,\
    ALLOWED_DISPATCHING_METHODS

import random
import torchvision
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
# torchvision.datasets.CIFAR10('./',download=True)

class CifarBlockDataset(BlockDistributedDataset):

    def transform_from_block(self,dataset,block:DataBlock):
        selected_items = [dataset[i] for i in block.data_item_indexes]
        return self.transform(selected_items)

    @staticmethod
    def transform(dataset):
        features = []
        labels = []
        for img_tensor,value in dataset:
            features.append(img_tensor)
            labels.append(value)
        features = torch.stack(features)
        labels = torch.Tensor(labels).long()
        return TensorDataset(features,labels)
    
    @staticmethod
    def get_all_labels(dataset):
        labels = []
        for _,value in dataset:
            labels.append(int(value))
        return labels
    
    def get_block_labels(self):
        block_labels = [

        ]
        for block in self.blocks:
            selected_items = [self.train_dataset[i] for i in block.data_item_indexes]
            labels = self.get_all_labels(selected_items)
            block_labels.append(labels)
        return block_labels

    def get_train_dataset(self, rank):
        selected_block = self.blocks[rank]
        return self.transform_from_block(self.train_dataset,selected_block)
    
    def get_test_dataset(self,rank):
        return self.transform(self.test_dataset)

    def get_labels(self):
        return [i for i in range(10)]


def get_distributed_dataset(dispatcher_args:DispatcherControlParams):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_st = torchvision.datasets.CIFAR10('./',transform=transform_train)
    test_st = torchvision.datasets.CIFAR10('./',train=False,transform=transform_test)
    if dispatcher_args.sample is not None:
        train_st = [ train_st[data_index] for data_index in random.sample([i for i in range(len(train_st))],dispatcher_args.sample)]
        test_st = [ test_st[data_index] for data_index in random.sample([i for i in range(len(test_st))],dispatcher_args.sample)]

    disp = Dispatcher(dispatcher_args,\
                    CifarBlockDataset.transform(train_st),\
                    CifarBlockDataset.transform(test_st))

    datastore = disp.run()

    cifar_block_ds = datastore.get_distributed_dataset(
        UsedClass=CifarBlockDataset,
        test_set=test_st
    )
    return cifar_block_ds,datastore