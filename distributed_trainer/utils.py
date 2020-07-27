import torch
import numpy as np
from typing import List
import os 
import pandas as pd
import tabulate
from datetime import datetime
from dataclasses import dataclass,field


#https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

import logging

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def create_logger(logger_name:str,level=logging.INFO):
    custom_logger = logging.getLogger(logger_name)
    ch1 = logging.StreamHandler()
    ch1.setLevel(level)
    ch1.setFormatter(formatter)
    custom_logger.addHandler(ch1)
    custom_logger.setLevel(level)    
    return custom_logger
    
def safe_mkdir(pth):
    try:
        os.makedirs(pth)
    except:
        return 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ConfusionMatrix:
    def __init__(self,labels:List,conf_mat=None):
        self.labels = labels
        # todo Parse to nparray
        if conf_mat is None:
            conf_mat = np.zeros((len(labels),len(labels)))
        self.conf_mat = np.array(conf_mat) # [Real][Predicted]
    
    def update(self,pred:torch.Tensor,target:torch.Tensor):
        """update 
        pred : shape(batch_size,1)
        target : shape(batch_size)
        """
        target = target.view(-1,1)
        conf_add_vec = torch.cat((target.long(),pred),1)
        for vec in conf_add_vec.cpu().detach().numpy():
            self.conf_mat[vec[0]][vec[1]]+=1

    def __str__(self):
        df = pd.DataFrame(self.conf_mat,['True_'+str(i) for i in self.labels],['Pred_'+str(i) for i in self.labels])
        return tabulate.tabulate(df,tablefmt='github', headers='keys')

    def to_json(self):
        return dict(
            labels = self.labels,
            conf_mat = self.conf_mat.tolist()
        )

    @classmethod
    def from_json(cls,json_object):
        if 'conf_mat' not in json_object or 'labels' not in json_object:
            raise Exception('Object Not Compatible')
        return cls(json_object['labels'],conf_mat=json_object['conf_mat'])

def get_accuracy(output:torch.Tensor, target:torch.Tensor,conf_matrix:ConfusionMatrix):
    with torch.no_grad():
        _,pred = output.topk(1,1,True,True) # Convert softmax logits argmax based selection of index to get prediction value
        conf_matrix.update(pred,target)
        pred = pred.t() # Transpose the pred value use in comparison
        num_correct_preds = pred.eq(target.view(1,-1).expand_as(pred)).view(-1).sum(0) # Compare the pred with the target
        return float(num_correct_preds)/output.shape[0]

@dataclass
class ExperimentResultsBundle:
    epoch:int=None
    losses:float=None
    accuracy:float=None
    batch_time:float=None
    confusion_matrix:dict=None
    created_on:str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

@dataclass
class ExperimentBundle:
    train_epoch_results:List[dict] = field(default_factory=lambda:[])
    validation_epoch_results:List[dict] = field(default_factory=lambda:[])
    train_args:dict=None
    dataset_metadata:dict = None
    created_on:str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    model:dict = None
    optimizer:dict = None
    rank:int = None
    distributed:bool=False
    model_args:dict=None
    optimizer_args:dict=None
    loss_fn:str=None


@dataclass
class DistributionArgs:
    sample:int=None
    uniform_label_distribution:bool=True
    label_split_values:List = field(default_factory=lambda:[])
    test_set_portion:float=0.3

@dataclass
class CheckpointingArgs:
    path:str=None
    checkpoint_all_ranks:bool=False
    checkpoint_rank:int=0
    checkpoint_frequency:int = 10
    save_experiment:bool=True


@dataclass
class TrainerArgs:
    batch_size:int=128
    shuffle:bool=True
    num_epochs:int=10
    checkpoint_args:CheckpointingArgs=None

@dataclass
class DistTrainerArgs:
    backend:str='gloo'
    master_ip:str='127.0.0.1'
    master_port:str='12355'
    world_size:int=5
    