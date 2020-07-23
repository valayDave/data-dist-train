import torch
import numpy as np
from typing import List
import os 
import pandas as pd
import tabulate


#https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix


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
    def __init__(self,labels:List):
        self.labels = labels
        self.conf_mat = np.zeros((len(labels),len(labels))) # [Real][Predicted]
    
    def update(self,pred:torch.Tensor,target:torch.Tensor):
        """update 
        pred : shape(batch_size,1)
        target : shape(batch_size)
        """
        target = target.view(-1,1)
        conf_add_vec = torch.cat((target.long(),pred),1)
        for vec in conf_add_vec:
            self.conf_mat[vec[0]][vec[1]]+=1

    def __str__(self):
        df = pd.DataFrame(self.conf_mat,['True_'+str(i) for i in self.labels],['Pred_'+str(i) for i in self.labels])
        return tabulate.tabulate(df,tablefmt='github', headers='keys')


def get_accuracy(output:torch.Tensor, target:torch.Tensor,conf_matrix:ConfusionMatrix):
    _,pred = output.topk(1,1,True,True) # Convert softmax logits argmax based selection of index to get prediction value
    conf_matrix.update(pred,target)
    pred = pred.t() # Transpose the pred value use in comparison
    num_correct_preds = pred.eq(target.view(1,-1).expand_as(pred)).view(-1).sum(0) # Compare the pred with the target
    return float(num_correct_preds)/output.shape[0]


