from utils import *
import torch.nn.functional as F
import torch.distributed as dist
import time
import datetime 
from torch.autograd import Variable
import os

DEFAULT_MODEL_PATH = os.path.join(\
    os.path.dirname(\
        os.path.abspath(__file__)
    ),
    'model_data',
    )
def sync_params(model):
    """ broadcast rank 0 parameter to all ranks """
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def sync_grads(model):
    """ all_reduce grads from all ranks """
    for param in model.parameters():
        dist.all_reduce(param.grad.data)


def class_train_loop(train_set,model,optimizer,conf_matrix,loss_fn = F.nll_loss):
    epoch_loss = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    end = time.time()
    model.train()
    for data, target in train_set:
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.view(-1,1))
        epoch_loss += loss.data
        loss.backward()
        acc_val = get_accuracy(output.float(),target.float(),conf_matrix)
        losses.update(loss.item(),target.shape[0])
        acc.update(acc_val,target.shape[0])
        batch_time.update(time.time() - end)
        # all_reduce grads
        end = time.time()
        sync_grads(model)
        optimizer.step()
    return (conf_matrix,batch_time,acc,losses)

def class_validation_loop(train_set,model,conf_matrix,loss_fn = F.nll_loss):
    epoch_loss = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    end = time.time()
    model.eval()
    print("Len Valset",len(train_set))
    for data, target in train_set:
        data, target = data, target
        output = model(data)
        loss = loss_fn(output, target.view(-1,1))
        epoch_loss += loss.data
        acc_val = get_accuracy(output.float(),target.float(),conf_matrix)
        losses.update(loss.item(),target.shape[0])
        acc.update(acc_val,target.shape[0])
        batch_time.update(time.time() - end)
        # all_reduce grads
        end = time.time()
    return (conf_matrix,batch_time,acc,losses)