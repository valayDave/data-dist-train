from utils import *
import torch.nn.functional as F
import torch.distributed as dist
import time
from typing import List
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
    print("Syncing Grads")
    for param in model.parameters():
        dist.all_reduce(param.grad.data)


def class_train_loop(train_set,model,optimizer,device,conf_matrix,loss_fn = F.nll_loss,print_every=10,rank=None):
    epoch_loss = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    end = time.time()
    model.train()
    curr_index = 0
    print("Train Loop Initiated For Rank %d"%rank)
    for data, target in train_set:
        data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
        print(data.shape,target.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.view(-1,1))
        epoch_loss += loss.data
        loss.backward()
        
        acc_val = get_accuracy(output.float(),target.float(),conf_matrix)
        losses.update(float(loss.item()),target.shape[0])
        acc.update(acc_val,target.shape[0])
        batch_time.update(time.time() - end)
        # all_reduce grads
        end = time.time()
        sync_grads(model)
        optimizer.step()
        if curr_index % print_every == 0:
            if rank is None or rank == 0:
                print_meters([losses,acc,batch_time],conf_matrix,rank=rank,batch_idx=curr_index)
        curr_index+=1
    print("\n\nCompleted Training Loop")
    print_meters([losses,acc,batch_time],conf_matrix)
    return (conf_matrix,batch_time,acc,losses)

def print_meters(meters:List[AverageMeter],conf_matrix,rank=None,batch_idx=None):
    meter_vals = ''.join(['\n\t'+str(meter) for meter in meters])
    print_str= '\n\n'
    if rank is not None:
        if batch_idx is not None:
            print_str = "\nMeter Readings On Rank %d And BatchIdx %d\n"%(rank,batch_idx)
        else:
            print_str = "\nMeter Readings On Rank %d\n"%rank
    else:
        if batch_idx is not None:
            print_str = "\nMeter Readings On BatchIdx %d\n"%(batch_idx)
    print(print_str)
    print(print_str)
    print(meter_vals)
    print('Confusion Matrix')
    print(str(conf_matrix))

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