import os
import torch
import pandas
import time
import torch.nn as nn
import torch.nn.functional as F
from ..trainer import *
from ..utils import *

class ClassificationTrainer(DistributedTrainer):
    def __init__(self,dataset,print_every=1000,**kwargs):
        self.print_every = print_every
        super(ClassificationTrainer, self).__init__(dataset=dataset,**kwargs)

    def train_loop(self,train_data_loader):
        end = time.time()
        conf_matrix,acc,losses,batch_time = self.get_meters()
        self.neural_network.train()
        curr_index = 0
        print_every = self.print_every
        print_checkpoints = [int(len(train_data_loader)*i/print_every) for i in range(print_every)]
        for data, target in train_data_loader:
            
            if self.gpu_enabled:
                data, target = data.to(self.rank), target.to(self.rank)
            
            self.optimizer.zero_grad()
            
            output = self.neural_network(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            acc_val = get_accuracy(output.float(),target.float(),conf_matrix)
            losses.update(float(loss.item()),target.shape[0])
            acc.update(acc_val,target.shape[0])
            batch_time.update(time.time() - end)
            # all_reduce grads
            end = time.time()

            # self.sync_grads()
            self.optimizer.step()
            curr_index+=1
            if curr_index in print_checkpoints:
                self.logger.info('%s %s %s For Rank : %s'%(str(losses),str(acc),str(batch_time),str(self.rank)))
                self.logger.info(self.conf_matrix_str(conf_matrix))

        self.logger.info("Completed Training Loop")
        self.logger.info('%s %s %s For Rank : %s'%(str(losses),str(acc),str(batch_time),str(self.rank)))
        self.logger.info(self.conf_matrix_str(conf_matrix))
        # self.sync_grads()
        return ExperimentResultsBundle(
            losses=losses.avg,
            accuracy=acc.avg,
            batch_time=batch_time.avg,
            confusion_matrix=conf_matrix.to_json()
        )

    def conf_matrix_str(self,conf_matrix):
        log_line = '''
        Confusion Matrix For Rank : {rank}\n
        {matrix_op}
        '''.format(matrix_op=str(conf_matrix).replace('\n','\n\t'),rank=str(self.rank))
        return log_line