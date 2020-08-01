import os
import torch
import pandas
import time
import torch.nn as nn
import torch.nn.functional as F
from ..trainer import *
from ..utils import *

class DistributedClassificationTrainer(DistributedTrainer):
    def __init__(self,dataset,print_every=10,**kwargs):
        self.print_every = print_every
        super(DistributedClassificationTrainer, self).__init__(dataset=dataset,**kwargs)

    def train_loop(self,train_data_loader):
        end = time.time()
        conf_matrix,acc,losses,batch_time = self.get_meters()
        self.neural_network.train()
        curr_index = 0
        print_every = self.print_every
        print_checkpoints = [int(len(train_data_loader)*i/print_every) for i in range(print_every)]
        for data, target in train_data_loader:
            
            if self.gpu_enabled:
                data, target = data.to(self.rank,non_blocking=True), target.to(self.rank,non_blocking=True)
            
            self.optimizer.zero_grad()
            
            output = self.neural_network(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            acc_val = self.get_accuracy(output.float(),target.float(),conf_matrix)
            losses.update(float(loss.item()),target.shape[0])
            acc.update(acc_val,target.shape[0])
            batch_time.update(time.time() - end)
            # all_reduce grads
            end = time.time()

            self.sync_grads()
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
    
    def test_loop(self,test_data_loader):
        with torch.no_grad():
            end = time.time()
            conf_matrix,acc,losses,batch_time = self.get_meters()
            self.neural_network.eval()
            curr_index = 0
            print_every = self.print_every
            print_checkpoints = [int(len(test_data_loader)*i/print_every) for i in range(print_every)]
            for data, target in test_data_loader:
                
                if self.gpu_enabled:
                    data, target = data.to(self.rank,non_blocking=True), target.to(self.rank,non_blocking=True)                
                output = self.neural_network(data)
                loss = self.loss_fn(output, target)
                acc_val = self.get_accuracy(output.float(),target.float(),conf_matrix)
                losses.update(float(loss.item()),target.shape[0])
                acc.update(acc_val,target.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
                curr_index+=1
                if curr_index in print_checkpoints:
                    self.logger.info('%s %s %s For Rank : %s'%(str(losses),str(acc),str(batch_time),str(self.rank)))
                    self.logger.info(self.conf_matrix_str(conf_matrix))

            self.logger.info("Completed Testing Loop")
            self.logger.info('%s %s %s For Rank : %s'%(str(losses),str(acc),str(batch_time),str(self.rank)))
            self.logger.info(self.conf_matrix_str(conf_matrix))
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

class MonolithClassificationTrainer(BaseTrainer):

    def __init__(self,dataset,print_every=10,**kwargs):
        self.print_every = print_every
        super(MonolithClassificationTrainer, self).__init__(dataset=dataset,**kwargs)

    def train_loop(self,train_data_loader):
        end = time.time()
        conf_matrix,acc,losses,batch_time = self.get_meters()
        self.neural_network.train()
        curr_index = 0
        print_every = self.print_every
        print_checkpoints = [int(len(train_data_loader)*i/print_every) for i in range(print_every)]
        for data, target in train_data_loader:
            
            if self.gpu_enabled:
                data, target = data.to(self.rank,non_blocking=True), target.to(self.rank,non_blocking=True)
            
            self.optimizer.zero_grad()
            
            output = self.neural_network(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            acc_val = self.get_accuracy(output.float(),target.float(),conf_matrix)
            losses.update(float(loss.item()),target.shape[0])
            acc.update(acc_val,target.shape[0])
            batch_time.update(time.time() - end)
            # all_reduce grads
            end = time.time()
            self.optimizer.step()
            curr_index+=1
            if curr_index in print_checkpoints:
                self.logger.info('%s %s %s'%(str(losses),str(acc),str(batch_time)))
                self.logger.info(self.conf_matrix_str(conf_matrix))

        self.logger.info("Completed Training Loop")
        self.logger.info('%s %s %s '%(str(losses),str(acc),str(batch_time)))
        self.logger.info(self.conf_matrix_str(conf_matrix))
        return ExperimentResultsBundle(
            losses=losses.avg,
            accuracy=acc.avg,
            batch_time=batch_time.avg,
            confusion_matrix=conf_matrix.to_json()
        )

    def test_loop(self,test_data_loader):
        with torch.no_grad():
            end = time.time()
            conf_matrix,acc,losses,batch_time = self.get_meters()
            self.neural_network.eval()
            curr_index = 0
            print_every = self.print_every
            print_checkpoints = [int(len(test_data_loader)*i/print_every) for i in range(print_every)]
            for data, target in test_data_loader:
                if self.gpu_enabled:
                    data, target = data.to(self.rank,non_blocking=True), target.to(self.rank,non_blocking=True)     
                output = self.neural_network(data)
                loss = self.loss_fn(output, target)
                acc_val = self.get_accuracy(output.float(),target.float(),conf_matrix)
                losses.update(float(loss.item()),target.shape[0])
                acc.update(acc_val,target.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
                curr_index+=1
                if curr_index in print_checkpoints:
                    self.logger.info('%s %s %s'%(str(losses),str(acc),str(batch_time)))
                    self.logger.info(self.conf_matrix_str(conf_matrix))

            self.logger.info("Completed Testing Loop")
            self.logger.info('%s %s %s '%(str(losses),str(acc),str(batch_time)))
            self.logger.info(self.conf_matrix_str(conf_matrix))
            return ExperimentResultsBundle(
                losses=losses.avg,
                accuracy=acc.avg,
                batch_time=batch_time.avg,
                confusion_matrix=conf_matrix.to_json()
            )

    def conf_matrix_str(self,conf_matrix):
        log_line = '''
        Confusion Matrix \n
        {matrix_op}
        '''.format(matrix_op=str(conf_matrix).replace('\n','\n\t'))
        return log_line
