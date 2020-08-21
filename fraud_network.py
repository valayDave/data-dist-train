import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudCNNModel(nn.Module):
    def __init__(self,input_length=30):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, input_length,kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm1d(input_length),
            nn.Dropout(0.1),
            nn.Conv1d(input_length, 64,kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128,kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(3456,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,1),
        )

    def forward(self,x):
        op = self.model(x)
        return torch.sigmoid(op)


class FraudFFNetwork(nn.Module):
    def __init__(self,input_length=30,hidden_dims=200):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_length,hidden_dims),
            nn.Linear(hidden_dims,hidden_dims),
            nn.Linear(hidden_dims,1)
        )
    
    def forward(self,x):
        '''
        x : (batchsize,1,N)
        '''
        x = x.squeeze(1)
        op = self.model(x)
        return torch.sigmoid(op)

class ModelFactory:
    models = ['CNN','FF']
    NETWORKS = {
        'CNN' : FraudCNNModel,
        'FF': FraudFFNetwork
    }
    default_args = {
        'CNN' : {
            'input_length':30,
        },
        'FF' : {
            'input_length':30,
            'hidden_dims':200
        }
    }
    DEFAULT = 'CNN'

    @property
    def default(self):
        return self.get_model(self.DEFAULT)

    def get_model(self,model_name):
        model = self.NETWORKS[model_name]
        args = self.default_args[model_name]
        return model,args


