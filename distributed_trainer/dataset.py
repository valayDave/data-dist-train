import torch
from typing import List
class Dataset:
    """
    Base class which needs to be implemeted for working with the DistributedTrainer Process. 
    """    
    def get_train_dataset(self)->torch.utils.data.TensorDataset:
        raise NotImplementedError()
    
    def get_test_dataset(self)->torch.utils.data.TensorDataset:
        raise NotImplementedError()

    def get_metadata(self)->dict: # To help get Quick dataset Related Metadata.
        raise NotImplementedError()

    def get_labels(self)->List:
        raise NotImplementedError()

    def model_alterations(self,model): # This is to change the model configurations acc to dataset. 
        return model

class DistributedDataset(Dataset):
    
    def get_train_dataset(self,rank)->torch.utils.data.TensorDataset:
        raise NotImplementedError()
    
    def get_test_dataset(self,rank)->torch.utils.data.TensorDataset:
        raise NotImplementedError()

