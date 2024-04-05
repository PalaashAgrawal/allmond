


import numpy as np

from fastai.text.all import *


class OWTData():
    def __init__(self, path, block_size=1024, dtype = None):
        """
        Dataset class for OpenWebText.

        Parameters:
        path (str): The path to the data file.
        block_size (int, optional): The block size for reading the data file. Defaults to 128.
        dtype: np datatype: VERY IMPORTANT: the format with which to read data. Make sure that this is the same format with which data was saved. 
        
        """
        
        dtype = dtype or np.uint64
        self.path = path
        
        self.data = np.memmap(self.path, dtype=dtype, mode='r')
        self.block_size = block_size
        
    def __len__(self): 
        
        data_len = len(self.data)
        assert self.block_size < data_len, "block_size can't be larger than the data size"
        return data_len - self.block_size - 1


    def __getitem__(self, idx):
        f'I think fastai expects input to be int64 format. to do .type, you first need to convert to np int64 (because it doesnt support uint64 conversion directly)'
        
        x,y = np.array(self.data[idx:idx+self.block_size]), np.array(self.data[idx+1: idx+self.block_size+1]) 
        return torch.from_numpy(x.astype(np.int64)).type(torch.long), torch.from_numpy(y.astype(np.int64)).type(torch.long)
    
    

class RandomSubsetSampler():
    """
    A custom sampler that generates random subset of a given dataset.
    Used specifically for the valid dataset.
    In our case, we use validation dataset only for checking if the model is training properly. 
    Every n  training iterataions (here, 10,000), we check val_loss on this subset. 
    If the val_loss is better than the best val_loss, we save the model in the form of a checkpoint. 

    Args:
        dataset (Dataset): The dataset to sample from.
        subset_size (int): The size of the random subset to generate.

    Attributes:
        dataset (Dataset): The dataset to sample from.
        subset_size (int): The size of the random subset to generate.
    """
    def __init__(self, dataset, subset_size=1000):
        self.dataset = dataset
        self.subset_size = subset_size
    
    def __len__(self):
        """
        Returns the size of the random subset.

        Returns:
            int: The size of the random subset.
        """
        return self.subset_size
    
    def __iter__(self):
        """
        Generates an iterator over the indices of the random subset.

        Returns:
            iter: An iterator over the indices of the random subset.
        """
        idxs = torch.randint(len(self.dataset), size=(self.subset_size,))
        return iter(idxs)

    
    