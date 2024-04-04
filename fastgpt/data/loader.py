


import numpy as np

from fastai.text.all import *


class OWTData():
    def __init__(self, path, block_size=1024):
        """
        Dataset class for OpenWebText.

        Parameters:
        path (str): The path to the data file.
        block_size (int, optional): The block size for reading the data file. Defaults to 128.
        
        TODO: is del necessary? 
        """
        self.path = path
        
        self.block_size = block_size
        self.l = None
        # assert self.block_size<=len(self.data), "block_size can't be larger than the data size"
        
    def __len__(self): 
        if self.l is None:
            data = np.memmap(self.path, dtype=np.uint16, mode='r')
            self.l = len(data)
            del data
        return self.l - self.block_size
    
    def __getitem__(self, idx):
        
        data = np.memmap(self.path, dtype=np.uint16, mode='r')

        x,y = np.array(data[idx:idx+self.block_size]).astype(np.int64), np.array(data[idx+1: idx+self.block_size+1]).astype(np.int64) 
        
        del data
        return torch.from_numpy(x).type(torch.long), torch.from_numpy(y).type(torch.long)
    
    
    




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

    
    