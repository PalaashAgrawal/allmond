from data.unlabeled import OpenWebTextConfig 
from model.gpt2 import GPT
import numpy as np

from fastai.text.all import *
from torch.utils.data import DataLoader


class OWTData():
    def __init__(self, path, block_size=1024):
        """
        Initialize the FastGPT model.

        Parameters:
        path (str): The path to the data file.
        block_size (int, optional): The block size for reading the data file. Defaults to 128.
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
    
    #def __getitems__(self, idx): for batched access
    


if __name__ == "__main__":
    
    model = GPT()
    train_dl = DataLoader(OWTData(OpenWebTextConfig().default_cache_dir/'train.bin'), batch_size=8)
    valid_dl = DataLoader(OWTData(OpenWebTextConfig().default_cache_dir/'val.bin'), batch_size=16)
    

    dls = DataLoaders(train_dl, valid_dl)
    dls.c = model.head.out_features

    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, Perplexity()])
    
    learn.fit_one_cycle(1, 1e-3)