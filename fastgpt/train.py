from data.unlabeled import TiktokenTokenizer, unlabeledDataset, dataConfig
from model.gpt2 import GPT

import numpy as np
from pathlib import Path

#first write script for single GPU, vanilla training
#then write script for multi-GPU, vanilla training
#then multinode


from fastai.text.all import *

class OWTData(dataConfig):
    f'openwebtext data'
    def __init__(self, bin_path: str):
        self.bin_path = bin_path
    
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')
    
    def __len__(self):
        #get length of data in train.bin file 
        return len(self.data)
    


