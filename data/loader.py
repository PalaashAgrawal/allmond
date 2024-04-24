import numpy as np
from fastai.text.all import *
from fastai.data.load import _FakeLoader
import mmap

from fastai.distributed import *


class dataloader(DataLoader):

    def __init__(self, file, block_size, bs, dtype, device:str = 'cuda', seed:int = 42, sample_size = None):
        """
        sample_size is specifically used for valid_dl, where you only want to test the dataset on a subset of the valid dl, solely to heck progress of the model.
        We dont want to iterate through the entire valid_dl, since that takes time. 
        just a couple samples is enough to check and checkpoint the training model at certain stages. 
        """
        
        self.file = file
        self.dataset = np.memmap(file, dtype = dtype, mode = 'r')
        self.block_size = block_size
        self.bs = bs
        self.dtype = dtype
        
        self.n = sample_size or len(self.dataset)
        
        self.device = device
        self._device_type = 'cuda' if 'cuda' in device else 'cpu'        
        
        # #other - required by distributed Trainer class initialization
        self.drop_last =True
        self.num_workers = 1
        self.rng = random.Random(random.randint(0,2**32-1))
        self.offs = 0
        self.pin_memory = True
        self.fake_l = _FakeLoader(self, pin_memory=True, num_workers=self.num_workers, 
                                  timeout=0, persistent_workers=False,
                                  pin_memory_device='')
        self.indexed = False
        self.shuffle = True
        
        torch.manual_seed(seed)
        

    
    def __iter__(self):
        'PAg: custom n: how many iterations do you want. By default, it is self.n//self.bs'
        return self.batch_generator()
    
    
    def batch_generator(self):
        """
        We almost sample almost all elements. 
        If in each random iteration, i were to choose only one element, I would encounter only about N(1-1/e) = 0.63N items uniquely (about 63% of the dataset)
        But since in each iteration, I sample bs items, the total elements encountered goes significantly up. We end up covering almost the entire dataset
       
        ACCORDING TO GPT4
        import numpy as np

        # Constants for sequential sampling
        M = 9e9  # 9 billion elements (approx) #number of draws
        N = 9e9  # 9 billion times (approx) #dataset size
        K = 20   # 20 items each draw #batch size

        # Assume the dataset does not wrap around
        max_start_index = M - K + 1

        # Probability that a specific item is in one particular draw (assuming random start index)
        prob_in_one_draw = K / M

        # Probability that a specific item is not in one particular draw
        prob_not_in_one_draw = 1 - prob_in_one_draw

        # Probability that a specific item is not in any draw across all N times
        prob_not_in_any_draw = prob_not_in_one_draw**N

        # Probability that a specific item is in at least one draw
        prob_in_at_least_one_draw = 1 - prob_not_in_any_draw

        # Expected number of unique items sampled at least once
        expected_unique_samples_sequential = M * prob_in_at_least_one_draw
        expected_unique_samples_sequential 
        8999999981.449612
        """
        
        self.before_iter()
    
        # while True:  # This loop will make it a generator that yields indefinitely
        for _ in range(len(self)):
            # We recreate np.memmap every batch to avoid a memory leak
            data = np.memmap(self.file, dtype=self.dtype, mode='r')
            ix = torch.randint(len(data) - self.block_size - 1, (self.bs,))
            
            # Prepare the input batch (x) and the target batch (y)
            x = torch.stack([torch.from_numpy(data[i:i+self.block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
            
            
            if self._device_type == 'cuda':
            # if 'cuda' in self.device:
                # Pin arrays x and y for asynchronous GPU transfer
                x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
            else:
                x, y = x.to(self.device), y.to(self.device)
            b = (x,y)
            # if self.device is not None: b = to_device(b, self.device)
            yield self.after_batch(b)
        
        
            # yield x, y  # Yield the batches instead of returning them
            
        self.after_iter()
        if hasattr(self, 'it'): del(self.it)
        
    def __getattr__(self, k):
        attr = getattr(self.__dict__, k, None)
        if attr is not None: return attr
        
        raise AttributeError(k)
        


class customDistributedDL(DistributedDL):
    _default='dl' #in the parent class(es), _default is dataset. which is not defined in our case
    
    def __iter__(self):
        return iter(self.dl)



