import numpy as np
from fastai.text.all import *
from fastai.data.load import _FakeLoader
import mmap

from fastai.distributed import *


# class OWTData:
#     """
#     np.memmap leads to memory leaks. This python mmap implementation is better. 
#     TODO:
#     1. valid dataloader raises
#     in __getitem__
#         array = np.frombuffer(buffer, dtype=self.dtype, count=self.shape[1])
#         ValueError: buffer is smaller than requested size 
        
#             """
#     def __init__(self, path, block_size=512, dtype=None):
#         self.file_path = path
#         self.dtype = np.dtype(dtype)
#         self.block_size = block_size

#         # Open the file and create the mmap object
#         self.file = open(self.file_path, 'rb')
#         self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
#         self.mm.madvise(mmap.MADV_RANDOM)
#         self.length = (os.path.getsize(self.file_path)//self.dtype.itemsize) - self.block_size - 1
        
    
#     def __len__(self):
#         f'ONLY IN distributed training: Works okay till 400 million, on a 125GB RAM card. After, that memory explodes, and kills processes'
#         f'Actual length of dataset is slightly greater than 9billion. '
#         return 400_000_000
#         # return self.length
    
#     def __getitem__(self, index):
        
#         # Calculate the byte offset for the requested index
#         offset = (index) * self.dtype.itemsize
#         # Calculate the number of bytes to read (block_size elements)
#         num_bytes = (self.block_size + 1) * self.dtype.itemsize # +1 because we are extracting both x and y from the same block
#         # Read the bytes and create a NumPy array from them
#         self.mm.seek(offset)
#         buffer = self.mm.read(num_bytes)
#         array = np.frombuffer(buffer, dtype=self.dtype)
#         x = array[:-1]
#         y = array[1:]
        
#         x_tensor = torch.from_numpy(x.astype(np.int64)).type(torch.long)
#         y_tensor = torch.from_numpy(y.astype(np.int64)).type(torch.long)
#         return x_tensor, y_tensor


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
        

    
    def __iter__(self, n= None):
        'PAg: custom n: how many iterations do you want. By default, it is self.n//self.bs'
        return self.batch_generator(n)
    
    
    def batch_generator(self, n= None):
        f'only uniquely samples 63% of all the data'
        
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
        



def _round_to_multiple(number,multiple): return int(math.ceil(number/multiple)*multiple)

class customDistributedDL(DistributedDL):
    _default='dl'
    
    # def __init__(self,dl,rank=None,world_size=None,device=None):
    #     super().__init__(dl, rank=rank, world_size=world_size, device = device)
        # self.dl.n = _round_to_multiple(self.dl.n, self.world_size)// self.world_size
    
    def __iter__(self):
        return iter(self.dl)



# class RandomSubsetSampler():
#     """
#     A custom sampler that generates random subset of a given dataset.
#     Used specifically for the valid dataset.
#     In our case, we use validation dataset only for checking if the model is training properly. 
#     Every n  training iterataions (here, 10,000), we check val_loss on this subset. 
#     If the val_loss is better than the best val_loss, we save the model in the form of a checkpoint. 

#     Args:
#         dataset (Dataset): The dataset to sample from.
#         subset_size (int): The size of the random subset to generate.

#     Attributes:
#         dataset (Dataset): The dataset to sample from.
#         subset_size (int): The size of the random subset to generate.
#     """
#     def __init__(self, dataset, subset_size=1000):
#         self.dataset = dataset
#         self.subset_size = subset_size
    
#     def __len__(self):
#         """
#         Returns the size of the random subset.

#         Returns:
#             int: The size of the random subset.
#         """
#         return self.subset_size
    
#     def __iter__(self):
#         """
#         Generates an iterator over the indices of the random subset.

#         Returns:
#             iter: An iterator over the indices of the random subset.
#         """
#         idxs = torch.randint(len(self.dataset), size=(self.subset_size,))
#         return iter(idxs)
    

# if __name__ =='__main__':
    
    #Just for testing

    # from config import OpenWebTextConfig
    # data = OWTData(OpenWebTextConfig().default_cache_dir/'train.bin', block_size=512, dtype=np.uint16)
    # print(data.length)
    # x,y = data[len(data)-1]
    # print(x.shape, y.shape)
    