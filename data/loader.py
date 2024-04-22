import numpy as np
from fastai.text.all import *
from fastai.data.load import _FakeLoader
import mmap

from fastai.distributed import *


# class OWTData():
#     def __init__(self, path, block_size=1024, dtype = None):
#         """
#         Dataset class for OpenWebText.

#         Parameters:
#         path (str): The path to the data file.
#         block_size (int, optional): The block size for reading the data file. Defaults to 128.
#         dtype: np datatype: VERY IMPORTANT: the format with which to read data. Make sure that this is the same format with which data was saved. 
        
#         """
        
#         dtype = dtype or np.uint64
#         self.path = path
        
#         self.data = np.memmap(self.path, dtype=dtype, mode='r')
        
#         self.block_size = block_size
        
#     def __len__(self): 
        
#         data_len = len(self.data)
#         assert self.block_size < data_len, "block_size can't be larger than the data size"
#         return data_len - self.block_size - 1
        


#     def __getitem__(self, idx):
#         f'I think fastai expects input to be int64 format. to do .type, you first need to convert to np int64 (because it doesnt support uint64 conversion directly)'
        
#         x,y = np.array(self.data[idx:idx+self.block_size]), np.array(self.data[idx+1: idx+self.block_size+1]) 
#         return torch.from_numpy(x.astype(np.int64)).type(torch.long), torch.from_numpy(y.astype(np.int64)).type(torch.long)
    

class OWTData:
    """
    np.memmap leads to memory leaks. This python mmap implementation is better. 
    TODO:
    1. valid dataloader raises
    in __getitem__
        array = np.frombuffer(buffer, dtype=self.dtype, count=self.shape[1])
        ValueError: buffer is smaller than requested size 
        
            """
    def __init__(self, path, block_size=512, dtype=None):
        self.file_path = path
        self.dtype = np.dtype(dtype)
        self.block_size = block_size

        # Open the file and create the mmap object
        # self.file = open(self.file_path, 'rb')
        # self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        # self.mm.madvise(mmap.MADV_RANDOM)
        # self.length = (os.path.getsize(self.file_path)//self.dtype.itemsize) - self.block_size - 1
        
    
    def __len__(self):
        f'ONLY IN distributed training: Works okay till 400 million, on a 125GB RAM card. After, that memory explodes, and kills processes'
        f'Actual length of dataset is slightly greater than 9billion. '
        return 400_000_000
        # return self.length
    
    def __getitem__(self, index):
        return None
        
        # # Calculate the byte offset for the requested index
        # offset = (index) * self.dtype.itemsize
        # # Calculate the number of bytes to read (block_size elements)
        # num_bytes = (self.block_size + 1) * self.dtype.itemsize # +1 because we are extracting both x and y from the same block
        # # Read the bytes and create a NumPy array from them
        # self.mm.seek(offset)
        # buffer = self.mm.read(num_bytes)
        # array = np.frombuffer(buffer, dtype=self.dtype)
        # x = array[:-1]
        # y = array[1:]
        
        # x_tensor = torch.from_numpy(x.astype(np.int64)).type(torch.long)
        # y_tensor = torch.from_numpy(y.astype(np.int64)).type(torch.long)
        # return x_tensor, y_tensor


# def generate_batches(file, block_size, batch_size, dtype):
#     # Determine the appropriate data file based on the split
#     # filename = 'train.bin' if split == 'train' else 'val.bin'
#     # data_path = os.path.join(data_dir, filename)
    
    
#     while True:  # This loop will make it a generator that yields indefinitely
#         # We recreate np.memmap every batch to avoid a memory leak
#         data = np.memmap(file, dtype=dtype, mode='r')
#         ix = torch.randint(len(data) - block_size, (batch_size,))
        
#         # Prepare the input batch (x) and the target batch (y)
#         x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
#         y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        
#         if device_type == 'cuda':
#             # Pin arrays x and y for asynchronous GPU transfer
#             x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#             # x, y = x.pin_memory(), y.pin_memory()
#         else:
#             x, y = x.to(device), y.to(device)
        
#         yield x, y  # Yield the batches instead of returning them

class dataloader(DataLoader):
    """
    attrgetter('bs','drop_last','dataset','fake_l','num_workers','offs','pin_memory')(self.dl)
    """
    # _noop_methods = 'wif before_iter after_item before_batch after_batch after_iter'.split()
    # for o in _noop_methods: exec(f"def {o}(self, x=None, *args, **kwargs): return x")
    # _methods = _noop_methods + 'create_batches create_item create_batch retain \
    #     get_idxs sample shuffle_fn do_batch create_batch'.split()
    # _default = 'dataset'
    def __init__(self, file, block_size, bs, dtype, device:str='cuda', seed:int = 42):
        # self.generator = self.batch_generator()
        # super().__init__(OWTData(file, block_size, dtype), bs = bs, device = device)
        
        self.file = file
        self.dataset = np.memmap(file, dtype = dtype, mode = 'r')
        self.block_size = block_size
        self.bs = bs
        self.dtype = dtype
        
        self.n = len(self.dataset)
        # self.n = 500_000_000
        self.device = device
        
        self._device_type = 'cuda' if 'cuda' in device else 'cpu'        
        
        # #other
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
        
        
    
    def __len__(self):
        # return 9_000_000_000//bs
        return self.n//self.bs
    
    def __iter__(self):
        return self.batch_generator()
    
    
    def batch_generator(self):
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
                # Pin arrays x and y for asynchronous GPU transfer
                x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
            else:
                x, y = x.to(self.device), y.to(self.device)
            b = (x,y)
            if self.device is not None: b = to_device(b, self.device)
            yield self.after_batch(b)
        
        
            # yield x, y  # Yield the batches instead of returning them
            
        self.after_iter()
        if hasattr(self, 'it'): del(self.it)
        
        




class customDistributedDL(DistributedDL):
    def __init__(self,dl,rank=None,world_size=None,device=None):
        super().__init__(dl, rank=rank, world_size=world_size, device = device)
    
    def __iter__(self):
        return iter(self.dl)


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
    

if __name__ =='__main__':
    
    #Just for testing

    from config import OpenWebTextConfig
    data = OWTData(OpenWebTextConfig().default_cache_dir/'train.bin', block_size=512, dtype=np.uint16)
    print(data.length)
    x,y = data[len(data)-1]
    print(x.shape, y.shape)
    
    
    
    # from unlabeled import TiktokenTokenizer
    
    # print(TiktokenTokenizer().decode([[t.item() for t in r] for r in (x,y)]))