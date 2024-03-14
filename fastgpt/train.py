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
        
    
    
    
class save_model_checkpoints(Callback):
    def __init__(self, dir = None, model_name = None, checkpoint_name = 'checkpoint', every_iters = 10000):
        self.path = Path(dir)
        self.model_name = Path(model_name)
        self.checkpoint_name = checkpoint_name
        self.every_iters = every_iters
        self.best_valid_loss = float('inf')
        
    
    def after_step(self): 
        
        f'check if validation loss is better than best_valid_loss'
                
        #I HATE THAT I WROTE THIS CODE. CHANGE THIS USING FASTAI'S API. self.learn.validate() #THIS CREATES AN ERROR. WIPES OUT SOME CALLBACKS RELATED TO BF16
                
        if self.path: self.learn.path = self.path
        if self.model_name: self.learn.model_dir = self.model_name
        
        if self.learn.training and self.learn.iter and self.learn.iter% self.every_iters ==0:
            
            # print(self.learn.iter)
            # self.learn.validate()

            # val_res = self.learn.validate() #THIS CREATES AN ERROR. WIPES OUT SOME CALLBACKS RELATED TO BF16
            # print('done')
            # val_loss = val_res[0]
            
            # if val_loss < self.best_valid_loss:
            #     self.best_valid_loss = val_loss
            #     self.learn.save(f'{self.checkpoint_name}', with_opt=True)


            accumulated_loss = 0.0
            count = 0
        
            for xb,yb in self.learn.dls.valid:
                
                with torch.no_grad():
                    pred = self.learn.model(xb.cuda())
                    loss_grad = self.learn.loss_func(pred, yb.cuda())
                    loss = loss_grad.clone()
                    accumulated_loss += loss
                    count+=yb.shape[0]
            
            loss = accumulated_loss/count

            if loss < self.best_valid_loss:
                self.best_valid_loss = loss
                self.learn.save(f'{self.checkpoint_name}', with_opt=True)
                

class RandomSubsetSampler():
    def __init__(self, dataset, subset_size=1000):
        self.dataset = dataset
        self.subset_size = subset_size
    
    def __len__(self): return self.subset_size
    
    def __iter__(self): 
        idxs = torch.randint(len(self.dataset), size=(self.subset_size,))
        return iter(idxs)


if __name__ == "__main__":
    
    
    bs = 16
    block_size = 512
    valid_sampler_size = 1000 #how many samples to use for validation. This is only used to check if validation loss is better than best_valid_loss, so that a checkpoint can be saved. Karpathy uses 200 random points
    
    
    
    model = GPT(block_size=block_size)
    
    train_ds = OWTData(OpenWebTextConfig().default_cache_dir/'train.bin'    ,block_size=block_size)
    valid_ds = OWTData(OpenWebTextConfig().default_cache_dir/'val.bin'      ,block_size=block_size)
                       
    train_dl = DataLoader(train_ds, batch_size=bs)
    
    valid_dl = DataLoader(valid_ds, batch_size=2*bs,
                          sampler = RandomSubsetSampler(valid_ds, subset_size=valid_sampler_size),
                          )
    

    dls = DataLoaders(train_dl, valid_dl)
    dls.c = model.head.out_features
    
    check_and_save_model = save_model_checkpoints(dir = Path('checkpoints/models'), 
                                                  model_name = str(model), 
                                                  checkpoint_name = 'gpt2', 
                                                  every_iters = 10000)

    
    learn = Learner(dls, 
                    model, 
                    loss_func=CrossEntropyLossFlat(), 
                    metrics=[accuracy, Perplexity()],
                    cbs = [check_and_save_model],
                    ).to_bf16()
    
    learn.fit_one_cycle(1, 1e-4)