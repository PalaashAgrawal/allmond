from data.unlabeled import OpenWebTextConfig 
from data.loader import OWTData, RandomSubsetSampler

from model.gpt2 import GPT
from model.callbacks import save_model_checkpoints


from fastai.text.all import *
from torch.utils.data import DataLoader



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
