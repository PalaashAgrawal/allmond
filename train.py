from data.unlabeled import TiktokenTokenizer
from data.config import OpenWebTextConfig
from data.loader import OWTData, RandomSubsetSampler

from model.gpt2 import GPT
from model.callbacks import save_model_checkpoints
from model.customLearner import customLearner

from fastai.text.all import *
from fastai.distributed import *

from torch.utils.data import DataLoader



if __name__ == "__main__":
    
    
    bs = 20 #distributed training
    block_size = 512
    valid_sampler_size = 1000 #how many samples to use for validation. This is only used to check if validation loss is better than best_valid_loss, so that a checkpoint can be saved. Karpathy uses 200 random points
    
    
    
    
    model = GPT(block_size=block_size)
    
    
    tokenizer = TiktokenTokenizer(from_model = "gpt2")
    
    train_ds = OWTData(OpenWebTextConfig().default_cache_dir/'train.bin'    ,block_size=block_size, dtype=tokenizer._get_numpy_dtype())
    valid_ds = OWTData(OpenWebTextConfig().default_cache_dir/'val.bin'      ,block_size=block_size, dtype=tokenizer._get_numpy_dtype())
                       
    train_dl = DataLoader(train_ds, batch_size=bs)
    
    valid_dl = DataLoader(valid_ds, batch_size=2*bs,
                          sampler = RandomSubsetSampler(valid_ds, subset_size=valid_sampler_size),
                          )
    
    

    dls = DataLoaders(train_dl, valid_dl)
    dls.c = model.head.out_features
    
    check_and_save_model = save_model_checkpoints(dir = Path('checkpoints'), 
                                                  model_name = str(model), 
                                                  checkpoint_name = 'gpt2', 
                                                  every_iters = 5000)
    
    
    class check_training_stages(Callback):
        order=80
        def before_fit(self):
            print('before_fit')
        def before_epoch(self):
            print('before_epoch')
        def before_train(self):
            print('before_train')
        def before_batch(self):
            print('before_batch')
        def after_batch(self):
            print('after_batch')
        def after_train(self):
            print('after_train')
        def after_epoch(self):
            print('after_epoch')
        def after_fit(self):
            print('after_fit')
        
        


    
    learn = customLearner(dls, 
                    model, 
                    loss_func=CrossEntropyLossFlat(), 
                    metrics=[accuracy, Perplexity()],
                    # cbs = [check_and_save_model
                        #    , check_training_stages()
                        # ],
                    ).to_bf16()
    
    
    
    with learn.distrib_ctx():
        learn.fit_one_cycle(1, 1e-4)
