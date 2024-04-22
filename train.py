from data.unlabeled import TiktokenTokenizer
from data.config import OpenWebTextConfig
from data.loader import dataloader, customDistributedDL

from model.gpt2 import GPT
from model.callbacks import save_and_load_model_checkpoints
from model.customLearner import customLearner

from fastai.text.all import *
from fastai.distributed import *

# from torch.utils.data import DataLoader



# import torch
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# from contextlib import nullcontext

os.environ['NCCL_P2P_DISABLE']='1' #without this, NCCL (via accelerate.prepare) gets stuck during synchronization. 
#You see an error like
# RuntimeError: Exception occured in `DistributedTrainer` when calling event `before_fit`:  DDP expects same model across all ranks, but Rank 1 has 221 params, while rank 0 has inconsistent 0 params.
#This seems like a NVIDIA problem due to the "asynchronous nature of CUDA kernels". Well then how can i make them synchronous? Asynchronous anyways does not look like a sound choice for parallel CUDA operations.
#Try updating CUDA version later.

bs=20 #each GPU gets bs = 20
block_size = 512
valid_sampler_size = 1000 #how many samples to use for validation. This is only used to check if validation loss is better than best_valid_loss, so that a checkpoint can be saved. Karpathy uses 200 random points
validate_every = 25 #1000 iterations, each iteration is bs*total_GPUs inputs

model = GPT(block_size=block_size)

tokenizer = TiktokenTokenizer(from_model = "gpt2")

train_dl = dataloader(OpenWebTextConfig().default_cache_dir/'train.bin', bs = bs, block_size=block_size, 
                      dtype=tokenizer._get_numpy_dtype())
valid_dl = dataloader(OpenWebTextConfig().default_cache_dir/'val.bin', bs = bs, block_size=block_size, 
                      dtype=tokenizer._get_numpy_dtype(), 
                      sample_size = valid_sampler_size//bs)

if num_distrib(): #distributed training
    train_dl, valid_dl = customDistributedDL(train_dl), customDistributedDL(valid_dl)
    



dls = DataLoaders(train_dl, valid_dl)
dls.c = model.head.out_features
    
check_and_save_model = save_and_load_model_checkpoints(dir = Path('checkpoints'), 
                                                model_name = str(model), 
                                                checkpoint_name = 'gpt2', 
                                                every_iters = validate_every)

    

learn = customLearner(dls, 
                model, 
                loss_func=CrossEntropyLossFlat(), 
                metrics=[accuracy, Perplexity()],
                cbs = [check_and_save_model],
                ).to_bf16()


with learn.distrib_ctx():
    learn.fit_one_cycle(1, 1e-4)
