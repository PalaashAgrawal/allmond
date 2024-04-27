from data.unlabeled import TiktokenTokenizer, download_dataset
from data.config import OpenWebTextConfig, WikipediaSimpleConfig
from data.loader import dataloader, customDistributedDL

from model.gpt import GPT

from learner.callbacks import save_checkpoints
from learner.LLMLearner import LLMLearner

from fastai.text.all import *
from fastai.distributed import *
from fastai.callback.wandb import *
import wandb


# os.environ['NCCL_P2P_DISABLE']='1' #without this, NCCL (via accelerate.prepare) gets stuck during synchronization. 
# os.environ['NCCL_P2P_LEVEL']='NVL'
#You see an error like
# RuntimeError: Exception occured in `DistributedTrainer` when calling event `before_fit`:  DDP expects same model across all ranks, but Rank 1 has 221 params, while rank 0 has inconsistent 0 params.
#This seems like a NVIDIA problem due to the "asynchronous nature of CUDA kernels". Well then how can i make them synchronous? Asynchronous anyways does not look like a sound choice for parallel CUDA operations.
#Try updating CUDA version later.



#________________________________________wandb_____________________________________
log_wandb = False

project = 'tinylm'
dataset = "wikisimple"
mode = 'scratch'

# ________________________________________hyperparams and settings_________________________

bs=20 #each GPU gets bs = 20, works good for a 24GB GPU
block_size = 512
valid_sampler_size = 1000 #how many samples to use for validation. This is only used to check if validation loss is better than best_valid_loss, so that a checkpoint can be saved. Karpathy uses 200 random points
validate_every = 1000 #1000 iterations, each iteration is bs*total_GPUs inputs

#________________________________________Model_____________________________________


model = GPT(block_size=block_size)
tokenizer = TiktokenTokenizer(from_model = "gpt2")

#________________________________________data_____________________________________

rank0_first(lambda: download_dataset(dataset = dataset, encoder = tokenizer)) #check if data exists

train_dl = dataloader(WikipediaSimpleConfig().default_cache_dir/'train.bin', bs = bs, block_size=block_size, 
                      dtype=tokenizer._get_numpy_dtype())
valid_dl = dataloader(WikipediaSimpleConfig().default_cache_dir/'val.bin', bs = bs, block_size=block_size, 
                      dtype=tokenizer._get_numpy_dtype(), 
                      sample_size = valid_sampler_size)

if num_distrib(): #distributed training
    train_dl, valid_dl = customDistributedDL(train_dl), customDistributedDL(valid_dl)
    
if not rank_distrib(): print(f'training {str(model)} on {train_dl.n} tokens')
    

dls = DataLoaders(train_dl, valid_dl)
dls.c = model.head.out_features

#________________________________________Trainer_____________________________________

check_and_save_model = save_checkpoints(dir = Path('checkpoints'), 
                                                model_name = 'gpt', 
                                                checkpoint_name = str(model), 
                                                every_iters = validate_every)
cbs = [check_and_save_model]
if log_wandb:
    cbs.append(WandbCallback())
    wandb.init(project=project, name = f"{id}_{str(model)}_{mode}")


learn = LLMLearner(dls, 
                model, 
                loss_func=CrossEntropyLossFlat(), 
                metrics=[accuracy, Perplexity()],
                cbs = cbs,
                path = check_and_save_model.path,
                model_dir=check_and_save_model.model_dir, 
                ).to_bf16()

#check and load previous checkpoint. Doesnt make sense to do it within the callback, because all callbacks are initialized in the Learner before they are even called
learn.check_and_load_learner(check_and_save_model.checkpoint_name, device = rank_distrib() if num_distrib() else None)


with learn.distrib_ctx():
    learn.fit_one_cycle(1, 1e-4)
