#script to evaluate phi3 without any finetuning

import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT
from fastai.text.all import *
from data.unlabeled import download_dataset
from fastai.distributed import *
from learner.LLMLearner import LLMLearner
from data.loader import memmapDL



model = GPT.from_hf('microsoft/Phi-3-mini-4k-instruct', enable_qlora = True)
dataset = "orcamath"
bs = 1
valid_sampler_size = 1000 #how many samples to use for validation. This is only used to check if validation loss is better than best_valid_loss, so that a checkpoint can be saved. Karpathy uses 200 random points
validate_every = 1000 #1000 iterations, each iteration is bs*total_GPUs inputs
qlora = True

train_path, valid_path = rank0_first(lambda: download_dataset(dataset = dataset, encoder = model.tokenizer)) #check if data exists, download only for rank0 GPU. 
train_dl = memmapDL(train_path, bs = bs, block_size=model.block_size, 
                      dtype=model.tokenizer._get_numpy_dtype())
valid_dl = memmapDL(valid_path, bs = bs, block_size=model.block_size, 
                      dtype=model.tokenizer._get_numpy_dtype(), 
                      sample_size = valid_sampler_size)

dls = DataLoaders(train_dl, valid_dl)
dls.c = model.vocab_size

learn = LLMLearner(dls, 
                model, 
                opt_func = partial(OptimWrapper, opt=torch.optim.AdamW),
                loss_func=CrossEntropyLossFlat(), 
                metrics=[accuracy, Perplexity()],
                ).to_bf16()
learn.path = Path('checkpoints/') #local path to save/load checkpoints
learn.model_dir = 'gpt'

learn.load(Path('Phi-3-mini-25.2M'))




tasks = [
    'mmlu_abstract_algebra', 
         'mmlu_college_mathematics',
         'mmlu_elementary_mathematics',
         'mmlu_high_school_mathematics',
         'mmlu_formal_logic',
         'arithmetic',
        #  'minerva_math',
        #  'hendrycks_math',
         'mathqa', 
        #  'gsm8k',
         ]
learn.model.evaluate(tasks = tasks, batch_size = 16)

