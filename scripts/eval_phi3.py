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


# model_id = 'microsoft/Phi-3-mini-4k-instruct'
model_id = 'meta-llama/Meta-Llama-3-8B'

model = GPT.from_hf(model_id, enable_qlora = True)


# state_dict = torch.load('checkpoints/gpt/Phi-3-mini-25.2M.pth')
# model.load_state_dict(state_dict['model'])


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


model.train()
model.evaluate(tasks = tasks)

