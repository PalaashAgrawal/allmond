#script to evaluate phi3 without any finetuning

import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT



model = GPT.from_hf('microsoft/Phi-3-mini-4k-instruct', enable_qlora = False)

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
model.evaluate(tasks = tasks)

