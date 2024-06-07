#script to evaluate phi3 without any finetuning
#_________________IMPORTS_______________

import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT

#________________________________


model = GPT.from_hf('microsoft/Phi-3-mini-4k-instruct', enable_qlora = False)
# mmlu_results = simple_evaluate(model, tasks = ['mmlu_high_school_computer_science'])
model.evaluate(tasks = ['mmlu_high_school_computer_science'], save_path = '/home/agrawalp2/allmond/.eval_results' )
#save mmlu results dict
# with open('/home/agrawalp2/allmond/.eval_results/phi3_mmlu_results.json', 'w') as f: json.dump(mmlu_results, f)
    
# print(make_table(mmlu_results))


#

#you can probably modify cli_evaluate in https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py to make everything easier, and then run this entire script using torch run to parallelize


