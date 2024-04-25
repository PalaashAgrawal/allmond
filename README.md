PAg: Implementation of GPT using fastai. Logic -- from nanogpt by Karpathy: https://github.com/karpathy/nanoGPT

TODO:

- ~~Implement GPT as is on Wikipedia data~~
- ~~Wandb callback integration~~
- Import a pretrained model of ~1B, and match model design specs



TODO this week:
- (Data): an estimate of relevant topic tree size in Wikipedia simple 
    - (Whats the size of Just Science related topics?)
    - what Wiki simple anyways? whats the alternative? i think wikipedia is much bigger than 50M tokens
    - What size do you get when you initialize multiple topics? (eg bio, physics, chemistry?)
- (Model):
    - solve model checkpoint loading problems
    - Implement LoRA finetuning
    - Test on Phi2 (2.7B). 
        - Setup pipeline (download, load weights, Lora initialization, testing (validation, does it throw any errors?))
        - We are interested in seeing the jump in performance when finetuned. Can we close in the gap of the perf of much larger models (eg. 56B or 70B?)
        - LorA weights = 500M maybe? then try 1B if even feasible. 
    - Uploading on huggingface? how does it work?
- (Eval):
    - Setup MMLU benchmarks

-(Other):
    - send draft mail to Colin regarding SGD5000 sharon approval. 





TODO Changes in fastai PR
1. in TrainEvalCallback - need to adjust pct_train
2. in fit_one_cycle - need to include start_iter argument . 


