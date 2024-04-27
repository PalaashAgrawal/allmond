# Trainer (aka Learner)
This handles training of a model on a given dataset. 

- `learner.py`

    Modified fastai Learner class to incorporate efficient training of LLMs. 

    Features

        - Iter/epoch info can be saved along with model/opt states.
        - Learner can resume training from a given epoch and iter
        - bug-free distributed training in the case of resume-from-particular-iter case. (There is a bug in the Learner class  in the `_with_events` function). 

- `callback.py`

callback definition, that checks the model performance every `n` iterations on `m` randomly chosen data points from the validation dataset, and saves the model/opt/iter states if the current validation loss is lower than the best encountered validation loss


### For Distributed Training...

This repo uses [huggingface 🤗 accelerate](https://huggingface.co/docs/accelerate/en/index) to handle distributed training. 
Before running your LLM training in distributed settings, you need to write some config using `accelerate config`. `accelerate_config_example.yml`  shows an example of the values for a standard multi-gpu training run. Configs are saves as yaml file in a default cache directory `~/.cache/huggingface/accelerate/default_config.yml`
