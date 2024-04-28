# FastLLM
## LLM Training Made Quick and Easy


A concise template to train LLMs, using fast.ai and PyTorch. Only focus on the most important parts (data and model, and maybe training strategies), without writing, or scrolling through unnecessary code. 

<p align="center">
  <img src=".assets/cover.webp" width="35%">
</p>


## Whats special about this repo?
This repo is a very _clean pythonic implementation_ of LLM pipelines.
- _Clean_ means that different components are clearly separated, and initialized using intuitive function arguments. 
- _Clean_ also means that training scripts are very minimal, which is a result of high level abstractions. This ensures you don't have to scroll through unnecessary code. 

Without going into much details here to keep this README concise, see [here](/.docs/whatMakesThisRepoUnique.md) for more details and design choice justifications. 

## This Repo's components in a nutshell


<table>
<tr>
<td>


- Data source: huggingface datasets
- Tokenizer: Tiktoken 
- Model: standard GPT architecture with Flash Attention

</td>
<td>

- Trainer function:  Learner class provided by fast.ai
- Distributed Training: HuggingFace Accelerate (based on Torch DDP)
- Distributed Backend: NCCL

</td>

<td>

- Progress Logging: Weights and Biases
- Precision: bf16
- Loss function: CrossEntropyLossFlat (pytorch)

</td>

</tr>
</table>





###

# Get Started. 

## Prerequisites

#### Hardware

- (Extensively Tested on, but not strictly required) NVIDIA GeForce: RTX 3090, or similar GPU hardware. More number of GPUs, the better. 
- CUDA: 12.4
- Driver Version: 550.54.14

#### Software
- Python 3.10 (preferably in a Conda environment)
- Install necessary libraries: `pip install -r requirements.txt`

## Run LLM Training

 ### Want to quickly run the training, on a single GPU, with no adjustments? 

- `python train.py` 

- This runs a ~125M standard GPT architecture model on [Simple Wiki](https://huggingface.co/datasets/wikipedia) (~51M tokens) using a standard Adam Optimizer. Model is checked for performance on validation set every 1000 iterations, and saved if the validation loss is the best one encountered yet. 

### But if you have multiple GPUs... (Distributed Training)

- First you need to set up configurations for distributed training. 

    `accelerate config`

    Most defaults work well. Just set up  "distributed_type" (multi-gpu/multi-node?),  "num_machines",  "num_processes" (accross all the machines combined), and "machine_rank". An example of the values that may be suitable for your config is shown in `learner/accelerate_config_example.yml`


- Next, simply run

    `accelerate launch train.py`

    LLM training runs can be very long, so you may want to just launch this process in the background. `nohup` is a good solution for that. 

    `nohup accelerate launch train.py &`

### Optional (but very useful): Log progress to W&B 

- [W&B](https://wandb.ai/) logging

    -  Before logging any info to W&B (Weights and Biases), you need to setup a config file. This is very simple. 
    
        `wandb init`

        If you're setting up for the first time, you will have to paste an API key, which you will get by navigating to the [wandb website](https://wandb.ai/), under User Settings (on the top right corner). Then, Create (or set) a project, where all the different training runs will be logged. Each project is like a folder, that organizes multiple training runs together, and separates them from a different project. 
    
    - Edit `train.py`
        Set `log_wandb` to `True`. 
        Make sure that `project` is set to the same project you used in `wandb init` configuration. 




## Customize your training run

### Customize Model
See `model/README.md` to see guidelines of building your own pytorch model.

### Customize Data
See `data/README.md` to see guidelines of downloading custom datasets from huggingface datasets. 

### Customize Training process

- If you wish to introduce changes to the training process itself (including optimization strategy, grad accumulation, etc etc), you need to do that using the Learner Class. Visit [fast.ai documentation](https://docs.fast.ai/) to explore this. For people unfamiliar with fast.ai, just open a github issue, I'll try to look into it and incorporate it as a training option. 



## TODO
1. Model Distribution across GPUs?
2. Instruct Finetuning of model
3. Evaluation on benchmarks
4. Combining multiple datasets for training?

## Known Issues
1. CUDA version 12.3, and Driver Version: 525.x doesn't seem to work well with the NCCL framework. Apparantly, data can't be synchronized between GPUs via P2P. 

    However, updating to the above mentioned Hardware Config works well. If you are not willing to do that, one way to get around this (although training is much slower), is to disable P2P sharing, which means, data will be synchronized via CPU. Do this by setting env variable in the python script `os.environ['NCCL_P2P_LEVEL']='NVL'`. 



## Cite
If you find this repository useful in your research or work, please consider citing it:
```
@misc{fastllm,
  title={LLM Training Made Quick and Easy},
  author={Palaash Agrawal},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PalaashAgrawal/fastllm}},
}
```



## Star History
<div style="display: flex; justify-content: center;">
  <a href="https://star-history.com/#palaashagrawal/fastllm&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=palaashagrawal/fastllm&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=palaashagrawal/fastllm&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=palaashagrawal/fastllm&type=Date" style="width: 75%;" />
    </picture>
  </a>
</div>

