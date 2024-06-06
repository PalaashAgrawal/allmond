# aLLMond

## LLM Training Made Quick, Flexible and Easy



A concise framework to train LLMs quickly. You only need focus on the most important parts (data and model, and maybe training strategies), without writing, or scrolling through unnecessary code. 


<p align="center">
  <img src=".assets/cover.webp" width="35%">
</p>



## Whats special about this repo?
This repo is a very _clean pythonic implementation_ of LLM pipelines.
- _Clean_ means that different components are clearly separated, and initialized using intuitive function arguments. 
- _Clean_ also means that training scripts are very minimal, which is a result of high level abstractions. This ensures you don't have to scroll through unnecessary code. 


Without going into much details here to keep this README concise, see [here](/docs/whatMakesThisRepoUnique.md) for more details and design choice justifications. 

## Key Features of this repo


- Download any HF dataset quickly. 
- Define custom architectures, or download any HF model (for continual pretraining).
- Automatic setup of Data, Model and Tokenizer using high-level APIs.
- Easy setup of Distributed Training (single-GPU, DDP, FSDP, CPU Offloading etc.).
- Inbuilt support for Mixed Precision Training.
- Automatic checkpointing of Model based on best validation loss.



## This Repo's components in a nutshell


<table>
<tr>
<td>


- Data source: huggingface datasets
- Tokenizer: Tiktoken 
- Model: standard GPT architecture with Flash Attention
- Trainer function:  Learner class provided by fast.ai
- Loss function: CrossEntropyLossFlat (pytorch)


</td>
<td>


- Distributed Training: HuggingFace Accelerate
- Distributed Backend: NCCL
- Progress Logging: Weights and Biases
- Precision: bf16

</td>



</tr>
</table>





###

# Get Started. 

## Prerequisites

#### Hardware

- (Extensively Tested on, but not strictly required) NVIDIA GeForce: RTX 3090, or similar GPU hardware. More number of GPUs, the better. 
- CUDA: 12.4 (or higher)
- Driver Version: 550.X (or higher)

#### Software
- Python 3.10 (preferably in a Conda environment)
- Install necessary libraries: `pip install -r requirements.txt`

## Run LLM Training

 ### Want to quickly run the training, on a single GPU, with no adjustments? 

- `python train.py` 

- This runs a ~125M standard GPT architecture model on [Simple Wiki](https://huggingface.co/datasets/wikipedia) (~51M tokens) using a standard Adam Optimizer. Model is checked for performance on validation set every 1000 iterations, and saved if the validation loss is the best one encountered yet. 

### But if you have multiple GPUs... (Distributed Training)

- We use Huggingface Accelerate to carry out distributed training (which itself is built on top of PyTorch DDP). 
- Before you can launch distributed training using accelerate, you need to create certain configurations (using `accelerate config`) that tell acclerate the nature of distributed training For example, 
  - whether the training is distributed across multiple GPUs in a single node, or multiple nodes are involved, 
  - which machine is the main machine, 
  - whether you want DDP or FSDP based distributed training, etc. 

Navigate to [this](docs/accelerate.md) document for details.


- Next, simply run

    `accelerate launch train.py` 

    LLM training runs can be very long, so you may want to just launch this process in the background. `nohup` is a good solution for that. 

- We have provided some configs for both DDP and FSDP settings in the directory `configs`. To run using a specific config (without running `accelerate config` all over again), use the `--config_file` arg. 
  - to run Phi-3 with QLoRA, using DDP,  run `accelerate launch --config_file configs/singlemachine_DDP.yml train_phi3.py` 

### Optional: Log progress to W&B 

- [W&B](https://wandb.ai/) logging

    -  There is a very short process you need to first carry out to setup W&B your system. Navigate to [this](docs/wandb.md) document for deets.
    
    - Edit `train.py`
        Set `log_wandb` to `True`. 
        Make sure that `project` is set to the same project you used in `wandb init` configuration. 




## Customize your training run

### Customize Model
See `model/README.md` to see guidelines of building your own pytorch model OR import from an existing huggingface model.

### Customize Data
See `data/README.md` to see guidelines of downloading custom datasets from huggingface datasets. 

### Customize Training process

- If you wish to introduce changes to the training process itself (including optimization strategy, grad accumulation, etc etc), you need to do that using the Learner Class. Visit [fast.ai documentation](https://docs.fast.ai/) to explore this. For people unfamiliar with fast.ai, just open a github issue, I'll try to look into it and incorporate it as a training option. 

## TODO
See [this document](docs/TODO.md)


## Known Issues
see [this document](docs/knownIssues.md)


## Cite
If you find this repository useful in your research or work, please consider citing it:
```
@misc{allmond,
  title={aLLMond: LLM Training Made Quick and Easy},
  author={Palaash Agrawal},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PalaashAgrawal/allmond}},
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

