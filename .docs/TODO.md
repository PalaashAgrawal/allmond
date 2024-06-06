# TODO

- ~~Model parallelization across GPUs (instead of data parallelization) ?~~
- ~~Quantization~~
- ~~FSDP (instead of 1.)~~ (Done)
- ~~CPU Offloading in FSDP.~~
- ~~(HIGH PRIORITY) QLoRA (instead of (2). Quantization during training isnt without LoRA doesnt make sense, because gradients will be zero) (IS IT EVEN REQUIRED FOR SMALLER MODELS?)~~
    - FDSP+QLoRA Support (Currently, we train QLoRA in DDP mode)

- (TOP PRIORITY) Instruct Finetuning of model
- (TOP PRIORITY) Evaluation on benchmarks ([eleutherAI/lm-harness](https://github.com/EleutherAI/lm-evaluation-harness))
- Combining multiple datasets for training? 
- Tokenize and collate on the fly (you shouldnt have to save on disk)
- (TOP PRIORITY) gradient checkpointing?
- Automatic selection of largest batch size? (ElutherAI has functionality for that. Check the source code @[this link](https://github.com/EleutherAI/lm-evaluation-harness/blob/b24ac4b8eb7b32e30f45c16a5be78670dcb25f47/lm_eval/models/huggingface.py#L674)
- What components required to achieve OpenAI's [model spec guidelines](https://openai.com/index/introducing-the-model-spec/)?





