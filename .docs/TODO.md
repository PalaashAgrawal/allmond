# TODO

1. ~~Model parallelization across GPUs (instead of data parallelization) ?~~
2. ~~Quantization~~
3. Instruct Finetuning of model
4. Evaluation on benchmarks
5. Combining multiple datasets for training? 
6. ~~FSDP (instead of 1.)~~ (Done)
7. CPU Offloading in FSDP. 
    - `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cpu! (when checking argument for argument target in method wrapper_CUDA_nll_loss_forward)`
7. QLoRA (instead of (2). Quantization during training isnt without LoRA doesnt make sense, because gradients will be zero)


