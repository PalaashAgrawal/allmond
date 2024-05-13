## Setting up Huggingface Accelerate for distributed training across GPUs

First you need to set up configurations for distributed training. In your terminal, run

`accelerate config`

This will create a yaml configuration file in a default location (such as `~/.cache/huggingface/accelerate/default_config.yml`)

Most defaults work well. Just set up  "distributed_type" (multi-gpu/multi-node?),  "num_machines",  "num_processes" (accross all the machines combined), and "machine_rank". An example of the yaml values that may be suitable for your config is shown below. 


### Example 1
This particular example is meant for DDP based multi-GPU training, within a single node.

```
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",
  "downcast_bf16": false,
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "bf16",
  "num_machines": 1,
  "num_processes": 4,
  "rdzv_backend": "c10d",
  "same_network": True,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false
}
```

### Example 2
This example shows FSDP based multi-GPU, single node training (with CPU offloading)

```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: true
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: Phi3DecoderLayer #in Phi3, the basic block is named Phi3DecoderLayer. This is the class that will be wrapped by FSDP
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```


## NOTE: Usage of FSDP
For larger models, you might want to use a FSDP strategy, where model layers are also split accross different GPUs, along with the data. In my experience, you should always define a wrap policy `fsdp_auto_wrap_policy` as `TRANSFORMER_BASED_WRAP`, and define the name of the class to wrap `fsdp_transformer_layer_cls_to_wrap`, with the name of the basic transformer block class used in your code. 

For example, 
- in the default architecture (defined in `model/gpt.py`), the basic transformer class is named as `TransformerBlock`. 
- In Huggingface based Phi3 model, the basic block is named Phi3DecoderLayer


TODO:
check if phi3 can work without CPU offloading