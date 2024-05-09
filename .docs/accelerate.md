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
This example shows FSDP based multi-GPU, single node training (with no CPU offloading)

```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: SIZE_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_min_num_params: 100000000
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```