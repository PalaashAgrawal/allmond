## Setting up Huggingface Accelerate for distributed training across GPUs

First you need to set up configurations for distributed training. In your terminal, run

`accelerate config`

This will create a yaml configuration file in a default location (such as `~/.cache/huggingface/accelerate/default_config.yml`)

Most defaults work well. Just set up  "distributed_type" (multi-gpu/multi-node?),  "num_machines",  "num_processes" (accross all the machines combined), and "machine_rank". An example of the yaml values that may be suitable for your config is shown below. This particular example is meant for multi-GPU training, within a single node.

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