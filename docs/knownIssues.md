1. CUDA version 12.3, and/or Driver Version: 525.x (or below) doesn't seem to work well with the NCCL framework. Apparantly, data can't be synchronized between GPUs via P2P. 

- However, updating to the Hardware Config mentioned in Prerequisites works well. I personally find [this](https://www.gpu-mart.com/blog/update-nvidia-drivers-on-windows-linux) guide the easiest way to update your NVIDIA drivers. 
    
- If you are not willing to do that, one way to get around this (although training is much slower), is to disable P2P sharing, which means, data will be synchronized via CPU. Do this by setting env variable in the main python script (train.py) `os.environ['NCCL_P2P_LEVEL']='NVL'`. 
