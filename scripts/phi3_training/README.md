script to finetune a Phi3 based model, using FSDP and CPU Offloading. 

Also provided is the config file that enables multi-GPU training using FSDP in the correct configuration. 

Make sure to copy the python script to the base directory level in order to run the script. 

accelerate launch --config_file <relative_path_to_this_config_file> train_phi3.py
