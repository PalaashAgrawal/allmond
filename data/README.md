# Data 

- unlabeled.py contains scripts to download a dataset from huggingface datasets, and also for tokenizer methods (to save the text as tokenized data in .bin files)


- loader.py is meant to retrieve data from  saved .bin files (containing tokenized data), and return dataloaders. s



## Helpers related to Data (downloading, creating dataloaders,etc).

1. `unlabeled.py` contains functions to download unlabeled text datasets (each dataset needs to have a config in the form of a class in `cofig.py`). 
    - use the `download_dataset(...)` function to download data using an id (string that represents name of the dataset, see config.py to see list of supported datasets. You can obviously add in more datasets there)
    - the tokenizer class is a template to be used to tokenize the dataset before saving to disk, AS WELL AS to find dtype of a given dataset. It is also used for inference (text decoding)
    


2. `loader.py` contains dataloader functions to create dataloader corresponding to a dataset.
    - `class memmapDL(DataLoader)` This is not a standard pytorch dataloader, since dataloaders dont retrieve items from an underlying dataset item. Instead, dataloaders randomly index a numpy memory map. 
    - We have also defined a dataloader specifically for fastai style distributed training dataloaders `class distributedMemmapDL(DistributedDL)`. It is a very basic wrapper around the `memmapDL` class to indicate to the underlying `__getitem__` function not to access a dataset object, but the `__iter__` method of memmapDL. 



## Customize your Data

- For other datasets (assuming they exist on HuggingFace Datasets), simply modify the `download_dataset` function arguments. 
    - But make sure that a corresponding config class exists in `data/config.py` and is included in the `config_dict` dictionary (in the same python script). This dictionary will be accessed by `download_dataset` to perform necessary operations on the dataset. 


- Your config class should have the following attributes
    - `dataset_name`: string id as required by huggingface datasets to identify a dataset
    -  `default_cache_dir`: Path on your local machine, where data will be stored/retrieved from
    - split_into_train_val: bool, whether to split dataset into validation dataset as well, or just get one single training text file. 
    - (Optional) `split_name`: what name to give to the split file. val and test are some popular names. 
    - kwargs: dict, Used to provide other arguments needed by huggingface datasets. Eg, in the [wikipedia dataset](https://huggingface.co/datasets/wikipedia), there are many versions of the dataset. In order to get a particular dataset, you need to specify the `name` argument. 
    - `split_pct`: what percentage to split the training and testing dataset on.

