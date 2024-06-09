"""
Contains functionalities to download and tokenize unlabeled text datasets
"""

import datasets
from tqdm import tqdm 
import numpy as np
import os
from pathlib import Path


from .config import config_dict
from model.tokenizer import Tokenizer



class unlabeledDataset():
    '''
    given huggingface dataset name, download the dataset using the datasets library
    dataset_name (str): name of the dataset to download (usually in the format "author/dataset_name")
    n_proc (int): number of parallel processes to use for downloading the dataset
    
    returns:
    datasets.Dataset: The downloaded dataset. Containing the train, and optionally the test and validation splits, each in the form of <class 'datasets.arrow_dataset.Dataset'>
    
    TODO:
    1. Add support for batch processing in datasets.map (batched = True) (see source code https://github.com/huggingface/datasets/blob/2.18.0/src/datasets/arrow_dataset.py#L2867)
    '''
    
    def __init__(self, datasetConfig , n_proc=8, cache_dir=None, force_redownload = False):
        """
        Download the dataset, tokenize it using an encoder and save it back to disk for fast retrieval during LLM training. 
        By default, saves the dataset into train.bin and val.bin (if config.split_into_train_val = True)

        Args:
            dataset_name (str): The name of the dataset to load. 
                See dataset.list_datasets (https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.list_datasets) 
                for list of all available datasets
            n_proc (int): The number of processes to use for loading the dataset.
            cache_dir (str): The directory to cache (save/read/write) the dataset. Defaults to None. If cache_dir is None, value from config class is used. 
            **kwargs: Additional keyword arguments to pass to the dataset loader.
        """
        
        
        self.config = datasetConfig
        self.n_proc = n_proc
        self.cache_dir = Path(cache_dir or datasetConfig.default_cache_dir).expanduser()
        
        
        self.force_redownload = force_redownload
        self.splits = ['train'] + [getattr(self.config, 'split_name', 'val')] if getattr(self.config, 'split_into_train_val', True) else []
        
        
        
        try: 
            #Assuming that datasets always returns "train" and "test"
            self.dataset = datasets.load_dataset(
                                                self.config.dataset_name,
                                                data_dir=self.cache_dir,
                                                num_proc=self.n_proc,
                                                trust_remote_code=True,
                                                cache_dir=self.cache_dir,
                                                download_mode = 'force_redownload' if self.force_redownload else 'reuse_cache_if_exists',
                                                **getattr(self.config, 'kwargs',{}),
                                                )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        

        self.cache_dir.mkdir(parents=True, exist_ok=True)
            

        
        
    def split(self, pct = 0.9995, val_name = 'val'):
        '''
        Split the dataset into training and validation sets.
        pct (float): The percentage of the dataset to use for the training set. Defaults to 0.9995.
        val_name (str): The name to use for the validation set. Defaults to 'val'. Other popular names "validation", "dev", "test". 
        
        returns:
        datasets.Dataset: The training set.
        datasets.Dataset: The validation set.
        '''
        
        
        pct = getattr(self, 'split_pct', None) or pct
        
        self.val_name = val_name
        
        self.dataset = self.dataset['train'].train_test_split(test_size = 1-pct) #splits into train and "test", which we later rename to val
        self.dataset[f'{self.val_name}'] = self.dataset.pop('test') #rename the test set to val
        
        
        return (self.dataset[key] for key in self.splits)
    
    
    def tokenize(self, encoder_fn):
        """
        Tokenizes the dataset using the given encoder and saves the tokenized dataset to disk.

        Args:
            encoder_fn: A function that takes an example and returns the tokenized version of the example.
                This function is used to tokenize the text in each example of the dataset. 
            save_tokens_to_disk (bool, optional): Whether to save the tokens to disk. Defaults to True.
            save_path (str or Path, optional): The path to save the tokenized dataset. Defaults to None.

        Returns:
            list or None: If `save_tokens_to_disk` is True, returns a list of paths where the tokenized dataset is saved.
                If `save_tokens_to_disk` is False, returns the tokenized dataset.

        Raises:
            AssertionError: If `encoder_fn` is not callable.

        Note:
            The tokenized dataset is saved as binary files using the `dtype` format of the encoder.

        """
        
        def _text_extractor(f, example):
            """
            A wrapper function that can be customized to work with different process functions.
            It directly takes an example, allowing process_function to work on the text.
            """
            return f(example['text'])

        tokens = self.dataset.map(lambda example: _text_extractor(encoder_fn, example),
                                       remove_columns=['text'],
                                       desc="tokenizing the splits",
                                       num_proc=self.n_proc)
        return tokens

        
         
        
    def process_dataset(self, encoder, save_tokens_to_disk=True, save_path=None):
        """
        Split and tokenize dataset.

        Args:
            encoder (Tokenizer): An instance of the Tokenizer class used for tokenization.
            save_tokens_to_disk (bool, optional): Whether to save the tokens to disk. Defaults to True.
            save_path (str, optional): The path to save the tokens. Defaults to None.

        Returns:
            list: A list of paths where the tokenized dataset is saved.

        Raises:
            AssertionError: If the encoder is not an instance of the Tokenizer class.

        """
        
        assert isinstance(encoder, Tokenizer), f"encoder must be an instance of Tokenizer class. Got {type(encoder)}"
        self.encoder = encoder        
        
        save_path = save_path or self.cache_dir
        self.paths = [save_path/f'{o}.bin' for o in self.splits]
        
        if self._check_data_on_disk(): return self.paths
        
        #split dataset
        if getattr(self.config, 'split_into_train_val', True):
            self.split_pct = getattr(self.config, 'split_pct', None)
            self.train, self.val = self.split(val_name=getattr(self.config, 'split_name', 'val'))
            
        
        #tokenize dataset
        tokens =  self.tokenize(encoder_fn=self.encoder.tokenize_dataset)
        if not save_tokens_to_disk: return tokens
        
        #save tokens to disk
        self._save_tokens_to_disk(tokens, path = save_path, dtype = self.encoder._get_numpy_dtype())
        return self.paths
        
        
        
        
        
        
    def _save_tokens_to_disk(self,  tokens, num_shards=1024, path =None, dtype = None):
        """
        Save the tokenized dataset to disk. 

        Args:
            tokens (dict): A dictionary containing the tokenized dataset.
            num_shards (int, optional): The number of shards to split the dataset into. Defaults to 1024.
            path (Optional, str or Pathlike): location where tokens will be saved. By default, it is saved in the self.cache_dir.

        Returns: None

        Notes:
            This method saves the tokenized dataset to disk by splitting it into multiple shards.
            Each shard is saved as a binary file with the format '<split>.bin', where '<split>' is the name of the dataset split.
            The tokenized data is stored in a numpy memmap array for efficient storage and retrieval.

        """
        
        save_pth = Path(path or self.cache_dir)
        
        dtype = dtype or np.uint64
        
        for split, dset in tokens.items():
            
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = save_pth/f'{split}.bin'
            
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            idx = 0
            for batch_idx in tqdm(range(num_shards), desc=f'writing {filename}'):
                # batch together samples for faster write
                batch = dset.shard(num_shards=num_shards, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # write into memmap
                arr[idx:idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)

            # save to disk
            arr.flush()
        
        #also save the tokenizer name used to save the tokens
        with open(save_pth/'tokenizer_name.txt', 'w+') as f:
            f.write(self.encoder.encoder_model)

    def _check_data_on_disk(self):
        """
        Check if the tokenized dataset already exists on disk.
        Also check if the tokenizer used is the same as the one used to save the tokens. If not, we retokenize the dataset
        """
        
        #check if tokenizer used is the same as the one to previously save the tokens
        if not (self.cache_dir/'tokenizer_name.txt').exists(): 
            print("Can't find any tokenizer info. Forcing (Re)tokenization of dataset.")
            return False
        
        with open(self.cache_dir/'tokenizer_name.txt', 'r') as f:
            tokenizer_name = f.read()
        if tokenizer_name != self.encoder.encoder_model: 
            print("Tokenizer does not match the tokenizer used to save the dataset. Retokenizing.")
            return False
        
        
        
        
        return not self.force_redownload and all((self.cache_dir/f'{split}.bin').exists() for split in self.splits)
    




def download_dataset(dataset:str, encoder: Tokenizer, force_redownload = False):
    f'download dataset in tokenized form using encoder'    
    
    assert dataset in config_dict, f"{dataset} is not supported. Available datasets: {config_dict.keys()}"
    dataset_config = config_dict[dataset]()
    n_procs = max(1, int(os.cpu_count()-2)) #leave atleast 2 cores for other processes
    ds = unlabeledDataset(dataset_config, n_procs, force_redownload=force_redownload)
    paths = ds.process_dataset(encoder = encoder, save_tokens_to_disk = True)
    return paths



    
    