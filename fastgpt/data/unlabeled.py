import datasets
from tqdm import tqdm 
import numpy as np
import os
from pathlib import Path

class OpenWebTextConfig():
    dataset_name = 'openwebtext'
    default_cache_dir = Path('~/.cache/tinyUniverse/pretraining_data/').expanduser()
    split_into_train_val = True
    split_name = 'val' #Optional
    
    
    
    
    
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
    
    def __init__(self, datasetConfig: OpenWebTextConfig, n_proc=8, cache_dir=None, force_redownload = False, **kwargs):
        """
        Initialize the UnlabeledDataLoader object.

        Args:
            dataset_name (str): The name of the dataset to load. 
                See dataset.list_datasets (https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.list_datasets) 
                for list of all available datasets
            n_proc (int): The number of processes to use for loading the dataset.
            cache_dir (str): The directory to cache (save/read/write) the dataset. Defaults to None. If cache_dir is None, value from config class is used. 
            **kwargs: Additional keyword arguments to pass to the dataset loader.
        """
        
        # Load the dataset
        super().__init__()
        
        self.config = datasetConfig
        self.n_proc = n_proc
        self.cache_dir = Path(cache_dir or datasetConfig.default_cache_dir)
        self.force_redownload = force_redownload
        
        if self.cache_dir is not None: self.cache_dir.mkdir(parents=True, exist_ok=True)
                
        
        self.dataset = datasets.load_dataset(
                                            self.config.dataset_name,
                                            num_proc=self.n_proc,
                                            trust_remote_code=True,
                                            cache_dir=self.cache_dir,
                                            download_mode = 'force_redownload' if self.force_redownload else 'reuse_cache_if_exists',
                                            **kwargs
                                            )
        if getattr(self.config, 'split_into_train_val', True): 
            self.train, self.val = self.split(val_name = getattr(self.config, 'split_name', 'val')) #by default, split the dataset into train and val sets
        
        
    def split(self, pct = 0.9995, val_name = 'val'):
        '''
        Split the dataset into training and validation sets.
        pct (float): The percentage of the dataset to use for the training set. Defaults to 0.9995.
        val_name (str): The name to use for the validation set. Defaults to 'val'. Other popular names "validation", "dev", "test". 
        
        returns:
        datasets.Dataset: The training set.
        datasets.Dataset: The validation set.
        '''
        
        self.val_name = val_name
        
        self.dataset = self.dataset['train'].train_test_split(test_size = 1-pct)
        self.dataset[f'{self.val_name}'] = self.dataset.pop('test') #rename the test set to val
        
        
        self.splits = self.dataset.keys()
        
        return (self.dataset[key] for key in self.splits)
    
    
    def tokenize(self, encoder_fn, save_tokens_to_disk = True, save_path = None):
        """
        Applies the given process_function (tokenizer) to the dataset,
        with text extracted automatically inside the wrapper function.
        
        Returns: None if save_tokens_to_disk is True. (saves tokens to disk. Use these files directly)
        Else, returns the tokenized dataset.
        """
        
        def _text_extractor(f, example):
            """
            A wrapper function that can be customized to work with different process functions.
            It directly takes an example, allowing process_function to work on the text.
            """
            return f(example['text'])
        
        
        if self._check_data_on_disk():
            print(f'Tokenized dataset already exists at {self.cache_dir}. You can load the tokenized dataset directly from disk.')
            return None
            

        self.tokens = self.dataset.map(lambda example: _text_extractor(encoder_fn, example), 
                                       remove_columns=['text'], 
                                       desc="tokenizing the splits", 
                                       num_proc=self.n_proc, )
        
        if save_tokens_to_disk: self._save_tokens_to_disk(self.tokens, path = save_path)
        else: return self.tokens
        
    
    def _save_tokens_to_disk(self,  tokens, num_shards=1024, path =None):
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

        TODO:
            1. if .bin files already exist, directly load tokenized dataset. See if self._load_tokenized_from_bin works?
        """
        
        save_pth = Path(path or self.cache_dir)
        
        
        for split, dset in tokens.items():
            
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = save_pth/f'{split}.bin'
            
            arr = np.memmap(filename, dtype=np.uint64, mode='w+', shape=(arr_len,))

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

    def _check_data_on_disk(self):
        return not self.force_redownload and all((self.cache_dir/f'{split}.bin').exists() for split in self.splits)
    
                
    def _load_tokenized_from_bin(self, split, path=None):
        """
        Load the tokenized dataset from disk.

        Args:
            split (str): The name of the dataset split to load.
            path (Optional, str or Pathlike): The location where the tokenized dataset is saved. Defaults to None.

        Returns:
            np.memmap: The tokenized dataset loaded from disk.

        Raises:
            FileNotFoundError: If the tokenized dataset is not found on disk.

        Notes:
            This method loads the tokenized dataset from disk.
            The tokenized data is stored in a numpy memmap array for efficient storage and retrieval.
        """
            
        load_pth = Path(path or self.cache_dir)
        filename = load_pth/f'{split}.bin'
        
        if not filename.exists():
            raise FileNotFoundError(f"Tokenized dataset not found at {filename}")
        
        return np.memmap(filename, dtype=np.uint64, mode='r', shape=(len(self.dataset[split]),))










class TiktokenTokenizer():
    "Tiktoken tokenizer for `lang`"
    def __init__(self, from_model = "gpt2"):
        """
        Initialize the TiktokenTokenizer class.
        from_model (str): The model to use for encoding.
        """
        try: import tiktoken
        except ImportError:
            raise Exception('Tiktoken module is missing: run `pip install tiktoken==0.6.0`')
        
        self.encoder_model = from_model
        self.encoder = tiktoken.get_encoding(self.encoder_model)
    
    def get_vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.
        Returns:
            int: The vocabulary size.
        """
        
        return self.encoder.n_vocab()
    
    
    def encode(self, text: str, ignore_special_tokens = True, batch=False):
        
        """
        Encodes the given text into tokenized format.
        
        Args:
            text (str): The input text to be encoded.
            ignore_special_tokens (bool): Whether to ignore special tokens during encoding. Defaults to True.
            batch (bool): Whether to encode the text in batch mode. Defaults to True.
        
        Returns:
            list or list of lists: The encoded tokenized representation of the input text.

        TODO: 
        1. accomodate all methods from tiktoken. 
        2. write function to automatically detect whether batches arrive as single peice or in batches. Batches can be both lists or dataloader generator objecst. How do you unify the representation of lists and dataloader generator objects?
        3. Make API compatible with fastai text API
        
        """
        
        if batch: return self.encoder.encode_ordinary_batch(text) if ignore_special_tokens else self.encoder.encode_batch(text) 
        else: return self.encoder.encode_ordinary(text) if ignore_special_tokens else self.encoder.encode(text)
        
    
    def tokenize_dataset(self, text):
        f'text is a row from Datasets.Dataset object.'
            
            
        ids = self.encode(text, ignore_special_tokens = True)
        ids.append(self.encoder.eot_token)#add the end of text token, e.g. 50256 for gpt2 bpe
        # note acc to Karpath. BUT WHY?: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out
        
             
    def decode(self, tokens: list, ignore_special_tokens = True, batch=True):
        """
        Decodes the given tokens into text.
        
        Args:
            tokens (list): The input tokens to be decoded.
            ignore_special_tokens (bool): Whether to ignore special tokens during decoding. Defaults to True.
            batch (bool): Whether to decode the tokens in batch mode. Defaults to True.
        
        Returns: str or list of str: The decoded text representation of the input tokens.
        """
        
        if batch: return self.encoder.decode_batch(tokens) if ignore_special_tokens else self.encoder.decode_batch(tokens) 
        else: return self.encoder.decode_ordinary(tokens) if ignore_special_tokens else self.encoder.decode(tokens)
        

if __name__ == "__main__":
    n_procs = max(1, int(os.cpu_count()-2)) #leave atleast 2 cores for other processes

    encoder = TiktokenTokenizer()    

    ds = unlabeledDataset(OpenWebTextConfig(), n_procs)
    ds.tokenize(encoder.tokenize_dataset, save_tokens_to_disk = True) 