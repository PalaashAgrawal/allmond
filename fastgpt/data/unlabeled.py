import datasets
from tqdm import tqdm 
import numpy as np
import os


dset = 'openwebtext'


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
    
    def __init__(self, dataset_name: str, n_proc = 8, **kwargs):
        # Check if the dataset exists
        
        # Load the dataset
        self.n_proc = n_proc
        self.dataset = datasets.load_dataset(dataset_name, num_proc = self.n_proc, **kwargs)
        self.train, self.val = self.split()
        
    def split(self, pct = 0.9995, val_name = 'val'):
        '''
        Split the dataset into training and validation sets.
        pct (float): The percentage of the dataset to use for the training set. Defaults to 0.9995.
        val_name (str): The name to use for the validation set. Defaults to 'val'. Other popular names "validation", "dev", "test". 
        
        returns:
        datasets.Dataset: The training set.
        datasets.Dataset: The validation set.
        '''
        
        self.dataset = self.dataset['train'].train_test_split(test_size = 1-pct)
        self.dataset[f'{val_name}'] = self.dataset.pop('test') #rename the test set to val
        
        return self.dataset['train'], self.dataset['val']
    
    
    def tokenize(self, encoder_fn, save_tokens_to_disk = True):
        """
        Returns a function that applies the given process_function (tokenizer) to the dataset,
        with text extracted automatically inside the wrapper function.
        """
        
        def _token_mapper(f, example):
            """
            A wrapper function that can be customized to work with different process functions.
            It directly takes an example, allowing process_function to work on the text.
            """
            return f(example['text'])

        self.tokens = self.dataset.map(lambda example: _token_mapper(encoder_fn, example), 
                                remove_columns=['text'], 
                                desc="tokenizing the splits", 
                                # num_proc=self.n_proc, #n_proc>1 throws internal error
                                )
        
        if save_tokens_to_disk: self._save_tokens_to_disk(self.tokens)
        
        return self.tokens
        
        
    def _save_tokens_to_disk(self, tokens, num_shards = 1024):
        """
        Save the tokenized dataset to disk.
        TODO: save tokens to .cache or /data directory
        """
        for split, dset in tokens.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
            arr = np.memmap(filename, dtype=np.uint64, mode='w+', shape=(arr_len,))
            
            idx = 0
            for batch_idx in tqdm(range(num_shards), desc = f'writing {filename}'):
                #batch together samples for faster write
                batch = dset.shard(num_shards = num_shards, index = batch_idx, contiguous = True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                #write into memmap
                arr[idx:idx+len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            
            #save to disk
            arr.flush()        
            del arr #close the memmap
                


    



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
        
        return self.n_vocab()
    
    
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
        
        if batch: return self.encode_ordinary_batch(text) if ignore_special_tokens else self.encode_batch(text) 
        else: return self.encode_ordinary(text) if ignore_special_tokens else self.encode(text)
        
    
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
        
        Returns:
            str or list of str: The decoded text representation of the input tokens.
        """
        
        if batch: return self.decode_batch(tokens) if ignore_special_tokens else self.decode_batch(tokens) 
        else: return self.decode_ordinary(tokens) if ignore_special_tokens else self.decode(tokens)
        
        
        
    
        
        

    #import all functionalities from self.encoder as methods of this class, if they dont exist in the class already
    def __getattr__(self, name):
        """
        Get the attribute from self.encoder if it exists, otherwise raise an AttributeError.
        Args:
            name (str): The name of the attribute.
        Returns:
            Any: The attribute value.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.encoder, name): return getattr(self.encoder, name)
        else: raise AttributeError(f"Missing attribute {name} for object of class 'TiktokenTokenizer'")
    
    
    

if __name__ == "__main__":
    # n_procs = os.cpu_count()//2
    n_procs =15
    
    encoder = TiktokenTokenizer()

    # for dataset in unlabeled_text_datasets: 
    ds = unlabeledDataset(dset, n_procs)
    toks = ds.tokenize(encoder.tokenize_dataset, save_tokens_to_disk = True)
    
        