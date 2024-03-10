# This file contains the code for downloading and processing the unlabeled text datasets.
import os
import sys
import time
import argparse


import datasets



unlabeled_text_datasets = ['openwebtext']


def get_dataset(dataset_name: str, n_proc = 8):
    '''
    given huggingface dataset name, download the dataset using the datasets library
    dataset_name (str): name of the dataset to download (usually in the format "author/dataset_name")
    n_proc (int): number of parallel processes to use for downloading the dataset
    
    returns:
    datasets.Dataset: The downloaded dataset. Containing the train, and optionally the test and validation splits, each in the form of <class 'datasets.arrow_dataset.Dataset'>
    '''
    
    # Check if the dataset exists
    
    # Load the dataset
    dataset = datasets.load_dataset(dataset_name, num_proc = n_proc, trust_remote_code=True)
    

    return dataset    




class TiktokenTokenizer():
    "SentencePiece tokenizer for `lang`"
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
    
    
    def encode(self, text: str, ignore_special_tokens = True, batch=True):
        
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
    
        
        """
        
        if batch: return self.encode_batch(text) if ignore_special_tokens else self.encode_batch(text) 
        else: return self.encode_ordinary(text) if ignore_special_tokens else self.encode(text)
        
        
        
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
        if hasattr(self, name): return getattr(self, name)
        elif hasattr(self.encoder, name): return getattr(self.encoder, name)
        else: raise AttributeError(f"Missing attribute {name} for object of class 'TiktokenTokenizer'")
    
    
    

if __name__ == "__main__":
    n_procs = os.cpu_count()//2
    

    for dataset in unlabeled_text_datasets: 
        ds = get_dataset(dataset, n_procs)
        print(ds["train"].__class__)
        