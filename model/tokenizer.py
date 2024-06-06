
import numpy as np
from transformers import AutoTokenizer
from functools import wraps



def check_huggingface_tokenizer_validity(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tokenizer = func(*args, **kwargs)
        
        #check if tokenizer has attributes eot_tojen, n_vocab, and functions encode, decode, tokenize_dataset
        assert hasattr(tokenizer, 'eot_token'), f"Tokenizer must have an attribute eot_token, which is the end-of-text token. This is used for encoding and decoding text. "
        assert hasattr(tokenizer, 'n_vocab'), f"Tokenizer must have an attribute n_vocab, which is the vocabulary size. This is used for encoding and decoding text. "
        assert hasattr(tokenizer, 'encoder_model'), f"Tokenizer must have an attribute encoder_model, which is the name of the underlying encoding model. This is used for loading dataset from disk"
        assert hasattr(tokenizer, 'encode'), f"Tokenizer must have a method encode, which is used for encoding text. "
        assert hasattr(tokenizer, 'decode'), f"Tokenizer must have a method decode, which is used for decoding text. "
        assert hasattr(tokenizer, 'tokenize_dataset'), f"Tokenizer must have a method tokenize_dataset, which is used for tokenizing a dataset. "
        
        
        #TODO
        # verify format of encode,decode and tokenize_dataset methods of tokenizer
        return tokenizer
    return wrapper

    
    
class BaseTokenizer:
    def _get_numpy_dtype(self):
        f'given the vocab size get the correct numpy dtype. Eg. You dont need uint64 for a vocab size of 50K, but only uint16.'
        vocab_size = self.get_vocab_size()
        dtype_limits = [(np.uint8, np.iinfo(np.uint8).max),
                        (np.uint16, np.iinfo(np.uint16).max),
                        (np.uint32, np.iinfo(np.uint32).max),
                        (np.uint64, np.iinfo(np.uint64).max)]
    
        # Select the smallest data type that can handle max_tokens
        for dtype, max_limit in dtype_limits:
            if vocab_size <= max_limit:
                return dtype.__name__

        # If no suitable type found, raise an exception (unlikely with uint64)
        raise ValueError("Value is too large for available data types.")
    
    
    def get_vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.
        Returns:
            int: The vocabulary size.
        """
        assert hasattr(self, 'n_vocab'), f"Tokenizer must have an attribute n_vocab, which is the vocabulary size. This is missing in {self.__class__.__name__}"
        
        return self.n_vocab
    
    def tokenize_dataset(self, *args, **kwargs):
        f'this class is very important for the download_dataset function'
        raise NotImplementedError
    
    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
    def decode(self, *args, **kwargs):
        raise NotImplementedError
    
    
    def _check_tiktoken_validity(self, tokenizerID):
        """
        Check if the tokenizer is a valid Tiktoken tokenizer.
        
        Args:
            tokenizer (object): The tokenizer object to be checked.
        
        Returns:
            str: The final encoding model name. (not necessarily the tokenizerID)
        """
        from tiktoken.model import MODEL_TO_ENCODING
        
        valid_tokenizerIDs = set(list(MODEL_TO_ENCODING.keys()) + list(MODEL_TO_ENCODING.values()))
        assert tokenizerID in valid_tokenizerIDs, f"Invalid tokenizerID. Valid tokenizerIDs are: {list(valid_tokenizerIDs)}"
        
        #return the final encoding model name. 
        # Doesnt make sense to simply return tokenizerID, because multiple tokenizerIDs are linked to the same tokenizer.
        # So we return the final encoding model name, so that during loading, we avoid downloading and tokenizing again, in case a different tokenizerID (with the same underlying tokenizer) was used, 
        return MODEL_TO_ENCODING[tokenizerID] if tokenizerID in MODEL_TO_ENCODING else tokenizerID 
        
        
    
    
    
    

class Tokenizer(BaseTokenizer):
    "tokenizer. By default, we use the tiktoken tokenizer (by OpenAI), which is used as the default tokenizer for from-scratch-defined models. Use other tokenizers for huggingface models. "
    def __init__(self, from_model = "gpt2"):
        """
        Initialize the TiktokenTokenizer class.
        from_model (str): The model to use for encoding.
        
        by default, we use gpt2 tokenizer, which has 50,257 tokens
        """
        try: import tiktoken
        except ImportError:
            raise Exception('Tiktoken module is missing: run `pip install tiktoken==0.6.0`')
        
        
        from_model = self._check_tiktoken_validity(from_model)
        self.encoder_model = from_model

        
        self.encoder = tiktoken.get_encoding(self.encoder_model)
        
        self.n_vocab = self.encoder.n_vocab #required
        self.eot_token = self.encoder.eot_token #required. eot_token is an integer (not string)
    
    
    
    
    def encode(self, text: str, ignore_special_tokens = True, batch=False, **kwargs):
        
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
        
        if hasattr(self, 'module'): return self.module.encode(text, **kwargs) #if the tokenizer is from huggingface, use the encode method of the huggingface tokenizer
        
        if batch: return self.encoder.encode_ordinary_batch(text) if ignore_special_tokens else self.encoder.encode_batch(text) 
        else: return self.encoder.encode_ordinary(text) if ignore_special_tokens else self.encoder.encode(text)
        
    
    def tokenize_dataset(self, text):
        """
        Tokenize a dataset so that it can saved to disk before training model to save time
        text is a row from Datasets.Dataset object.
        """
            
            
        ids = self.encode(text, ignore_special_tokens = True, verbose = False) #verbose is a common kwarg in huggingface transformers tokenizers. We dont want unnecessary warnings to be printed. Our code takes care of major guardrails
        
        ids.append(self.eot_token)#add the end of text token, e.g. 50256 for gpt2 bpe
        # note acc to Karpath. BUT WHY?: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out
        
             
    def decode(self, tokens: list, batch=False, **kwargs):
        """
        Decodes the given tokens into text.
        
        Args:
            tokens (list): The input tokens to be decoded.
            batch (bool): Whether to decode the tokens in batch mode. Defaults to True.
            
            Note: there is no concept of ignore_special_tokens in decode. Whatever tokens we get, we can simply look them up in the vocabulary and return the corresponding string token.  
        
        Returns: str or list of str: The decoded text representation of the input tokens.
        """
        if hasattr(self, 'module'): return self.module.batch_decode(tokens, **kwargs) #if the tokenizer is from huggingface, use the decode method of the huggingface tokenizer
        
        return self.encoder.decode_batch(tokens) if batch else self.encoder.decode(tokens)
    
    
    @classmethod
    @check_huggingface_tokenizer_validity
    def from_huggingface(self, hf_tokenizer: AutoTokenizer, 
                         eot_token_name = 'eos_token',
                         n_vocab_name = 'vocab_size',):
        """
        Create a Tokenizer object from a Huggingface tokenizer, which comes from huggingface_wrappers.py
        
        eot_token_name (str): The name of the end-of-text token (int) attribute in the Huggingface tokenizer. Defaults to 'eos_token'.
        n_vocab_name (str): The name of the vocabulary size (int) attribute in the Huggingface tokenizer. Defaults to 'vocab_size'.
        
        """
        
        instance = Tokenizer.__new__(Tokenizer)
        
        assert hasattr(hf_tokenizer, eot_token_name), f"{eot_token_name} not found in {hf_tokenizer}. The corresponding EOT token must be by a different name."
        assert hasattr(hf_tokenizer, n_vocab_name), f"{n_vocab_name} not found in {hf_tokenizer}. The corresponding vocabulary size must be by a different name." 
         
        instance.module = hf_tokenizer #the hugginface tokenizer is saved in the class by the name "module"
        instance.eot_token = getattr(hf_tokenizer, eot_token_name)
        instance.n_vocab = getattr(hf_tokenizer, n_vocab_name)
        instance.encoder_model = hf_tokenizer.__class__.__name__
        
        
        
        return instance
        
        
        
    
    
    
            
