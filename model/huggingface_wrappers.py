from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch import nn
from functools import wraps

from .tokenizer import Tokenizer    



def check_model_validity(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        
    
        assert isinstance(model,nn.Module), 'Not a valid model. Model should be a nn.Module instance. Huggingface models by default are nn.Module instances'
        assert hasattr(model,'cfg_dict') and isinstance(model.cfg_dict, dict), f"Each Huggingface model has a config dict, although it may come with different names. Make sure that your model's configs are stored in the `cfg_dict` attribute as a dictionary."
        assert 'model_name' in model.cfg_dict, f'Make sure that config should have a key "model_name" which is a string. This is used for Logging purposes. '
        assert 'block_size' in model.cfg_dict, f'Make sure that config should have a key "block_size" which is an integer. This is essentially the maximum sequence length of the model. Common names for this are "max_position_embeddings" or "max_seq_len" '
        assert 'tokenizer' in model.cfg_dict and isinstance(model.cfg_dict["tokenizer"], Tokenizer), f'Make sure that config should have a key "tokenizer" which is an instance of the Tokenizer class (see Tokenizer.from_huggingface in tokenizer.py). This is used for encoding and decoding text. '
        return model
    return wrapper





class HF_base:
    supported_hf_models = ['microsoft/Phi-3-mini-4k-instruct',
                           #add more models here
                        ]
    
    @check_model_validity
    def get_hf_model(self, model_identifier):
        if model_identifier not in self.supported_hf_models: raise ValueError(f"Unsupported model identifier. Supported model identifiers are: {self.supported_hf_models}")
        
        if model_identifier=='microsoft/Phi-3-mini-4k-instruct':    return self.phi3_mini(model_identifier)
        #create more methods like above for each model
    
    
    def phi3_mini(self, model_identifier):
        
        #Model
        _model =  AutoModelForCausalLM.from_pretrained(model_identifier, trust_remote_code = True)
        
        #Model Configs
        cfg = AutoConfig.from_pretrained(model_identifier, trust_remote_code = True)
        
        #Model Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code = True)
        tokenizer.n_vocab = len(tokenizer.vocab) #get_vocab_size does not account for special added tokens, somehow. So we custom define the vocab size.
        tokenizer = Tokenizer.from_huggingface(tokenizer, eot_token_name='eos_token_id', n_vocab_name='n_vocab') 
        
                
        #encoder and decoder functions already defined. 
        
        
        
        _model.cfg_dict = {**cfg.to_dict(),
                         "model_name": "Phi-3-mini",
                         "block_size": cfg.max_position_embeddings, #phi3Config defines context length as max_position_embeddings. "block_size" is required for dataloader declaration. 
                         "tokenizer": tokenizer,
                         }
        
        # _model._attn_implementation = 'eager'
        
        
        return _model

    
    #add more methods like above for each model. Make sure to overwrite the config attribute of the model with the model's config as dict, along with a model_name key (which is used for logging purposes)


    
        
        
        