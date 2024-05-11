from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch import nn
from functools import wraps



def check_validity(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        
    
        assert isinstance(model,nn.Module), 'Not a valid model. Model should be a nn.Module instance. Huggingface models by default are nn.Module instances'
        assert hasattr(model,'config'), f"Each Huggingface model has a config dict, although it may come with different names. Make sure that your model's configs are stored in the `config` attribute as a dict"
        assert isinstance(model.config, dict), f'Make sure that config should be a dict'
    
        return model
    return wrapper


class HF_base:
    supported_hf_models = ['microsoft/Phi-3-mini-4k-instruct',
                        ]
    @check_validity
    def get_hf_model(self, model_identifier):
        if model_identifier not in self.supported_hf_models: raise ValueError(f"Unsupported model identifier. Supported model identifiers are: {self.supported_hf_models}")
        
        if model_identifier=='microsoft/Phi-3-mini-4k-instruct':    return self.phi3_mini(model_identifier)
    
    
    def phi3_mini(self, model_identifier):
        _model =  AutoModelForCausalLM.from_pretrained(model_identifier, trust_remote_code = True)
        _model.config = {**AutoConfig.from_pretrained(model_identifier, trust_remote_code = True).to_dict(),
                         "model_name": "Phi-3-mini"}
        
        return _model
    


    
        
        
        