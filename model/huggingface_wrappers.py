from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch import nn
import torch
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



class HuggingFaceModelLoader:
    def get_hf_components(self, model_identifier, tokenizer_identifier=None, enable_qlora=False, **kwargs):
            """
            Any huggingface models has 3 major components: the model, its configs (which usually comes in the form of a class instance, and the tokenizer.)
            This function returns the model, tokenizer, and config for the given model identifier.

            Args:
                model_identifier (str): The identifier of the model.
                tokenizer_identifier (str, optional): The identifier of the tokenizer. 
                    ONLY applicable in cases when tokenizer identifier is different than model identifier, which isn't usually the case. An example of the exception is apple OpenELM.
                    Defaults to the model identifier itself.
                    
                enable_qlora (bool, optional): Flag to enable QLORA. Defaults to False.

            Returns:
                tuple: A tuple containing the model, tokenizer, and config.

            Raises:
                ValueError: If the model identifier is not supported.

            """
        
            model =  self._get_QLoRA_model(model_identifier, **kwargs) if enable_qlora else AutoModelForCausalLM.from_pretrained(model_identifier, trust_remote_code = True)        
            cfg = AutoConfig.from_pretrained(model_identifier, trust_remote_code = True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier if tokenizer_identifier else model_identifier, trust_remote_code = True)
            
            return (model, cfg, tokenizer)
        
    
    def _get_QLoRA_model(self, model_identifier, use_4bit_quantization:bool = True, r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"):
        """Method to prepare the model for QLoRA training.

        Args:
            model_identifier (str): The identifier of the pre-trained model.
            ___________________________________________________________________________________________________________________________________________________________________________________
            BNB CONFIGURATION (for quantization)
            use_4bit_quantization = True: Flag to enable 4-bit quantization. If False, we use 8-bit quantization.
            
            ___________________________________________________________________________________________________________________________________________________________________________________
            #PEFT CONFIGURATION (for LoRA adapter addition)
            
            r (int): Rank of LoRA update matrices.
            lora_alpha (int): LoRA scaling factor
            target_modules (str): Name of the module parallel to which the LoRA adapter is added. The common practice is to apply LoRA parallel to linear layers. But if you want to apply LoRA to additional modules, you can specify the name of the module here.
            lora_dropout (float): The dropout rate for LoRA layers.
            bias (str): Specifies if the bias parameters should be trained. Can be ‘none’, ‘all’ or ‘lora_only’.
            task_type (str): The type of task for the model. In this case, it is a causal language modeling task.

        Returns:
            model: The prepared QLoRA model.

        Raises:
            ImportError: If the required packages (bitsandbytes and peft) are not installed.

        TODO: WTF is low_cpu_mem_usage 
        (https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained.low_cpu_mem_usage(bool,)
        I'm so confused. Does it use the CPU more, or less? And if less, why?
        """
        try:
            # import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)

        except ImportError:
            raise ImportError("To enable QLORA, please install the bitsandbytes and peft packages.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit_quantization,
            load_in_8bit=not use_4bit_quantization,
            
            load_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type
        )

        # Get model
        model = AutoModelForCausalLM.from_pretrained(model_identifier, trust_remote_code=True, quantization_config=bnb_config)  # Internally uses bitsandbytes

        # Prepare model for kbit quantization
        model = prepare_model_for_kbit_training(model)

        # Get LoRA model
        model = get_peft_model(model, lora_config)

        return model
        
        


















class HuggingfaceModelWrappers(HuggingFaceModelLoader):
    supported_hf_models = ['microsoft/Phi-3-mini-4k-instruct',
                           #add more models here
                        ]
    
    
    @check_model_validity
    def get_hf_model(self, model_identifier, **kwargs):
        """kwargs for this model are: enable_qlora: bool, tokenizer_identifier: str. 
        See self._get_hf_components for more details on these kwargs.
        """
        if model_identifier not in self.supported_hf_models: raise ValueError(f"Unsupported model identifier. Supported model identifiers are: {self.supported_hf_models}")
        if model_identifier=='microsoft/Phi-3-mini-4k-instruct':    return self.phi3_mini(model_identifier, **kwargs)
        #create more methods like above for each model
    
    
    def phi3_mini(self, model_identifier, **kwargs):

        _model, cfg, tokenizer = self.get_hf_components(model_identifier, **kwargs)
        tokenizer.n_vocab = len(tokenizer.vocab) #get_vocab_size does not account for special added tokens, somehow. So we custom define the vocab size.
        tokenizer = Tokenizer.from_huggingface(tokenizer, eot_token_name='eos_token_id', n_vocab_name='n_vocab') #instance of our custom defined Tokenizer class

        #encoder and decoder functions already defined. 
        
        _model.cfg_dict = {**cfg.to_dict(),
                         "model_name": "Phi-3-mini",
                         "block_size": cfg.max_position_embeddings, #phi3Config defines context length as max_position_embeddings. "block_size" is required for dataloader declaration. 
                         "tokenizer": tokenizer,
                         }        
        
        return _model
    
    #add more methods like above for each model. Make sure to overwrite the config attribute of the model with the model's config as dict, along with a model_name key (which is used for logging purposes)

    
        
            
            


    
        
        
        