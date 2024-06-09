from torch import nn
import torch

#parent classes

from .eval.eval import evalBase
from .tokenizer import Tokenizer
from .utils.all import GenerationBase, HuggingfaceModelWrappers, TransformerBlock


class gptBase(HuggingfaceModelWrappers, evalBase, GenerationBase):
    def __str__(self): 
        f'get model name along with number of parameters in millions/billions'
        def _format_number(num):
            if num >= 1_000_000_000:    return f"{num / 1_000_000_000:.1f}B"
            elif num >= 1_000_000:      return f"{num / 1_000_000:.1f}M"
            else:                       return str(num)
        model_name = self.model_name if hasattr(self,'model_name') else 'GPT'   
        return f'{model_name}-{_format_number(self.num_params)}'
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = self.num_params
        assert hasattr(self, 'wpe') and isinstance(self.wpe, nn.Embedding), f'Positional Encoding Embedding (wpe) not defined in the model. Define `self.wpe` as a nn.Embedding'
        assert hasattr(self, 'wte') and isinstance(self.wte, nn.Embedding), f'Token Embedding layer (wte) not defined in the model. Define `self.wte` as a nn.Embedding'

        if non_embedding: n_params-=self.wpe.weight.numel()
        params_excl_embeddings = n_params - self.wte.weight.numel()
        
        print(f'Number of parameters: {n_params/1e6:.2f}M. Number of parameters (excluding embeddings): {params_excl_embeddings/1e6:.2f}M. Embeddings occupy {params_excl_embeddings/n_params*100:.2f}% of the total parameter count. ')
        
        return n_params
    
    @torch.no_grad()
    def _residual_init_weights(self):
        """
        Initialize weights for residual connections. Reweight std deviation according to GPT2 paper.
        The remaining layers are default initialized according to Pytorch. I don't think 0.02 stddev is a necessary condition (Acc to original GPT paper (2018), they say 0.02 "works well", without any proper justification)
        """
        for param_name, param in self.named_parameters():
            if param_name.endswith(('residual_fc.weight', 'residual_projection.weight')): param.div_(torch.sqrt(torch.tensor(2*self.n_layer)))
        
    @property
    def num_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def device(self): return next(self.parameters()).device
    
    
    def _detect_batch_size(self):
        """
        Detect largest batch size for training
        """
        max_length = self.block_size

        # Starting with a relatively high batch size and reducing it in case of OOM errors
        batch_size = 64  # Initial batch size
        step_size = 2  # Reduction factor in case of OOM

        def can_allocate_memory(batch_size):
            try:
                # Create a dummy input to test memory allocation
                test_batch = torch.ones((batch_size, max_length), device=self.device).long()
                # Run a forward pass
                with torch.no_grad(): self(test_batch)
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower(): return False
                else: raise e

        # Adjust the batch size until it fits into memory
        while batch_size > 0:
            if can_allocate_memory(batch_size): break
            batch_size //= step_size

        # Ensure at least one batch can be processed
        return max(batch_size, 1)
        

    
class GPT(nn.Module, gptBase):
    
    model_name = 'GPT'
    def __init__(self,
                block_size: int = 1024,
                vocab_size: int = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency,
                n_layer: int = 12,
                n_head: int = 12,
                n_embd: int = 768,
                dropout: float = 0.0,
                bias: bool = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster,
                
                tokenizer_from: str = 'gpt2',
                ):
        """
        Initializes the GPT-2 model.

        Args:
            block_size (int): The size of each block.
            vocab_size (int): The vocabulary size.
            n_layer (int): The number of transformer layers.
            n_head (int): The number of attention heads.
            n_embd (int): The dimension of the token embeddings and the positional embeddings.
            dropout (float): The dropout rate.
            bias (bool): Whether to include bias in Linears and LayerNorms.
            
            tokenizer_from (str): By default, we use the Tiktoken Tokenizer. This parameter is used to specify  which model to source the tokenizer from (as supported by TikToken). Default to the gpt2 tokenizer, which contains 50,304 tokens.
        """
        
        super().__init__() #nn.Module
        

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        
        self.wte = nn.Embedding(vocab_size, n_embd) # token embedding
        self.wpe = nn.Embedding(block_size, n_embd) # positional embedding
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout, bias, 
                                                      apply_causal_mask = True, 
                                                      block_size=self.block_size) 
                                     for _ in range(n_layer)])
        
        self.layernorm_final = nn.LayerNorm(n_embd, bias = bias)
        self.head = nn.Linear(n_embd, vocab_size, bias = bias)
        #weight tying
        self.wte.weight = self.head.weight
        self._residual_init_weights() #per GPT2 paper
        self.tokenizer = Tokenizer(tokenizer_from)
        self.forward_fn = self._gpt_forward_impl #we separately define forward_fn so that custom defined huggingface models can easily implement their forwarding.
        
        
        
        
    def _gpt_forward_impl(self, idx):
        f'implentation of the forward function for the generic GPT class.'
        f'Educational Note: as you see, this function in invariant to the sequence length. The only reason padding is done, is so that input sequences can be processed in batches.'
        
        _,t = idx.shape #idx = b,t
        assert t<=self.block_size, f'Cannot forward -- model block size is exhausted. Model block size is {self.block_size}, but input sequence length is {t}.'
        pos = torch.arange(t, dtype = torch.long, device = idx.device) #shape (t,)
        
        x = self.wpe(pos) + self.wte(idx) #t,n_embd + b,t,n_embd --> b,t,n_embd
        x = self.dropout(x) #b,t,n_embd
        
        
        # attention_mask = (idx != self.tokenizer.pad_token).float().unsqueeze(1).unsqueeze(2)  # shape (batch_size, 1, 1, sequence_length)
        # attention_mask = attention_mask.expand(-1, -1, t, -1)  # shape (batch_size, 1, sequence_length, sequence_length)
         # Create attention mask
        attention_mask = (idx != self.tokenizer.pad_token).float().unsqueeze(1)# shape (batch_size, 1, sequence_length)
        attention_mask = attention_mask.expand(-1, t, -1)  # shape (batch_size, seq_length, seq_length)
    
    
        for block in self.layers: x = block(x, attention_mask) #b,t,n_embd
        x = self.layernorm_final(x) #b,t,n_embd
        return self.head(x) #b,t,vocab_size
        
        
    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    def forward(self, idx):
        #TODO:
        #1. Add support for custom attention mask, extendable to huggingface models. 
        
        assert hasattr(self, 'forward_fn'), f'You need to implement a forward_fn function as attribute for the model to process inputs.'
        assert isinstance(idx, torch.Tensor), f'forward function should only have one argument as input, i.e., the input tensor of shape (bs, seq_len)'
       
        ret =  self.forward_fn(idx)
        assert isinstance(ret, torch.Tensor), f'forward function should return a tensor. Instead got {type(ret)}'
        
        return ret
        
    
    
    @classmethod
    def as_variant(cls, model_type:str, override_args:dict = None):
        f'to TEST'
        """
        used to create an instance of the GPT model with a specific configuration based on the model_type parameter. 
        The model_type should be one of the following: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'. 
        These correspond to different configurations of the GPT model with varying numbers of layers, embedding dimensions, heads, vocabulary size, block size, and whether to use bias or not.
        """
        supported_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        aliases = {'medium': 'gpt2-medium',
                   'large': 'gpt2-large',
                   'xl':    'gpt2-xl' 
                   }
        assert model_type in supported_models or model_type in aliases, f'Unsupported model type. Supported variant model types are: {supported_models}'
        if override_args is None: override_args = {} 
        assert all(k=='dropout' for k in override_args) #only dropout is overridable for now. According to Karpathy's repo. 
        
        config_args  = {'gpt2':         {'n_layer': 12, 'n_embd': 768,  'n_head': 12,   'vocab_size': 50257,  'block_size': 1024, 'bias': True}, #124M params
                        'gpt2-medium':  {'n_layer': 24, 'n_embd': 1024, 'n_head': 16,   'vocab_size': 50257,  'block_size': 1024, 'bias': True}, #345M params
                        'gpt2-large':   {'n_layer': 36, 'n_embd': 1280, 'n_head': 20,   'vocab_size': 50257,  'block_size': 1024, 'bias': True}, #774M params
                        'gpt2-xl':      {'n_layer': 48, 'n_embd': 1600, 'n_head': 25,   'vocab_size': 50257,  'block_size': 1024, 'bias': True}, #1558M params
                        }[model_type]
        
        _model =  cls(**config_args, **override_args)
        _model.model_name = model_type
        return _model
    
    
    
    @classmethod
    def from_hf(cls, model_identifier, enable_qlora:bool = False, **kwargs):
        """Create an instance of the GPT model from a Huggingface model identifier. 
        The model_identifier should be a string that corresponds to a model in the Huggingface model hub.
        Basically this model will behave exactly like the GPT class, except that the model parameters will be loaded from the Huggingface model hub.
        
        kwargs in this case are set as attributes of the GPT model instance.
        """
                
        instance = cls.__new__(cls)
        super(cls, instance).__init__() #for nn.Module
        instance.qlora = enable_qlora==True
        
        hf_model  = instance.get_hf_model(model_identifier, enable_qlora = enable_qlora)
        for key, value in hf_model.cfg_dict.items(): setattr(instance, key, value)
        #storing the model parameters in the class instance
        instance.base_model = hf_model  
        #defining the forward_fn for proper forwarding. 
        instance.forward_fn = lambda x: instance.base_model(x).logits
        
        for key, value in kwargs.items(): setattr(instance, key, value)
        
        return instance
    
    
        

        
        
    
    
        