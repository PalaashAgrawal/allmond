from torch import nn
import torch
import torch.nn.functional as F

from .transformer_components import TransformerBlock
from .huggingface_wrappers import HF_base
from .tokenizer import Tokenizer
    
class BaseModel:
    def __str__(self): 
        f'get model name along with number of parameters in millions/billions'
        def _format_number(num):
            if num >= 1_000_000_000:
                return f"{num / 1_000_000_000:.1f}B"
            elif num >= 1_000_000:
                return f"{num / 1_000_000:.1f}M"
            else:
                return str(num)
        
        model_name = self.model_name if hasattr(self,'model_name') else 'GPT'
            
        return f'{model_name}-{_format_number(self.num_params)}'
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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

   
   
   
    
    
    
    
    
    
    
    
    
class GPT(nn.Module, BaseModel, HF_base):
    
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
        
        super().__init__()

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
        

        
        
    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    def forward(self, idx): 
        assert hasattr(self, 'forward_fn'), f'You need to implement a forward_fn function as attribute for the model to process inputs.'
        assert isinstance(idx, torch.Tensor), f'forward function should only have one argument as input, i.e., the input tensor of shape (bs, seq_len)'
        return self.forward_fn(idx)
        
    def _gpt_forward_impl(self, idx):
        f'implentation of the forward function for the generic GPT class'
        
        _,t = idx.shape
        assert t<=self.block_size, f'Cannot forward -- model block size is exhausted. Model block size is {self.block_size}, but input sequence length is {t}.'
        pos = torch.arange(t, dtype = torch.long, device = idx.device) #shape (t,)
        
        x = self.wpe(pos)
        x = x + self.wte(idx)
        x = self.dropout(x) 
        
        for block in self.layers: x = block(x)
        x = self.layernorm_final(x)
    
        return self.head(x)
    
          
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        """
        Generate new tokens from the model., given a conditioning sequence of indices idx (LongTensor of shape (b,t)), and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time. 
        Most likely you'll want to make sure to be in model.eval() mode when calling this function.))
        
        """
        
        for _ in range(max_new_tokens):
            #if the sequence context is growing too long we must crop it at block size
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            #take the logits of the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            #optinally choose only the top_k tokens
            if top_k is not None:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits<v[:, [-1]]] = -float('Inf')
            #apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim = -1)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim = 1)
        
        return idx 
    
    
    
    @classmethod
    def as_variant(cls, model_type:str, override_args:dict = None):
        f'to TEST'
        f"""
        used to create an instance of the GPT model with a specific configuration based on the model_type parameter. 
        The model_type should be one of the following: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'. 
        These correspond to different configurations of the GPT model with varying numbers of layers, embedding dimensions, heads, vocabulary size, block size, and whether to use bias or not.
        """
        supported_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        assert model_type in supported_models, f'Unsupported model type. Supported model types are: {supported_models}'
        if override_args is None: override_args = {} 
        assert all(k=='dropout' for k in override_args) #only dropout is overridable for now. According to Karpathy's repo. 
        
        config_args  = {'gpt2': {'n_layer': 12, 'n_embd': 768, 'n_head': 12, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #124M params
                        'gpt2-medium': {'n_layer': 24, 'n_embd': 1024, 'n_head': 16, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #345M params
                        'gpt2-large': {'n_layer': 36, 'n_embd': 1280, 'n_head': 20, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #774M params
                        'gpt2-xl': {'n_layer': 48, 'n_embd': 1600, 'n_head': 25, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #1558M params
        }[model_type]
        
        _model =  cls(**config_args, **override_args)
        _model.model_name = model_type
        return _model
    
    
    
    @classmethod
    def from_hf(cls, model_identifier, **kwargs):
        f"""Create an instance of the GPT model from a Huggingface model identifier. 
        The model_identifier should be a string that corresponds to a model in the Huggingface model hub.
        Basically this model will behave exactly like the GPT class, except that the model parameters will be loaded from the Huggingface model hub.
        """
                
        instance = cls.__new__(cls)
        super(cls, instance).__init__() #for nn.Module
        
        hf_model  = instance.get_hf_model(model_identifier)
        for key, value in hf_model.cfg_dict.items(): setattr(instance, key, value)
        
        #storing the model parameters in the class instance
        instance.layers = hf_model  
        #defining the forward_fn for proper forwarding. 
        instance.forward_fn = lambda x: instance.layers(x).logits
        
        for key, value in kwargs.items(): setattr(instance, key, value)
        
        return instance
    
        