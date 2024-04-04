from torch import nn
import torch
import torch.nn.functional as F



class MultiHeadSelfAttention(nn.Module):
    'Why the fuck is self attention not defined in Pytorch API??'
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, 
                 batch_first=True, #by default is TRUE 
                 device=None, dtype=None,
                 is_causal = True,
                 block_size = None):
        
         
        
        super().__init__()
        
        self.k = nn.Linear(embed_dim, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        
        self.is_causal = is_causal
        
        self.multiheadattn = nn.MultiheadAttention( embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, 
                                                    add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, 
                                                    batch_first=batch_first, #by default is TRUE 
                                                    device=device,  dtype=dtype)
        
        if self.is_causal:
            
            assert block_size, 'block_size must be provided for causal attention'
            assert batch_first, 'Causal attention is only implemented for batch_first = True'
            self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))
            
        
    def forward(self, x):
        attn_mask = self.causal_mask if self.is_causal else None
        # print(attn_mask)
        
        k,q,v = self.k(x), self.q(x), self.v(x) #B, T, n_embd
        return self.multiheadattn(q, k, v, attn_mask = attn_mask)
    
        
        
        
        
<<<<<<< HEAD
        
        
        
=======
>>>>>>> c8e2abc49ec715c163dd087067043ee01e672fa1
    
class MLP(nn.Module):
    def __init__(self, n_embd: int, 
                 n_hidden: int = None,
                 dropout: float = 0.0, 
                 bias = False):
        
        """
        Initializes the GPT2 model.
        
        Args:
            n_embd (int): The size of the embedding layer.
            n_hidden (int, optional): The size of the hidden layer. Defaults to None. If None, defaults to n_embd * 4. (According to the GPT-2 paper.)
            dropout (float, optional): The dropout rate. Defaults to 0.0. 
            bias (bool, optional): Whether to include bias terms. Defaults to False. bias = False is a bit better and faster.
        """
            
        super().__init__()
        if n_hidden is None: n_hidden = n_embd * 4
        
        self.residual_fc = nn.Linear(n_embd, n_hidden, bias = bias)
        self.gelu = nn.GELU()
        self.residual_projection = nn.Linear(n_hidden, n_embd, bias = bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        x = self.residual_fc(x)
        x = self.gelu(x)
        x = self.residual_projection(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    f'Implementation of the (Decoder only) block of the transformer'
    def __init__(self, n_embd, n_heads, dropout = 0.0, bias = False, apply_causal_mask = True, block_size = None):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_embd, bias = bias)
        self.attn = MultiHeadSelfAttention(n_embd, n_heads, dropout = 0.0, bias = bias, is_causal=apply_causal_mask, block_size = block_size)
        self.layernorm2 = nn.LayerNorm(n_embd, bias = bias)
        self.mlp = MLP(n_embd, dropout = dropout, bias = bias)
        
    
    def forward(self, x, ):
        """
        Forward pass for the transformer block.
        if apply_causal_mask is True, then the attention layer will only attend to the previous tokens, not the future tokens.
        
        TODO: should flash attention be enabled once globally in GPT.forward() or should it be applied in each transformer block?
        """
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        x = x + self.attn(self.layernorm1(x))[0] #pre-normalization
        x = x + self.mlp(self.layernorm2(x)) 
        return x            
        
    
class GPT(nn.Module):
    def __init__(self,
                block_size: int = 1024,
                vocab_size: int = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency,
                n_layer: int = 12,
                n_head: int = 12,
                n_embd: int = 768,
                dropout: float = 0.0,
                bias: bool = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster,
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
            flash_attention (bool): Whether to use FlashAttention optimized kernel for attention. If False, use standard dot-product attention.
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
        
        
    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    def forward(self, idx, targets = None):
        _,t = idx.shape
        assert t<=self.block_size, f'Cannot forward, model block size is exhausted. Model block size is {self.block_size}, but input sequence length is {t}.'
        pos = torch.arange(t, dtype = torch.long, device = idx.device) #shape (t,)
        
        x = self.wpe(pos)
        x = x+ self.wte(idx)
        x = self.dropout(x) 
        
        for block in self.layers: x = block(x)
        x = self.layernorm_final(x)
        
        
        
        # if targets is not None:
        #     logits = self.head(x)
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # else:
        #     #during inference, we don't have targets. We just return the logits.
        #     logits = self.head(x[:, [-1], :]) #note: using list[-1] to preserve the time dimension. out shape (b,1,vocab_size)
        #     loss = None
            
        # return logits, loss
<<<<<<< HEAD
=======
    
    
    
    
>>>>>>> c8e2abc49ec715c163dd087067043ee01e672fa1
        return self.head(x)
            
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding: n_params -= self.wpe.weight.numel()
        
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
    def from_pretrained(cls, model_type:str, override_arge:dict = None):
        supported_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        assert model_type in supported_models, f'Unsupported model type. Supported model types are: {supported_models}'
        if override_args is None: override_args = {} 
        assert all(k=='dropout' for k in override_args) #only dropout is overridable for now. According to Karpathy's repo. 
        
        try:
            from transformers import GPT2LMHeadModel
        except ImportError: 
            raise ImportError('Huggingface transformers is not installed. Please install it using pip install transformers')
    
        config_args  = {'gpt2': {'n_layer': 12, 'n_embd': 768, 'n_head': 12, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #124M params
                        'gpt2-medium': {'n_layer': 24, 'n_embd': 1024, 'n_head': 16, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #345M params
                        'gpt2-large': {'n_layer': 36, 'n_embd': 1280, 'n_head': 20, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #774M params
                        'gpt2-xl': {'n_layer': 48, 'n_embd': 1600, 'n_head': 25, 'vocab_size': 50257, 'block_size': 1024, 'bias': True}, #1558M params
        }[model_type]
        
        model = self(**config_args, **override_args)
        
        
        return model
<<<<<<< HEAD
=======
    
    
    def __str__(self): return 'gpt2'
>>>>>>> c8e2abc49ec715c163dd087067043ee01e672fa1
        
        
        
        
        
        
        
        
        
    

    
        