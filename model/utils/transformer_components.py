from torch import nn
import torch


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
            
    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    def forward(self, x, attention_mask=None):
        batch_size, seq_length, _ = x.size()
        
        # Create the causal mask if required
        if self.is_causal: causal_mask = self.causal_mask[:seq_length, :seq_length]  # (seq_length, seq_length)
        else: causal_mask = None
        
        # Combine causal mask and attention mask if necessary
        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, seq_length, seq_length)  # (batch_size, seq_length, seq_length)
            if causal_mask is not None:
                attention_mask = causal_mask.unsqueeze(0) * attention_mask
            # else:
            #     attention_mask = attention_mask
        else:
            attention_mask = causal_mask.unsqueeze(0) if causal_mask is not None else None
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask.expand(batch_size, self.multiheadattn.num_heads, seq_length, seq_length)  # (batch_size, num_heads, seq_length, seq_length)
            attention_mask = attention_mask.reshape(batch_size * self.multiheadattn.num_heads, seq_length, seq_length)  # (batch_size * num_heads, seq_length, seq_length)
        
        k, q, v = self.k(x), self.q(x), self.v(x)  # B, T, n_embd
        return self.multiheadattn(q, k, v, attn_mask=attention_mask)
    
        
        
    
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
        
    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    def forward(self, x, attention_mask = None ):
        """
        Forward pass for the transformer block.
        if apply_causal_mask is True, then the attention layer will only attend to the previous tokens, not the future tokens.
        
        TODO: should flash attention be enabled once globally in GPT.forward() or should it be applied in each transformer block?
        """
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        x = x + self.attn(self.layernorm1(x), attention_mask = attention_mask)[0] #pre-normalization
        x = x + self.mlp(self.layernorm2(x)) 
        return x            
        