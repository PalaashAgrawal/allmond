from models.nanoGPT import nanoGPT 

from dataclasses import dataclass





def build_nanoGPT(config):
    @dataclass
    class GPTConfig:
        block_size: int = 1024
        vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        dropout: float = 0.0
        bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    model_config = GPTConfig(
        block_size = config.arch.context_window,
        vocab_size = config.arch.vocab_size,
        n_layer = config.arch.depth,
        n_head = config.arch.num_heads,
        n_embd = config.arch.hidden_dim,
        dropout = config.arch.dropout,
        bias = config.arch.bias
    )

    return nanoGPT(model_config)
