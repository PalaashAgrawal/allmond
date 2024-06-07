"""
functions to help in generation (inference) of text using the model. 
Decided to put in a separate file because there were many functions solely dedicated to generation, 
that it made sense to separate them from the main model file.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional


class GenerationBase:
    def _prepare_generation_input(self, input):
        """
        We support multiple kinds of inputs in generate function. Check format, and convert to tensor for proceesing.
        """
        
        if isinstance(input, str):
            input = [self.tokenizer.encode(input)]  # list of list of ints
            self.ret_type = 'str'
        elif isinstance(input, list) and isinstance(input[0], str):
            input = [self.tokenizer.encode(i) for i in input]  # list of list of ints
            self.ret_type = 'str'
        elif isinstance(input, list) and isinstance(input[0], int):
            input = [input]  # list of list of ints
        elif isinstance(input, list) and isinstance(input[0], list):
            pass  # already in the required format
        else:
            assert isinstance(input, torch.Tensor), f'Unsupported input type. input can be a single string, single list of ints, list of strings, tensor of shape (b,t) or list of list of ints. Instead got {type(input)}'
            input = input.tolist()  # Convert tensor to list of list of ints

        # Pad the input to the block size
        input = self._pad_sequences(input)
        return input.to(self.device)
    
    def _pad_sequences(self, sequences):
        """
        Pad the input sequences to the maximum length in the batch. This is only done so that sequences can be batched for generation. In the forward function of the model, we will again pad the batch to the block_size(context window).
        These two padding parts are separated so that one can also pass a sequence of length<=block_size to the model forward pass directly. 
        """
        max_len = max(len(seq) for seq in sequences)
        pad_token_id = self.tokenizer.pad_token
        padded_sequences = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)
    
    
    @torch.no_grad()
    def generate(self, inp, max_new_tokens, temperature = 1.0, top_k = None, 
                 return_input = False,  
                 return_logprobs: Optional[bool] = False):
        """
        Generate new tokens from the model., given a conditioning sequence of indices idx (LongTensor of shape (b,t)), and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time. 
        Most likely you'll want to make sure to be in model.eval() mode when calling this function.)
        
        By default, returns list of newly generated tokens (along with input tokens), unless one of return_* is specified

        Parameters:
        inp : str, List[str], torch.Tensor() of size (b,t),  List[int], List[List[int]]
            The input tokens to condition on. If a string, it will be tokenized using the model's tokenizer. 
            If a tensor, it should be of shape (b,t) where b is the batch size and t is the sequence length.
            
        max_new_tokens : int
            The maximum number of new tokens to generate.
        temperature : float, optional, default=1.0
            The sampling temperature to use.
        top_k : int, optional, default=None
            If specified, only consider the top_k most probable tokens at each step.
            
        return_input : bool, optional, default=True
            Whether to return the input tokens along with the generated tokens. 
            
        
        Returns:
        torch.Tensor or str depending on input type. 

        #in some cases, you might want to return log probabilities instead of tokens. For eval, you also need to know if the returned logprob is greedy sampled? (ie, if in torch.mutinomial, the highest prob value is sampled or not)
        return_logprobs : bool, optional, default=False
            Whether to return the log probabilities of the generated tokens.
    
        
        
        
        TODO: 
        
        MOST IMP: need to modify probability calculation for input tokens themselves (as in openai example), for lm-eval-harness. 
        
        1. padding mask for the input sequence., so that batch processing is possible
        2. Versatility to return tokens, logits or logprobs
        
        
        
        Testcases
        1. Check if the function returns the correct number of tokens
        2. does it work for temperture 0 
        3. does it work for top_k
        
        4. why doesnt it use GPU? only for HF model. s
        
        """
        
        #input can be a single string, single list of ints, list of strings, tensor of shape (b,t) or list of list of ints
        self.ret_type = None
        idx = self._prepare_generation_input(inp)

        # Initialize logprobs if required
        if return_logprobs:
            self.is_greedy = True
            logprobs = np.array([])

            # Calculate log probabilities for input tokens
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)  # b, t, vocab_size
            logits = logits[:, -idx.shape[1]:, :] / (temperature + 1e-20)  # To avoid 0 division error if temperature = 0.0
            input_probs = F.log_softmax(logits, dim=-1)
            input_logprobs = input_probs.gather(2, idx.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            logprobs = np.append(logprobs, input_logprobs.flatten())

        # Generate new tokens
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)  # b, t, vocab_size
            logits = logits[:, -1, :] / (temperature + 1e-20)  # To avoid 0 division error if temperature = 0.0

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) if temperature > 0 else torch.argmax(probs, dim=-1).unsqueeze(-1)  # Greedy search if temperature is 0

            if return_logprobs:
                logprobs = np.append(logprobs, torch.log(probs[torch.arange(probs.shape[0]), idx_next.squeeze()]).cpu().numpy())
                self.is_greedy = self.is_greedy and torch.argmax(probs, dim=-1) == idx_next.squeeze()

            idx = torch.cat((idx, idx_next), dim=1)

        if return_logprobs:
            return logprobs
        ret = idx if return_input else idx[:, -max_new_tokens:]
        if self.ret_type == 'str':
            #covert ret tensor to list of ints (or list of list of ints if batched)
            ret = ret.cpu().numpy().tolist()
            ret = self.tokenizer.decode(ret, batch = True)
        return ret
    
    