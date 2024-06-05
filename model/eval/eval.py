from lm_eval.api.model import LM 
from lm_eval.api.instance import Instance
from lm_eval.utils import Collator


import importlib
import pathlib
from copy import deepcopy
from typing import List, Literal
from fastai.torch_core import rank_distrib, num_distrib
import torch
from tqdm import tqdm
import numpy as np




class evalBase(LM):
    
    #PAg -- so that I dont have to rename the functions sourced from 
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/070d31df4edad2d8263b64b3ace7ec0cbaeb7ae8/lm_eval/models/nemo_lm.py#L333
    def __init__(self):
        super().__init__()
        self.tok_encode = self.tokenizer.encode
        self.eot_token_id = self.tokenizer.eot_token
        self.max_length = self.block_size
        
        
        
    #utils for evaluation. These functions are required by lm-evaluation-harness
    
    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            ctxlens = []
            contlens = []

            for _, context_enc, continuation_enc in chunk:
                # Leave one token for generation. Tokens_to_generate = 0 breaks NeMo.
                inp = (context_enc + continuation_enc)[-(self.max_length - 1) :]

                ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - (self.max_length - 1))
                ctxlens.append(ctxlen)
                contlens.append(len(continuation_enc))

                inps.append(self.tok_decode(inp))

            # output = self.generate(
            #     self.model,
            #     inputs=inps,
            #     tokens_to_generate=1,
            #     min_tokens_to_generate=1,
            #     compute_logprob=True,
            #     all_probs=True,
            # )
            
            
            # logits = self(inps)
            # logits = logits[:]
            logprobs = self.generate(inps, return_logprobs=True)
            
            

            # batch_token_ids = np.asarray(output["token_ids"])[:, :-1]
            # batch_logprobs = output["logprob"][:, :-1]
            # batch_full_logprob = output["full_logprob"][:, :-1, :]

            # Compute greedy tokens for entire batch rather than calling it with proper ctxlen for each sample.
            # Additional tokens for each sample will be trimmed later.
            min_ctxlen = min(ctxlens)

            # Use min_ctxlen-1 instead of min_ctxlen since full_logprobs are not returns for the first token.
            batch_greedy_tokens = (
                torch.argmax(batch_full_logprob[:, min_ctxlen - 1 :, :], -1)
                .cpu()
                .numpy()
            )

            for token_ids, greedy_tokens, logprobs, ctxlen, contlen, (
                cache_key,
                _,
                _,
            ) in zip(
                batch_token_ids,
                batch_greedy_tokens,
                batch_logprobs,
                ctxlens,
                contlens,
                chunk,
            ):
                # Trim at contlen since shorter contexts in a batch will have more than one token generated.
                # Use ctxlen-1 instead of ctxlen same as for full_logprob in batch_greedy_tokens calculation
                logprobs = (logprobs[ctxlen - 1 :])[:contlen]
                logprob = sum(logprobs).tolist()

                continuation_tokens = (token_ids[ctxlen:])[:contlen]
                len_diff = ctxlen - min_ctxlen
                is_greedy = continuation_tokens == (greedy_tokens[len_diff:])[:contlen]
                if not isinstance(is_greedy, bool):
                    is_greedy = is_greedy.all()
                answer = (logprob, is_greedy)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def loglikelihood(self, requests:list[Instance])-> list[tuple[float, bool]]:
        """
        Each request contains Instance.args : Tuple[str, str] containing 1. an input string to the LM and 2. a target string on which the loglikelihood of the LM producing this target, conditioned on the input, will be returned.
        Each request will have, as result, (ll, is_greedy): Tuple[float, int] returned, where ll is a floating point number representing the log probability of generating the target string conditioned on the input, and is_greedy being either the value 0 or 1, with it being 1 if and only if the target string would be generated by greedy sampling from the LM (that is, if the target string is the most likely N-token string to be output by the LM given the input. )
        """
        
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation) #more efficient to encode them together, rather than individually. You can always concat and split

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)
    
    
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:

    
    
    
    def generate_until(self, requests):

        
        
    

    @property
    def rank(self): return rank_distrib() #or 1
    
    @property
    def world_size(self): return num_distrib() or 1
    

    