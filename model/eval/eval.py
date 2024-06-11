#utils for evaluation. These functions are required by lm-evaluation-harness

from lm_eval.api.model import CacheHook
from lm_eval.api.instance import Instance
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import make_table
from lm_eval.models.utils import Collator,pad_and_concat

from lm_eval.models.huggingface import HFLM

import torch.nn.functional as F 
from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm
import json
from fastai.torch_core import rank_distrib, num_distrib
import gc
import warnings

class evalUtils(HFLM):
    """
    Generously taken from EleutherAI's Huggingface interface in lm-evaluation-harness, since Huggingface's model interface is very similar to ours. (ie, model forward fn returns raw logits. )
    """
    
        
    def _detect_batch_size(self, requests = None, pos=0):

        """
        Detect largest possible batch_size specifically for eval
        """
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1])
        else: 
            max_length = self.max_length

            
        def can_allocate_memory(batch_size):
            gc.collect()
            torch.cuda.empty_cache()
            try:
                x_test = torch.ones((batch_size, max_length), device=self.device).long()
                for _ in range(5): 
                    output = self(x_test)
                return True
            
            except RuntimeError as e:
                if "out of memory" in str(e): 
                    try:
                        del x_test
                        del output
                    except: pass
                    
                    return False
                else: raise e
                
        # Start with a batch size of 2 and increase in powers of 2
        batch_size = 1
        while can_allocate_memory(batch_size): batch_size *= 2

        # Decrement phase: fine-tune the batch '/  tsize by decreasing in steps of 1
        while not can_allocate_memory(batch_size) and batch_size > 1: batch_size-=1

        # Ensure at least one batch can be processed
        final_batch_size =  max(batch_size, 1)
        return final_batch_size
    
    
    def _batch_scheduler(self, pos, n_reordered_requests):
        """
        This function is called by the Collator in each iteration to determine the batch size for a given position in the reordered requests
        Determine the batch size for a given position in the reordered requests
        Save the batch size in self.batch_sizes for future use, so that we don't have to recompute it for each iteration. 
        """
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes: return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (self.batch_sizes[sched - 1] == self.max_batch_size):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(f"Detecting largest batch size")
        bs = self._detect_batch_size(n_reordered_requests, pos)
        self.batch_sizes[sched] = bs
        print(f"Determined largest batch size: {bs}")
        return bs
    
    
    def _select_cont_toks(self, logits: torch.Tensor, contlen: int = None, inplen: int = None) -> torch.Tensor:
        assert (contlen and inplen), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]
        return logits

    
    def _loglikelihood_tokens(self, requests, disable_tqdm = False):
        res = []
        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)
        
        re_ord = Collator(requests, sort_fn = _collate)
        chunks = re_ord.get_batched(n = 0, batch_fn = self._batch_scheduler)
        pbar = tqdm(total=len(requests),
                    disable=(disable_tqdm or (self.rank != 0)),
                    desc="Running loglikelihood requests",
                    )
        
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []


            padding_len_inp = None
            
            
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying
            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],dtype=torch.long,device=self.device)
                (inplen,) = inp.shape
                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )
                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
                
            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")  # [batch, padding_len_inp]
            multi_logits = F.log_softmax(self(batched_inps), dim=-1)  # [batch, padding_length (inp or cont), vocab]
            
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (inplen + (logits.shape[0] - padding_len_inp))
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]
                
                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                
                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()
                    
                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)
                
        pbar.close()
        return re_ord.get_original(res)
                
                
    def get_model_info(self):
        return {"model_num_parameters": self.num_params,
                "model_name": str(self)}
        
    def tok_encode(self, text): return self.tokenizer.encode(text)
        
    @property
    def eot_token_id(self): return self.tokenizer.eot_token
    
    @property
    def max_length(self): return self.block_size
    
    #give ability to set max_length property
    @max_length.setter
    def max_length(self, value): self.block_size = value
    
    @property
    def _model(self): return self
    
    @property
    def rank(self): return rank_distrib()
    
    @property
    def world_size(self): return num_distrib() or 1

class evalBase(evalUtils):
    
    def loglikelihood(self, requests:list[Instance])-> list[tuple[float, bool]]:
        """
        This function is required by simple_evaluate to carry out evaluation.
        Directly taken by lm-evaluation-harness sample codes. 
        
        Each request contains Instance.args : Tuple[str, str] containing 1. an input string to the LM and 2. a target string on which the loglikelihood of the LM producing this target, conditioned on the input, will be returned.
        Each request will have, as result, (ll, is_greedy): Tuple[float, int] returned, where ll is a floating point number representing the log probability of generating the target string conditioned on the input, and is_greedy being either the value 0 or 1, with it being 1 if and only if the target string would be generated by greedy sampling from the LM (that is, if the target string is the most likely N-token string to be output by the LM given the input. )
        """
        with torch.no_grad(): #this is very important in order to achieve an optimal batch size for evaluation
            return super().loglikelihood(requests) 
       
    
    
    def evaluate(self, tasks:list[str], save_path = None):
        """
        run model evaluation on bencharks (as defined by Eleuther AI's lm-evaluation-harness)
        define your tasks as list of strings
        
        save results as dict (json) at `save_path` for later visualization. Defaults to None
        if save_path is None: results are not saved, but still displayed as a table
        """
        #insert assertions to verify if strings in benchmarks is valid
        #TODO: make this cleaner. save_path should automatically be sourced from Learner's save paths/model_dir
        
        
        
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = 128
        self.cache_hook = CacheHook(None)
        
        #if device is CPU, raise warning
        if self.device == torch.device('cpu'):
            warnings.warn('Evaluation on CPU is not recommended, since it is Extremely Slow. Please run on GPU for optimal performance by initializing the model on GPU using .cuda()')
        
        print('running eval. This may take a few minutes just to setup...')    
        results = simple_evaluate(self, tasks = tasks, batch_size = 'auto')
        if save_path is not None:
            pth = Path(save_path)/f'{str(self)}_eval.json'
            print('saving results at', pth)
            with open(pth, 'w') as f: json.dump(results, f)
        
        print(make_table(results))
        
        
    
    
    
    
        
    
    
    
    