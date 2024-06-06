#cleaned iteration: check for working

from lm_eval.api.model import LM 
from lm_eval.api.instance import Instance
# from lm_eval.utils import Collator


import importlib
import pathlib
from copy import deepcopy
from typing import List, Literal
from fastai.torch_core import rank_distrib, num_distrib
import torch
from tqdm import tqdm
import numpy as np


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
    condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list: return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len

def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b

class evalBase(LM):
    

    #utils for evaluation. These functions are required by lm-evaluation-harness
    
    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer.encode(context + continuation)
        context_enc = self.tokenizer.encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):        
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        
        for (context, continuation), context_enc, continuation_enc in requests:
            inp = (context_enc + continuation_enc)[-(self.max_length - 1) :]
            output = self.generate(inp, max_new_tokens = 0, return_logprobs=True)
            logprob = sum(output).tolist()
            is_greedy = self.is_greedy
            res.append((logprob, is_greedy))
            pbar.update(1)
        
        pbar.close()
        return res

    @torch.no_grad()
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
                    # self.tok_encode(continuation),
                    self.tokenizer.encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation) #more efficient to encode them together, rather than individually. You can always concat and split

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)
    
    
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        """
        Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: BOS/EOS
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  BOS   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the BOS/EOS token.
                Can also be overridden for custom cases by `prefix_token_id`.
        """
      
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            # Tokenize the input string
            token_list = self.tokenizer.encode(string)
            
            # Get rolling token windows
            rolling_token_windows = list(get_rolling_token_windows(
                token_list=token_list,
                prefix_token=self.eot_token_id,
                max_seq_len=self.max_length,
                context_len=1,
            ))

            # Make disjoint windows
            rolling_token_windows = [make_disjoint_window(x) for x in rolling_token_windows]

            # Compute log-likelihood for each window
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # Discard is_greedy
            string_nll = [x[0] for x in string_nll]

            # Sum the log-likelihoods for the entire string
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    
    
    def generate_until(self, requests):
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        if not requests: return []
    
        res = []

        def get_until(req_args):
            until = req_args.get("until", [])
            until = deepcopy(until)  # prevent from modifying req_args for cache_key
            if self.tokenizer.ids_to_tokens([self.eot_token_id])[0] not in until:
                until.append(self.tokenizer.ids_to_tokens([self.eot_token_id])[0])
            return until

        for request in requests:
            context, req_args = request.args
            until = get_until(req_args)
            max_gen_toks = req_args.get("max_gen_toks", self.max_gen_toks)
            
            remaining_length = self.max_length - max_gen_toks
            encoded_context = self.tokenizer.encode(context)
            encoded_context = encoded_context[-remaining_length:]
            context_str = self.tokenizer.decode(encoded_context)

            generated_text = context_str
            for _ in range(max_gen_toks):
                output = self.generate(
                    generated_text,
                    max_new_tokens=1,
                    temperature=0,  # greedy generation
                    top_k=None,
                    return_input=False
                )

                generated_text += self.tokenizer.decode(output)

                stop = False
                for term in until:
                    if term in generated_text:
                        generated_text = generated_text.split(term)[0]
                        stop = True
                        break

                if stop: break

            self.cache_hook.add_partial("greedy_until", request, generated_text)
            res.append(generated_text)

        return res

        

    @property
    def rank(self): return rank_distrib() #or 1
    
    @property
    def world_size(self): return num_distrib() or 1
    
    @property
    def eot_token_id(self): return self.tokenizer.eot_token
    
    @property
    def max_length(self): return self.block_size
    
    #give ability to set max_length property
    @max_length.setter
    def max_length(self, value): self.block_size = value
    
    def tok_encode(self, text): return self.tokenizer.encode(text)
    
    
    

    
    
    
# from lm_eval.api.model import LM 
# from lm_eval.api.instance import Instance
# # from lm_eval.utils import Collator


# import importlib
# import pathlib
# from copy import deepcopy
# from typing import List, Literal
# from fastai.torch_core import rank_distrib, num_distrib
# import torch
# from tqdm import tqdm
# import numpy as np


# def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
#     """
#     - context_len allows for a rolling window context, allowing each prediction window to potentially
#     condition on some context

#     :param token_list: list
#         List of tokens to be PREDICTED
#     :param max_seq_len: int
#         max_seq_len of model (or max_seq_len we want to use)
#     :param context_len: int
#         Amount of desired token context for prediction. Needs to be at least 1.
#     :param prefix_token: token
#         Dummy token like <eos> so the first token has something to condition on
#     :return: generator
#         Generator of tuples
#             (input_tokens, pred_tokens)
#         Note: Score only the last len(pred_tokens) logits of the LM
#     """
#     assert 1 <= context_len <= max_seq_len
#     if not token_list:
#         return
#     # +1 offset, going from input->preds
#     pred_len = max_seq_len - context_len + 1
#     predicted = 0

#     # Special handling for first window: predict all tokens
#     first_seq_len = min(max_seq_len, len(token_list))
#     yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
#     predicted += first_seq_len

#     while predicted < len(token_list):
#         window_pred_len = min(len(token_list) - predicted, pred_len)
#         window_end = predicted + window_pred_len

#         yield (
#             token_list[window_end - max_seq_len - 1 : window_end - 1],
#             token_list[window_end - window_pred_len : window_end],
#         )
#         predicted += window_pred_len

# def make_disjoint_window(pair):
#     """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
#     a, b = pair
#     return a[: len(a) - (len(b) - 1)], b

# class evalBase(LM):
    
#     #PAg -- so that I dont have to rename the functions sourced from 
#     # https://github.com/EleutherAI/lm-evaluation-harness/blob/070d31df4edad2d8263b64b3ace7ec0cbaeb7ae8/lm_eval/models/nemo_lm.py#L333
#     def __init__(self):
#         # super().__init__()
#         print('welp i have been initialized mmmhmmm')
#         self.tok_encode = self.tokenizer.encode
#         self.eot_token_id = self.tokenizer.eot_token
#         self.max_length = self.block_size
        
        
        
#     #utils for evaluation. These functions are required by lm-evaluation-harness
    
#     def _encode_pair(self, context, continuation):
#         n_spaces = len(context) - len(context.rstrip())
#         if n_spaces > 0:
#             continuation = context[-n_spaces:] + continuation
#             context = context[:-n_spaces]
#         # whole_enc = self.tok_encode(context + continuation)
#         # context_enc = self.tok_encode(context)
#         whole_enc = self.tokenizer.encode(context + continuation)
#         context_enc = self.tokenizer.encode(context)
#         context_enc_len = len(context_enc)
#         continuation_enc = whole_enc[context_enc_len:]
#         return context_enc, continuation_enc

    
#     def _loglikelihood_tokens(self, requests, disable_tqdm=False):
#         #my generate function does not support batch processing
        
#         res = []
#         pbar = tqdm(
#             total=len(requests),
#             disable=(disable_tqdm or (self.rank != 0)),
#             desc="Running loglikelihood requests",
#         )
        
#         for (context, continuation), context_enc, continuation_enc in requests:
#         # for req in requests:
#             # print('checking', req)
#             # (context_enc, continuation_enc) = req
#             inp = (context_enc + continuation_enc)[-(self.max_length - 1) :]
#             output = self.generate(inp, max_new_tokens = 0, return_logprobs=True)
#             logprob = sum(output).tolist()
#             # continuation_tokens = output["token_ids"]
#             # is_greedy = continuation_tokens == output["greedy_tokens"]
#             is_greedy = self.is_greedy
#             res.append((logprob, is_greedy))
#             pbar.update(1)
        
#         pbar.close()
#         return res
    
#         # res = []

#         # def _collate(x):
#         #     toks = x[1] + x[2]
#         #     return -len(toks), tuple(toks)

#         # re_ord = Collator(requests, sort_fn=_collate)
#         # chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
#         # pbar = tqdm(
#         #     total=len(requests),
#         #     disable=(disable_tqdm or (self.rank != 0)),
#         #     desc="Running loglikelihood requests",
#         # )
#         # for chunk in chunks:
#         #     inps = []
#         #     ctxlens = []
#         #     contlens = []

#         #     for _, context_enc, continuation_enc in chunk:
#         #         # Leave one token for generation. Tokens_to_generate = 0 breaks NeMo.
#         #         inp = (context_enc + continuation_enc)[-(self.max_length - 1) :]

#         #         ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - (self.max_length - 1))
#         #         ctxlens.append(ctxlen)
#         #         contlens.append(len(continuation_enc))

#         #         inps.append(self.tok_decode(inp))
            
#         #         output = self.generate(inp, max_new_tokens = 0, return_logprobs=True)
                

#             # output = self.generate(
#             #     self.model,
#             #     inputs=inps,
#             #     tokens_to_generate=1,
#             #     min_tokens_to_generate=1,
#             #     compute_logprob=True,
#             #     all_probs=True,
#             # )
            
            
#             # logits = self(inps)
#             # logits = logits[:]
#             # logprobs = self.generate(inps, max_new_tokens = , return_logprobs=True) #what should be the max_new_tokens? in the openai example, max_new_tokens = 0
            
            

#             # batch_token_ids = np.asarray(output["token_ids"])[:, :-1]
#             # batch_logprobs = output["logprob"][:, :-1]
#             # batch_full_logprob = output["full_logprob"][:, :-1, :]

#             # Compute greedy tokens for entire batch rather than calling it with proper ctxlen for each sample.
#             # Additional tokens for each sample will be trimmed later.
#         #     min_ctxlen = min(ctxlens)

#         #     # Use min_ctxlen-1 instead of min_ctxlen since full_logprobs are not returns for the first token.
#         #     batch_greedy_tokens = (
#         #         torch.argmax(batch_full_logprob[:, min_ctxlen - 1 :, :], -1)
#         #         .cpu()
#         #         .numpy()
#         #     )

#         #     for token_ids, greedy_tokens, logprobs, ctxlen, contlen, (
#         #         cache_key,
#         #         _,
#         #         _,
#         #     ) in zip(
#         #         batch_token_ids,
#         #         batch_greedy_tokens,
#         #         batch_logprobs,
#         #         ctxlens,
#         #         contlens,
#         #         chunk,
#         #     ):
#         #         # Trim at contlen since shorter contexts in a batch will have more than one token generated.
#         #         # Use ctxlen-1 instead of ctxlen same as for full_logprob in batch_greedy_tokens calculation
#         #         logprobs = (logprobs[ctxlen - 1 :])[:contlen]
#         #         logprob = sum(logprobs).tolist()

#         #         continuation_tokens = (token_ids[ctxlen:])[:contlen]
#         #         len_diff = ctxlen - min_ctxlen
#         #         is_greedy = continuation_tokens == (greedy_tokens[len_diff:])[:contlen]
#         #         if not isinstance(is_greedy, bool):
#         #             is_greedy = is_greedy.all()
#         #         answer = (logprob, is_greedy)

#         #         if cache_key is not None:
#         #             self.cache_hook.add_partial("loglikelihood", cache_key, answer)

#         #         res.append(answer)
#         #         pbar.update(1)

#         # pbar.close()

#         # return re_ord.get_original(res)
    

#     # Provided helper functions
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     def loglikelihood(self, requests:list[Instance])-> list[tuple[float, bool]]:
#         """
#         Each request contains Instance.args : Tuple[str, str] containing 1. an input string to the LM and 2. a target string on which the loglikelihood of the LM producing this target, conditioned on the input, will be returned.
#         Each request will have, as result, (ll, is_greedy): Tuple[float, int] returned, where ll is a floating point number representing the log probability of generating the target string conditioned on the input, and is_greedy being either the value 0 or 1, with it being 1 if and only if the target string would be generated by greedy sampling from the LM (that is, if the target string is the most likely N-token string to be output by the LM given the input. )
#         """
        
#         new_reqs = []
#         for context, continuation in [req.args for req in requests]:
#             if context == "":
#                 # end of text as context
#                 context_enc, continuation_enc = (
#                     [self.eot_token_id],
#                     # self.tok_encode(continuation),
#                     self.tokenizer.encode(continuation),
#                 )
#             else:
#                 context_enc, continuation_enc = self._encode_pair(context, continuation) #more efficient to encode them together, rather than individually. You can always concat and split

#             new_reqs.append(((context, continuation), context_enc, continuation_enc))

#         return self._loglikelihood_tokens(new_reqs)
    
    
#     def loglikelihood_rolling(
#         self, requests: List[Instance], disable_tqdm: bool = False
#     ) -> List[float]:
#         """
#         Compute full log-likelihood of a string, with no truncation, for perplexity computation
#         - We will use the full max context length of the model.
#         - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
#         the max context length.
#         - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
#           which may simply concatenate multiple documents together.
#         - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
#           multiple chunks, the last input will still a full-sized context.
#           Example:
#             Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
#             Prefix: BOS/EOS
#             Max context length: 4
#             Resulting input/prediction pairs:

#                 INPUT:  BOS   0   1   2
#                 PRED:     0   1   2   3

#                 INPUT:    3   4   5   6
#                 PRED:     4   5   6   7

#                 INPUT:    5   6   7   8
#                 PRED:             8   9

#           Observe that:
#             1. Each token is predicted exactly once
#             2. For the last pair, we provide the full context, but only score the last two tokens

#         :param requests: list[Instance]
#             A list of Instance objects with property `args` which returns a tuple (context,).
#             string: str
#                 String for which we are computing overall loglikelihood
#         :return: list[tuple[float]]
#             A list of tuples (logprob,)
#             logprob: float
#                 The log probability of `context` conditioned on the BOS/EOS token.
#                 Can also be overridden for custom cases by `prefix_token_id`.
#         """
      
#         loglikelihoods = []

#         for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
#             # Tokenize the input string
#             # token_list = self.tok_encode(string)
#             token_list = self.tokenizer.encode(string)
            
#             # Get rolling token windows
#             rolling_token_windows = list(get_rolling_token_windows(
#                 token_list=token_list,
#                 prefix_token=self.eot_token_id,
#                 max_seq_len=self.max_length,
#                 context_len=1,
#             ))

#             # Make disjoint windows
#             rolling_token_windows = [make_disjoint_window(x) for x in rolling_token_windows]

#             # Compute log-likelihood for each window
#             string_nll = self._loglikelihood_tokens(
#                 rolling_token_windows,
#             )

#             # Discard is_greedy
#             string_nll = [x[0] for x in string_nll]

#             # Sum the log-likelihoods for the entire string
#             string_nll = sum(string_nll)
#             loglikelihoods.append(string_nll)

#         return loglikelihoods

        
        
        

    
    
    
#     def generate_until(self, requests):
#         """Generate greedily until a stopping sequence

#         :param requests: list[Instance]
#             A list of Instance objects with property `args` which returns a tuple (context, until).
#             context: str
#                 Context string
#             until: [str]
#                 The string sequences to generate until. These string sequences
#                 may each span across multiple tokens, or may be part of one token.
#         :return: list[str]
#             A list of strings continuation
#             continuation: str
#                 The generated continuation.
#         """
#         if not requests: return []
    
#         res = []

#         def get_until(req_args):
#             until = req_args.get("until", [])
#             until = deepcopy(until)  # prevent from modifying req_args for cache_key
#             if self.tokenizer.ids_to_tokens([self.eot_token_id])[0] not in until:
#                 until.append(self.tokenizer.ids_to_tokens([self.eot_token_id])[0])
#             return until

#         for request in requests:
#             context, req_args = request.args
#             until = get_until(req_args)
#             max_gen_toks = req_args.get("max_gen_toks", self.max_gen_toks)
            
#             remaining_length = self.max_length - max_gen_toks
#             # encoded_context = self.tok_encode(context)
#             encoded_context = self.tokenizer.encode(context)
#             encoded_context = encoded_context[-remaining_length:]
#             context_str = self.tokenizer.decode(encoded_context)

#             generated_text = context_str
#             for _ in range(max_gen_toks):
#                 output = self.generate(
#                     generated_text,
#                     max_new_tokens=1,
#                     temperature=0,  # greedy generation
#                     top_k=None,
#                     return_input=False
#                 )

#                 generated_text += self.tokenizer.decode(output)

#                 stop = False
#                 for term in until:
#                     if term in generated_text:
#                         generated_text = generated_text.split(term)[0]
#                         stop = True
#                         break

#                 if stop:
#                     break

#             self.cache_hook.add_partial("greedy_until", request, generated_text)
#             res.append(generated_text)

#         return res

        
        
    

#     @property
#     def rank(self): return rank_distrib() #or 1
    
#     @property
#     def world_size(self): return num_distrib() or 1
    

    