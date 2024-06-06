from model.gpt import GPT
from lm_eval.evaluator import simple_evaluate
model_id = 'microsoft/Phi-3-mini-4k-instruct'
qlora = False

model = GPT.from_hf(model_id, enable_qlora = qlora)
# simple_evaluate(model, tasks = ['boolq'])
# print(model.generate('hello world', max_new_tokens = 10, temperature = 0))
# print(model.generate([0,45,12,12,2], max_new_tokens = 10))


simple_evaluate(model, tasks = ['boolq'])
