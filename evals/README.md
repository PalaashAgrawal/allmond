# Evals for Tiny LM Project
Contains code for evaluation of our LLMs on:
- ARC - [arc.py](evals/arc.py), [link](https://allenai.org/data/arc)
- Hellaswag - [hellaswag.py](evals/hellaswag.py), [link](https://rowanzellers.com/hellaswag/)
- MMLU - [mmlu.py](evals/mmlu.py), [link](https://arxiv.org/pdf/2009.03300.pdf)
- MTEB - [mteb.py](evals/mteb.py), [link](https://arxiv.org/abs/2210.07316)
- nonsense-words-grammar - [nonsense.py](evals/nonsense.py), [link](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/nonsense_words_grammar)
- winograd schema challenge - [winograd.py](evals/winograd.py), [link](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)
- vitaminC - [vitaminC.py](evals/vitaminC.py), [link](https://aclanthology.org/2021.naacl-main.52/)

## Installation
```bash
cd evals
pip install -e .
```
This installs the editable package which you can then import in your code.
## Usage
You must first implement an interface for your model to be able to use the code.
This requires implementing:
- a predict method that takes in a batch of strings, and a batch of answer options, and returns a batch of predictions [for classification tasks]
- an embed method that takes in a batch of strings, and returns a batch of embeddings [for similarity tasks]
- a generate method that takes in a batch of strings, and returns a batch of generated strings [for generation tasks]

