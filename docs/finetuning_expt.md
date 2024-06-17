Expt:
Fine tune a model on a domain 

Goal:
To check what kind of results are theoretically possible, in how much time. 
Can an already good model (like Phi-3, which performs 70%-ish accross the board), be pushed accross ALL relevant benchmarks SIGNIFICANTLY?



1. (math, because you can easily select datasets and benchmarks manually and quickly evaluate). 
Also, you can directly use SFT datasets, because there isnt much diversity of information in the domain itself. A few concepts can solve all benchmarks. 

But, downside is that math is harder for LLMs by default. 

Benchmarks:
MMLU:
    - mmlu_abstract_algebra
    - mmlu_college_mathematics
    - mmlu_elementary_mathematics
    - mmlu_high_school_mathematics
    - mmlu_formal_logic

arithmetic
minerva_math (hard!!!)
hendrycks_math
mathqa
gsm8K


Datasets (SFT): see (https://huggingface.co/collections/HuggingFaceH4/awesome-sft-datasets-65788b571bf8e371c4e4241a)
1. tiedong/goat #this is a labeled dataset, we currently dont support that 
2. meta-math/MetaMathQA
3. TIGER-Lab/MathInstruct
3. openai/gsm8k

4. GAIR/MathPile (but this dataset is 9.5B tokens, so in itself is very large)
5. ArtifactAI/arxiv-math-instruct-50k (but contains high level concepts, and mathematical reasoning tokens)
6. microsoft/orca-math-word-problems-200k  



2. Logic (can directly use SFT datasets, not much diversity of information in the domain itself. A few concepts can solve all benchmarks)

Benchmarks:
MMLU:
    - mmlu_formal_logic

