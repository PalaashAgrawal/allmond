Scripts for training/evaluating code

1.  `train.py`: train a simple GPT architecture from scratch on the wiki dataset. Modify architecture parameters and dataset to customize training run. 
2. `train_phi3.py`: finetune a phi-3 model using a QLORA adapter on wiki dataset. Modify dataset and base model (in replacement of Phi3) to customize

3. `eval_phi3.py`: evaluate a phi-3 model (raw, without QLORA) on the MMLU benchmark. Modify benchmark list (as defined by EleutherAI's lm-evaluation-harness) and model (either saved (as in saved finetuned QLORA-phi3), or directly from huggingface (as in pretrained model weights)) to customize. 

