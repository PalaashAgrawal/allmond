# Model

Standard GPT architecture based on Karpathy's NanoGPT code. 

Use the `GPT` class to define a pytorch based model. 



### To Customize the Model

- `transformer_components.py` contains various standard blocks used in a transformer architecture. You can use them, along with custom Implementations to create a custom model architecture class. 

- `gpt.py` brings together the components to build a standard GPT architecture. 


- In a nutshell, your custom model class should have the following features. 

    - Should inherit from `nn.Module`
    - Should contain a `forward` function, which ONLY takes in one argument (input tensor `x`), and returns one single output prediction tensor. 
        - No other value should be passed or returned. Many implementations, such as Karpathy's implementation, take output tensor `y` as input and/or return loss along with model predictions. This is NOT CORRECT. The model is only meant to forward the input through the model layers, and return a prediction. Calculation of loss values is the job of the `Learner` class after the model predictions are obtained. 
    - Should include a `generate` function, that takes a sequence of tokens (as list), detokenizes them, and returns a string. Useful for inference.
    - Should preferably include a `get_num_params` function, to calculate number of trainable parameters in the model. As you know, larger models can easily lead to CUDA OOM errors. So for larger models, you would have to adjust batch_size or block_size (context window length)
    - Should preferably include a `model_name` attribute (string) that the function can use to log to wandb (say, if you're experimenting with different model variations). 

- `huggingface_wrappers.py`
    - You can also import pretrained models from huggingface. Class `GPT` (in `gpt.py`) will automatically wrap the model to the standard format (i.e., the format expected by the trainer class).
    - For any huggingface model, you only have to create a wrapper in `huggingface_wrappers.py` under the `HF_base` class. Simply return the huggingface model with a config dict (which are stored as attributes in the GPT class). 
    - (In progress), I'm adding model support slowly, but you can always create your own custom model definition




### DevLogs

- I decided to move the tokenizer to model submodule (from the data submodule). This decision stems from the following 
    - For huggingface based models, each model comes with a (potentially) unique tokenizer. So for a developer who wishes to add a new model's support in `huggingface_wrappers`, they should be able to define all the related dependencies (model params, configs AND tokenizer) in a single script. They shouldn't have to jump between scripts. 
    - This means that in your main script, you pass a tokenizer as `model.tokenizer` instead of a custom defined tokenizer. 
    - By default (a standard GPT architecture) will use a TikToken Tokenizer. You can define which model's tokenizer you wish to use (as supported by TikToken)
    - TODO: create functionality for custom tokenizer, even for default GPT based architectures. 
    