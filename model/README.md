# Model

Standard GPT architecture based on Karpathy's NanoGPT code. 

Use the `GPT` class to define a pytorch based model. 



### To Customize the Model

- `gpt.py` contains various standard blocks used in a transformer architecture. You can use them, along with custom Implementations to create a custom model architecture class. 
- In a nutshell, your custom model class should have the following features. 

    - Should inherit from `nn.Module`
    - Should contain a `forward` function, which ONLY takes in one argument (input tensor `x`), and returns one single output prediction tensor. 
        - No other value should be passed or returned. Many implementations, such as Karpathy's implementation, take output tensor `y` as input and/or return loss along with model predictions. This is NOT CORRECT. The model is only meant to forward the input through the model layers, and return a prediction. Calculation of loss values is the job of the `Learner` class after the model predictions are obtained. 
    - Should include a `generate` function, that takes a sequence of tokens (as list), detokenizes them, and returns a string. Useful for inference.
    - Should preferably include a `get_num_params` function, to calculate number of trainable parameters in the model. As you know, larger models can easily lead to CUDA OOM errors. So for larger models, you would have to adjust batch_size or block_size (context window length)
    - Should preferably include a `__str__` function that you can probably use to log to wandb (say, if you're experimenting with different model variations). 
