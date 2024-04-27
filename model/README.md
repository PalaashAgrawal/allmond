standard GPT architecture based on Karpathy's NanoGPT code. 

Use the `GPT` class to define a pytorch based model. Use the property `.num_params` to get the total number of trainable parameters in the model. 
Since LLMs can get quite large and may not fit in the GPU(s), it is advised to first play around with the model size to ensure that the model is trainable on given GPU(s) with appropriate `batch_size`, `block_size` (ie context windoe length)
