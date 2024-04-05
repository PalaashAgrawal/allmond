import numpy as np 
import os 

data = np.memmap('train.bin', dtype=np.uint16, mode='r') 


# count the number of tokens
num_tokens = len(data)
print(f"Number of tokens: {num_tokens}")

# print first 15 tokens, in a pretty f-string
print(f"First 15 tokens: {data[:15]}")