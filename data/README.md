unlabeled.py contains scripts to download a dataset from huggingface datasets, and also for tokenizer methods (to save the text as tokenized data in .bin files)

download_X.py contains specific scripts for running unlabeled.py on specific datasets

loader.py is meant to retrieve data from  saved .bin files (containing tokenized data), and return dataloaders. s