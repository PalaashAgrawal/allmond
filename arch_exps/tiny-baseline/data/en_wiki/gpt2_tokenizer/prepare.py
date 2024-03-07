
import os 
from tqdm import tqdm 
import numpy as np 
import tiktoken

from datasets import load_dataset


# number workers 
num_proc = 8

# load encoder 
encoder = tiktoken.get_encoding("gpt2")


# load the dataset 
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")

# create a test split with 0.0001 of the data
split_dataset = dataset["train"].train_test_split(
    test_size=0.001, 
    seed=489, 
    shuffle=True
)

def process(example):
    ids = encoder.encode_ordinary(example["text"])
    ids.append(encoder.eot_token)
    return {"ids": ids, "len": len(ids)}

# tokenize dataset
tokenized = split_dataset.map(
    process,
    remove_columns=["url", "title", "text"],
    desc="Tokenizing the dataset",
    num_proc=num_proc
)

# concatenate all the ids in each dataset into one large file for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(
        os.path.dirname(__file__), f'{split}.bin'
    )
    dtype = np.uint16 # possible since enc.max_token_value ==50256 is < 2**16
    arr = np.memmap(
        filename,
        dtype=dtype,
        mode="w+",
        shape=(arr_len,)
    )
    total_batches = 1024

    idx = 0 
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        batch = dset.shard(
            num_shards=total_batches,
            index=batch_idx,
            contiguous=True
        ).with_format('numpy')
        arr_batch = np.concatenate(batch["ids"])

        # write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush() 