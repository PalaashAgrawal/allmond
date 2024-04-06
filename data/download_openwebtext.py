from unlabeled import unlabeledDataset, TiktokenTokenizer
from config import OpenWebTextConfig
from pathlib import Path
import os



    
    

n_procs = max(1, int(os.cpu_count()-2)) #leave atleast 2 cores for other processes

encoder = TiktokenTokenizer()    

ds = unlabeledDataset(OpenWebTextConfig(), n_procs)

ds.tokenize(encoder.tokenize_dataset, 
            save_tokens_to_disk = True, 
            dtype = encoder._get_numpy_dtype()) 