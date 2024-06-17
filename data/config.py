from pathlib import Path

#Documnentation:
# Each class is a configuration for a dataset.
# Arguments:
# dataset: Name of the dataset
# default_cache_dir: Path to the cache directory, where datasets will be stored
# columns: columns to extract text from. If multiple columns are provided, they will be concatenated. 
# split_into_train_val: If True, the dataset will be split into train and validation
# split_name: After splitting training set, rename the split to this name. Default is 'val'
# split_pct: Percentage of the dataset to keep for training
# kwargs: Additional arguments for the dataset, that will directly be passed to load_dataset function


class OpenWebTextConfig():
    dataset = 'openwebtext'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/openwebtext').expanduser()
    columns = 'text'
    split_into_train_val = True
    split_name = 'val' #Optional
    
    

class WikipediaSimpleConfig():
    dataset = 'wikipedia'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/wikipedia_simple').expanduser()
    columns = 'text'
    split_into_train_val = True
    split_name = 'val' #Optional
    kwargs = {'name':'20220301.simple'}
    split_pct = 0.99
    
    
class WikipediaConfig():
    dataset = 'wikipedia'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/wikipedia').expanduser()
    columns = 'text'
    split_into_train_val = True
    split_name = 'val' #Optional
    kwargs = {'name':'20220301.en'}
    
    
class OrcaMath:
    dataset = 'microsoft/orca-math-word-problems-200k'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/orcamath').expanduser()
    columns = ('question', 'answer')
    split_into_train_val = True
    split_name = 'val' #Optional
    split_pct = 0.99
    


config_dict = { 'openwebtext':  OpenWebTextConfig,
                'wikisimple':   WikipediaSimpleConfig, 
                'wiki-en':      WikipediaConfig,
                'orcamath':     OrcaMath,
              }
    
    
    
    
    
    