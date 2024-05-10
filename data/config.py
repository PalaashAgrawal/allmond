from pathlib import Path

class OpenWebTextConfig():
    dataset_name = 'openwebtext'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/openwebtext').expanduser()
    split_into_train_val = True
    split_name = 'val' #Optional
    
    

class WikipediaSimpleConfig():
    dataset_name = 'wikipedia'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/wikipedia_simple').expanduser()
    split_into_train_val = True
    split_name = 'val' #Optional
    kwargs = {'name':'20220301.simple'}
    split_pct = 0.99
    
    
class WikipediaConfig():
    dataset_name = 'wikipedia'
    default_cache_dir = Path('~/.cache/allmond/pretraining_data/wikipedia').expanduser()
    split_into_train_val = True
    split_name = 'val' #Optional
    kwargs = {'name':'20220301.en'}
    


config_dict = {'openwebtext':OpenWebTextConfig,
                'wikisimple': WikipediaSimpleConfig, 
                'wiki-en': WikipediaConfig }
    
    
    