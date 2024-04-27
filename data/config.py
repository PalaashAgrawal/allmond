from pathlib import Path

class OpenWebTextConfig():
    dataset_name = 'openwebtext'
    default_cache_dir = Path('~/.cache/tinyUniverse/pretraining_data/openwebtext').expanduser()
    split_into_train_val = True
    split_name = 'val' #Optional
    
    

class WikipediaSimpleConfig():
    dataset_name = 'wikipedia'
    default_cache_dir = Path('~/.cache/tinyUniverse/pretraining_data/wikipedia_simple').expanduser()
    split_into_train_val = True
    split_name = 'val' #Optional
    kwargs = {'name':'20220301.simple'}
    split_pct = 0.99
    


config_dict = {'openwebtext':OpenWebTextConfig,
           'wikisimple': WikipediaSimpleConfig}
    
    
    