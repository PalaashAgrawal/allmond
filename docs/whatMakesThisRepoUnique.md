## Whats special about this repo?
This repo is a very _clean pythonic implementation_ of LLM pipelines. 

- _Clean_ means that different components are clearly separated, and initialized using intuitive function arguments. 
    - Most LLM implementation use yml config files to pass configurations to python files, but I personally don't like that, because it makes the code unnecessarily longer, and forces you to navigate to a different file to see config details. I rather prefer defining a minimal set of variables within the python script itself passed directly to model/data/learner initializations, and defining/importing new functions to the main script directly, to create different types of architectures, datasets, etc. 

- _Clean_ also means that training scripts are very minimal, which is a result of high level abstractions. This ensures you don't have to scroll through unnecessary code. 
    - Standardized models and data structures. This repo implements models in standard pytorch nn.Module formats, and corresponding data in pytorch dataloader formats. Distributed training is also well handled in our repo.  
    - Most people don't care about the training strategy, a standard training (optimization) pipeline usually works well. So we hide away code related to optimizer steps, learning rate scheduling, etc from the scripts. We use _[fast.ai](https://docs.fast.ai/)_ to handle the training process.
        - What is _fast.ai_? fast.ai is a library built on top of PyTorch that provides high-level functions to train deep learning models quickly. Basically given a model and dataloaders, it will carry out the training process super efficiently, including learning rate/momentum scheduling,loss calculation, weight update, distributed parallelization,etc. 
        
        - It is highly flexible as well. As you will notice, we add various kinds of custom functionalities, such as evaluating on the validation dataset at regular intervals during training. We also write a few wrappers around fast.ai functions to incorporate LLM training efficiently. This is needed because 
        fast.ai expects the components to be in a standard format. The problem with LLM training is that the underlying components  don't always fit the standard pytorch data structure format that can be accomodated in a typical computer, simply because of the sheer scale. So in this repo, we build simple wrappers around functions to accomodate LLM training. Dataloaders access data using memory maps instead of a Dataset class object, for example. 



In short, we hide away implementation details in scripts, and expose hyperparameters that actually affect training (data format, model architectures, batch_size, etc etc). On the contrary, most repos, expose unnecessary implementation details (like the training process (iterating through dataloaders, forwarding through model, calculating loss, etc)), and hide away important hyperparameters in config files. 
