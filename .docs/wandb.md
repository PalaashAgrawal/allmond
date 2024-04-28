## Setting up your system to log progress to Weights and Biases

Before logging any info to W&B (Weights and Biases), you need to setup a config file. This is very simple. In your terminal, run 
    
`wandb init`

(_Ensure that wandb is pip-installed in your environment_)


### Steps:

- If you're setting up for the first time, you will have to paste an API key, which you will get by navigating to the [wandb website](https://wandb.ai/), under User Settings (on the top right corner). 
- Then, in your terminal, Create (or set) a project, where all the different training runs will be logged. Each project is like a folder, that organizes multiple training runs together, and separates them from a different project. 