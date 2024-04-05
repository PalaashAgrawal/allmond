

from fastai.text.all import *


class save_model_checkpoints(Callback):
    """
    Callback to save model checkpoints during training.

    Args:
        dir (str): The directory path where the model checkpoints will be saved.
        model_name (str): The name of the model directory.
        checkpoint_name (str): The name of the checkpoint file.
        every_iters (int): The number of iterations after which a checkpoint will be saved.

    Attributes:
        path (Path): The directory path where the model checkpoints will be saved.
        model_name (Path): The name of the model directory.
        checkpoint_name (str): The name of the checkpoint file.
        every_iters (int): The number of iterations after which a checkpoint will be saved.
        best_valid_loss (float): The best validation loss achieved so far.

    Methods:
        after_step: Method called after each training step to save model checkpoints. 
        
        
    
    TODO: 
    remove hardcoding of cuda., and include device parameter
    Implement DDP device handling. 
    

    """
    def __init__(self, dir=None, 
                 model_name=None, 
                 checkpoint_name='checkpoint', 
                 every_iters=10000,):
        
        self.path = Path(dir)
        self.model_name = Path(model_name)
        self.checkpoint_name = checkpoint_name
        self.every_iters = every_iters
        self.best_valid_loss = float('inf')
        
        
    
        
    def after_step(self):
        """
        Method called after each training step to save model checkpoints.

        """
        if self.path: self.learn.path = self.path
        if self.model_name: self.learn.model_dir = self.model_name
        
        if self.learn.training and self.learn.iter and self.learn.iter % self.every_iters == 0:
            accumulated_loss = 0.0
            count = 0
        
            for xb, yb in self.learn.dls.valid:
                with torch.no_grad():
                    pred = self.learn.model(xb.cuda())
                    loss_grad = self.learn.loss_func(pred, yb.cuda())
                    loss = loss_grad.clone()
                    accumulated_loss += loss
                    count += yb.shape[0]
            
            loss = accumulated_loss / count

            if loss < self.best_valid_loss:
                self.best_valid_loss = loss
                self.learn.save(f'{self.checkpoint_name}', with_opt=True)
                

