from fastai.text.all import *
from fastai.distributed import *


class save_and_load_model_checkpoints(Callback):

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
    (DONE) remove hardcoding of cuda., and include device parameter 
    (I think this is taken care of). Implement DDP device handling. 
    


    """
    
    def __init__(self, dir=None, 
                 model_name=None, 
                 checkpoint_name='checkpoint', 
                 every_iters=10000,):
        
        self.path = Path(dir)
        self.model_dir = Path(model_name)
        self.checkpoint_name = checkpoint_name
        self.every_iters = every_iters
        self.best_valid_loss = float('inf')
        
    
    # def before_fit(self):
        
    #     if self.path: self.learn.path = self.path
    #     if self.model_dir: self.learn.model_dir = self.model_dir
        
        
    #     checkpoint = self.path/self.model_dir/f'{self.checkpoint_name}.pth'
    #     if checkpoint.exists():
    #         print(f'Resuming training using checkpoint {checkpoint}')
    #         self.learn.load(self.checkpoint_name)
        
    
    def after_step(self):
        """
        Method called after each training step to save model checkpoints.

        """
        
        if self.path: self.learn.path = self.path
        if self.model_dir: self.learn.model_dir = self.model_dir
    
    
        if self.learn.training and self.learn.iter>0 and self.learn.iter % self.every_iters == 0:
            pct = (self.learn.iter/self.learn.n_iter)%1.
            # accumulated_loss = 0.0
            # count = 0
            
            # print('len', len(self.learn.dls.valid))
        
            # for b in self.learn.dls.valid:
            #     xb,yb = self.learn._set_device(b)
            #     print('shape', xb.shape, yb.shape)
            #     with torch.no_grad():
            #         pred = self.learn.model(xb)
            #         loss_grad = self.learn.loss_func(pred, yb)
            #         loss = loss_grad.clone()
            #         accumulated_loss += loss
            #         count += yb.shape[0]
            
            # loss = accumulated_loss / count
            
            res = self.learn.validate()
            
            if res[0] < self.best_valid_loss:
                self.best_valid_loss = res[0]
                self.learn.save(f'{self.checkpoint_name}', with_opt=True, with_iter = True)
                
        
            #need to set the model back to training setting
            # self.learn.pct_train=(self.learn.epoch + pct/self.learn.n_epoch)/self.learn.n_epoch
            # self.model.train()
            # self.learn.training=True
            
            self.learn(f'before_fit')
            self.learn.pct_train=(self.learn.epoch + pct/self.learn.n_epoch)/self.learn.n_epoch
        
                

def rank0_decorator(func):
    def wrapper(self):
        return func(self)
    return wrapper
         

