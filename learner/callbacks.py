from fastai.text.all import *
from fastai.distributed import *
from torch import nn

class save_checkpoints(Callback):

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
        
        
    
    def after_step(self):
        """
        Method called after each training step to save model checkpoints.

        """
        
        
        if self.path: self.learn.path = self.path
        if self.model_dir: self.learn.model_dir = self.model_dir
        
    
    
        if (self.learn.training and 
            self.learn.iter>0 and self.learn.iter % self.every_iters == 0): #only execute for rank==0
            
            # pct = (self.learn.iter/self.learn.n_iter)%1.
            accumulated_loss = 0.0
            count = 0
                    
            for b in self.learn.dls.valid:
                xb,yb = self.learn._set_device(b)
                with torch.no_grad():
                    pred = self.learn.model(xb)
                    loss_grad = self.learn.loss_func(pred, yb)
                    loss = loss_grad.clone()
                    accumulated_loss += loss
                    count += yb.shape[0]
            
            loss = accumulated_loss / count
                        
           
            
            if not rank_distrib() and loss< self.best_valid_loss: #only save for 
                self.best_valid_loss = loss
                self.learn.save(f'{self.checkpoint_name}', with_opt=True, with_iter = True)                
        


class qlora_resolve(Callback):
    """
    This Callback is essential to run before running any QLORA models. 
    Essentially, here we introduce minor Changes in the remove_hook_from_module from `accelerater.hooks.py`. 
    Why is this necessary?
    By default, accelerate runs the AlignDevicesHook on top of your model to ensure model is in the correct device. 
    But For Quantized modes, this function creates a lot of problems. Specifically, it doesnt remove the AlignDevicesHook hook from the model automatically after running an interation. 
    This hook somehow misaligns the input to LLM and the model (it always puts the input to cuda:0)
    So we have to remove the hook using this function by force.
     
    But the problem is that, delattr, for some reason, doesnt remove the hook from the model (this is a bug from Huggingface. OR It may just be an issue related to quantization).
    So we do a few changes (like first reassigning the variables, and then deleting them) to remove the hook.
        
    """
    
    def __init__(self, enable_qlora = True):
        self.enable_qlora = enable_qlora
        
    def remove_hook_from_module(self, module: nn.Module, recurse = False):
       
        
        try:
            from accelerate.hooks import AlignDevicesHook
        except:
            print("accelerate package not founds")

        module._hf_hook = AlignDevicesHook()
        
        if hasattr(module, "_hf_hook"):
            module._hf_hook.detach_hook(module)
            delattr(module, "_hf_hook")

        if hasattr(module, "_old_forward"):
            # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
            # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
            if "GraphModuleImpl" in str(type(module)):
                module.__class__.forward = module._old_forward
            else:
                module.forward = module._old_forward
            
            module._old_forward = None
            delattr(module, "_old_forward")

        # if recurse:
        #     for child in module.children():
        #         remove_hook_from_module(child, recurse)

        return module
    
    
    def before_fit(self):
        if self.enable_qlora:
            self.learn.model.base_model = self.remove_hook_from_module(self.learn.model.base_model)
        