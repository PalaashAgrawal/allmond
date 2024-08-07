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
                        
           
            
            if not rank_distrib() and loss<self.best_valid_loss: #only save for 
                self.best_valid_loss = loss
                print(f'saving checkpoint: {self.path/self.model_dir/(str(self.checkpoint_name)+".pth")} at iteration {self.learn.iter} with validation loss {loss}')
                self.learn.save(f'{self.checkpoint_name}', with_opt=True, with_iter = True)                
        


class QLORA_resolve(Callback):
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
        



#PAg: consider pushing to fastai PR
class SkipToIter(Callback):
    "Skip training up to   `iter`th iteration in `epoch`th epoch"
    "if epoch and iter passed during initialization are not 0, they override values derived from loaded hyperparameters in learn.load"
    order = 51
    
    def __init__(self, epoch:int, iter: int):
        self._skip_to_epoch = epoch
        self._skip_to_iter = iter
        
    def before_epoch(self):
        if self.epoch < self._skip_to_epoch:
            raise CancelEpochException
        
    def before_batch(self):
        if self.iter < self._skip_to_iter:
            raise CancelBatchException
        

# class GetLargestBatchSize(Callback):
#     """
#     TODO: can get_largest_bs be merged with _detect_batch_size in model.eval.eval.py?
#     """
#     order = 100  # we want this to be executed at the very end. At this point, model has been assigned to appropriate 
#     def can_allocate_memory(self, batch_size, block_size = None):
#         try:
#             model = self.learn.model
#             block_size = block_size or (model.module.block_size if isinstance(model,nn.parallel.distributed.DistributedDataParallel) else model.block_size)
                
#             x_test, y_test = torch.ones((batch_size, block_size), device=self.device).long(), torch.ones((batch_size, block_size), device=self.device).long()
#             # Run a forward pass
#             output = model(x_test)
#             loss = self.learn.loss_func(output, y_test)
#             loss.backward()
#             return True
#         except RuntimeError as e:
            
#             if "out of memory" in str(e): 
#                 return False
#             else: raise e
#         finally:
#             self.learn.opt.zero_grad()
            
    
#     def find_optimal_bs_binary_search(self, max_bs, block_size = None):
#          # Decrement phase: fine-tune the batch '/  tsize by decreasing in steps of 1
#         # while not can_allocate_memory(batch_size) and batch_size > 1: batch_size-=1
#         high, low = max_bs, max_bs // 2
#         while low < high - 1:
#             mid = (low + high) // 2
#             if self.can_allocate_memory(mid, block_size): low = mid
#             else: high = mid
            
#         return low
    
#     def find_optimal_block_size_binary_search(self, max_bs, block_size = None):
#         high, low = block_size, block_size // 2
#         while low < high - 1:
#             mid = (low + high) // 2
#             if self.can_allocate_memory(max_bs, mid): low = mid
#             else: high = mid
            
#         return low
        
        
#     def get_largest_bs(self, max_bs = 64):
        
#         # Start with a batch size of 2 and increase in powers of 2
#         batch_size = 1
#         while batch_size<=max_bs and  self.can_allocate_memory(batch_size): batch_size *= 2
        
#         if batch_size>max_bs: return batch_size
#         batch_size = self.find_optimal_bs_binary_search(batch_size)

#         # Ensure at least one batch can be processed
#         final_batch_size =  max(batch_size, 1)
#         return final_batch_size
    
    
#     def before_fit(self):
#         print(f'Calculating maximum batch_size that can fit the device {self.learn.model.device}')
#         bs = self.get_largest_bs()
#         print(f'Detected largest batch_size to fit {self.learn.model.device} = {bs}')
#         for dl in self.dls.loaders: dl.bs = bs
#         #also store the batch size in the model, for evaluation purposes
#         self.learn.model.bs = bs
        
        


class GetLargestBatchSize(Callback):
    """
    TODO: can get_largest_bs be merged with _detect_batch_size in model.eval.eval.py?
    """
    order = 100  # we want this to be executed at the very end. At this point, model has been assigned to appropriate 

    def can_allocate_memory(self, batch_size, block_size=None):
        try:
            
            model = self.learn.model
            block_size = block_size or (model.module.block_size if isinstance(model, nn.parallel.distributed.DistributedDataParallel) else model.block_size)
            print('trying batch_size =', batch_size, 'and window size', block_size)
            
            
            x_test = torch.ones((batch_size, block_size), device=self.device).long()
            y_test = torch.ones((batch_size, block_size), device=self.device).long()
            # Run a forward pass
            output = model(x_test)
            loss = self.learn.loss_func(output, y_test)
            loss.backward()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            else:
                raise e
        finally:
            self.learn.opt.zero_grad()
            
    def find_optimal_bs(self, max_bs, block_size=None):
        if max_bs == 1: return 1
        high, low = max_bs, max_bs // 2
        while low < high - 1:
            mid = (low + high) // 2
            if self.can_allocate_memory(mid, block_size):
                low = mid
            else:
                high = mid
        return low
    
    def find_optimal_block_size(self, batch_size, max_block_size):
        while max_block_size > 1 and not self.can_allocate_memory(batch_size, max_block_size): max_block_size//=2
        
        high, low = max_block_size*2, max_block_size
        while low < high - 1:
            mid = (low + high) // 2
            if self.can_allocate_memory(batch_size, mid): low = mid
            else: high = mid
            
        return low

    def get_largest_bs(self, max_bs=64):
        block_size = self.learn.model.block_size
        batch_size = 1
        
        # Check if even batch size 1 causes OOM
        if not self.can_allocate_memory(1, block_size):
            # Find the optimal block size using binary search
            # print('Block size 1 causes OOM. Adjusting optimal block size')
            block_size = self.find_optimal_block_size(1, block_size)
            self.learn.model.block_size = block_size
            
        
        # Start with a batch size of 2 and increase in powers of 2
        while batch_size < max_bs and self.can_allocate_memory(batch_size, block_size): batch_size *= 2
        
        if batch_size==max_bs and self.can_allocate_memory(max_bs): return max_bs
        
        batch_size = self.find_optimal_bs(batch_size, block_size)

        # # Ensure at least one batch can be processed
        # final_batch_size = max(batch_size, 1)
        return batch_size, block_size
    
    def before_fit(self):
        print(f'Calculating maximum batch_size that can fit the device {self.learn.model.device}')
        bs, block_size = self.get_largest_bs()
        print(f'Detected largest batch_size to fit {self.learn.model.device} = {bs} with block_size = {block_size}')
        
        for dl in self.dls.loaders:
            dl.bs = bs
            dl.block_size = block_size
        
        self.learn.model.bs = bs
        self.learn.model.block_size = block_size

        
        
        
        