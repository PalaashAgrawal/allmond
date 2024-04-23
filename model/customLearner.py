from fastai.learner import *
# from fastcore.all import *
# from fastai.callback.all import *
from fastai.vision.all import * #placeholder to resolve all errors



def save_model(file, model, opt, iteration, with_opt = True, with_iter = True, pickle_protocol = 2, **torch_save_kwargs):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`), and iteration information (epoch and iteration) (if available, and if `with_epoch_iter`. This allows automatically resumable training)"
    if with_iter: assert with_opt, f'Optimizer state must be saved for epoch/iteration resumable training. Set with_opt= True  if with_epoch_iter=True. '
    
    if rank_distrib(): return # don't save if child proc
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    if with_iter: state['iter'] = iteration
    torch.save(state, file, pickle_protocol=pickle_protocol, **torch_save_kwargs)
        

def load_model(file, model, opt, with_opt=True, with_iter = True, device=None, strict=True, **torch_load_kwargs):
    
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`), and iteration information (epoch and iteration, saved as a dictionary in self.resumeIter) (if available, and if `with_epoch_iter`. This allows automatically resumable training. "
    if isinstance(device, int): device = torch.device('cuda', device)
    elif device is None: device = 'cpu'
    state = torch.load(file, map_location=device, **torch_load_kwargs)
    
    hasopt = 'opt' in state
    hasiter = 'iter' in state
    
    
    model_state = state['model'] if hasopt else state
    
    get_model(model).load_state_dict(model_state, strict=strict)
    
    if hasopt and with_opt:
        try: opt.load_state_dict(state['opt'])
        except:
            if with_opt: warn("Could not load the optimizer state.")
    elif with_opt: warn("Saved file doesn't contain an optimizer state.")
    
    
    if hasiter and with_iter:
        try: 
            return state['iter'] #unclean solution. How do i change this to a reference modification instead of return value?
            
        except:
            if with_iter: warn("Could not load the iteration state.")
            
    elif with_iter: warn("Saved file doesn't contain iteration state.")
        
        
        
class customLearner(Learner):
    """
    The goal of this learner is to
    A. learner should automatically resume training using a checkpoint file
    B. 
    
    1. Save training checkpoints ALONG WITH epoch and iteration value. (save and save_model functions)
        a. in the fit function, IF THE USER WANTS TO OVERRIDE SAVED VALUES, they can specify value, but then model params will be different
        b. SaveModelCallback args should be modified to incorporate iter and epoch values as well)
    2. Similarly loading should be compatible.  (load and load_model functions)
    3. Custom SkipToIter(Callback) (akin to SkiptoEpoch)
    """
    
    
    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0, start_iter = 0):
        
        if hasattr(self, 'resumeIter'):
            start_epoch  = start_epoch or self.resumeIter['epoch']
            start_iter = start_iter or self.resumeIter['iter']    
            
        if start_epoch != 0 or start_iter != 0:
            cbs = L(cbs) + SkipToIter(start_epoch, start_iter)
        
        
        with self.added_cbs(cbs):
            if reset_opt or not self.opt: self.create_opt()
            if wd is None: wd = self.wd
            if wd is not None: self.opt.set_hypers(wd=wd)
            self.opt.set_hypers(lr=self.lr if lr is None else lr)
            self.n_epoch = n_epoch
                        
            
            self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
        
    def _do_fit(self):
        for epoch in range(self.n_epoch):
            self.epoch=epoch
            self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    
    
    #PAg: not for PR
    def check_and_load_learner(self, file, device = 'cuda'):
        f'check if a checkpoint exists'
        checkpoint = self.path/self.model_dir/f'{file}.pth'
        if checkpoint.exists():
            print(f'Resuming training using checkpoint {str(checkpoint)}')
            self.load(file, device = device)

@patch
@delegates(save_model)
def save(self:Learner, file, **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    
    iteration = {'epoch':getattr(self,'epoch',0), 'iter': getattr(self,'iter',0) }
    
    save_model(file, self.model, getattr(self,'opt',None), iteration, **kwargs)
    return file


    
@patch
@delegates(load_model)
def load(self:Learner, file, device=None, **kwargs):
    "Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`"
    if device is None and hasattr(self.dls, 'device'): device = self.dls.device
    if self.opt is None: self.create_opt()
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    
    distrib_barrier()
    
    iteration = load_model(file, self.model, self.opt, device=device, **kwargs)
    if iteration is not None: self.resumeIter = iteration
    
    
    return self
    
    
class SkipToIter(Callback):
    "Skip training up to   `iter`th iteration in `epoch`th epoch"
    "if epoch and iter passed during initialization are not 0, they override values derived from loaded hyperparameters in learn.load"
    order = 70
    
    def __init__(self, epoch:int, iter: int = 0):
        self._skip_to_epoch = epoch
        self._skip_to_iter = iter
        
    def before_epoch(self):
        if self.epoch < self._skip_to_epoch:
            raise CancelEpochException
        
    def before_batch(self):
        if self.iter < self._skip_to_iter:
            raise CancelBatchException
