from fastai.text.all import * #this is a very lazy import. Basically imports everything
from fastai.learner import *    


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
        
        
        
class LLMLearner(Learner):
    """
    Custom Learner Class specially designed for LLMs
    Features:
    1.  The Learner can save/load checkpoints with epoch/iteration values last logged. Useful for parallel distributed training, where hardware often fails abruptly. Learner will automatically resume training from last saved epoch AS WELL AS iteration. 
    2.  Added capability to resume from an Iteration (by default, only resume from epoch was supported)
    3.  slightly modified _with_events definition, that carries out after_{event_type} within the try block. 
        The default definition causes error during distributed training. Particularly in the recorder class. See github issue link in _with_event definition
    """
    
    
    
    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0, start_iter = 0):
        'start_epoch and start_iter override values loaded from checkpoints.'
        
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
            
            
    #PAg: not for pR
    def _with_events(self, f, event_type, ex, final=noop):
        """
        Not a fool-proof solution. This function is a very core function in Learner. Changing this isnt ideal. 
        But its required for resumable training (training resumed from a particular iteration). 
        I even submitted a PR for this: https://github.com/fastai/fastai/issues/4030. 
        """
        try: self(f'before_{event_type}');  f();  self(f'after_{event_type}')
        except ex: self(f'after_cancel_{event_type}')
        final()
        
        
    def _do_one_batch(self):
        f'PAg: modified to support CPU offloading, by casting xb,yb to the device of the model prediction'
        #cast xb, yb to the device of the model prediction

        self.xb = tuple(map(lambda x: x.to(self.model.device), self.xb))
        self.yb = tuple(map(lambda y: y.to(self.model.device), self.yb))
        
        
        
        
        self.pred = self.model(*self.xb)
        self('after_pred')
        if len(self.yb):
            # self.yb = tuple(map(lambda y: y.to(self.pred.device), self.yb)) #required for CPU offloading in FSDP
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return
        self._do_grad_opt()


    
    #PAg: not for PR
    def check_and_load_learner(self, file, device = 'cuda'):
        f'check if a checkpoint exists'
        checkpoint = self.path/self.model_dir/f'{file}.pth'
        if checkpoint.exists():
            self.load(file, device = device)
            print(f"Resuming training from iteration {getattr(self, 'resumeIter', {}).get('iter', 0)} of epoch {getattr(self, 'resumeIter', {}).get('epoch', 0)}  using checkpoint {str(checkpoint)}")


# %% ../../nbs/14_callback.schedule.ipynb 46
@patch
def fit_one_cycle(self:Learner, n_epoch, lr_max=None, div=25., div_final=1e5, pct_start=0.25, wd=None,
                  moms=None, cbs=None, reset_opt=False, start_epoch=0, start_iter = 0):
    "PAg: added support start_iter"
    "Fit `self.model` for `n_epoch` using the 1cycle policy."
    if self.opt is None: self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
    lr_max = np.array([h['lr'] for h in self.opt.hypers])
    scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
              'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    
    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch, start_iter = start_iter)
    
    
    
    
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
    
