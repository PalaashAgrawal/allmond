#not for PR
from .callbacks import SkipToIter, QLORA_resolve, GetLargestBatchSize
from .fastai_learner_mod import LLMLearnerBase
from fastai.learner import *   
from fastai.text.all import * #this is a very lazy import. Basically imports everything


class LLMLearner(LLMLearnerBase):
    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0, 
            start_iter = 0,
            find_largest_batch_size = False):
        """
        All definitions are the same as fastai.Learner, except a few new definitions
        
        start_epoch and start_iter override values loaded from checkpoints
        find_largest_batch_size: If True, automatically finds largest batch size that can fit on model device. If False, we use the default batch_size specified in the dataloaders. 
        """        

        if hasattr(self, 'resumeIter'):
            start_epoch  = start_epoch or self.resumeIter['epoch']
            start_iter = start_iter or self.resumeIter['iter']    
            
        if start_epoch != 0 or start_iter != 0:
            cbs = L(cbs) + SkipToIter(start_epoch, start_iter) 
        
        if find_largest_batch_size:
            cbs = L(cbs) + GetLargestBatchSize()
            
        #PAg: not for PR
        cbs = L(cbs)+QLORA_resolve(getattr(self.model, 'qlora', False))
        
        with self.added_cbs(cbs):
            if reset_opt or not self.opt: self.create_opt()
            if wd is None: wd = self.wd
            if wd is not None: self.opt.set_hypers(wd=wd)
            self.opt.set_hypers(lr=self.lr if lr is None else lr)
            self.n_epoch = n_epoch    
            self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

    
    #PAg: not for PR
    def check_and_load_learner(self, file, device = 'cuda'):
        f'check if a checkpoint exists'
        checkpoint = self.path/self.model_dir/f'{file}.pth'
        if checkpoint.exists():
            self.load(file, device = device)
            print(f"Resuming training from iteration {getattr(self, 'resumeIter', {}).get('iter', 0)} of epoch {getattr(self, 'resumeIter', {}).get('epoch', 0)}  using checkpoint {str(checkpoint)}")

       