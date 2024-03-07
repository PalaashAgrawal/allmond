import os, time, math, pickle, hydra
from omegaconf import DictConfig, OmegaConf
from contextlib import nullcontext

import numpy as np
import torch

from models.build_models import * 



class DataLoader:
    def __init__(self, data_dir, device, context_window=512, batch_size=8):
        self.data_dir = data_dir 
        self.device = device
        self.context_window = context_window
        self.batch_size = batch_size

    def get_batch(self, split="train")
        data = np.memmap(os.path.join(self.data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')

        ix = torch.randint(len(data) - self.context_window, (self.batch_size,))
        X = torch.stack([torch.from_numpy((data[i:i+self.context_window]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.context_window]).astype(np.int64)) for i in ix])


        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)

        return X, y
    

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters, ctx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataloader.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@hydra.main(config_path="configs/train/", config_name="nanoGPT.yaml")
def main(model_cfg: DictConfig) -> None:
    # Load the general config file
    general_cfg_path = hydra.utils.to_absolute_path("configs/general_config.yaml")
    general_cfg = OmegaConf.load(general_cfg_path)
    
    # Merge the general configuration with the nanoGPT configuration
    cfg = OmegaConf.merge(general_cfg, model_cfg)

    # set the random seed
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    



    # specify the device and dtype for training
    device = "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = True 

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))



    tokens_pre_iteration = cfg.training.gradient_accumulation_steps * cfg.training.batch_size * cfg.arch.context_window
    print(f"Tokens per iteration: {tokens_pre_iteration}")


    # get dataloader
    dataloader = DataLoader(
        data_dir=os.path.join(
            cfg.data_path, 
            cfg.training.dataset,
            cfg.training.tokenizer
        ),
        device=device,
    )


    iter_num = 0
    best_val_loss = 1e9
    
    # model
    model = build_nanoGPT(cfg.arch)
    model.to(device)

    # optimizer 
    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.optimizer.weight_decay, 
        learning_rate=cfg.training.optimizer.lr, 
        betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2), 
        device_type=device
    )


    # start logging 
    if cfg.logging.wandb_log:
        import wandb
        wandb.init(project=cfg.logging.wandb_project, config=cfg)

    

    # start training
    X,y = dataloader.get_batch()
    t0 = time.time()
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(
            iter_num, 
            cfg.training.warmup_iters, 
            cfg.training.lr_decay_iters, 
            cfg.training.optimizer.lr, 
            cfg.training.optimizer.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if not iter_num % cfg.training.eval_interval:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg.logging.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })

        for micro_step in range(cfg.training.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, y)
                loss = loss / cfg.training.gradient_accumulation_steps
            
            X, y = dataloader.get_batch()
            scaler.scale(loss).backward()

        if cfg.training.optimizer.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.optimizer.grad_clip)

        






if __name__ == "__main__":
    main()
