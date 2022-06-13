import os
import re
import json
import numpy as np
import random
import shutil
from pathlib import Path
from utils.optim import set_optim

import torch
import torch.distributed as dist


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save(model, optimizer, scheduler, step, best_metric, args, checkpoint_path, name, keep_last_ckpt=-1):
    os.makedirs(checkpoint_path, exist_ok=True)

    fp = os.path.join(checkpoint_path, f"checkpoint_{name}.pth.tar")
    state_dict = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'checkpoint_path': checkpoint_path,
        'args': args,
        'best_metric': best_metric
    }
    torch.save(state_dict, fp)
    shutil.copyfile(fp, os.path.join(checkpoint_path, f"checkpoint_last.pth.tar"))
    if keep_last_ckpt>0:
        remove_past_checkpoint(checkpoint_path, keep_last_ckpt=keep_last_ckpt)


def load(model, load_model_path, args, reset_optim=False):
    checkpoint = torch.load(load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    prev_args = checkpoint["args"]
    args = update_args(new_args=args, prev_args=prev_args)

    step = checkpoint["step"]
    best_metric = checkpoint["best_metric"]
    if not reset_optim:
        optimizer, scheduler = set_optim(args, model)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer, scheduler = set_optim(args, model)

    return model, optimizer, scheduler, args, step, best_metric


def remove_past_checkpoint(checkpoint_path, keep_last_ckpt):
    # number of checkpoints to keep
    all_files = [str(path) for path in sorted(Path(checkpoint_path).iterdir(), key=os.path.getmtime) if 'checkpoint' in str(path) and ('last' not in str(path) and 'best' not in str(path))]
    if keep_last_ckpt>0:
        files_to_remove = all_files[:-keep_last_ckpt]
    else:
        files_to_remove = all_files
    for f in files_to_remove:
        os.remove(f)


def update_args(new_args, prev_args):
    for arg in vars(prev_args):
        if arg not in new_args:
            setattr(new_args, arg, getattr(prev_args, arg))
    return new_args

