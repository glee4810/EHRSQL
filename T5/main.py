import sys, os
sys.path.append(os.getcwd())
import json
import numpy as np
import argparse
import yaml

import torch
from utils.model_utils import set_seeds, load, update_args
from utils.optim import set_optim
from utils.dataset import EHRSQL_Dataset, DataCollator
from utils.logger import init_logger
from config import Config

from T5.model import load_model, load_tokenizer


if __name__ == '__main__':
    args = Config()
    args.get_param(use_model_param=True,
                   use_eval_param=True,
                   use_optim_param=True)
    args.parser.add_argument('--config', required=True, type=str)
    args.parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str)
    args = args.parse()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    for k, v in config.items():
        if config[k]:
            setattr(args, k, config.get(k))

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES if args.device == 'cuda' else ""
    if torch.cuda.is_available():
        print(f'Current device: cuda:{args.CUDA_VISIBLE_DEVICES}')
    else:
        print('Current device: cpu')
    set_seeds(args.random_seed)
    
    output_path = os.path.join(args.output_dir, args.exp_name)
    if os.path.exists(output_path):
        raise Exception(f"directory already exists ({output_path})")

    logger = None
    if args.mode=='train':
        logger = init_logger(output_path, args)
        logger.info(args)

    if args.use_wandb:
        import wandb
        from wandb_api_key import WANDB_API_KEY
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        if args.mode == 'train':
            wandb.init(project=args.wandb_project, name=args.exp_name)

    tokenizer = load_tokenizer(args.model_name)
    data_collator = DataCollator(tokenizer=tokenizer, return_tensors='pt')

    if args.mode=='train':
        train_dataset = EHRSQL_Dataset(path=args.train_data_path, tokenizer=tokenizer, args=args, data_ratio=args.training_data_ratio)
        valid_dataset = EHRSQL_Dataset(path=args.valid_data_path, tokenizer=tokenizer, args=args)
        if logger:
            logger.info(f"loaded {len(train_dataset)} training examples from {args.train_data_path}")
            logger.info(f"loaded {len(valid_dataset)} valid examples from {args.valid_data_path}")
    elif args.mode=='eval':
        test_dataset = EHRSQL_Dataset(path=args.test_data_path, tokenizer=tokenizer, args=args, include_impossible=True)
        print(f"loaded {len(test_dataset)} test examples from {args.test_data_path}")

    model = load_model(model_name=args.model_name)
    if args.bf16:
        model = model.to(torch.bfloat16)
    if args.init_weights:
        model.init_weights()
    if args.load_model_path is None:
        model = model.to(args.device)
        optimizer, scheduler = set_optim(args, model)
        step = 0
        if args.eval_metric == 'loss':
            best_metric = np.inf
        elif args.eval_metric == 'esm':
            best_metric = -np.inf
    else:
        model, optimizer, scheduler, args, step, best_metric = load(model, args.load_model_path, args)
        if logger:
            logger.info("loading checkpoint %s" %args.load_model_path)
        else:
            print("loading checkpoint %s" %args.load_model_path)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.mode=='eval':
        import json
        from T5.generate import generate_sql
        print("start inference")
        out_eval = generate_sql(model=model, eval_dataset=test_dataset, args=args, collator=data_collator, verbose=1)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, args.output_file), 'w') as f:
            json.dump(out_eval, f)
    else:
        from trainer_t5 import train
        print("start training")
        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            args=args,
            collator=data_collator,
            best_metric=best_metric,
            checkpoint_path=output_path,
            logger=logger
        )