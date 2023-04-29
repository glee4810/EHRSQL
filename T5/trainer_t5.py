import wandb
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.model_utils import save

from generate import generate_sql

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, args, collator, best_metric, checkpoint_path, logger=None):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
                                train_dataset,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                drop_last=True,
                                num_workers=args.num_workers,
                                collate_fn=collator
                            )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
                                eval_dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size, 
                                drop_last=False,
                                num_workers=args.num_workers,
                                collate_fn=collator
                                )
    tokenizer = train_dataset.tokenizer

    cur_loss = 0.0
    epoch, step = 0, 0
    patience, early_stop_flag = 0, False
    if args.total_epoch == -1:
        setattr(args, 'total_epoch', np.inf)
    if args.total_step == -1:
        setattr(args, 'total_step', np.inf)
    model.train()
    while epoch < args.total_epoch and step < args.total_step and early_stop_flag==False :
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            input_ids = batch['inputs'].to(args.device)
            labels = batch['labels'].to(args.device)
            train_loss = model(
                               input_ids=input_ids, 
                               labels=labels
                               )[0]

            train_loss = torch.mean(train_loss) / args.accumulation_steps
            train_loss.backward()

            if step % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            cur_loss += train_loss.item()
            lr = scheduler.get_last_lr()[0]

            if step % args.report_every_step == 0:
                if logger is not None:
                    log = f"epoch: {epoch} (step: {step}) | "
                    log += f"train loss: {cur_loss/args.report_every_step:.6f} | "
                    log += f"lr: {lr:.6f}"
                    logger.info(log)
                cur_loss = 0.0
                if args.use_wandb:
                    wandb.log({f"train_loss": train_loss}, step=step)
                    wandb.log({f"learning_rate": lr}, step=step)

            if step % args.eval_every_step==0:
                with torch.no_grad():
                    valid_loss_list = []
                    for i, batch in enumerate(eval_dataloader):
                        input_ids = batch['inputs'].to(args.device)
                        labels = batch['labels'].to(args.device)
                        valid_loss = model(
                                        input_ids=input_ids,
                                        labels=labels
                                        )[0]
                        valid_loss_list.append(valid_loss.item())
                    valid_loss = sum(valid_loss_list)/len(valid_loss_list)

                    if args.eval_metric == 'loss':
                        valid_metric = valid_loss
                    elif args.eval_metric == 'esm':
                        out_eval = generate_sql(model, eval_dataset, args, collator, verbose=1)
                        valid_esm = []
                        for id_ in out_eval:
                            if out_eval[id_]['real'] == out_eval[id_]['pred']:
                                valid_esm.append(1)
                            else:
                                valid_esm.append(0)
                        valid_metric = sum(valid_esm)/len(valid_esm)*100
                    else:
                        NotImplementedError

                    if logger is not None:
                        log = f"epoch: {epoch} (step: {step})"
                        log += f" | valid_loss: {valid_loss:.6f}"
                        log += f" | valid_{args.eval_metric}: {valid_metric:.6f}"
                        logger.info(log)
                    if args.use_wandb:
                        wandb.log({f"valid_loss": valid_loss}, step=step)
                        wandb.log({f"valid_{args.eval_metric}": valid_metric}, step=step)

                    if (args.eval_metric=='loss' and valid_metric < best_metric) or \
                       (args.eval_metric=='esm' and valid_metric > best_metric):
                        best_metric = valid_metric
                        patience = 0
                        save(model, optimizer, scheduler, step, best_metric, args, checkpoint_path, name='best')
                        if logger is not None:
                            log = f"epoch: {epoch} (step: {step})"
                            log += f" | best metric updated ({args.eval_metric}) - {best_metric:.6f}"
                            logger.info(log)
                    else:
                        patience += 1

                    if args.show_eval_sample:
                        input_sample = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        gt_sample = tokenizer.decode(labels[0], skip_special_tokens=True)
                        pred_sample = tokenizer.decode(model.generate(
                                                                        input_ids=input_ids, 
                                                                        num_beams=args.num_beams,
                                                                        max_length=args.max_length,
                                                                        repetition_penalty=args.repetition_penalty,
                                                                        length_penalty=args.length_penalty,
                                                                        early_stopping=args.early_stopping
                                                                    )[0], skip_special_tokens=True)
                    if logger is not None:
                        log = f'\nevaluation sample:\ninput: {input_sample}\nreal: {gt_sample}\npred: {pred_sample}'
                        logger.info(log)
                        
                    if args.early_stop_patience > 0:
                        if patience == args.early_stop_patience:
                            if logger is not None:
                                logger.info(f'early stopping! (patience: {patience})')
                            early_stop_flag = True
                            break

            if args.save_every_step != -1 and step % args.save_every_step == 0:
                save(model, optimizer, scheduler, step, best_metric, args, checkpoint_path, name=f'{step}', keep_last_ckpt=args.keep_last_ckpt)

            if early_stop_flag or step > args.total_step:
                break

        if args.save_every_epoch:
            save(model, optimizer, scheduler, step, best_metric, args, checkpoint_path, name=f'{epoch}_{step}', keep_last_ckpt=args.keep_last_ckpt)

    # save(model, optimizer, scheduler, step, best_metric, args, checkpoint_path, name='last', keep_last_ckpt=args.keep_last_ckpt)
    if logger is not None:
        logger.info('training completed!')
