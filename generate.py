import re
import csv
import os
import numpy as np
import sqlite3
import time
import multiprocessing as mp
from utils.process import post_process_sql

import torch
from torch.utils.data import DataLoader, SequentialSampler


def generate_sql(model, eval_dataset, args, collator, logger, verbose=0):

    if not hasattr(args, 'current_time'):
        Exception(f'current_time must be specified!')

    file_name = args.config.split('/')[-1]
    start_time = time.time()
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(
                                eval_dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size, 
                                drop_last=False,
                                num_workers=args.num_workers,
                                collate_fn=collator
                                )
    tokenizer = eval_dataset.tokenizer
    model.eval()
    

    do_sample = True if args.num_beams == 1 else False
    with torch.no_grad():
        for idx, batch in enumerate(dataloader, 1):

            input_ids = batch['inputs'].to(args.device)
            labels = batch['labels'].to(args.device)
            db_ids = batch['db_id']
            q_tags = batch['q_tag']
            t_tags = batch['t_tag']
            o_tags = batch['o_tag']
            para_types = batch['para_type']
            imps = batch['imp']
            is_impossibles = batch['is_impossible']

            generation_output = model.generate(
                                    input_ids=input_ids, 
                                    num_beams=args.num_beams,
                                    max_length=args.max_length,
                                    do_sample=do_sample,
                                    num_return_sequences=args.num_samples,
                                    repetition_penalty=args.repetition_penalty,
                                    length_penalty=args.length_penalty,
                                    early_stopping=args.early_stopping, 
                                    return_dict_in_generate=True, 
                                    output_scores=True
                                    )

            preds = generation_output['sequences'].cpu() if args.device == 'cuda' else generation_output['sequences']
            sequences_scores = generation_output['sequences_scores'].cpu() if args.device == 'cuda' else generation_output['sequences_scores']
            logits = torch.stack(generation_output['scores'], dim=1)[::int(args.num_beams/args.num_samples)]
            logits = logits.cpu() if args.device == 'cuda' else logits
            output_prob = torch.softmax(logits, dim=2)
            log_prob = torch.log_softmax(logits, dim=2)
            sequences_entropy = ( torch.sum(output_prob * log_prob, dim=2) * (-1) ).numpy()

            for i in range(len(preds)):
                
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                real = tokenizer.decode(labels[i], skip_special_tokens=True)
                pred = tokenizer.decode(preds[i], skip_special_tokens=True)

                real_executable = post_process_sql(real, current_time=args.current_time)
                pred_executable = post_process_sql(pred, current_time=args.current_time)

                pred_tensor = preds[i][1:]
                entropy = sequences_entropy[i].tolist()
                if tokenizer.eos_token_id in pred_tensor:
                    pred_eos_idx = torch.nonzero(pred_tensor==tokenizer.eos_token_id)[0].item()
                    entropy = entropy[:pred_eos_idx+1]

                log = {}
                log['db_id'] = db_ids[i]
                log['question'] = text
                log['real'] = real_executable
                log['pred'] = pred_executable
                log['is_impossible'] = is_impossibles[i]
                log['sequence_entropy'] = tuple(entropy)
                log['para_type'] = para_types[i]
                if log['is_impossible']==False:                    
                    log['q_tag'] = q_tags[i]
                    log['t_tag'] = tuple(t_tags[i])
                    log['o_tag'] = tuple(o_tags[i])                    
                    log['imp'] = imps[i]
                logger.info(log)

            if verbose>0:
                print(f'{idx}/{len(dataloader)} ({round(idx/len(dataloader)*100, 4)}%) --- {file_name}', end='\r')

    end_log = f"inference took {round(time.time() - start_time, 6)} secs"
    logger.info(end_log)


