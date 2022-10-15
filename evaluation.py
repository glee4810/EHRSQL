'''Official evaluation script for EHRSQL

Description
The goal of the model is to return as many correct answers to the answerable questions as possible while disregarding unanswerable questions.
- Precision measures how well the model generates SQL queries while it abstains from answering unanswerable questions.
- Recall measures how comprehensive the model can handle different question templates.
- F1 is a combination of precision and recall.
'''
import os
import sys
import json
import numpy as np
import argparse
import sqlite3
import warnings
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--infernece_result_path', required=True, type=str, help='path for inference')
    args.add_argument('--db_path', required=True, type=str, help='path database')    
    args.add_argument("--num_workers", type=int, default=-1)
    args.add_argument("--timeout", type=int, default=5.0, help='execution time limit in sec')
    args.add_argument("--threshold", type=float, default=-1, help='entropy threshold to abstrain from answering')
    args.add_argument("--out_file", type=str, default=None, help='path to save the output file')
    return args.parse_args()


exec_result = []
def result_tracker(result):
    exec_result.append(result)

def process_answer(ans):
    return str(set([ret[0] for ret in ans]))

def execute(sql, db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result

def execute_wrapper(sql, args, tag, skip_indicator='null'):
    if sql != skip_indicator:
        try:
            result = func_timeout(args.timeout, execute, args=(sql, args.db_path))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout_{tag}',)]
        except:
            result = [(f'error_{tag}',)] # possibly len(query) > 512 or not executable
        result = process_answer(result)
    else:
        result = skip_indicator
    return result

def execute_query(sql1, sql2, args, data_idx=None):
    '''
    Execute the query. Time out if it exceeds {args.timeout} seconds
    '''
    result1 = execute_wrapper(sql1, args, tag='real')
    result2 = execute_wrapper(sql2, args, tag='pred')
    result = {'data_idx': data_idx, 'real': result1, 'pred': result2}
    return result

def execute_query_distributed(real, pred, db_path, num_workers):
    pool = mp.Pool(processes=num_workers)
    for data_idx, (sql1, sql2) in enumerate(zip(real, pred)):
        pool.apply_async(execute_query, args=(sql1, sql2, args, data_idx), callback = result_tracker)
    pool.close()
    pool.join()


def main(args):

    num_workers = mp.cpu_count() if args.num_workers==-1 else args.num_workers
    with open(args.infernece_result_path, 'r') as f:
        data = json.load(f)
    print(f'[result] {len(data)} lines loaded')

    data_id = []
    query_real = []
    query_pred = []
    entropy = []
    impossible = []
    cnt = 0
    for idx_, line in data.items():
        data_id.append(idx_)
        query_real.append(line['real'])
        query_pred.append(line['pred'])
        entropy.append(max(line['sequence_entropy']))
        impossible.append(line['is_impossible'])

    if args.threshold == -1:
        warnings.warn("Threshold value is not set! All predictions are sent to the database.")

    threshold = args.threshold if args.threshold != -1 else np.inf
    query_real = [label if label!='nan' else 'null' for label in query_real]
    query_pred = [pred if ent <= threshold else 'null' for pred, ent in zip(query_pred, entropy)]

    exec_real = []
    exec_pred = []
    if num_workers>1:
        execute_query_distributed(query_real, query_pred, args.db_path, num_workers)
        indices = []
        for ret in exec_result:
            exec_real.append(ret['real'])
            exec_pred.append(ret['pred'])
            indices.append(ret['data_idx'])
        exec_real = np.array(exec_real)[np.argsort(indices)]
        exec_pred = np.array(exec_pred)[np.argsort(indices)]
    else:
        for sql1, sql2 in zip(query_real, query_pred):
            ret = execute_query(sql1, sql2, args)
            exec_real.append(ret['real'])
            exec_pred.append(ret['pred'])
        exec_real = np.array(exec_real)
        exec_pred = np.array(exec_pred)

    ans_precision_list = []
    ans_recall_list = []
    ans_precision_exec_list = []
    ans_recall_exec_list = []   
    for idx in range(len(exec_real)):
        ans_real, ans_pred = exec_real[idx], exec_pred[idx]
        if ans_pred!='null': # calculate the score over predicted answerable queries
            ans_precision_list.append(1 if ans_real != 'null' else 0)
            ans_precision_exec_list.append(1 if ans_real == ans_pred else 0)
        if ans_real!='null': # calculate the score over GT answerable queries
            ans_recall_list.append(1 if ans_pred != 'null' else 0)
            ans_recall_exec_list.append(1 if ans_real == ans_pred else 0)

    ans_precision = sum(ans_precision_list) / len(ans_precision_list)
    ans_recall = sum(ans_recall_list) / len(ans_recall_list)
    ans_f1 = 2*((ans_precision*ans_recall)/(ans_precision+ans_recall+1e-10))
    ans_precision_exec = sum(ans_precision_exec_list) / len(ans_precision_exec_list)
    ans_recall_exec = sum(ans_recall_exec_list) / len(ans_recall_exec_list)
    ans_f1_exec = 2*((ans_precision_exec*ans_recall_exec)/(ans_precision_exec+ans_recall_exec+1e-10))

    out_eval = OrderedDict([
        ('precision', 100.0 * ans_precision),
        ('recall', 100.0 * ans_recall),
        ('f1', 100.0 * ans_f1),
        ('precision_exec', 100.0 * ans_precision_exec),
        ('recall_exec', 100.0 * ans_recall_exec),
        ('f1_exec', 100.0 * ans_f1_exec),
    ])

    if args.out_file:
        with open(args.out_file, 'w') as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))

if __name__ == '__main__':
    args = parse_args()
    main(args)




