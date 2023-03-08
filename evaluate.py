'''Official evaluation script for EHRSQL'''

import os
import re
import sys
import json
import argparse
import sqlite3
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_file', metavar='data.json', help='input data json file')
    args.add_argument('--pred_file', metavar='pred.json', help='model predictions')
    args.add_argument('--db_path', required=True, type=str, help='path database')
    args.add_argument("--num_workers", type=int, default=-1)
    args.add_argument("--timeout", type=int, default=60.0, help='execution time limit in sec')
    args.add_argument("--out_file", type=str, default=None, help='path to save the output file')
    args.add_argument("--ndigits", type=int, default=2, help='scores rounded to ndigits')
    args.add_argument("--current_time", type=str, default='2105-12-31 23:59:00')
    return args.parse_args()

def post_process_sql(query,
                     current_time="2105-12-31 23:59:00",
                     precomputed_dict={
                                'temperature': (35.5, 38.1),
                                'sao2': (95.0, 100.0),
                                'heart rate': (60.0, 100.0),
                                'respiration': (12.0, 18.0),
                                'systolic bp': (90.0, 120.0),
                                'diastolic bp':(60.0, 90.0),
                                'mean bp': (60.0, 110.0)
                            }):
    query = query.lower()
    if "current_time" in query:
        query = query.replace("current_time", f"'{current_time}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list)==1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
    query = query.replace("''", "'").replace('< =', '<=')
    query = query.replace("%y", "%Y").replace('%j', '%J')
    query = query.replace("'now'", f"'{current_time}'")
    return query

exec_result = []
def result_tracker(result):
    exec_result.append(result)

def process_answer(ans):
    return str(sorted([str(ret) for ret in ans[:100]])) # check only up to 100th record

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
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    with open(args.pred_file, 'r') as f:
        pred = json.load(f)

    data_id = []
    query_real = []
    query_pred = []
    for line in data:
        id_ = line['id']
        data_id.append(id_)
        real = post_process_sql(line['query'], current_time=args.current_time)
        query_real.append(real)
        if id_ in pred:
            query_pred.append(post_process_sql(pred[id_], current_time=args.current_time))
        else:
            query_pred.append('n/a')

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

    precision_ans_list = []
    precision_exec_list = []
    recall_ans_list = []
    recall_exec_list = []
    for idx in range(len(exec_real)):
        ans_real, ans_pred = exec_real[idx], exec_pred[idx]
        if ans_pred!='null': # calculate the score over predicted answerable queries
            precision_ans_list.append(1 if ans_real != 'null' else 0)
            precision_exec_list.append(1 if ans_real == ans_pred else 0)
        if ans_real!='null': # calculate the score over GT answerable queries
            recall_ans_list.append(1 if ans_pred != 'null' else 0)
            recall_exec_list.append(1 if ans_real == ans_pred else 0)

    precision_ans = sum(precision_ans_list) / (len(precision_ans_list)+1e-10)
    recall_ans = sum(recall_ans_list) / (len(recall_ans_list)+1e-10)
    f1_ans = 2*((precision_ans*recall_ans)/(precision_ans+recall_ans+1e-10))
    precision_exec = sum(precision_exec_list) / (len(precision_exec_list)+1e-10)
    recall_exec = sum(recall_exec_list) / (len(recall_exec_list)+1e-10)
    f1_exec = 2*((precision_exec*recall_exec)/(precision_exec+recall_exec+1e-10))

    out_eval = OrderedDict([
        ('precision_ans', round(100.0 * precision_ans, args.ndigits)),
        ('recall_ans', round(100.0 * recall_ans, args.ndigits)),
        ('f1_ans', round(100.0 * f1_ans, args.ndigits)),
        ('precision_exec', round(100.0 * precision_exec, args.ndigits)),
        ('recall_exec', round(100.0 * recall_exec, args.ndigits)),
        ('f1_exec', round(100.0 * f1_exec, args.ndigits)),
    ])

    if args.out_file:
        with open(args.out_file, 'w') as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))

if __name__ == '__main__':
    args = parse_args()
    main(args)