import os
import sys
import numpy as np
import pandas as pd
import argparse
import sqlite3
import multiprocessing as mp
from collections import Counter

from func_timeout import func_timeout, FunctionTimedOut
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.process import post_process_sql


exec_result = []
def result_tracker(result):
    exec_result.append(result)
    print(f'execution processed {len(exec_result)}/{len(processed_lines)}', end='\r')

def execute_distributed(real, pred, db_path, num_workers):
    pool = mp.Pool(processes=num_workers)
    for data_idx, (sql1, sql2) in enumerate(zip(real, pred)):
        pool.apply_async(execute_query, args=(sql1, sql2, data_idx, args), callback = result_tracker)
    pool.close()
    pool.join()

def execute_sql(sql, args):
    con = sqlite3.connect(args.db_path)
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result

def execute_query(sql1, sql2, data_idx, args):

    try:
        result1 = func_timeout(args.timeout, execute_sql, args=(sql1, args))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result1 = [(f'timeout_real (>{args.timeout} sec)',)]
    except:
        result1 = [('error_real',)] # possibly len(query) > 512 or not executable

    try:
        result2 = func_timeout(args.timeout, execute_sql, args=(sql2, args))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result2 = [(f'timeout_pred (>{args.timeout} sec)',)]
    except:
        result2 = [('error_pred',)]

    result = {'data_idx': data_idx, 'real': result1, 'pred': result2}
    return result

def process_answer(ans):
    return str(set([ret[0] for ret in ans]))

def esm_score(gt, pred):
    gt_tokens = word_tokenize(gt)
    pred_tokens = word_tokenize(pred)
    sm_flag = (gt_tokens==pred_tokens)
    return sm_flag


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--infernece_result_path', required=True, type=str, help='path for inference')
    args.add_argument('--db_path', required=True, type=str, help='path database')    
    args.add_argument("--num_workers", type=int, default=-1)
    args.add_argument("--timeout", type=int, default=5.0, help='execution time limit in sec')
    args.add_argument("--entropy_threshold", type=float, default=-1, help='-1 means no threshold value is set'))
    args = args.parse_args()

    num_workers = mp.cpu_count() if args.num_workers==-1 else args.num_workers
    with open(args.infernece_result_path, 'r') as f:
        data = f.read().split('\n')
    processed_lines = []
    for line in data:
        try:
            processed_lines.append(eval(line))
        except:
            pass
    print(f'[result] {len(processed_lines)} lines loaded')


    question_ = [] 
    query_real_ = []
    query_pred_ = []
    impossible_real_ = []
    para_type_ = []
    entropy_ = []

    q_tags_ = []
    t_tags_ = []
    importance_ = []

    for row_idx, line in enumerate(processed_lines):
        question_.append(line['question'])
        query_real_.append(line['real'])
        query_pred_.append(line['pred'])
        para_type_.append(line['para_type'])
        entropy_.append(max(line['sequence_entropy']))
        impossible_real_.append(line['is_impossible'])
        if line['is_impossible'] == False:
            q_tags_.append(line['q_tag'])
            t_tags_.append(line['t_tag'])
            importance_.append(line['imp'])
        else:
            q_tags_.append('nan')
            t_tags_.append('nan')
            importance_.append('nan')      

    execute_distributed(query_real_, query_pred_, args.db_path, num_workers)
    exec_real_ = []
    exec_pred_ = []
    indices_ = []
    for ret in exec_result:
        exec_real_.append(process_answer(ret['real']))
        exec_pred_.append(process_answer(ret['pred']))
        indices_.append(ret['data_idx'])
    exec_real_ = np.array(exec_real_)[np.argsort(indices_)]
    exec_pred_ = np.array(exec_pred_)[np.argsort(indices_)]

    
    ientropy_threshold = args.entropy_threshold if args.entropy_threshold != -1 else np.inf
    impossible_pred_ = [ent > entropy_threshold for ent in entropy_]


    question_ = np.array(question_)
    query_real_ = np.array(query_real_)
    query_pred_ = np.array(query_pred_)
    impossible_real_ = np.array(impossible_real_)  
    impossible_pred_ = np.array(impossible_pred_)
    entropy_ = np.array(entropy_)
    para_type_ = np.array(para_type_)
    exec_real_modified_ = np.array([res if impossible_real_[idx]==False else 'nan' for idx, res in enumerate(exec_real_)])
    exec_pred_modified_ = np.array([res if impossible_pred_[idx]==False else 'nan' for idx, res in enumerate(exec_pred_)])
    query_real_modified_ = np.array([sql if impossible_real_[idx]==False else 'nan' for idx, sql in enumerate(query_real_)])
    query_pred_modified_ = np.array([sql if impossible_pred_[idx]==False else 'nan' for idx, sql in enumerate(query_pred_)])

    AccEX_ = (exec_real_modified_==exec_pred_modified_)
    AccESM_ = np.array([esm_score(gt=real, pred=pred) for real, pred in zip(query_real_modified_, query_pred_modified_)])

    AccEX_ans_ = np.array(AccEX_)[impossible_pred_==False]
    AccESM_ans_ = np.array(AccESM_)[impossible_pred_==False]
    para_type_ans_ = para_type_[impossible_pred_==False]

    num_ans_real = sum(impossible_real_==False)
    num_unans_real = sum(impossible_real_==True)
    num_ans_pred = sum(impossible_pred_==False)
    num_unans_pred = sum(impossible_pred_==True)

    AccEX = np.mean(AccEX_)
    AccEX_human = np.mean(AccEX_[para_type_=='human'])
    AccEX_machine = np.mean(AccEX_[para_type_=='machine'])

    AccESM = np.mean(AccESM_)
    AccESM_human = np.mean(AccESM_[para_type_=='human'])
    AccESM_machine = np.mean(AccESM_[para_type_=='machine'])

    # erroneous
    exec_erroneous = exec_pred_[(impossible_real_==True) & (impossible_real_!=impossible_pred_)]
    question_erroneous = question_[(impossible_real_==True) & (impossible_real_!=impossible_pred_)]
    query_pred_erroneous = query_pred_[(impossible_real_==True) & (impossible_real_!=impossible_pred_)]

    # skipped
    exec_skipped = exec_pred_[(impossible_real_==False) & (impossible_real_!=impossible_pred_)]


    acc = np.mean(impossible_real_==impossible_pred_)*100
    acc_human = np.mean(impossible_real_[para_type_=='human']==impossible_pred_[para_type_=='human'])*100
    acc_machine = np.mean(impossible_real_[para_type_=='machine']==impossible_pred_[para_type_=='machine'])*100

    recall = recall_score(impossible_real_, impossible_pred_)*100
    recall_human = recall_score(impossible_real_[para_type_=='human'], impossible_pred_[para_type_=='human'])*100
    recall_machine = recall_score(impossible_real_[para_type_=='machine'], impossible_pred_[para_type_=='machine'])*100

    precision = precision_score(impossible_real_, impossible_pred_, zero_division=0)*100
    precision_human = precision_score(impossible_real_[para_type_=='human'], impossible_pred_[para_type_=='human'], zero_division=0)*100
    precision_machine = precision_score(impossible_real_[para_type_=='machine'], impossible_pred_[para_type_=='machine'], zero_division=0)*100

    f1 = f1_score(impossible_real_, impossible_pred_)*100
    f1_human = f1_score(impossible_real_[para_type_=='human'], impossible_pred_[para_type_=='human'])*100
    f1_machine = f1_score(impossible_real_[para_type_=='machine'], impossible_pred_[para_type_=='machine'])*100

    AccESM_ans = np.mean(AccESM_ans_)*100
    AccESM_ans_human = np.mean(AccESM_ans_[para_type_ans_=='human'])*100
    AccESM_ans_machine = np.mean(AccESM_ans_[para_type_ans_=='machine'])*100

    AccEX_ans = np.mean(AccEX_ans_)*100
    AccEX_ans_human = np.mean(AccEX_ans_[para_type_ans_=='human'])*100
    AccEX_ans_machine = np.mean(AccEX_ans_[para_type_ans_=='machine'])*100

    log =  f"--------------------------------All Queries--------------------------------"
    log += f"\neval_size: {num_ans_real+num_unans_real} (all), {num_ans_real} (ans), {num_unans_real} (unans) "    
    log += f"\nentropy  : >{entropy_threshold:.4f} "
    log += f"\npred ood : {num_ans_pred} (ans), {num_unans_pred} (unans) "
    log += f"\nESM              : {AccESM:.3f} (human: {AccESM_human:.3f}, machine: {AccESM_machine:.3f}) "
    log += f"\nEX               : {AccEX:.3f} (human: {AccEX_human:.3f}, machine: {AccEX_machine:.3f}) "    
    log += f"\nFNR = FN/(TP+FN) : {len(exec_erroneous)/(len(exec_erroneous) + sum((impossible_real_==True) & (impossible_pred_==True))):.3f} "
    log += f"\n|real:ans   & pred:unans (FP) : {len(exec_skipped)} (query_skipped_wrong) " # (impossible_real_==False) & (impossible_pred_==True)
    log += f"\n|real:ans   & pred:ans   (TN) : {sum((impossible_real_==False) & (impossible_pred_==False))} (query_skipped_ok) "
    log += f"\n|real:unans & pred:unans (TP) : {sum((impossible_real_==True) & (impossible_pred_==True))} (query_executed_ok) "
    log += f"\n|real:unans & pred:ans   (FN) : {len(exec_erroneous)} (query_executed_wrong) " # (impossible_real_==True) & (impossible_pred_==False)
    log += f"\n-------------------------Unanswerability Detection-------------------------"
    log += f"\nacc      : {acc:.2f} (human: {acc_human:.2f}, machine: {acc_machine:.2f}) "    
    log += f"\nrecall   : {recall:.2f} (human: {recall_human:.2f}, machine: {recall_machine:.2f}) "
    log += f"\nprecision: {precision:.2f} (human: {precision_human:.2f}, machine: {precision_machine:.2f}) "
    log += f"\nf1       : {f1:.2f} (human: {f1_human:.2f}, machine: {f1_machine:.2f})"
    log += f"\n-----------------------------Answered Queries------------------------------"
    log += f"\nacc_lf   : {AccESM_ans:.2f} (human: {AccESM_ans_human:.2f}, machine: {AccESM_ans_machine:.2f}) "
    log += f"\nacc_ex   : {AccEX_ans:.2f} (human: {AccEX_ans_human:.2f}, machine: {AccEX_ans_machine:.2f}) "
    log += f"\n---------------------------------------------------------------------------"
    print(log)

