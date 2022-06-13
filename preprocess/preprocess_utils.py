import os
import sys
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta


class Sampler:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def condition_value_shuffler(self, table, target_cols):
        original_idx = np.arange(len(table))
        rng_val = np.random.default_rng(0)
        shuffled_idx = rng_val.choice(original_idx, len(original_idx), replace=False).tolist()
        table.iloc[original_idx, [table.columns.get_loc(col) for col in target_cols]] = table[target_cols].values[shuffled_idx]
        return table

    def first_admit_year_sampler(self, start_year, span_year, earliest_year=None):
        end_year = start_year + span_year
        prob = np.array(range(1,span_year+2))/(sum(np.array(range(1, span_year+2))))
        sampled_year = self.rng.choice(range(start_year,end_year+1), p=prob)
        if earliest_year is not None:
            year_adjustment = int(sampled_year - earliest_year)
            return year_adjustment*365*24*60 # in minute
        else:
            return sampled_year

    def sample_date_given_year(self, year, num_split=1, frmt = '%Y-%m-%d'):
        start_time = time.mktime(time.strptime(f'{year}-01-01', frmt))
        end_time = time.mktime(time.strptime(f'{year}-12-31', frmt))    
        dts = []
        for split in range(num_split):
            split_seed = split/num_split + self.rng.random()/num_split
            ptime = start_time + split_seed * (end_time - start_time)
            dt = datetime.fromtimestamp(time.mktime(time.localtime(ptime)))
            dts.append(dt.strftime("%Y-%m-%d"))
        return dts


def adjust_time(table, time_col, patient_col, current_time=None, offset_dict=None):

    shifted_time = []
    for idx, time_val in enumerate(table[time_col].values):
        if pd.notnull(time_val) and time_val!='':
            if offset_dict is not None:
                id_ = table[patient_col].iloc[idx]
                if id_ in offset_dict:
                    if type(time_val)==str: # mimic3
                        time_val = str(datetime.strptime(time_val, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=int(offset_dict[id_])))
                    else: # eicu
                        time_val = str(datetime.strptime(offset_dict[id_], '%Y-%m-%d %H:%M:%S') + timedelta(minutes=int(time_val)))
                else:
                    time_val = None
            if time_val is not None and current_time is not None and current_time < time_val:
                time_val = None
        else:
            time_val = None
        shifted_time.append(time_val)

    return shifted_time



def read_csv(data_dir, filename, columns=None, lower=True, filter=None, dtype=None, memory_efficient=False):
    filepath = os.path.join(data_dir, filename)
    if memory_efficient:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar
        ProgressBar().register()

        if filepath.endswith('gz'):
            compression='gzip'
        else:
            compression=None

        if dtype:
            df = dd.read_csv(filepath, blocksize=25e6, dtype=dtype, compression=compression)
        else:
            df = dd.read_csv(filepath, blocksize=25e6, compression=compression)
        if columns is not None:
            df = df[columns]
        if filter is not None:
            for key in filter:
                df = df[df[key].isin(filter[key])]
        df = df.compute()
    else:
        df = pd.read_csv(filepath, usecols=columns)
        if filter is not None:
            for key in filter:
                df = df[df[key].isin(filter[key])]
    if lower:
        df = df.applymap(lambda x: x.lower().strip() if pd.notnull(x) and type(x)==str else x)
    return df


