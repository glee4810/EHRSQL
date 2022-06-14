import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import sqlite3

import warnings
warnings.filterwarnings("ignore")

from preprocess_utils import Sampler, adjust_time, read_csv


class Build_MIMIC_III(Sampler):
    def __init__(self, data_dir, out_dir, db_name,
                num_patient, sample_icu_patient_only,
                deid=False,
                timeshift=False,
                cur_patient_ratio=0.0,
                start_year=None, time_span=None,
                current_time = None,
                ):

        super().__init__()

        self.data_dir = data_dir
        self.out_dir = os.path.join(out_dir, db_name)
 
        self.deid = deid
        self.timeshift = timeshift

        self.sample_icu_patient_only = sample_icu_patient_only
        self.num_patient = num_patient
        self.num_cur_patient = int(self.num_patient * cur_patient_ratio)
        self.num_not_cur_patient = self.num_patient - int(self.num_patient * cur_patient_ratio)

        if timeshift:
            self.start_year = start_year
            self.start_pivot_datetime = datetime(year=self.start_year, month=1, day=1)
            self.time_span = time_span
            self.current_time = current_time

        self.conn = sqlite3.connect(os.path.join(self.out_dir, db_name+'.db'))
        self.cur = self.conn.cursor()
        with open(os.path.join(self.out_dir, db_name+'.sqlite'), 'r') as sql_file:
            sql_script = sql_file.read()
        self.cur.executescript(sql_script)

        self.chartevent2itemid = {
            'Temperature C (calc)'.lower(): '677', # body temperature
            'SaO2'.lower(): '834', # Sao2
            'heart rate'.lower(): '211', # heart rate
            'Respiratory Rate'.lower(): '618', # respiration rate
            'Arterial BP [Systolic]'.lower(): '51', # systolic blood pressure
            'Arterial BP [Diastolic]'.lower(): '8368', # diastolic blood pressure
            'Arterial BP Mean'.lower(): '52', # mean blood pressure

            'Admit Wt'.lower(): '762', # weight
            'Admit Ht'.lower(): '920' # height
        }


    def build_admission_table(self):

        print('Processing PATIENTS, ADMISSIONS, ICUSTAYS, TRANSFERS')
        start_time = time.time()

        # read patients
        PATIENTS_table = read_csv(self.data_dir, 'PATIENTS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'GENDER', 'DOB', 'DOD'], lower=True)
        subjectid2dob = {pid: dob for pid, dob in zip(PATIENTS_table['SUBJECT_ID'].values, PATIENTS_table['DOB'].values)}

        # read admissions
        ADMISSIONS_table = read_csv(self.data_dir, 'ADMISSIONS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 
                                                                              'DISCHTIME', 'ADMISSION_TYPE', 
                                                                              'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 
                                                                              'LANGUAGE', 'MARITAL_STATUS', 'ETHNICITY'], lower=True)
        ADMISSIONS_table['AGE'] = [int((datetime.strptime(admtime, '%Y-%m-%d %H:%M:%S') - datetime.strptime(subjectid2dob[pid], '%Y-%m-%d %H:%M:%S')).days/365.25) for pid, admtime in zip(ADMISSIONS_table['SUBJECT_ID'].values, ADMISSIONS_table['ADMITTIME'].values)]

        # remove age outliers
        ADMISSIONS_table = ADMISSIONS_table[(ADMISSIONS_table['AGE']>10) & (ADMISSIONS_table['AGE']<90)]

        # remove hospital stay outlier
        hosp_stay_dict = {hosp: (datetime.strptime(dischtime, '%Y-%m-%d %H:%M:%S') - datetime.strptime(admtime, '%Y-%m-%d %H:%M:%S')).days for hosp, admtime, dischtime in zip(ADMISSIONS_table['HADM_ID'].values, ADMISSIONS_table['ADMITTIME'].values, ADMISSIONS_table['DISCHTIME'].values)}
        threshold_offset = np.percentile(list(hosp_stay_dict.values()), q=95) # remove greater than 95% ≈ 28 days
        ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table['HADM_ID'].isin([hosp for hosp in hosp_stay_dict if hosp_stay_dict[hosp] < threshold_offset])]

        # save original admittime
        self.HADM_ID2admtime_dict = {hadm: admtime for hadm, admtime in zip(ADMISSIONS_table['HADM_ID'].values, ADMISSIONS_table['ADMITTIME'].values)}
        self.HADM_ID2dischtime_dict = {hadm: dischtime for hadm, dischtime in zip(ADMISSIONS_table['HADM_ID'].values, ADMISSIONS_table['DISCHTIME'].values)}

        # get earlist admission time
        ADMITTIME_earliest = {subj_id: min(ADMISSIONS_table['ADMITTIME'][ADMISSIONS_table['SUBJECT_ID']==subj_id].values) for subj_id in ADMISSIONS_table['SUBJECT_ID'].unique()}
        if self.timeshift:
            self.subjectid2admittime_dict = {subj_id: self.first_admit_year_sampler(self.start_year, self.time_span, datetime.strptime(ADMITTIME_earliest[subj_id], '%Y-%m-%d %H:%M:%S').year) for subj_id in ADMISSIONS_table['SUBJECT_ID'].unique()}            
    

        # read icustays
        ICUSTAYS_table = read_csv(self.data_dir, 'ICUSTAYS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
                                                                          'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID',
                                                                          'INTIME', 'OUTTIME'], lower=True)
        # subset only icu patients
        if self.sample_icu_patient_only:
            ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table['SUBJECT_ID'].isin(set(ICUSTAYS_table['SUBJECT_ID']))]

        # read transfer
        TRANSFERS_table = read_csv(self.data_dir, 'TRANSFERS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'EVENTTYPE', 'CURR_CAREUNIT', 'CURR_WARDID', 'INTIME', 'OUTTIME'], lower=True)
        TRANSFERS_table = TRANSFERS_table.rename(columns={"CURR_CAREUNIT": "CAREUNIT", "CURR_WARDID": "WARDID"})
        TRANSFERS_table = TRANSFERS_table.dropna(subset=['INTIME'])


        # process patients
        if self.timeshift:
            PATIENTS_table['DOB'] = adjust_time(PATIENTS_table, 'DOB', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            PATIENTS_table['DOD'] = adjust_time(PATIENTS_table, 'DOD', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            PATIENTS_table = PATIENTS_table.dropna(subset=['DOB'])

        # process admissions
        if self.timeshift:
            ADMISSIONS_table['ADMITTIME'] = adjust_time(ADMISSIONS_table, 'ADMITTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            ADMISSIONS_table['DISCHTIME'] = adjust_time(ADMISSIONS_table, 'DISCHTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            ADMISSIONS_table = ADMISSIONS_table.dropna(subset=['ADMITTIME'])

        # process icustays
        if self.timeshift:
            ICUSTAYS_table['INTIME'] = adjust_time(ICUSTAYS_table, 'INTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            ICUSTAYS_table['OUTTIME'] = adjust_time(ICUSTAYS_table, 'OUTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            ICUSTAYS_table = ICUSTAYS_table.dropna(subset=['INTIME'])

        # process transfers
        if self.timeshift:
            TRANSFERS_table['INTIME'] = adjust_time(TRANSFERS_table, 'INTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            TRANSFERS_table['OUTTIME'] = adjust_time(TRANSFERS_table, 'OUTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            TRANSFERS_table = TRANSFERS_table.dropna(subset=['INTIME'])
        
        if self.timeshift:
            num_valid_cur_patient = len(ADMISSIONS_table['SUBJECT_ID'][ADMISSIONS_table['DISCHTIME'].isnull()].unique())
            if self.num_cur_patient > num_valid_cur_patient:
                warnings.warn(f"{self.num_cur_patient}>{num_valid_cur_patient}")
            self.cur_patient_list = self.rng.choice(ADMISSIONS_table['SUBJECT_ID'][ADMISSIONS_table['DISCHTIME'].isnull()].unique(), self.num_cur_patient, replace=False).tolist()
        else:
            self.cur_patient_list = []
        self.not_cur_patient = self.rng.choice(ADMISSIONS_table['SUBJECT_ID'][(ADMISSIONS_table['DISCHTIME'].notnull()) & (~ADMISSIONS_table['SUBJECT_ID'].isin(self.cur_patient_list))].unique(), self.num_not_cur_patient, replace=False).tolist()
        self.patient_list = self.cur_patient_list + self.not_cur_patient
        
        PATIENTS_table = PATIENTS_table[PATIENTS_table['SUBJECT_ID'].isin(self.patient_list)]
        ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table['SUBJECT_ID'].isin(self.patient_list)]

        self.hadm_list = list(set(ADMISSIONS_table['HADM_ID']))
        ICUSTAYS_table = ICUSTAYS_table[ICUSTAYS_table['HADM_ID'].isin(self.hadm_list)]
        TRANSFERS_table = TRANSFERS_table[TRANSFERS_table['HADM_ID'].isin(self.hadm_list)]

        if self.deid:
            icu2careunit = {}
            icu2wardid = {}
            random_indices = self.rng.choice(len(ICUSTAYS_table), len(ICUSTAYS_table), replace=False).tolist()
            for idx, icu in enumerate(ICUSTAYS_table['ICUSTAY_ID']):
                icu2careunit[icu] = {}
                icu2careunit[icu][ICUSTAYS_table['FIRST_CAREUNIT'][ICUSTAYS_table['ICUSTAY_ID']==icu].values[0]] = ICUSTAYS_table['FIRST_CAREUNIT'].iloc[random_indices[idx]]
                icu2careunit[icu][ICUSTAYS_table['LAST_CAREUNIT'][ICUSTAYS_table['ICUSTAY_ID']==icu].values[0]] = ICUSTAYS_table['LAST_CAREUNIT'].iloc[random_indices[idx]]
                ICUSTAYS_table['FIRST_CAREUNIT'][ICUSTAYS_table['ICUSTAY_ID']==icu] = ICUSTAYS_table['FIRST_CAREUNIT'].iloc[random_indices[idx]]
                ICUSTAYS_table['LAST_CAREUNIT'][ICUSTAYS_table['ICUSTAY_ID']==icu] = ICUSTAYS_table['LAST_CAREUNIT'].iloc[random_indices[idx]]
                icu2wardid[icu] = {}
                icu2wardid[icu][ICUSTAYS_table['FIRST_WARDID'][ICUSTAYS_table['ICUSTAY_ID']==icu].values[0]] = ICUSTAYS_table['FIRST_WARDID'].iloc[random_indices[idx]]
                icu2wardid[icu][ICUSTAYS_table['LAST_WARDID'][ICUSTAYS_table['ICUSTAY_ID']==icu].values[0]] = ICUSTAYS_table['LAST_WARDID'].iloc[random_indices[idx]]
                ICUSTAYS_table['FIRST_WARDID'][ICUSTAYS_table['ICUSTAY_ID']==icu] = ICUSTAYS_table['FIRST_WARDID'].iloc[random_indices[idx]]
                ICUSTAYS_table['LAST_WARDID'][ICUSTAYS_table['ICUSTAY_ID']==icu] = ICUSTAYS_table['LAST_WARDID'].iloc[random_indices[idx]]

            for icu in ICUSTAYS_table['ICUSTAY_ID']:
                TRANSFERS_table['CAREUNIT'][TRANSFERS_table['ICUSTAY_ID']==icu] = TRANSFERS_table['CAREUNIT'][TRANSFERS_table['ICUSTAY_ID']==icu].replace(icu2careunit[icu])
                TRANSFERS_table['WARDID'][TRANSFERS_table['ICUSTAY_ID']==icu] = TRANSFERS_table['WARDID'][TRANSFERS_table['ICUSTAY_ID']==icu].replace(icu2wardid[icu])

        PATIENTS_table['ROW_ID'] = range(len(PATIENTS_table))
        ADMISSIONS_table['ROW_ID'] = range(len(ADMISSIONS_table))
        ICUSTAYS_table['ROW_ID'] = range(len(ICUSTAYS_table))
        TRANSFERS_table['ROW_ID'] = range(len(TRANSFERS_table))

        PATIENTS_table.to_csv(os.path.join(self.out_dir, 'PATIENTS.csv'), index=False)
        ADMISSIONS_table.to_csv(os.path.join(self.out_dir, 'ADMISSIONS.csv'), index=False)
        ICUSTAYS_table.to_csv(os.path.join(self.out_dir, 'ICUSTAYS.csv'), index=False)
        TRANSFERS_table.to_csv(os.path.join(self.out_dir, 'TRANSFERS.csv'), index=False)

        print(f'PATIENTS, ADMISSIONS, ICUSTAYS, TRANSFERS processed (took {round(time.time() - start_time, 4)} secs)')



    def build_dictionary_table(self):

        print('Processing D_ICD_DIAGNOSES, D_ICD_PROCEDURES, D_LABITEMS, D_ITEMS')
        start_time = time.time()

        D_ICD_DIAGNOSES_table = read_csv(self.data_dir, 'D_ICD_DIAGNOSES.csv', columns=['ROW_ID', 'ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE'], lower=True)
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.astype({'ICD9_CODE': str})
        self.D_ICD_DIAGNOSES_dict = {item: val for item, val in zip(D_ICD_DIAGNOSES_table['ICD9_CODE'].values, D_ICD_DIAGNOSES_table['SHORT_TITLE'].values)}

        D_ICD_PROCEDURES_table = read_csv(self.data_dir, 'D_ICD_PROCEDURES.csv', columns=['ROW_ID', 'ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE'], lower=True)
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.astype({'ICD9_CODE': str}).drop_duplicates(subset=['ICD9_CODE'])
        self.D_ICD_PROCEDURES_dict = {item: val for item, val in zip(D_ICD_PROCEDURES_table['ICD9_CODE'].values, D_ICD_PROCEDURES_table['SHORT_TITLE'].values)}

        D_LABITEMS_table = read_csv(self.data_dir, 'D_LABITEMS.csv', columns=['ROW_ID', 'ITEMID', 'LABEL'], lower=True)
        self.D_LABITEMS_dict = {item: val for item, val in zip(D_LABITEMS_table['ITEMID'].values, D_LABITEMS_table['LABEL'].values)}

        D_ITEMS_table = read_csv(self.data_dir, 'D_ITEMS.csv', columns=['ROW_ID', 'ITEMID', 'LABEL', 'LINKSTO'], lower=True)
        D_ITEMS_table = D_ITEMS_table.dropna(subset=['LABEL'])
        D_ITEMS_table = D_ITEMS_table[D_ITEMS_table['LINKSTO'].isin(['inputevents_cv', 'outputevents', 'chartevents'])]
        self.D_ITEMS_dict = {item: val for item, val in zip(D_ITEMS_table['ITEMID'].values, D_ITEMS_table['LABEL'].values)}

        D_ICD_DIAGNOSES_table['ROW_ID'] = range(len(D_ICD_DIAGNOSES_table))
        D_ICD_PROCEDURES_table['ROW_ID'] = range(len(D_ICD_PROCEDURES_table))
        D_LABITEMS_table['ROW_ID'] = range(len(D_LABITEMS_table))
        D_ITEMS_table['ROW_ID'] = range(len(D_ITEMS_table))

        D_ICD_DIAGNOSES_table.to_csv(os.path.join(self.out_dir, 'D_ICD_DIAGNOSES.csv'), index=False)
        D_ICD_PROCEDURES_table.to_csv(os.path.join(self.out_dir, 'D_ICD_PROCEDURES.csv'), index=False)
        D_LABITEMS_table.to_csv(os.path.join(self.out_dir, 'D_LABITEMS.csv'), index=False)
        D_ITEMS_table.to_csv(os.path.join(self.out_dir, 'D_ITEMS.csv'), index=False)

        print(f'D_ICD_DIAGNOSES, D_ICD_PROCEDURES, D_LABITEMS, D_ITEMS processed (took {round(time.time() - start_time, 4)} secs)')



    def build_diagnosis_table(self):

        print('Processing DIAGNOSES_ICD table')
        start_time = time.time()

        DIAGNOSES_ICD_table = read_csv(self.data_dir, 'DIAGNOSES_ICD.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'], lower=True)
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.astype({'ICD9_CODE': str})
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.dropna(subset=['ICD9_CODE'])
        DIAGNOSES_ICD_table['CHARTTIME'] = [self.HADM_ID2admtime_dict[hadm] if hadm in self.HADM_ID2admtime_dict else None for hadm in DIAGNOSES_ICD_table['HADM_ID'].values] # assume charttime is at the hospital admission

        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table[DIAGNOSES_ICD_table['ICD9_CODE'].isin(self.D_ICD_DIAGNOSES_dict)]
        if self.deid:
            DIAGNOSES_ICD_table = self.condition_value_shuffler(DIAGNOSES_ICD_table, target_cols=['ICD9_CODE'])
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table[DIAGNOSES_ICD_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            DIAGNOSES_ICD_table['CHARTTIME'] = adjust_time(DIAGNOSES_ICD_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.dropna(subset=['CHARTTIME'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in DIAGNOSES_ICD_table['CHARTTIME'].values])
            DIAGNOSES_ICD_table = DIAGNOSES_ICD_table[TIME >= self.start_pivot_datetime]

        DIAGNOSES_ICD_table['ROW_ID'] = range(len(DIAGNOSES_ICD_table))
        DIAGNOSES_ICD_table.to_csv(os.path.join(self.out_dir, 'DIAGNOSES_ICD.csv'), index=False)

        print(f'DIAGNOSES_ICD processed (took {round(time.time() - start_time, 4)} secs)')



    def build_procedure_table(self):

        print('Processing PROCEDURES_ICD table')
        start_time = time.time()

        PROCEDURES_ICD_table = read_csv(self.data_dir, 'PROCEDURES_ICD.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'], lower=True)
        PROCEDURES_ICD_table = PROCEDURES_ICD_table.astype({'ICD9_CODE': str})
        PROCEDURES_ICD_table['CHARTTIME'] = [self.HADM_ID2dischtime_dict[hadm] if hadm in self.HADM_ID2dischtime_dict else None for hadm in PROCEDURES_ICD_table['HADM_ID'].values] # assume charttime is at the hospital discharge

        PROCEDURES_ICD_table = PROCEDURES_ICD_table[PROCEDURES_ICD_table['ICD9_CODE'].isin(self.D_ICD_PROCEDURES_dict)]
        if self.deid:
            PROCEDURES_ICD_table = self.condition_value_shuffler(PROCEDURES_ICD_table, target_cols=['ICD9_CODE'])
        PROCEDURES_ICD_table = PROCEDURES_ICD_table[PROCEDURES_ICD_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            PROCEDURES_ICD_table['CHARTTIME'] = adjust_time(PROCEDURES_ICD_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            PROCEDURES_ICD_table = PROCEDURES_ICD_table.dropna(subset=['CHARTTIME'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in PROCEDURES_ICD_table['CHARTTIME'].values])
            PROCEDURES_ICD_table = PROCEDURES_ICD_table[TIME >= self.start_pivot_datetime]

        PROCEDURES_ICD_table['ROW_ID'] = range(len(PROCEDURES_ICD_table))
        PROCEDURES_ICD_table.to_csv(os.path.join(self.out_dir, 'PROCEDURES_ICD.csv'), index=False)

        print(f'PROCEDURES_ICD processed (took {round(time.time() - start_time, 4)} secs)')



    def build_labevent_table(self):

        print('Processing LABEVENTS table')
        start_time = time.time()

        LABEVENTS_table = read_csv(self.data_dir, 'LABEVENTS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM'], lower=True)
        LABEVENTS_table = LABEVENTS_table.dropna(subset=['HADM_ID', 'VALUENUM', 'VALUEUOM'])

        LABEVENTS_table = LABEVENTS_table[LABEVENTS_table['ITEMID'].isin(self.D_LABITEMS_dict)]
        if self.deid:
            LABEVENTS_table = self.condition_value_shuffler(LABEVENTS_table, target_cols=['ITEMID', 'VALUENUM', 'VALUEUOM'])
        LABEVENTS_table = LABEVENTS_table[LABEVENTS_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:    
            LABEVENTS_table['CHARTTIME'] = adjust_time(LABEVENTS_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            LABEVENTS_table = LABEVENTS_table.dropna(subset=['CHARTTIME'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in LABEVENTS_table['CHARTTIME'].values])
            LABEVENTS_table = LABEVENTS_table[TIME >= self.start_pivot_datetime]

        LABEVENTS_table['ROW_ID'] = range(len(LABEVENTS_table))
        LABEVENTS_table.to_csv(os.path.join(self.out_dir, 'LABEVENTS.csv'), index=False)

        print(f'LABEVENTS processed (took {round(time.time() - start_time, 4)} secs)')



    def build_prescriptions_table(self):

        print('Processing PRESCRIPTIONS table')
        start_time = time.time()

        PRESCRIPTIONS_table = read_csv(self.data_dir, 'PRESCRIPTIONS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'STARTDATE', 'ENDDATE', 'DRUG', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'ROUTE'], lower=True)
        PRESCRIPTIONS_table = PRESCRIPTIONS_table.dropna(subset=['STARTDATE', 'ENDDATE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'ROUTE'])
        PRESCRIPTIONS_table['DOSE_VAL_RX'] = [int(str(v).replace(',', '')) if str(v).replace(',', '').isnumeric() else None for v in PRESCRIPTIONS_table['DOSE_VAL_RX'].values]
        PRESCRIPTIONS_table = PRESCRIPTIONS_table.dropna(subset=['DOSE_VAL_RX']) # remove not int elements

        drug2unit_dict = {}
        for item, unit in zip(PRESCRIPTIONS_table['DRUG'].values, PRESCRIPTIONS_table['DOSE_UNIT_RX'].values):
            if item in drug2unit_dict:
                drug2unit_dict[item].append(unit)
            else:
                drug2unit_dict[item] = [unit]
        drug_name2unit_dict = {item: Counter(drug2unit_dict[item]).most_common(1)[0][0] for item in drug2unit_dict} # pick only the most frequent unit of measure

        PRESCRIPTIONS_table = PRESCRIPTIONS_table[PRESCRIPTIONS_table['DRUG'].isin(drug2unit_dict)]
        PRESCRIPTIONS_table = PRESCRIPTIONS_table[PRESCRIPTIONS_table['DOSE_UNIT_RX']==[drug_name2unit_dict[drug] for drug in PRESCRIPTIONS_table['DRUG']]]
        if self.deid:
            PRESCRIPTIONS_table = self.condition_value_shuffler(PRESCRIPTIONS_table, target_cols=['DRUG', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'ROUTE'])
        PRESCRIPTIONS_table = PRESCRIPTIONS_table[PRESCRIPTIONS_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            PRESCRIPTIONS_table['STARTDATE'] = adjust_time(PRESCRIPTIONS_table, 'STARTDATE', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            PRESCRIPTIONS_table['ENDDATE'] = adjust_time(PRESCRIPTIONS_table, 'ENDDATE', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            PRESCRIPTIONS_table = PRESCRIPTIONS_table.dropna(subset=['STARTDATE'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in PRESCRIPTIONS_table['STARTDATE'].values])
            PRESCRIPTIONS_table = PRESCRIPTIONS_table[TIME >= self.start_pivot_datetime]

        PRESCRIPTIONS_table['ROW_ID'] = range(len(PRESCRIPTIONS_table))
        PRESCRIPTIONS_table.to_csv(os.path.join(self.out_dir, 'PRESCRIPTIONS.csv'), index=False)

        print(f'PRESCRIPTIONS processed (took {round(time.time() - start_time, 4)} secs)')



    def build_cost_table(self):

        print('Processing COST table')
        start_time = time.time()

        DIAGNOSES_ICD_table = read_csv(self.out_dir, 'DIAGNOSES_ICD.csv').astype({'ICD9_CODE': str})
        LABEVENTS_table = read_csv(self.out_dir, 'LABEVENTS.csv')
        PROCEDURES_ICD_table = read_csv(self.out_dir, 'PROCEDURES_ICD.csv').astype({'ICD9_CODE': str})
        PRESCRIPTIONS_table = read_csv(self.out_dir, 'PRESCRIPTIONS.csv')


        cnt = 0
        data_filter = []
        mean_costs = self.rng.poisson(lam=10, size=4)

        cost_id = cnt+np.arange(len(DIAGNOSES_ICD_table))
        person_id = DIAGNOSES_ICD_table['SUBJECT_ID'].values
        hospitaladmit_id = DIAGNOSES_ICD_table['HADM_ID'].values
        cost_event_table_concept_id = DIAGNOSES_ICD_table['ROW_ID'].values
        charge_time = DIAGNOSES_ICD_table['CHARTTIME'].values
        diagnosis_cost_dict = {item: round(self.rng.normal(loc=mean_costs[0], scale=1.0),2) for item in sorted(DIAGNOSES_ICD_table['ICD9_CODE'].unique())}
        cost = [diagnosis_cost_dict[item] for item in DIAGNOSES_ICD_table['ICD9_CODE'].values]
        temp = pd.DataFrame(data={'ROW_ID': cost_id,
                                'SUBJECT_ID': person_id, 
                                'HADM_ID': hospitaladmit_id,
                                'EVENT_TYPE': 'DIAGNOSES_ICD'.lower(),
                                'EVENT_ID': cost_event_table_concept_id,
                                'CHARGETIME': charge_time,
                                'COST': cost})
        cnt += len(DIAGNOSES_ICD_table)
        data_filter.append(temp)

        cost_id = cnt+np.arange(len(LABEVENTS_table))
        person_id = LABEVENTS_table['SUBJECT_ID'].values
        hospitaladmit_id = LABEVENTS_table['HADM_ID'].values
        cost_event_table_concept_id = LABEVENTS_table['ROW_ID'].values
        charge_time = LABEVENTS_table['CHARTTIME'].values
        lab_cost_dict = {item: round(self.rng.normal(loc=mean_costs[1], scale=1.0),2) for item in sorted(LABEVENTS_table['ITEMID'].unique())}
        cost = [lab_cost_dict[item] for item in LABEVENTS_table['ITEMID'].values]
        temp = pd.DataFrame(data={'ROW_ID': cost_id, 
                                'SUBJECT_ID': person_id, 
                                'HADM_ID': hospitaladmit_id, 
                                'EVENT_TYPE': 'LABEVENTS'.lower(),
                                'EVENT_ID': cost_event_table_concept_id,
                                'CHARGETIME': charge_time,
                                'COST': cost})
        cnt += len(LABEVENTS_table)
        data_filter.append(temp)

        cost_id = cnt+np.arange(len(PROCEDURES_ICD_table))
        person_id = PROCEDURES_ICD_table['SUBJECT_ID'].values
        hospitaladmit_id = PROCEDURES_ICD_table['HADM_ID'].values
        cost_event_table_concept_id = PROCEDURES_ICD_table['ROW_ID'].values
        charge_time = PROCEDURES_ICD_table['CHARTTIME'].values
        procedure_cost_dict = {item: round(self.rng.normal(loc=mean_costs[2], scale=1.0),2) for item in sorted(PROCEDURES_ICD_table['ICD9_CODE'].unique())}
        cost = [procedure_cost_dict[item] for item in PROCEDURES_ICD_table['ICD9_CODE'].values]        
        temp = pd.DataFrame(data={'ROW_ID': cost_id, 
                                'SUBJECT_ID': person_id, 
                                'HADM_ID': hospitaladmit_id, 
                                'EVENT_TYPE': 'PROCEDURES_ICD'.lower(),
                                'EVENT_ID': cost_event_table_concept_id,
                                'CHARGETIME': charge_time,
                                'COST': cost})
        cnt += len(PROCEDURES_ICD_table)
        data_filter.append(temp)

        cost_id = cnt+np.arange(len(PRESCRIPTIONS_table))
        person_id = PRESCRIPTIONS_table['SUBJECT_ID'].values
        hospitaladmit_id = PRESCRIPTIONS_table['HADM_ID'].values
        cost_event_table_concept_id = PRESCRIPTIONS_table['ROW_ID'].values
        charge_time = PRESCRIPTIONS_table['STARTDATE'].values
        prescription_cost_dict = {item: round(self.rng.normal(loc=mean_costs[3], scale=1.0),2) for item in sorted(PRESCRIPTIONS_table['DRUG'].unique())}
        cost = [prescription_cost_dict[item] for item in PRESCRIPTIONS_table['DRUG'].values]        
        temp = pd.DataFrame(data={'ROW_ID': cost_id, 
                                'SUBJECT_ID': person_id, 
                                'HADM_ID': hospitaladmit_id, 
                                'EVENT_TYPE': 'PRESCRIPTIONS'.lower(),
                                'EVENT_ID': cost_event_table_concept_id,
                                'CHARGETIME': charge_time,
                                'COST': cost})
        cnt += len(PRESCRIPTIONS_table)
        data_filter.append(temp)


        COST_table = pd.concat(data_filter, ignore_index=True)
        COST_table.to_csv(os.path.join(self.out_dir, 'COST.csv'), index=False)
        print(f'COST processed (took {round(time.time() - start_time, 4)} secs)')



    def build_chartevent_table(self):

        print('Processing CHARTEVENTS table')
        start_time = time.time()

        CHARTEVENTS_table = read_csv(self.data_dir, 'CHARTEVENTS.csv', dtype={'SUBJECT_ID': int, 'ITEMID': str, 'ICUSTAY_ID': 'float64', 'RESULTSTATUS': 'object', 'CGID': 'float64', 'ERROR': 'float64', 'STOPPED': 'object', 'VALUE': 'object', 'WARNING': 'float64'},
                                                                       columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM'], lower=True,
                                                                       filter={'ITEMID': self.chartevent2itemid.values(), 'SUBJECT_ID': self.patient_list},                                                                       
                                                                       memory_efficient=True)
        CHARTEVENTS_table = CHARTEVENTS_table.dropna(subset=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM'])
        CHARTEVENTS_table = CHARTEVENTS_table.astype({'ROW_ID': int, 'SUBJECT_ID': int, 'HADM_ID': int, 'ICUSTAY_ID': int, 'ITEMID': int, 'CHARTTIME': str, 'VALUENUM': float, 'VALUEUOM': str})

        if self.timeshift: # changed order due to the large number of rows in CHARTEVENTS_table
            CHARTEVENTS_table['CHARTTIME'] = adjust_time(CHARTEVENTS_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            CHARTEVENTS_table = CHARTEVENTS_table.dropna(subset=['CHARTTIME'])

        if self.deid:
            CHARTEVENTS_table = self.condition_value_shuffler(CHARTEVENTS_table, target_cols=['ITEMID', 'VALUENUM', 'VALUEUOM'])
        CHARTEVENTS_table = CHARTEVENTS_table[CHARTEVENTS_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in CHARTEVENTS_table['CHARTTIME'].values])
            CHARTEVENTS_table = CHARTEVENTS_table[TIME >= self.start_pivot_datetime]

        CHARTEVENTS_table['ROW_ID'] = range(len(CHARTEVENTS_table))
        CHARTEVENTS_table.to_csv(os.path.join(self.out_dir, 'CHARTEVENTS.csv'), index=False)
        print(f'CHARTEVENTS processed (took {round(time.time() - start_time, 4)} secs)')



    def build_inputevent_table(self):

        print('Processing INPUTEVENTS_CV table')
        start_time = time.time()

        INPUTEVENTS_table = read_csv(self.data_dir, 'INPUTEVENTS_CV.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM'], lower=True)
        INPUTEVENTS_table = INPUTEVENTS_table.dropna(subset=['HADM_ID', 'ICUSTAY_ID', 'AMOUNT', 'AMOUNTUOM'])
        INPUTEVENTS_table = INPUTEVENTS_table[INPUTEVENTS_table['AMOUNTUOM']=='ml']
        del INPUTEVENTS_table['AMOUNTUOM']

        INPUTEVENTS_table = INPUTEVENTS_table[INPUTEVENTS_table['ITEMID'].isin(self.D_ITEMS_dict)]
        if self.deid:
            INPUTEVENTS_table = self.condition_value_shuffler(INPUTEVENTS_table, target_cols=['ITEMID', 'AMOUNT'])
        INPUTEVENTS_table = INPUTEVENTS_table[INPUTEVENTS_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            INPUTEVENTS_table['CHARTTIME'] = adjust_time(INPUTEVENTS_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            INPUTEVENTS_table = INPUTEVENTS_table.dropna(subset=['CHARTTIME'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in INPUTEVENTS_table['CHARTTIME'].values])
            INPUTEVENTS_table = INPUTEVENTS_table[TIME >= self.start_pivot_datetime]

        INPUTEVENTS_table['ROW_ID'] = range(len(INPUTEVENTS_table))
        INPUTEVENTS_table.to_csv(os.path.join(self.out_dir, 'INPUTEVENTS_CV.csv'), index=False)

        print(f'INPUTEVENTS_CV processed (took {round(time.time() - start_time, 4)} secs)')



    def build_outputevent_table(self):

        print('Processing OUTPUTEVENTS table')
        start_time = time.time()

        OUTPUTEVENTS_table = read_csv(self.data_dir, 'OUTPUTEVENTS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'], lower=True)
        OUTPUTEVENTS_table = OUTPUTEVENTS_table.dropna(subset=['HADM_ID', 'ICUSTAY_ID', 'VALUE', 'VALUEUOM'])
        OUTPUTEVENTS_table = OUTPUTEVENTS_table[OUTPUTEVENTS_table['VALUEUOM']=='ml']
        del OUTPUTEVENTS_table['VALUEUOM']

        OUTPUTEVENTS_table = OUTPUTEVENTS_table[OUTPUTEVENTS_table['ITEMID'].isin(self.D_ITEMS_dict)]
        if self.deid:
            OUTPUTEVENTS_table = self.condition_value_shuffler(OUTPUTEVENTS_table, target_cols=['ITEMID', 'VALUE'])
        OUTPUTEVENTS_table = OUTPUTEVENTS_table[OUTPUTEVENTS_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            OUTPUTEVENTS_table['CHARTTIME'] = adjust_time(OUTPUTEVENTS_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            OUTPUTEVENTS_table = OUTPUTEVENTS_table.dropna(subset=['CHARTTIME'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in OUTPUTEVENTS_table['CHARTTIME'].values])
            OUTPUTEVENTS_table = OUTPUTEVENTS_table[TIME >= self.start_pivot_datetime]

        OUTPUTEVENTS_table['ROW_ID'] = range(len(OUTPUTEVENTS_table))
        OUTPUTEVENTS_table.to_csv(os.path.join(self.out_dir, 'OUTPUTEVENTS.csv'), index=False)

        print(f'OUTPUTEVENTS processed (took {round(time.time() - start_time, 4)} secs)')



    def build_microbiology_table(self):

        print('Processing MICROBIOLOGYEVENTS table')
        start_time = time.time()

        MICROBIOLOGYEVENTS_table = read_csv(self.data_dir, 'MICROBIOLOGYEVENTS.csv', columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'SPEC_TYPE_DESC', 'ORG_NAME'], lower=True)
        MICROBIOLOGYEVENTS_table['CHARTTIME'] = MICROBIOLOGYEVENTS_table['CHARTTIME'].fillna(MICROBIOLOGYEVENTS_table['CHARTDATE'])
        MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table.drop(columns=['CHARTDATE'])
        if self.deid:
            MICROBIOLOGYEVENTS_table = self.condition_value_shuffler(MICROBIOLOGYEVENTS_table, target_cols=['SPEC_TYPE_DESC', 'ORG_NAME'])
        MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table[MICROBIOLOGYEVENTS_table['HADM_ID'].isin(self.hadm_list)]

        if self.timeshift:
            MICROBIOLOGYEVENTS_table['CHARTTIME'] = adjust_time(MICROBIOLOGYEVENTS_table, 'CHARTTIME', current_time=self.current_time, offset_dict=self.subjectid2admittime_dict, patient_col='SUBJECT_ID')
            MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table.dropna(subset=['CHARTTIME'])
            TIME = np.array([datetime.strptime(tt, '%Y-%m-%d %H:%M:%S') for tt in MICROBIOLOGYEVENTS_table['CHARTTIME'].values])
            MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table[TIME >= self.start_pivot_datetime]

        MICROBIOLOGYEVENTS_table['ROW_ID'] = range(len(MICROBIOLOGYEVENTS_table))
        MICROBIOLOGYEVENTS_table.to_csv(os.path.join(self.out_dir, 'MICROBIOLOGYEVENTS.csv'), index=False)

        print(f'MICROBIOLOGYEVENTS processed (took {round(time.time() - start_time, 4)} secs)')


    def generate_db(self):

        rows = read_csv(self.out_dir, 'PATIENTS.csv')
        rows.to_sql('PATIENTS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'ADMISSIONS.csv')
        rows.to_sql('ADMISSIONS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'D_ICD_DIAGNOSES.csv').astype({'ICD9_CODE': str})
        rows.to_sql('D_ICD_DIAGNOSES', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'D_ICD_PROCEDURES.csv').astype({'ICD9_CODE': str})
        rows.to_sql('D_ICD_PROCEDURES', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'D_ITEMS.csv')
        rows.to_sql('D_ITEMS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'D_LABITEMS.csv')
        rows.to_sql('D_LABITEMS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'DIAGNOSES_ICD.csv').astype({'ICD9_CODE': str})
        rows.to_sql('DIAGNOSES_ICD', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'PROCEDURES_ICD.csv').astype({'ICD9_CODE': str})
        rows.to_sql('PROCEDURES_ICD', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'LABEVENTS.csv')
        rows.to_sql('LABEVENTS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'PRESCRIPTIONS.csv')
        rows.to_sql('PRESCRIPTIONS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'COST.csv')
        rows.to_sql('COST', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'CHARTEVENTS.csv')
        rows.to_sql('CHARTEVENTS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'INPUTEVENTS_CV.csv')
        rows.to_sql('INPUTEVENTS_CV', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'OUTPUTEVENTS.csv')
        rows.to_sql('OUTPUTEVENTS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'MICROBIOLOGYEVENTS.csv')
        rows.to_sql('MICROBIOLOGYEVENTS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'ICUSTAYS.csv')
        rows.to_sql('ICUSTAYS', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'TRANSFERS.csv')
        rows.to_sql('TRANSFERS', self.conn, if_exists='append', index=False)

        query = "SELECT * FROM sqlite_master WHERE type='table'"
        print(pd.read_sql_query(query, self.conn)['name']) # 17 tables

