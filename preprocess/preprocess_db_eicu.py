import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from preprocess_utils import Sampler, adjust_time, read_csv


class Build_eICU(Sampler):
    def __init__(self, data_dir, out_dir, db_name,
                num_patient, 
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

        self.num_patient = num_patient        
        self.num_cur_patient = int(self.num_patient * cur_patient_ratio)
        self.num_not_cur_patient = self.num_patient - int(self.num_patient * cur_patient_ratio)

        if timeshift:
            self.start_year = start_year
            self.time_span = time_span
            self.current_time = current_time

        self.conn = sqlite3.connect(os.path.join(self.out_dir, db_name+'.db'))
        self.cur = self.conn.cursor()
        with open(os.path.join(self.out_dir, db_name+'.sqlite'), 'r') as sql_file:
            sql_script = sql_file.read()
        self.cur.executescript(sql_script)

        # self.chartevent_dict = {
        #     'body temperature'.lower(): 'vitalPeriodic.temperature'.lower(),
        #     'SaO2'.lower(): 'vitalPeriodic.sao2'.lower(),
        #     'heart rate'.lower(): 'vitalPeriodic.heartrate'.lower(), 
        #     'respiration rate'.lower(): 'vitalPeriodic.respiration'.lower(),
        #     'systolic blood pressure'.lower(): 'vitalPeriodic.systemicsystolic'.lower(),
        #     'diastolic blood pressure'.lower(): 'vitalPeriodic.systemicdiastolic'.lower(),
        #     'mean blood pressure'.lower(): 'vitalPeriodic.systemicmean'.lower(),

        #     'weight'.lower(): 'patient.admissionweight'.lower(),
        #     'height'.lower(): 'patient.admissionheight'.lower()
        # }


    def build_admission_table(self):

        print('Processing patient')
        start_time = time.time()

        # read patients
        patient_table = read_csv(self.data_dir, 'patient.csv', columns=['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'gender', 'age', 
                                                                        'ethnicity', 'hospitalid', 'wardid', 'admissionheight', 'admissionweight', 'dischargeweight', 'hospitaladmitsource', 'hospitaldischargestatus',
                                                                        'hospitaladmitoffset', 'hospitaladmittime24', 'unitdischargeoffset', 'hospitaldischargeoffset', 'hospitaldischargeyear',
                                                                        ], lower=True)

        patient_table = patient_table.dropna(subset=['hospitaladmitsource'])
        # hospital-admission: patienthealthsystemstayid, icu-admission: patientunitstayid

        # gender filtering  
        patient_table = patient_table[patient_table['gender'].isin(['male', 'female'])]

        # remove age outliers
        patient_table = patient_table.replace({'age': '> 89'}, {'age': None}) # '>89' NaN replace
        patient_table = patient_table.dropna(subset=['age']) # Remove NA in age
        patient_table = patient_table.astype({'age': int}) # Change age type to int
        patient_table = patient_table[patient_table['age'] > 10]

        # remove hospital stay outlier
        hosp_stay = -patient_table['hospitaladmitoffset'].values + patient_table['hospitaldischargeoffset'].values
        threshold_offset = np.percentile(hosp_stay, q=95) # remove greater than 95% ≈ 25 days
        patient_table = patient_table[hosp_stay < threshold_offset]

        # Time shifting
        # patient: hospitaladmittime, unitadmittime, unitdischargetime, hospitaldischargetime
        # diagnosis: diagnosistime
        # treatment: treatmenttime
        # lab: labresulttime
        # medication: drugstarttime, drugstoptime
        # cost: chargetime
        # allergy: allergytime
        # intakeoutput: intakeoutputtime
        # microlab: culturetakentime
        # vitalperiodic: observationtime

        print('Processing patient table')
        self.unitstayid2time_dict = {} # icu_id2icu_admittime
        self.unitstayid2hadmid_dict = {} # icu_id2hosp_id
        self.unitstayid2uniquepid_dict = {} # icu_id2patient_id
        self.unitstayid2age_dict = {} # icu_id2age

        # for each patient
        uniquepid = patient_table['uniquepid'].unique()
        valid_uniquepid = []
        for pt_id in tqdm(uniquepid):

            cnt_total, cnt_processed = 0, 0

            msg = f"uniquepid=='{str(pt_id)}'"
            pt_data = patient_table.query(msg)
            hospitaldischargeyears = sorted(pt_data['hospitaldischargeyear'].unique()) # use discharge year not to exceed the year when sampling time
            init_dischargeyear = hospitaldischargeyears[0]
            init_age = min(pt_data['age'])
            sampled_year = self.first_admit_year_sampler(self.start_year, self.time_span)

            # for each year
            for yrs_seq, dischargeyear in enumerate(hospitaldischargeyears, 1):

                msg = f"hospitaldischargeyear=={str(dischargeyear)}"
                pt_data_given_same_year = pt_data.query(msg)
                patient_visit_year_cnt = len(pt_data_given_same_year['patienthealthsystemstayid'].unique())
                rand_dates = self.sample_date_given_year(sampled_year+(dischargeyear-init_dischargeyear), num_split=patient_visit_year_cnt)
                new_age = init_age+(dischargeyear-init_dischargeyear)

                # for each hospital visit
                for visit_seq, hos_id in enumerate(pt_data_given_same_year['patienthealthsystemstayid'].unique()):

                    msg = f"patienthealthsystemstayid=={str(hos_id)}"
                    hadm_data = pt_data_given_same_year.query(msg)
                    rand_date = rand_dates[visit_seq]

                    # for each icu visit
                    for icu_seq, icu_id in enumerate(hadm_data['patientunitstayid'].unique()):
                        
                        if icu_id not in self.unitstayid2hadmid_dict:
                            self.unitstayid2hadmid_dict[icu_id] = hos_id
                        if icu_id not in self.unitstayid2uniquepid_dict:
                            self.unitstayid2uniquepid_dict[icu_id] = pt_id
                        self.unitstayid2age_dict[icu_id] = int(new_age)

                        msg = f"patientunitstayid=={str(icu_id)}"
                        row = hadm_data.query(msg).iloc[0]

                        hospitaladmitoffset, hospitaladmittime24 = row['hospitaladmitoffset'], row['hospitaladmittime24']
                        unitadmitoffset = 0
                        unitdischargeoffset = row['unitdischargeoffset']
                        hospitaldischargeoffset = row['hospitaldischargeoffset']
                        cnt_total += 1

                        if hospitaladmitoffset <= unitadmitoffset and unitadmitoffset < unitdischargeoffset and unitdischargeoffset <= hospitaldischargeoffset:
                            hospitaladmittime24_shifted = datetime.strptime(rand_date + ' ' + hospitaladmittime24, '%Y-%m-%d %H:%M:%S')
                            self.unitstayid2time_dict[icu_id] = str(hospitaladmittime24_shifted - timedelta(hours=0, minutes=int(hospitaladmitoffset))) # sampled_hosp_admit - (icu_admit offset from hosp_admit) => sampled_icu_admit
                            cnt_processed += 1
                        if cnt_total!=cnt_processed:
                            break
                    if cnt_total!=cnt_processed:
                        break
                if cnt_total!=cnt_processed:
                    break
            if cnt_total==cnt_processed:
                valid_uniquepid.append(pt_id)
                

        patient_table = patient_table[[True if itm in self.unitstayid2age_dict else False for itm in patient_table['patientunitstayid'].values]]
        patient_table['age'] = [self.unitstayid2age_dict[itm] for itm in patient_table['patientunitstayid'].values]
        patient_table['unitadmitoffset'] = 0

        # filter out future patients
        patient_table['hospitaladmittime'] = adjust_time(patient_table, 'hospitaladmitoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        patient_table['unitadmittime'] = adjust_time(patient_table, 'unitadmitoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        patient_table['unitdischargetime'] = adjust_time(patient_table, 'unitdischargeoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        patient_table['hospitaldischargetime'] = adjust_time(patient_table, 'hospitaldischargeoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        patient_table = patient_table.drop(columns=['hospitaldischargeyear', 'hospitaladmittime24', 'hospitaladmitoffset', 'unitadmitoffset', 'unitdischargeoffset', 'hospitaldischargeoffset'])
        patient_table = patient_table.dropna(subset=['hospitaladmittime', 'unitadmittime'])

        if self.timeshift:
            num_valid_cur_patient = len(patient_table['uniquepid'][patient_table['hospitaldischargetime'].isnull()].unique())
            if self.num_cur_patient > num_valid_cur_patient:
                warnings.warn(f"{self.num_cur_patient}>{num_valid_cur_patient}")
            self.cur_patient_list = self.rng.choice(patient_table['uniquepid'][patient_table['hospitaldischargetime'].isnull()].unique(), self.num_cur_patient, replace=False).tolist()
        else:
            self.cur_patient_list = []
        self.not_cur_patient = self.rng.choice(patient_table['uniquepid'][(patient_table['hospitaldischargetime'].notnull()) & (~patient_table['uniquepid'].isin(self.cur_patient_list))].unique(), self.num_not_cur_patient, replace=False).tolist()
        self.patient_list = self.cur_patient_list + self.not_cur_patient
        
        patient_table = patient_table[patient_table['uniquepid'].isin(self.patient_list)]        
        self.icu_list = patient_table['patientunitstayid'].values.tolist()

        patient_table.to_csv(os.path.join(self.out_dir, 'patient.csv'), index=False)

        print(f'patient processed (took {round(time.time() - start_time, 4)} secs)')



    def build_diagnosis_table(self):

        print('Processing diagnosis table')
        start_time = time.time()

        diagnosis_table = read_csv(self.data_dir, 'diagnosis.csv', columns=['diagnosisid', 'patientunitstayid', 'diagnosisoffset', 'diagnosisstring', 'icd9code'], lower=True)

        # filtering out diagnosis
        paths12_34 = [['|'.join(path.split('|')[:2]), '|'.join(path.split('|')[2:4])] for path in diagnosis_table['diagnosisstring'].values]
        p12, p34 = list(zip(*paths12_34))

        p34_to_p12 = {}
        path1234 = []
        for p12, p34 in paths12_34:
            path1234.append('|'.join([p12, p34]))
            if p34 in p34_to_p12:
                p34_to_p12[p34].append(p12)
            else:
                p34_to_p12[p34] = [p12]

        filtered_path = []
        for p34 in p34_to_p12:
            if len(np.unique(p34_to_p12[p34]))>1:
                new_path = Counter(p34_to_p12[p34]).most_common(1)[0][0]+'|'+p34
            else:
                new_path = np.unique(p34_to_p12[p34])[0]+'|'+p34
            filtered_path.append(new_path)

        boolean = [True if p in filtered_path else False for p in path1234]
        diagnosis_table = diagnosis_table[boolean]
        diagnosis_table['diagnosisname'] = [' - '.join(path.split('|')[2:4]) if len(path.split('|'))>3 else path.split('|')[2] for path in diagnosis_table['diagnosisstring'].values]
        diagnosis_table = diagnosis_table.drop(columns=['diagnosisstring'])
        combined_table = diagnosis_table.merge(pd.read_csv(f'{self.data_dir}/patient.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            diagnosis_table = self.condition_value_shuffler(combined_table, target_cols=['diagnosisname', 'icd9code'])
        diagnosis_table = diagnosis_table[diagnosis_table['patientunitstayid'].isin(self.icu_list)]

        diagnosis_table['diagnosistime'] = adjust_time(diagnosis_table, 'diagnosisoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        diagnosis_table = diagnosis_table.drop(columns=['diagnosisoffset'])
        diagnosis_table = diagnosis_table.dropna(subset=['diagnosistime'])
        del diagnosis_table['uniquepid']

        diagnosis_table['diagnosisid'] = range(len(diagnosis_table))
        diagnosis_table.to_csv(os.path.join(self.out_dir, 'diagnosis.csv'), index=False)

        print(f'diagnosis processed (took {round(time.time() - start_time, 4)} secs)')



    def build_treatment_table(self):

        print('Processing treatment table')
        start_time = time.time()

        treatment_table = read_csv(self.data_dir, 'treatment.csv', columns=['treatmentid', 'patientunitstayid', 'treatmentoffset', 'treatmentstring'], lower=True)

        # filtering out treatment
        paths12_34 = [['|'.join(path.split('|')[:2]), '|'.join(path.split('|')[2:4])] for path in treatment_table['treatmentstring'].values]
        p12, p34 = list(zip(*paths12_34))

        p34_to_p12 = {}
        path1234 = []
        for p12, p34 in paths12_34:
            path1234.append('|'.join([p12, p34]))
            if p34 in p34_to_p12:
                p34_to_p12[p34].append(p12)
            else:
                p34_to_p12[p34] = [p12]

        filtered_path = []
        for p34 in p34_to_p12:
            if len(np.unique(p34_to_p12[p34]))>1:
                new_path = Counter(p34_to_p12[p34]).most_common(1)[0][0]+'|'+p34
            else:
                new_path = np.unique(p34_to_p12[p34])[0]+'|'+p34
            filtered_path.append(new_path)

        boolean = [True if p in filtered_path else False for p in path1234]
        treatment_table = treatment_table[boolean]
        treatment_table['treatmentname'] = [' - '.join(path.split('|')[2:4]) if len(path.split('|'))>3 else path.split('|')[2] for path in treatment_table['treatmentstring'].values]
        treatment_table = treatment_table.drop(columns=['treatmentstring'])
        combined_table = treatment_table.merge(pd.read_csv(f'{self.data_dir}/{"patient"}.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            treatment_table = self.condition_value_shuffler(combined_table, target_cols=['treatmentname'])
        treatment_table = treatment_table[treatment_table['patientunitstayid'].isin(self.icu_list)]

        treatment_table['treatmenttime'] = adjust_time(treatment_table, 'treatmentoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        treatment_table = treatment_table.drop(columns=['treatmentoffset'])
        treatment_table = treatment_table.dropna(subset=['treatmenttime'])
        del treatment_table['uniquepid']

        treatment_table['treatmentid'] = range(len(treatment_table))
        treatment_table.to_csv(os.path.join(self.out_dir, 'treatment.csv'), index=False)

        print(f'treatment processed (took {round(time.time() - start_time, 4)} secs)')



    def build_lab_table(self):
    
        print('Processing lab table')
        start_time = time.time()

        lab_table = read_csv(self.data_dir, 'lab.csv', columns=['labid', 'patientunitstayid', 'labresultoffset', 'labname', 'labresult'], lower=True)
        lab_table = lab_table.dropna(subset=['labresult'])

        combined_table = lab_table.merge(pd.read_csv(f'{self.data_dir}/{"patient"}.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            lab_table = self.condition_value_shuffler(combined_table, target_cols=['labname', 'labresult']) 
        lab_table = lab_table[lab_table['patientunitstayid'].isin(self.icu_list)]

        lab_table['labresulttime'] = adjust_time(lab_table, 'labresultoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        lab_table = lab_table.drop(columns=['labresultoffset'])
        lab_table = lab_table.dropna(subset=['labresulttime'])
        del lab_table['uniquepid']

        lab_table['labid'] = range(len(lab_table))
        lab_table.to_csv(os.path.join(self.out_dir, 'lab.csv'), index=False)

        print(f'lab processed (took {round(time.time() - start_time, 4)} secs)')



    def build_medication_table(self):
        
        print('Processing medication table')
        start_time = time.time()

        medication_table = read_csv(self.data_dir, 'medication.csv', columns=['medicationid', 'patientunitstayid', 'drugstartoffset', 'drugstopoffset', 'drugname', 'dosage', 'routeadmin', 'drugordercancelled'], lower=True)
        medication_table = medication_table.dropna(subset=['drugname', 'dosage', 'routeadmin'])
        medication_table = medication_table[medication_table['drugordercancelled']=='no']
        medication_table = medication_table.drop(columns=['drugordercancelled'])

        combined_table = medication_table.merge(pd.read_csv(f'{self.data_dir}/{"patient"}.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            medication_table = self.condition_value_shuffler(combined_table, target_cols=['drugname', 'dosage', 'routeadmin'])
        medication_table = medication_table[medication_table['patientunitstayid'].isin(self.icu_list)]

        medication_table['drugstarttime'] = adjust_time(medication_table, 'drugstartoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        medication_table['drugstoptime'] = adjust_time(medication_table, 'drugstopoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        medication_table = medication_table.drop(columns=['drugstartoffset', 'drugstopoffset'])
        medication_table = medication_table.dropna(subset=['drugstarttime'])
        del medication_table['uniquepid']

        medication_table['medicationid'] = range(len(medication_table))
        medication_table.to_csv(os.path.join(self.out_dir, 'medication.csv'), index=False)

        print(f'medication completed! (took {round(time.time() - start_time, 4)} secs)')



    def build_cost_table(self):

        print('Processing cost table')
        start_time = time.time()

        diagnosis_table = read_csv(self.out_dir, 'diagnosis.csv')
        lab_table = read_csv(self.out_dir, 'lab.csv')
        treatment_table = read_csv(self.out_dir, 'treatment.csv')
        medication_table = read_csv(self.out_dir, 'medication.csv')
 
        
        cnt = 0
        data_filter = []        
        mean_costs = self.rng.poisson(lam=10, size=4)
        
        cost_id = cnt+np.arange(len(diagnosis_table))
        person_id = [self.unitstayid2uniquepid_dict[itm] for itm in diagnosis_table['patientunitstayid'].values]
        hospitaladmit_id = [self.unitstayid2hadmid_dict[itm] for itm in diagnosis_table['patientunitstayid'].values]
        cost_event_table_concept_id = diagnosis_table['diagnosisid'].values
        charge_time = diagnosis_table['diagnosistime'].values
        diagnosis_cost_dict = {itm: round(self.rng.normal(loc=mean_costs[0], scale=1.0),2) for itm in diagnosis_table['diagnosisname'].unique()}
        cost = [diagnosis_cost_dict[itm] for itm in diagnosis_table['diagnosisname'].values]
        temp = pd.DataFrame(data={'costid': cost_id, 
                                'uniquepid': person_id, 
                                'patienthealthsystemstayid': hospitaladmit_id, 
                                'eventtype': 'diagnosis',
                                'eventid': cost_event_table_concept_id,
                                'chargetime': charge_time,
                                'cost': cost})
        cnt += len(diagnosis_table)
        data_filter.append(temp)

        cost_id = cnt+np.arange(len(lab_table))
        person_id = [self.unitstayid2uniquepid_dict[itm] for itm in lab_table['patientunitstayid'].values]
        hospitaladmit_id = [self.unitstayid2hadmid_dict[itm] for itm in lab_table['patientunitstayid'].values]
        cost_event_table_concept_id = lab_table['labid'].values
        charge_time = lab_table['labresulttime'].values
        lab_cost_dict = {itm: round(self.rng.normal(loc=mean_costs[0], scale=1.0),2) for itm in lab_table['labname'].unique()}
        cost = [lab_cost_dict[itm] for itm in lab_table['labname'].values]
        temp = pd.DataFrame(data={'costid': cost_id, 
                                'uniquepid': person_id,
                                'patienthealthsystemstayid': hospitaladmit_id, 
                                'eventtype': 'lab',
                                'eventid': cost_event_table_concept_id,
                                'chargetime': charge_time,
                                'cost': cost})
        cnt += len(lab_table)
        data_filter.append(temp)
            
        cost_id = cnt+np.arange(len(treatment_table))
        person_id = [self.unitstayid2uniquepid_dict[itm] for itm in treatment_table['patientunitstayid'].values]
        hospitaladmit_id = [self.unitstayid2hadmid_dict[itm] for itm in treatment_table['patientunitstayid'].values]
        cost_event_table_concept_id = treatment_table['treatmentid'].values
        charge_time = treatment_table['treatmenttime'].values
        treatment_cost_dict = {itm: round(self.rng.normal(loc=mean_costs[2], scale=1.0),2) for itm in treatment_table['treatmentname'].unique()}
        cost = [treatment_cost_dict[itm] for itm in treatment_table['treatmentname'].values]        
        temp = pd.DataFrame(data={'costid': cost_id, 
                                'uniquepid': person_id, 
                                'patienthealthsystemstayid': hospitaladmit_id, 
                                'eventtype': 'treatment',
                                'eventid': cost_event_table_concept_id,
                                'chargetime': charge_time,
                                'cost': cost})
        cnt += len(treatment_table)
        data_filter.append(temp)

        cost_id = cnt+np.arange(len(medication_table))
        person_id = [self.unitstayid2uniquepid_dict[itm] for itm in medication_table['patientunitstayid'].values]
        hospitaladmit_id = [self.unitstayid2hadmid_dict[itm] for itm in medication_table['patientunitstayid'].values]
        cost_event_table_concept_id = medication_table['medicationid'].values
        charge_time = medication_table['drugstarttime'].values
        medication_cost_dict = {itm: round(self.rng.normal(loc=mean_costs[3], scale=1.0),2) for itm in medication_table['drugname'].unique()}
        cost = [medication_cost_dict[itm] for itm in medication_table['drugname'].values]
        temp = pd.DataFrame(data={'costid': cost_id, 
                                'uniquepid': person_id, 
                                'patienthealthsystemstayid': hospitaladmit_id,
                                'eventtype': 'medication',
                                'eventid': cost_event_table_concept_id,
                                'chargetime': charge_time,
                                'cost': cost})
        cnt += len(medication_table)
        data_filter.append(temp)
            

        cost_table = pd.concat(data_filter, ignore_index=True)
        cost_table.to_csv(os.path.join(self.out_dir, 'cost.csv'), index=False)
        print(f'cost processed (took {round(time.time() - start_time, 4)} secs)')



    def build_allergy_table(self):

        print('Processing allergy table')
        start_time = time.time()

        allergy_table = read_csv(self.data_dir, 'allergy.csv', columns=['allergyid', 'patientunitstayid', 'allergyoffset', 'drugname', 'allergyname'], lower=True)

        combined_table = allergy_table.merge(pd.read_csv(f'{self.data_dir}/{"patient"}.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            allergy_table = self.condition_value_shuffler(combined_table, target_cols=['allergyname', 'drugname'])
        allergy_table = allergy_table[allergy_table['patientunitstayid'].isin(self.icu_list)]

        allergy_table['allergytime'] = adjust_time(allergy_table, 'allergyoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        allergy_table = allergy_table.drop(columns=['allergyoffset'])
        allergy_table = allergy_table.dropna(subset=['allergytime'])
        del allergy_table['uniquepid']

        allergy_table['allergyid'] = range(len(allergy_table))
        allergy_table.to_csv(os.path.join(self.out_dir, 'allergy.csv'), index=False)

        print(f'allergy processed (took {round(time.time() - start_time, 4)} secs)')



    def build_intakeoutput_table(self):
    
        print('Processing intakeOutput table')
        start_time = time.time()

        intakeoutput_table = read_csv(self.data_dir, 'intakeOutput.csv', columns=['intakeoutputid', 'patientunitstayid', 'intakeoutputoffset', 'cellpath', 'celllabel', 'cellvaluenumeric'], lower=True)

        valid_intake_unit = intakeoutput_table['cellpath'].apply(lambda x: True if 'intake (ml)' in x else False)
        valid_output_unit = intakeoutput_table['cellpath'].apply(lambda x: True if 'output (ml)' in x else False)
        intakeoutput_labels = intakeoutput_table['celllabel'].values[valid_intake_unit].tolist() + intakeoutput_table['celllabel'].values[valid_output_unit].tolist()
        intakeoutput_table = intakeoutput_table[intakeoutput_table['celllabel'].isin(intakeoutput_labels)]
        intakeoutput_table = intakeoutput_table[~intakeoutput_table['celllabel'].isin(['intake', 'output'])]
        intakeoutput_table['celllabel'] = intakeoutput_table['celllabel'].apply(lambda x: ' '.join(x.split()[:-1]) if x.split()[-1]=='intake' else x)
        intakeoutput_table['celllabel'] = intakeoutput_table['celllabel'].apply(lambda x: ' '.join(x.split()[:-1]) if x.split()[-1]=='output' else x)

        combined_table = intakeoutput_table.merge(pd.read_csv(f'{self.data_dir}/{"patient"}.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            intakeoutput_table = self.condition_value_shuffler(combined_table, target_cols=['celllabel', 'cellpath', 'cellvaluenumeric'])
        intakeoutput_table = intakeoutput_table[intakeoutput_table['patientunitstayid'].isin(self.icu_list)]

        intakeoutput_table['intakeoutputtime'] = adjust_time(intakeoutput_table, 'intakeoutputoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        intakeoutput_table = intakeoutput_table.drop(columns=['intakeoutputoffset'])
        intakeoutput_table = intakeoutput_table.dropna(subset=['intakeoutputtime'])
        del intakeoutput_table['uniquepid']

        intakeoutput_table['intakeoutputid'] = range(len(intakeoutput_table))
        intakeoutput_table.to_csv(os.path.join(self.out_dir, 'intakeoutput.csv'), index=False)

        print(f'intakeOutput processed (took {round(time.time() - start_time, 4)} secs)')



    def build_microlab_table(self):
        
        print('Processing microLab table')
        start_time = time.time()

        microlab_table = read_csv(self.data_dir, 'microLab.csv', columns=['microlabid', 'patientunitstayid', 'culturetakenoffset', 'culturesite', 'organism'], lower=True)

        combined_table = microlab_table.merge(pd.read_csv(f'{self.data_dir}/{"patient"}.csv')[['patientunitstayid', 'uniquepid']], how='inner', on='patientunitstayid')
        if self.deid:
            microlab_table = self.condition_value_shuffler(combined_table, target_cols=['culturesite', 'organism'])
        microlab_table = microlab_table[microlab_table['patientunitstayid'].isin(self.icu_list)]

        microlab_table['culturetakentime'] = adjust_time(microlab_table, 'culturetakenoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        microlab_table = microlab_table.drop(columns=['culturetakenoffset'])
        microlab_table = microlab_table.dropna(subset=['culturetakentime'])
        del microlab_table['uniquepid']

        microlab_table['microlabid'] = range(len(microlab_table))
        microlab_table.to_csv(os.path.join(self.out_dir, 'microlab.csv'), index=False)

        print(f'microLab processed (took {round(time.time() - start_time, 4)} secs)')



    def build_vital_table(self):
        
        print('Processing vitalPeriodic table')
        start_time = time.time()

        vital_table = read_csv(self.data_dir, 'vitalPeriodic.csv', dtype={'vitalperiodicid': int, 'patientunitstayid': int, 'observationoffset': int, 'temperature': float, 'sao2': float, 'heartrate': float, 'respiration': float, 'systemicsystolic': float, 'systemicdiastolic': float, 'systemicmean': float},
                                                                   columns=['vitalperiodicid', 'patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate', 'respiration', 'systemicsystolic', 'systemicdiastolic', 'systemicmean'], lower=True, 
                                                                   filter={'patientunitstayid': self.icu_list},                                                                    
                                                                   memory_efficient=True)

        vital_table['observationtime'] = adjust_time(vital_table, 'observationoffset', current_time=self.current_time, offset_dict=self.unitstayid2time_dict, patient_col='patientunitstayid')
        vital_table = vital_table.drop(columns=['observationoffset'])
        vital_table = vital_table.dropna(subset=['observationtime'])

        vital_table['vitalperiodicid'] = range(len(vital_table))
        vital_table.to_csv(os.path.join(self.out_dir, 'vitalperiodic.csv'), index=False)

        print(f'vitalperiodic processed (took {round(time.time() - start_time, 4)} secs)')



    def generate_db(self):

        rows = read_csv(self.out_dir, 'patient.csv')
        rows.to_sql('patient', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'diagnosis.csv')
        rows.to_sql('diagnosis', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'treatment.csv')
        rows.to_sql('treatment', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'lab.csv')
        rows.to_sql('lab', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'medication.csv')
        rows.to_sql('medication', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'cost.csv')
        rows.to_sql('cost', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'intakeoutput.csv')
        rows.to_sql('intakeoutput', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'microlab.csv')
        rows.to_sql('microlab', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'allergy.csv')
        rows.to_sql('allergy', self.conn, if_exists='append', index=False)

        rows = read_csv(self.out_dir, 'vitalperiodic.csv')
        rows.to_sql('vitalperiodic', self.conn, if_exists='append', index=False)

        query = "SELECT * FROM sqlite_master WHERE type='table'"
        print(pd.read_sql_query(query, self.conn)['name']) # 10 tables

