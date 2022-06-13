# EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records

EHRSQL is a large text-to-SQL dataset for Electronic Health Records (EHRs), where the questions are collected from 222 hospital staff—including physicians, nurses, insurance review and health records teams, etc. The questions are linked to two open-source EHR database schemas: [Medical Information Mart for Intensive Care III (MIMIC III)](https://physionet.org/content/mimiciii/1.4/) and [eICU Collaborative Research Database (eICU)](https://physionet.org/content/eicu-crd/2.0/). More details are provided below.



##  Requirments and Installation
- Python version >= 3.7
- Pytorch version == 1.7.1
- SQLite3 version >= 3.33.0

```
git clone https://github.com/glee4810/EHRSQL.git
cd EHRSQL
conda create -n ehrsql python=3.7
conda activate ehrsql
pip install dask
```



## Getting Started


To access the databases, PhysioNet’s credentialed access (see license) is needed. Below is the links to getting started pages.


- [MIMIC III](https://mimic.mit.edu/docs/gettingstarted/)
- [eICU](https://eicu-crd.mit.edu/gettingstarted/access/)

Once completed, run the code below to preprocess the databases (patient sampling, de-identification, time-shifting, etc.)

```
cd preprocess
python3 preprocess_db.py --data_dir <path_to_mimic_iii_csv_files> --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 &
python3 preprocess_db.py --data_dir <path_to_eicu_csv_files> --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 
```
