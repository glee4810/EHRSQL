# EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records

EHRSQL is a large-scale, high-quality text-to-SQL dataset for question answering (QA) on Electronic Health Records ([MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [eICU](https://physionet.org/content/eicu-crd/2.0/)). The questions are collected from 222 hospital staff, including physicians, nurses, insurance review and health records teams, etc. The dataset can be used to test three aspects of QA models: generating a wide range of SQL queries asked in the hospital workplace, understanding various types of time expressions (absolute, relative, or both), and the capability to abstain from answering (querying the database) when the model prediction is not confident (trustworthy Text-to-SQL task).

The dataset is released along with our paper: [EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records](https://arxiv.org/abs/2301.07695) (NeurIPS 2022 Datasets and Benchmarks). For more information, please refer to our paper.


# Under Construction

<!-- ##  Requirments and Installation
- Python version >= 3.7
- Pytorch version == 1.7.1
- SQLite3 version >= 3.33.0

```
git clone https://github.com/glee4810/EHRSQL.git
cd EHRSQL
conda create -n ehrsql python=3.7
conda activate ehrsql
pip install pandas
pip install dask
pip install wandb # if needed
pip install nltk
pip install scikit-learn
pip install func-timeout
```



## Getting Started


### Dataset

#### Question and SQL

For each database, `train.json` contains the following fields:
- `db_id`: the database id to which this question is addressed.
- `question`: the paraphrased question 
- `template`: the template question 
- `query`: the SQL query corresponding to the question. 
- `value`: sampled values from the database
- `q_tag`: the question template
- `t_tag`: sampled time template
- `o_tag`: sampled operation value
- `tag`: question template (q_tag) combined with time templates (t_tag) and operation values (o_tag)
- `department`: hospital department where the question is collected from
- `importance`: the importance of the question in the hospital (high, medium, low, n/a)
- `para_type`: the source of paraphrase (machine or human)
- `is_impossible`: whether the question is answerable or unanswerable
- `split`: data split (train, valid, test)
- `id`: unique id of each data instance

```json
 {
    "db_id": "mimic_iii",
    "question": "tell me the method of intake of clobetasol propionate 0.05% ointment?",
    "template": "what is the intake method of clobetasol propionate 0.05% ointment?",
    "query": "select distinct prescriptions.route from prescriptions where prescriptions.drug = 'clobetasol propionate 0.05% ointment'",
    "value": {"drug_name": "clobetasol propionate 0.05% ointment"},
    "q_tag": "what is the intake method of {drug_name}?",
    "t_tag": "["", "", "", "", ""]",
    "o_tag": "["", "", "", "", "", "", "", "", ""]",
    "tag": "what is the intake method of {drug_name}?",
    "department": "['nursing']",
    "importance": "medium",
    "para_type": "machine",
    "is_impossible": false,
    "split": "train",
    "id": "294c4222b4ad35fbe4fb9801"
}
```

For `valid.json`, answerable instances are structured in the same manner as `train.json`. But unanswerable instances have a smaller number of fields.
```json
 {
    "db_id": "mimic_iii",
    "question": "tell me what medicine to use to relieve a headache in hypertensive patients.",
    "query": "nan",
    "department": "['nursing']",
    "para_type": "human",
    "is_impossible": true,
    "split": "valid",
    "id": "9db3a82be08e143d7976b015"
}
```




#### Tables

We follow the same style of table information introduced in [Spider](https://github.com/taoyds/spider). `tables.json` contains the following information for each database:

- `db_id`: database id
- `table_names_original`: original table names stored in the database.
- `table_names`: cleaned and normalized table names.
- `column_names_original`: original column names stored in the database. Each column looks like: `[0, "id"]`. `0` is the index of table names in `table_names`. `"id"` is the column name. 
- `column_names`: cleaned and normalized column names.
- `column_types`: data type of each column
- `foreign_keys`: foreign keys in the database. `[7, 2]` means column indices in the `column_names`. These two columns are foreign keys of two different tables.
- `primary_keys`: primary keys in the database. Each number is the index of `column_names`.


```json
{
    "column_names": [
      [
        0,
        "row id"
      ],
      [
        0,
        "subject id"
      ],
      [
        0,
        "gender"
      ],
      [
        0,
        "dob"
      ],
      ...
    ],
    "column_names_original": [
      [
        0,
        "ROW_ID"
      ],
      [
        0,
        "SUBJECT_ID"
      ],
      [
        0,
        "GENDER"
      ],
      [
        0,
        "DOB"
      ],
      ...
    ],
    "column_types": [
      "number",
      "number",
      "text",
      "time",
      ...
    ],
    "db_id": "mimic_iii",
    "foreign_keys": [
      [
        7,
        2
      ],
      ...
    ],
    "primary_keys": [
      1,
      5,
      ...
    ],
    "table_names": [
      "patients",
      "admissions",
      ...
    ],
    "table_names_original": [
      "PATIENTS",
      "ADMISSIONS",
      ...
    ]
  }
```


### Database

To access the databases, PhysioNet’s credentialed access (see license) is needed. Below is the links to getting started pages.


- [MIMIC III](https://mimic.mit.edu/docs/gettingstarted/)
- [eICU](https://eicu-crd.mit.edu/gettingstarted/access/)

Once completed, run the code below to preprocess the databases (patient sampling, de-identification, time-shifting, etc.)

```
cd preprocess
python3 preprocess_db.py --data_dir <path_to_mimic_iii_csv_files> --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 &
python3 preprocess_db.py --data_dir <path_to_eicu_csv_files> --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
```

If the databases are not available, no execution accuracy (EX) is measured, but exact string matching (ESM) and false negative rate (FNR) are still measured. 



### T5 SQL Generation

To train T5-base models, run the code below.
```
python T5/main.py --config T5/config/ehrsql/training/t5_ehrsql_mimic3_natural_lr0.001.yaml --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/main.py --config T5/config/ehrsql/training/t5_ehrsql_mimic3_natural_lr0.001_schema.yaml --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/main.py --config T5/config/ehrsql/training/t5_ehrsql_eicu_natural_lr0.001.yaml --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/main.py --config T5/config/ehrsql/training/t5_ehrsql_eicu_natural_lr0.001_schema.yaml --CUDA_VISIBLE_DEVICES <gpu_id>
```

To generate SQL queries with abstention, run the code below.
```
python T5/main.py --config T5/config/ehrsql/eval/t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid --output_file prediction.json --threshold -1
python T5/main.py --config T5/config/ehrsql/eval/t5_ehrsql_mimic3_natural_lr0.001_schema_best__mimic3_natural_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_schema_best__mimic3_natural_valid --output_file prediction.json --threshold -1
python T5/main.py --config T5/config/ehrsql/eval/t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__eicu_natural_valid --output_file prediction.json --threshold -1
python T5/main.py --config T5/config/ehrsql/eval/t5_ehrsql_eicu_natural_lr0.001_schema_best__eicu_natural_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_schema_best__eicu_natural_valid --output_file prediction.json --threshold -1
```


### Codex SQL Generation

To generate SQL queries with Codex, run the code below. The capability to abstain is not implemented for Codex.
```
python gpt/codex.py --api_key_path <openai_api_key> --test_data_path dataset/ehrsql/mimic_iii/valid.json --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid --output_file prediction.json --prompt_path gpt/prompts/codex_apidoc.txt
python gpt/codex.py --api_key_path <openai_api_key> --test_data_path dataset/ehrsql/mimic_iii/valid.json --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_schema_best__mimic3_natural_valid --output_file prediction.json --prompt_path gpt/prompts/codex_apidoc.txt
python gpt/codex.py --api_key_path <openai_api_key> --test_data_path dataset/ehrsql/eicu/valid.json --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__eicu_natural_valid --output_file prediction.json --prompt_path gpt/prompts/codex_apidoc.txt
python gpt/codex.py --api_key_path <openai_api_key> --test_data_path dataset/ehrsql/eicu/valid.json --infernece_result_path outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_schema_best__eicu_natural_valid --output_file prediction.json --prompt_path gpt/prompts/codex_apidoc.txt
```


### Evaluation

To evaluate the generated SQL queries, run the code below. This code is compatible with both T5 and Codex SQL generation outputs.
```
python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid/prediction.json
python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_schema_best__mimic3_natural_valid/prediction.json
python evaluate.py --db_path ./dataset/ehrsql/eicu/eicu.db --data_file dataset/ehrsql/eicu/valid.json --pred_file ./outputs/eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid/prediction.json
python evaluate.py --db_path ./dataset/ehrsql/eicu/eicu.db --data_file dataset/ehrsql/eicu/valid.json --pred_file ./outputs/eval_t5_ehrsql_eicu_natural_lr0.001_schema_best__eicu_natural_valid/prediction.json
```


## Have Questions?

Ask us questions at our Github issues page or contact gyubok.lee@kaist.ac.kr. -->
