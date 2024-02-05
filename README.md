# EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records

## Overview

EHRSQL is a large-scale, high-quality dataset designed for text-to-SQL question answering on Electronic Health Records from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [eICU](https://physionet.org/content/eicu-crd/2.0/). The dataset includes questions collected from 222 hospital staff, such as physicians, nurses, insurance reviewers, and health records teams. It can be used to test three aspects of QA models: generating a wide range of SQL queries asked in the hospital workplace, understanding various types of time expressions (absolute, relative, or both), and the capability to abstain from answering (querying the database) when the model's prediction is not confident.

The dataset is released along with our paper titled [EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records](https://arxiv.org/abs/2301.07695) (NeurIPS 2022 Datasets and Benchmarks). For further details, please refer to our paper.



## News

`02/06/2024` We are currently working on building EHRSQL 1.1, which will be connected to the demo versions of MIMIC and eICU databases (released in a month!).

`01/29/2024` EHRSQL MIMIC-IV is being used as one of the shared tasks at [NAACL](https://2024.naacl.org/)-[ClinicalNLP 2024](https://clinical-nlp.github.io/2024/call-for-papers.html). For more information, please visit https://sites.google.com/view/ehrsql-2024.

`09/21/2022` EHRSQL has been accepted to NeurIPS 2022 Datasets and Benchmarks ([3/163](https://papercopilot.com/statistics/neurips-statistics/neurips-2022-statistics-datasets-benchmarks/))!


## Getting Started

###  Requirments and Installation
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
pip install scikit-learn
pip install func-timeout
pip install transformers==4.19.2 # 4.29.2 works too
pip install sentencepiece
pip install wandb # if needed
```

### Dataset

#### Question and SQL

The `train.json` file contains the following fields for each database:
- `db_id`: the ID of the database to which the question pertains
- `question`: the paraphrased version of the question 
- `template`: the original template question 
- `query`: the corresponding SQL query for the question 
- `value`: sampled values from the database
- `q_tag`: the question template
- `t_tag`: the sampled time template
- `o_tag`: the sampled operation value
- `tag`: the combination of the question template (q_tag) with the time templates (t_tag) and operation values (o_tag)
- `department`: the hospital department where the question was collected
- `importance`: the importance of the question in the hospital (high, medium, low, or n/a)
- `para_type`: the source of the paraphrase (machine or human)
- `is_impossible`: whether the question is answerable or unanswerable
- `split`: the data split (train, valid, or test)
- `id`: a unique ID for each data instance

```json
  {
    "db_id": "mimic_iii", 
    "question": "what is the ingesting method of methimazole?", 
    "template": "what is the intake method of methimazole?", 
    "query": "select distinct prescriptions.route from prescriptions where prescriptions.drug = 'methimazole'", 
    "value": {"drug_name": "methimazole"},
    "q_tag": "what is the intake method of {drug_name}?", 
    "t_tag": ["", "", "", "", ""], 
    "o_tag": ["", "", "", "", "", "", "", "", ""], 
    "tag": "what is the intake method of {drug_name}?",
    "department": "['nursing']",
    "importance": "medium", 
    "para_type": "machine", 
    "is_impossible": false,
    "split": "train", 
    "id": "75379177b6a56fb54e946591"
  }
```

In `valid.json`, answerable instances have the same structure as `train.json`. However, unanswerable instances have fewer fields.
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

We follow the same table information style used in [Spider](https://github.com/taoyds/spider). `tables.json` contains the following information for both databases:

- `db_id`: the ID of the database
- `table_names_original`: the original table names stored in the database.
- `table_names`: the cleaned and normalized table names.
- `column_names_original`: the original column names stored in the database. Each column has the format `[0, "id"]`. `0` is the index of the table name in `table_names`. `"id"` is the column name. 
- `column_names`: the cleaned and normalized column names.
- `column_types`: the data type of each column
- `foreign_keys`: the foreign keys in the database. `[7, 2]` indicates the column indices in `column_names`. that correspond to foreign keys in two different tables.
- `primary_keys`: the primary keys in the database. Each number represents the index of `column_names`.


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

To access the databases, PhysioNet’s credentialed access (see license) is needed. Below are the links to the download pages.


- [MIMIC-III-1.4](https://physionet.org/content/mimiciii/1.4/)
- [eICU-2.0](https://physionet.org/content/eicu-crd/2.0/)

Once completed, run the code below to preprocess the database. This step involves patient sampling, further de-identification, and time-shifting, and more.

```
cd preprocess
python3 preprocess_db.py --data_dir <path_to_mimic_iii_csv_files> --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
```


### T5 SQL Generation

To train T5-base models, run the code below.
```
python T5/main.py --config T5/config/ehrsql/training/ehrsql_mimic3_t5_base.yaml --CUDA_VISIBLE_DEVICES <gpu_id>
```

To generate SQL queries with abstention, run the code below.
```
python T5/main.py --config T5/config/ehrsql/eval/ehrsql_mimic3_t5_base__mimic3_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/abstain_with_entropy.py --inference_result_path outputs/eval_ehrsql_mimic3_t5_base__mimic3_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14923561
```


### Codex SQL Generation

To generate SQL queries with Codex, run the code below. It is important to note that the ability to abstain has not been implemented in the current version of the Codex run script.
```
python gpt/codex.py --api_key_path <api_key_path> --test_data_path dataset/ehrsql/mimic_iii/valid.json --inference_result_path outputs/eval_ehrsql_mimic3_codex__mimic3_valid --output_file prediction.json --prompt_path gpt/prompts/codex_apidoc.txt
```


### Evaluation

To evaluate the generated SQL queries, run the code below. This code is compatible with both T5 and Codex SQL generation outputs.
```
python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.sqlite --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_ehrsql_mimic3_t5_base__mimic3_valid/prediction.json
python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.sqlite --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_ehrsql_mimic3_codex__mimic3_valid/prediction.json
```



## Have Questions?

Ask us questions on our Github issues page or contact gyubok.lee@kaist.ac.kr.



## Citation

When you use the EHRSQL dataset, we would appreciate it if you cite the following:

```
@article{lee2022ehrsql,
  title={EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records},
  author={Lee, Gyubok and Hwang, Hyeonji and Bae, Seongsu and Kwon, Yeonsu and Shin, Woncheol and Yang, Seongjun and Seo, Minjoon and Kim, Jong-Yeup and Choi, Edward},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={15589--15601},
  year={2022}
}
```
