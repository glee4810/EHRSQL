# Examining EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records
## Project for UIUC CS598: Deep Learning for Healthcare
## Project by Manaswi (Mona) Kashyap, mkashy3@illinois.edu

EHRSQL is a large-scale, high-quality dataset designed for text-to-SQL question answering on Electronic Health Records from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [eICU](https://physionet.org/content/eicu-crd/2.0/). The dataset includes questions collected from 222 hospital staff, such as physicians, nurses, insurance reviewers and health records teams. It can be used to test three aspects of QA models: generating a wide range of SQL queries asked in the hospital workplace, understanding various types of time expressions (absolute, relative, or both), and the capability to abstain from answering (querying the database) when the model prediction is not confident (a trustworthy semantic parsing task).

Original paper reference: [EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records](https://arxiv.org/abs/2301.07695) (NeurIPS 2022 Datasets and Benchmarks). Please visit [the task website](https://glee4810.github.io/EHRSQL) for more general information on the project and a general introduction.

As part of the course CS598: Deep Learning for Healthcare in Spring 2023, I chose to reproduce this paper and evaluate its merits. 

### Steps followed to reproduce the original study: 
1) Created an AWS ec2 instance, Linux AMI, t3.2xlarge with 8vCPUs, volume of size 500 GiB. 
2) Cloned original EHRSQL repo:
 
 ```git clone git@github.com:MKASHY3/CS598-Spring-2023-EHRSQL.git . ```

3) Set up public key to access my Git repo in the ec2 instance: 
https://medium.com/coder-life/practice-2-host-your-website-on-github-pages-39229dc9bb1b
4) Made sure to use sudo access / root user on ec2: ```sudo su``
5) Connect to my Git repo with ssh on my ec2 instance 
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account 
6) Install anaconda on ec2 instance:

```
wget https://repo.continuum.io/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
```

NOTE: While installing anaconda on ec2, I got this error: Fixing AWS EC2 “No space left on device” issue on EBS volume. To resolve:
https://medium.com/@wlarch/no-space-left-on-device-on-my-ec2-aws-instance-cfbd69fba37a
  - Made a snapshot of the ec2 volume, increased the volume in AWS, and increased the size of the file system in my ec2 instance.
  - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/recognize-expanded-volume-linux.html
  - https://repost.aws/knowledge-center/ebs-volume-size-increase  

7) Run following commands to install necessary packages/libraries (from original README):
```
conda create -n ehrsql python=3.7
conda activate ehrsql
pip install pandas
pip install dask
pip install wandb # if needed
pip install nltk
pip install scikit-learn
pip install func-timeout
```

8) Follow steps provided by course instructors to gain credentialed access to PhysioNet
9) Once PhysioNet access activated, access datasets from MIMIC-III and eICU databases, using links found here:
  - [MIMIC-III-1.4](https://physionet.org/content/mimiciii/1.4/) 
  - [eICU-2.0](https://physionet.org/content/eicu-crd/2.0/)
10) Download and save both datasets locally, noting the location as well
11) If not already installed, download and install locally wget: https://builtvisible.com/download-your-website-with-wget/ 
12) Download the files from eICU using the terminal: 
```wget -r -N -c -np --user mkashy3 --ask-password https://physionet.org/files/eicu-crd/2.0/```
13) Download the files from MIMIC-III using the terminal: 
```wget -r -N -c -np --user mkashy3 --ask-password https://physionet.org/files/mimiciii/1.4/```
14) Unzip the gzip files in my ec2 terminal: 
```gunzip *.gz```
15) Once data has been downloaded and unzipped, preprocess it:
```cd preprocess```

#### Preprocess MIMIC-III data: 
```
python preprocess_db.py --data_dir /home/ec2-user/GitHub/physionet.org/files/mimiciii/1.4/ --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
```

#### Preprocess eICU data: 
```
python preprocess_db.py --data_dir /home/ec2-user/GitHub/physionet.org/files/eicu-crd/2.0/ --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
```

### T5 SQL Generation

Run the following code to generate SQL queries with abstention: 
```
python T5/main.py --config T5/config/ehrsql/eval/ehrsql_mimic3_t5_base__mimic3_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_id>
python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_ehrsql_mimic3_t5_base__mimic3_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14923561
```

### Evaluate 

Evaluate our generated SQL queries by running the following commands:
```
python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_ehrsql_mimic3_t5_base__mimic3_valid/prediction.json
python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_ehrsql_mimic3_codex__mimic3_valid/prediction.json
```


###  Requirements and Installation
- Python version >= 3.7
- Pytorch version == 1.7.1
- SQLite3 version >= 3.33.0

###  Have Questions?

Please contact me at mkashy3@illinois.edu




### Citation

The original paper citation:

```
@inproceedings{lee2022ehrsql,
  title     = {{EHRSQL}: A Practical Text-to-{SQL} Benchmark for Electronic Health Records},
  author    = {Gyubok Lee and Hyeonji Hwang and Seongsu Bae and Yeonsu Kwon and Woncheol Shin and Seongjun Yang and Minjoon Seo and Jong-Yeup Kim and Edward Choi},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://openreview.net/forum?id=B2W8Vy0rarw}
}
```
