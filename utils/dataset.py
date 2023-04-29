import json
from tqdm import tqdm
from random import Random
from typing import Optional, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

@dataclass
class AnnotatedSQL:
    question: str
    query: str
    db_id: Optional[str] = None
    is_impossible: Optional[str] = None
    id: Optional[str] = None


class EHRSQL_Dataset(Dataset):
    def __init__(self, path, tokenizer, args, include_impossible=False, data_ratio=1.0):

        self.dataset = args.dataset
        self.db_id = args.db_id
        self.tokenizer = tokenizer
        self.add_schema = args.add_schema
        self.shuffle_schema = args.shuffle_schema
        self.random = Random(args.random_seed)
        self.add_column_type = args.add_column_type
        self.tables_path = args.tables_path
        self.data_ratio = data_ratio
        self.include_impossible = include_impossible

        with open(path) as json_file:
            data = json.load(json_file)
        if self.data_ratio < 1.0:
            train_data_id_list_all = [instance['id'] for instance in data]
            self.random.shuffle(train_data_id_list_all)
            train_data_id_list_all = train_data_id_list_all[:max(int(len(data) * self.data_ratio), 1)]
            new_data = []
            for instance in data:
                if instance['id'] in train_data_id_list_all:
                    new_data.append(instance)
            data = new_data

        if self.add_schema:
            if self.tables_path is None:
                raise "tables_path must be provided for add_schema=True"
            with open(self.tables_path) as f:
                self.db_json = json.load(f)

        self.data = []
        for line in tqdm(data):
            if self.include_impossible==False and 'is_impossible' in line and line['is_impossible']:
                continue
            annotated_sql = AnnotatedSQL(
                question=line["question"].lower(),
                query=line["query"].lower() if "query" in line else 'null',
                db_id=line["db_id"],
                is_impossible=line["is_impossible"],
                id=line["id"]
            )
            instance = self.preprocess_sample(annotated_sql)
            self.data.append(instance)

    def preprocess_sample(self, annotated_sql: AnnotatedSQL) -> AnnotatedSQL:

        question = annotated_sql.question

        if self.add_schema:
            tables_json = [db for db in self.db_json if db["db_id"] == annotated_sql.db_id][0]
            schema_description = self.get_schema_description(tables_json, self.shuffle_schema, self.random)
            question += f" {schema_description}"

        processed_annotated_sql: AnnotatedSQL = AnnotatedSQL(
            question=question,
            query=annotated_sql.query,
            db_id=annotated_sql.db_id,
            is_impossible=annotated_sql.is_impossible,
            id=annotated_sql.id
        )

        return processed_annotated_sql


    def get_schema_description(self, tables_json: Dict, shuffle_schema: bool, random: Random):
        table_names = tables_json["table_names_original"]
        if shuffle_schema:
            random.shuffle(table_names)

        columns = [
            (column_name[0], column_name[1], column_type)
            for column_name, column_type in zip(tables_json["column_names_original"], tables_json["column_types"])
        ]

        schema_description = ""
        for table_index, table_name in enumerate(table_names):
            schema_description += f" | {table_name} : "
            
            table_columns = [column[1] for column in columns if column[0] == table_index]
            if shuffle_schema:
                random.shuffle(table_columns)

            schema_description += " , ".join(table_columns)

        return schema_description.lower().lstrip()


    def __getitem__(self, index):
        fields = {
            "inputs": self.data[index].question,
            "labels": self.data[index].query,
            "db_id": self.data[index].db_id,
            "is_impossible": self.data[index].is_impossible,
            "id": self.data[index].id,
        }
        return fields


    def __len__(self):
        return len(self.data)


class DataCollator(object):
    def __init__(self, tokenizer, return_tensors='pt', padding=True, truncation=True, max_length=512):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __call__(self, batch):

        input_ids, labels = [], []
        data_ids = []
        db_id = []
        is_impossibles = []
        for instance in batch:
            input_ids.append(instance['inputs'])
            labels.append(instance['labels'])
            db_id.append(instance['db_id'])
            is_impossibles.append(instance['is_impossible'])
            data_ids.append(instance['id'])

        inputs = self.tokenizer(input_ids, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation, max_length=self.max_length)
        outputs = self.tokenizer(labels, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation, max_length=self.max_length)

        fields = {
            "inputs": inputs.input_ids,
            "labels": outputs.input_ids,
            "db_id": db_id,
            "is_impossible": is_impossibles,
            "id": data_ids
        }

        return fields