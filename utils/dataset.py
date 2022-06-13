import json
from random import Random
from typing import Optional, Dict
from tqdm import tqdm
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

@dataclass
class AnnotatedSQL:
    question: str
    query: str
    db_id: Optional[str] = None
    q_tag: Optional[str] = None
    t_tag: Optional[str] = None
    o_tag: Optional[str] = None
    para_type: Optional[str] = None
    imp: Optional[str] = None
    is_impossible: Optional[str] = None


class EHRSQL_Dataset(Dataset):
    def __init__(self, path, tokenizer, args, include_impossible=False):

        self.dataset = args.dataset
        self.db_id = args.db_id
        self.tokenizer = tokenizer
        self.use_para = args.use_para # whether to use paraphrase
        self.add_schema = args.add_schema
        self.shuffle_schema = args.shuffle_schema
        self.random = Random(args.random_seed)
        self.add_column_type = args.add_column_type
        self.tables_path = args.tables_path
        self.include_impossible = include_impossible

        if self.dataset == "ehrsql":
            with open(path) as json_file:
                data = json.load(json_file)
            if self.use_para:
                question_key = "question"
            else:
                question_key = "template"
            query_key = "query"

        elif args.dataset == 'mimicsql' or args.dataset == 'mimicsqlstar':
            data = []
            with open(path) as json_file:
                for line in json_file:
                    data.append(json.loads(line))
            question_key = "question_refine"
            query_key = "sql"

        if self.add_schema:
            if self.tables_path is None:
                raise "tables_path must be provided for add_schema=True"
            with open(self.tables_path) as f:
                self.db_json = json.load(f)

        self.data = []
        for line in tqdm(data):
            if self.include_impossible==False and 'is_impossible' in line and line['is_impossible']:
                continue
            if self.dataset == "ehrsql":
                annotated_sql: AnnotatedSQL = AnnotatedSQL(
                    question=line[question_key].lower() if question_key in line else line['question'],
                    query=line[query_key].lower() if query_key in line else 'nan',
                    db_id=line["db_id"] if "db_id" in line else None,
                    q_tag=line["q_tag"] if "q_tag" in line else None,
                    t_tag=line["t_tag"] if "t_tag" in line else None,
                    o_tag=line["o_tag"] if "o_tag" in line else None,
                    para_type=line["para_type"] if "para_type" in line else None,
                    imp=line["importance"] if "importance" in line else None,
                    is_impossible=line["is_impossible"] if "is_impossible" in line else None,
                )
            elif args.dataset == 'mimicsql' or args.dataset == 'mimicsqlstar':
                annotated_sql: AnnotatedSQL = AnnotatedSQL(                
                    question=line[question_key].lower(),
                    query=line[query_key].lower(),
                    db_id=args.dataset,
                )
            else:
                raise NotImplementedError

            instance = self.preprocess_sample(annotated_sql)
            self.data.append(instance)

    def preprocess_sample(self, annotated_sql: AnnotatedSQL) -> AnnotatedSQL:

        question = annotated_sql.question

        if self.add_schema:
            tables_json = [db for db in self.db_json if db["db_id"] == annotated_sql.db_id][0]
            schema_description = self.get_schema_description(self.add_column_type, tables_json, self.shuffle_schema, self.random)
            question += f" {schema_description}"

        processed_annotated_sql: AnnotatedSQL = AnnotatedSQL(
            question=question,
            query=annotated_sql.query,
            db_id=annotated_sql.db_id,
            q_tag=annotated_sql.q_tag,
            t_tag=annotated_sql.t_tag,
            o_tag=annotated_sql.o_tag,
            para_type=annotated_sql.para_type,
            imp=annotated_sql.imp,
            is_impossible=annotated_sql.is_impossible
        )

        return processed_annotated_sql


    def get_schema_description(self, add_column_type: bool, tables_json: Dict, shuffle_schema: bool, random: Random):

        table_names = tables_json["table_names_original"]

        if shuffle_schema:
            random.shuffle(table_names)

        columns = [
            (column_name[0], column_name[1], column_type)
            for column_name, column_type in zip(tables_json["column_names_original"], tables_json["column_types"])
        ]
        
        schema_description = ""
        for table_index, table_name in enumerate(table_names):
            table_name = table_name.lower()
            if table_index == 0:
                schema_description += "<TAB> " + table_name
            else:
                schema_description += " <TAB> " + table_name
            schema_description += " <COL>"
            table_columns = [column for column in columns if column[0] == table_index]

            if shuffle_schema:
                random.shuffle(table_columns)

            for table_column in table_columns:
                if add_column_type:
                    column_description = (
                        f"<type: "
                        f"{table_column[2].lower()}> "
                        f"{table_column[1].lower()}"
                    )
                else:
                    column_description = f"{table_column[1].lower()}"
                schema_description += " " + column_description

        return schema_description



    def __getitem__(self, index):
        fields = {
            "inputs": self.data[index].question,
            "labels": self.data[index].query,
            "db_id": self.data[index].db_id,
            "q_tag": self.data[index].q_tag,
            "t_tag": self.data[index].t_tag,
            "o_tag": self.data[index].o_tag,
            "para_type": self.data[index].para_type,
            "imp": self.data[index].imp,
            "is_impossible": self.data[index].is_impossible
        }
        return fields


    def __len__(self):
        return len(self.data)


class DataCollator(object):
    def __init__(self, tokenizer, return_tensors='pt', padding=True, truncation=True):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.padding = padding
        self.truncation = truncation

    def __call__(self, batch):

        input_ids, labels = [], []
        db_id = []
        q_tags = []
        t_tags = []
        o_tags = []
        para_types = []
        imps = []
        is_impossibles = []
        for instance in batch:
            input_ids.append(instance['inputs'])
            labels.append(instance['labels'])
            db_id.append(instance['db_id'])
            q_tags.append(instance['q_tag'])
            t_tags.append(instance['t_tag'])
            o_tags.append(instance['o_tag'])
            para_types.append(instance['para_type'])
            imps.append(instance['imp'])
            is_impossibles.append(instance['is_impossible'])

        inputs = self.tokenizer(input_ids, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation)
        outputs = self.tokenizer(labels, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation)

        fields = {
            "inputs": inputs.input_ids,
            "labels": outputs.input_ids,
            "db_id": db_id,
            "q_tag": q_tags,
            "t_tag": t_tags,
            "o_tag": o_tags,
            "para_type": para_types,
            "imp": imps,            
            "is_impossible": is_impossibles
        }

        return fields
