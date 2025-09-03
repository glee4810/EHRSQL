"""Shared pytest fixtures for EHRSQL testing."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    file_path = os.path.join(temp_dir, "test_file.txt")
    with open(file_path, 'w') as f:
        f.write("test content")
    return file_path


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = MagicMock()
    config.dataset = "mimic_iii"
    config.db_id = "test_db"
    config.random_seed = 42
    config.add_schema = True
    config.shuffle_schema = False
    config.add_column_type = True
    config.tables_path = "/mock/tables.json"
    config.model_name = "t5-base"
    config.device = "cpu"
    config.batch_size = 4
    config.learning_rate = 1e-4
    config.num_epochs = 1
    config.max_input_length = 512
    config.max_output_length = 256
    return config


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test decoded text"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 32128
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock T5 model for testing."""
    model = MagicMock()
    model.config.vocab_size = 32128
    model.config.d_model = 768
    return model


@pytest.fixture
def sample_dataset_item():
    """Sample dataset item for testing."""
    return {
        "question": "What is the average age of patients?",
        "query": "SELECT AVG(age) FROM patients;",
        "db_id": "test_db",
        "is_impossible": "0",
        "id": "test_001"
    }


@pytest.fixture
def sample_json_data(temp_dir):
    """Create sample JSON data file for testing."""
    import json
    data = [
        {
            "question": "What is the average age of patients?", 
            "query": "SELECT AVG(age) FROM patients;",
            "db_id": "test_db"
        },
        {
            "question": "How many patients are there?",
            "query": "SELECT COUNT(*) FROM patients;", 
            "db_id": "test_db"
        }
    ]
    file_path = os.path.join(temp_dir, "test_data.json")
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_tables_json(temp_dir):
    """Create sample tables.json for testing."""
    import json
    tables_data = {
        "test_db": {
            "table_names_original": ["patients", "admissions"],
            "table_names": ["patients", "admissions"],
            "column_names_original": [
                [-1, "*"],
                [0, "patient_id"],
                [0, "age"],
                [1, "admission_id"],
                [1, "patient_id"]
            ],
            "column_names": [
                [-1, "*"],
                [0, "patient_id"], 
                [0, "age"],
                [1, "admission_id"],
                [1, "patient_id"]
            ],
            "column_types": ["text", "number", "number", "number", "number"],
            "primary_keys": [1, 3],
            "foreign_keys": [[4, 1]]
        }
    }
    file_path = os.path.join(temp_dir, "tables.json")
    with open(file_path, 'w') as f:
        json.dump(tables_data, f)
    return file_path


@pytest.fixture
def mock_sqlite_db(temp_dir):
    """Create a mock SQLite database for testing."""
    import sqlite3
    db_path = os.path.join(temp_dir, "test.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            age INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE admissions (
            admission_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)
    
    # Insert sample data
    cursor.execute("INSERT INTO patients (patient_id, age) VALUES (1, 65)")
    cursor.execute("INSERT INTO patients (patient_id, age) VALUES (2, 45)")
    cursor.execute("INSERT INTO admissions (admission_id, patient_id) VALUES (1, 1)")
    
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing."""
    wandb_mock = MagicMock()
    return wandb_mock


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_DISABLED", "true")


@pytest.fixture
def yaml_config_content():
    """Sample YAML configuration content."""
    return """
dataset: mimic_iii
db_id: test_db
model_name: t5-base
batch_size: 4
learning_rate: 0.0001
num_epochs: 1
max_input_length: 512
max_output_length: 256
random_seed: 42
add_schema: true
shuffle_schema: false
add_column_type: true
"""


@pytest.fixture
def yaml_config_file(temp_dir, yaml_config_content):
    """Create a YAML configuration file for testing."""
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, 'w') as f:
        f.write(yaml_config_content)
    return config_path