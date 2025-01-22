from pathlib import Path
import pandas as pd
from transformers import DistilBertTokenizer
from mlops_groupp_66.data import MyDataset, get_transformer_dataloaders
import os
from dotenv import load_dotenv
import pytest

# Load environment variables
load_dotenv()

@pytest.fixture
def raw_data_path():
    """Fixture to provide the path to the balanced_creditcard.csv file."""
    raw_data = os.getenv("RAW_DATA")
    if raw_data is None:
        pytest.fail("RAW_DATA environment variable is not set")
    return Path(raw_data).resolve()

def test_len(raw_data_path, tmp_path):
    """Test the length of the dataset after preprocessing."""
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(tmp_path)
    assert len(dataset) == 1000

def test_getitem(raw_data_path, tmp_path):
    """Test retrieving items from the dataset after preprocessing."""
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(tmp_path)
    item = dataset[0]
    assert "Time" in item  # Ensure the item has the expected columns
    assert "V1" in item

def test_preprocess(raw_data_path, tmp_path):
    """Test the preprocessing function to ensure it saves the preprocessed data."""
    dataset = MyDataset(raw_data_path)
    output_folder = tmp_path
    dataset.preprocess(output_folder)
    output_file = output_folder / "preprocessed_data.csv"
    assert output_file.exists()
