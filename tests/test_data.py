import pytest
from pathlib import Path
import pandas as pd
from transformers import DistilBertTokenizer
from mlops_groupp_66.data import MyDataset, get_transformer_dataloaders

@pytest.fixture
def sample_csv(tmp_path):
    """Fixture to create a sample CSV file for testing."""
    raw_data_path = tmp_path / "test_raw_data.csv"
    raw_data_path.write_text("col1,col2,label\n1,2,0\n3,4,1\n")
    return raw_data_path

def test_len(sample_csv):
    """Test the length of the dataset after preprocessing."""
    dataset = MyDataset(sample_csv)
    dataset.preprocess(Path("."))
    assert len(dataset) == 2

def test_getitem(sample_csv):
    """Test retrieving items from the dataset after preprocessing."""
    dataset = MyDataset(sample_csv)
    dataset.preprocess(Path("."))
    assert dataset[0]["col1"] == 1
    assert dataset[1]["col2"] == 4

def test_preprocess(sample_csv, tmp_path):
    """Test the preprocessing function to ensure it saves the preprocessed data."""
    dataset = MyDataset(sample_csv)
    output_folder = tmp_path
    dataset.preprocess(output_folder)
    output_file = output_folder / "preprocessed_data.csv"
    assert output_file.exists()

@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing."""
    return pd.DataFrame({
        "text1": ["sample text 1", "sample text 2"],
        "text2": ["more text 1", "more text 2"],
        "label": [0, 1]
    })

@pytest.fixture
def tokenizer():
    """Fixture to create a tokenizer for testing."""
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def test_get_transformer_dataloaders(sample_dataframe, tokenizer):
    """Test the creation of transformer dataloaders."""
    train_loader, test_loader = get_transformer_dataloaders(sample_dataframe, tokenizer)
    assert len(train_loader.dataset) == 1
    assert len(test_loader.dataset) == 1
