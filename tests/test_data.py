#from mlops_groupp_66.data import MyDataset

from pathlib import Path
import pandas as pd
import torch
from transformers import DistilBertTokenizer
from mlops_groupp_66.data import MyDataset, get_nn_dataloaders, get_transformer_dataloaders

def test_dummy():
    assert 1 + 1 == 2
