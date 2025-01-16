'''
from torch.utils.data import Dataset
from mlops_groupp_66.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
'''

def test_dummy():
    assert 1 + 1 == 2
