from mlops_groupp_66.data import MyDataset


def test_data():
    raw_data_path = "data/raw/balanced_creditcard.csv"
    dataset = MyDataset(raw_data_path)
    assert dataset is not None
