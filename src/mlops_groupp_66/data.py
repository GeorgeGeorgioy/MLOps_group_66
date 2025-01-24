# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, raw_data_path: Path) -> None:
#         self.data_path = raw_data_path
#         self.data = None

#     def __len__(self) -> int:
#         """Return the length of the dataset."""
#         if self.data is not None:
#             return len(self.data)
#         raise ValueError("Dataset not loaded. Run preprocess first.")

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""
#         if self.data is not None:
#             return self.data.iloc[index]
#         raise ValueError("Dataset not loaded. Run preprocess first.")

#     def preprocess(self, output_folder: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""
#         print("Loading raw data...")
#         self.data = pd.read_csv(self.data_path)

#         # Save preprocessed data
#         output_file = output_folder / "preprocessed_data.csv"
#         print(f"Saving preprocessed data to {output_file}...")
#         self.data.to_csv(output_file, index=False)


# # def get_nn_dataloaders(data: pd.DataFrame, batch_size=32):
# #     """Create dataloaders for the neural network."""
# #     X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     scaler = StandardScaler()
# #     X_train = scaler.fit_transform(X_train)
# #     X_test = scaler.transform(X_test)

# #     train_dataset = TensorDataset(
# #         torch.tensor(X_train, dtype=torch.float32),
# #         torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# #     )
# #     test_dataset = TensorDataset(
# #         torch.tensor(X_test, dtype=torch.float32),
# #         torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# #     )

# #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# #     return train_loader, test_loader


from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
import os
from dotenv import load_dotenv


class MyDataset(Dataset):
    """Custom dataset for loading raw data."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.data = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.data is not None:
            return len(self.data)
        raise ValueError("Dataset not loaded. Run preprocess first.")

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if self.data is not None:
            return self.data.iloc[index]
        raise ValueError("Dataset not loaded. Run preprocess first.")

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Loading raw data...")
        self.data = pd.read_csv(self.data_path)

        # Save preprocessed data
        output_file = output_folder / "preprocessed_data.csv"
        print(f"Saving preprocessed data to {output_file}...")
        self.data.to_csv(output_file, index=False)


def get_transformer_dataloaders(data: pd.DataFrame, tokenizer: DistilBertTokenizer, max_len=128, batch_size=8):
    """Create dataloaders for the transformer."""
    # Prepare the text and labels (last column as target)
    X, y = data.iloc[:, :-1].apply(lambda row: ' '.join(row.values.astype(str)), axis=1), data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class FraudDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Create DataLoader instances
    train_dataset = FraudDataset(X_train.tolist(), y_train, tokenizer, max_len)
    test_dataset = FraudDataset(X_test.tolist(), y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data."""
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    load_dotenv()
    raw_data_path = Path(os.getenv("RAW_DATA")).resolve()
    output_folder = Path(os.getenv("OUTPUT_FOLDER")).resolve()
    preprocess(raw_data_path, output_folder)
