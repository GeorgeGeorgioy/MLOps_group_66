import matplotlib.pyplot as plt
import pandas as pd
import typer
from pathlib import Path
import torch
from sklearn.preprocessing import LabelEncoder
from data import MyDataset


def dataset_statistics(datadir: Path, output_folder: Path) -> None:
    """Compute dataset statistics."""
    # Initialize and preprocess the dataset
    dataset = MyDataset(raw_data_path=datadir)
    dataset.preprocess(output_folder=output_folder)

    print(f"Dataset: MyDataset")
    print(f"Number of samples: {len(dataset)}")

    # Load the preprocessed data
    preprocessed_data = pd.read_csv(output_folder / "preprocessed_data.csv")

    # Display basic statistics
    print("Basic statistics:")
    print(preprocessed_data.describe())

    # Visualize label distribution
    label_col = preprocessed_data.columns[-1]  # Assuming the last column is the label
    labels = preprocessed_data[label_col]

    # Encode labels if they are not numeric
    if not pd.api.types.is_numeric_dtype(labels):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    label_distribution = torch.bincount(torch.tensor(labels))

    plt.bar(range(len(label_distribution)), label_distribution.numpy())
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(output_folder / "label_distribution.png")
    plt.close()

    print(f"Label distribution saved to {output_folder / 'label_distribution.png'}")


if __name__ == "__main__":
    typer.run(dataset_statistics)
