import matplotlib.pyplot as plt
import typer
from pathlib import Path
from mlops_groupp_66.data import MyDataset

def dataset_statistics(raw_data_path: str = "data/raw/balanced_creditcard.csv") -> None:
    """Compute dataset statistics."""
    raw_data_path = Path(raw_data_path).resolve()
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(Path("."))

    data = dataset.data
    print(f"Dataset: {raw_data_path.name}")
    print(f"Number of samples: {len(data)}")
    print(f"Columns: {data.columns.tolist()}")
    print("\n")

    # Show some sample data
    print("Sample data:")
    print(data.head())

    # Plot label distribution
    plt.figure(figsize=(20, 15))
    plt.subplot(4, 8, 1)
    label_distribution = data['Class'].value_counts()
    plt.bar(label_distribution.index, label_distribution.values)
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")

    # Plot feature distributions
    for i, column in enumerate(data.columns[:-1], start=2):  # Exclude the label column
        plt.subplot(4, 8, i)
        plt.hist(data[column], bins=50, alpha=0.75)
        plt.title(f"{column} distribution")
        plt.xlabel(column)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("data_statistics.png")
    plt.close()

if __name__ == "__main__":
    typer.run(dataset_statistics)
