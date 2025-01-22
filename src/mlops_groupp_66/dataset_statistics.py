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
    label_distribution = data['Class'].value_counts()
    plt.bar(label_distribution.index, label_distribution.values)
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("label_distribution.png")
    plt.close()

    # Plot some feature distributions
    for column in data.columns[:-1]:  # Exclude the label column
        plt.hist(data[column], bins=50)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.savefig(f"{column}_distribution.png")
        plt.close()

if __name__ == "__main__":
    typer.run(dataset_statistics)
