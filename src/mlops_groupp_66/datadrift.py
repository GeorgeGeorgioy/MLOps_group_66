import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable
from dotenv import load_dotenv
import os
from pathlib import Path


def detect_data_drift(main_data_path: str, balanced_data_path: str, drift_report_path: str):
    """Detect data drift between the main dataset and the balanced dataset."""
    print("Loading datasets...")

    # Load datasets
    main_data = pd.read_csv(main_data_path)
    balanced_data = pd.read_csv(balanced_data_path)

    # Prepare a representative subset from the main dataset
    target_column = main_data.columns[-1]  # Get the last column as the target
    frauds = main_data[main_data[target_column] == 1]
    non_frauds = main_data[main_data[target_column] == 0]
    non_frauds_sampled = non_frauds.sample(len(frauds) * 9, random_state=42)  # Adjust ratio if needed
    representative_main = pd.concat([frauds, non_frauds_sampled])

    print("Generating data drift report...")
    # Create Evidently data drift report
    data_drift_report = Report(metrics=[DataDriftTable()])
    data_drift_report.run(reference_data=representative_main, current_data=balanced_data)

    # Save the HTML report
    data_drift_report.save_html(str(drift_report_path))  # Convert Path to string
    print(f"Data drift report saved to {drift_report_path}")

    # Extract drift report data as a dictionary (which includes metrics)
    drift_report_data = data_drift_report.as_dict()
    print("Drift Report Data:")
    print(drift_report_data)  # Prints the drift metrics in a structured format

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Paths from .env
    main_data_path = Path(os.getenv("ORIGINAL_DATA")).resolve()
    balanced_data_path = Path(os.getenv("RAW_DATA")).resolve()
    drift_report_path = Path(os.getenv("DRIFT_REPORT")).resolve()

    # Run drift detection
    detect_data_drift(main_data_path, balanced_data_path, drift_report_path)
