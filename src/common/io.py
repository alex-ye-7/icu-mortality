# Alexander Ye
# Shared I/O utilities

import pandas as pd
from zipfile import ZipFile
from config import DATA_RAW

def extract_raw_data():
    """Extract raw zip files. Only needs to be run once.""" # python io
    for zip_name in ["set-a.zip", "set-b.zip"]:
        zip_path = DATA_RAW / zip_name
        print(f"Extracting {zip_path}...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_RAW)
        print(f"{zip_name} extracted")

def load_outcomes():
    """Load outcome labels for train and test sets.

    Returns:
        train_dict: {RecordID: In-hospital_death} for set-a
        test_dict: {RecordID: In-hospital_death} for set-b
    """
    outcomes_train = pd.read_csv(DATA_RAW / "Outcomes-a.txt")
    outcomes_test = pd.read_csv(DATA_RAW / "Outcomes-b.txt")
    train_dict = dict(zip(outcomes_train['RecordID'], outcomes_train['In-hospital_death']))
    test_dict = dict(zip(outcomes_test['RecordID'], outcomes_test['In-hospital_death']))
    return train_dict, test_dict

def get_patient_files():
    """Get sorted lists of patient file paths.

    Returns:
        train_files: Sorted list of paths in set-a
        test_files: Sorted list of paths in set-b
    """
    train_files = sorted((DATA_RAW / "set-a").glob("*.txt"))
    test_files = sorted((DATA_RAW / "set-b").glob("*.txt"))
    return train_files, test_files

if __name__ == "__main__":
    extract_raw_data()
