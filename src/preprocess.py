# Alexander Ye
# Standalone preprocessing script to generate .pt files

from pathlib import Path
import pandas as pd
from data_processing import build_dataset, save_processed_data
from config import DATA_RAW, DATA_PROCESSED, STATIC_VARS, TIME_SERIES_VARS

def main():
    # Get file lists
    train_dir = DATA_RAW / "set-a"
    test_dir = DATA_RAW / "set-b"

    print("Loading patient information...")
    train_files = sorted(train_dir.glob("*.txt"))
    test_files = sorted(test_dir.glob("*.txt"))
    
    print("Loading patient outcomes...")
    outcomes_train = pd.read_csv(DATA_RAW / "Outcomes-a.txt")
    outcomes_test = pd.read_csv(DATA_RAW / "Outcomes-b.txt")
    
    train_dict = dict(zip(outcomes_train['RecordID'], outcomes_train['In-hospital_death']))
    test_dict = dict(zip(outcomes_test['RecordID'], outcomes_test['In-hospital_death']))
    
    # Process data
    print("Processing training data...")
    X_train, y_train = build_dataset(train_files[:2000], train_dict, STATIC_VARS, TIME_SERIES_VARS)
    
    print("Processing test data...")
    X_test, y_test = build_dataset(test_files[:2000], test_dict, STATIC_VARS, TIME_SERIES_VARS)
    
    # Save
    save_processed_data(X_train, X_test, y_train, y_test, DATA_PROCESSED)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    main()