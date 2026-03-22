# Alexander Ye
# Standalone preprocessing script to generate .pt files

import pandas as pd
from utils.base_utils import *
from utils.triplets_utils import *
from utils.io import *
from config import DATA_RAW, DATA_PROCESSED, STATIC_VARS, TIME_SERIES_VARS

def main_baseline():
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
    X_train, y_train = build_dataset(train_files, train_dict, STATIC_VARS, TIME_SERIES_VARS)
    
    print("Processing test data...")
    X_test, y_test = build_dataset(test_files, test_dict, STATIC_VARS, TIME_SERIES_VARS)
    
    # Save
    save_processed_data(X_train, X_test, y_train, y_test, DATA_PROCESSED)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

def main_triplet():
    """Triplet-based preprocessing"""
    # Get file lists
    train_dir = DATA_RAW / "set-a"
    test_dir = DATA_RAW / "set-b"

    print("Loading patient files...")
    train_files = sorted(train_dir.glob("*.txt"))
    test_files = sorted(test_dir.glob("*.txt"))
    
    print("Loading patient outcomes...")
    outcomes_train = pd.read_csv(DATA_RAW / "Outcomes-a.txt")
    outcomes_test = pd.read_csv(DATA_RAW / "Outcomes-b.txt")
    
    train_dict = dict(zip(outcomes_train['RecordID'], outcomes_train['In-hospital_death']))
    test_dict = dict(zip(outcomes_test['RecordID'], outcomes_test['In-hospital_death']))
    
    print("\nProcessing training data (triplet format)...")
    triplets_train, y_train, feature_to_id = build_triplet_dataset(
        train_files[:800], train_dict, STATIC_VARS, TIME_SERIES_VARS
    )
    
    # Process test data
    print("\nProcessing test data (triplet format)...")
    triplets_test, y_test, _ = build_triplet_dataset(
        test_files[:800], test_dict, STATIC_VARS, TIME_SERIES_VARS
    )
    
    # Save
    save_triplet_data(triplets_train, y_train, feature_to_id, DATA_PROCESSED, split_name="train")
    save_triplet_data(triplets_test, y_test, feature_to_id, DATA_PROCESSED, split_name="test")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "triplet":
        main_triplet()
    elif len(sys.argv) > 1 and sys.argv[1] == "baseline":
        main_baseline()
    else:
        print("Usage: python preprocess.py [baseline|triplet]")