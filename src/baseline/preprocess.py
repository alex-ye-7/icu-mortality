# Alexander Ye
# Baseline preprocessing: imputation + fixed-length sequences
# Usage: python -m baseline.preprocess

from common.io import get_patient_files, load_outcomes
from baseline.utils import build_dataset, save_processed_data
from config import DATA_BASELINE, STATIC_VARS, TIME_SERIES_VARS

def main():
    train_files, test_files = get_patient_files()
    train_dict, test_dict = load_outcomes()

    # Process data
    print("Processing training data...")
    X_train, y_train = build_dataset(train_files, train_dict, STATIC_VARS, TIME_SERIES_VARS)
    
    print("Processing test data...")
    X_test, y_test = build_dataset(test_files, test_dict, STATIC_VARS, TIME_SERIES_VARS)
    
    # Save
    save_processed_data(X_train, X_test, y_train, y_test, DATA_BASELINE)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    main()
