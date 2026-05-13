# Alexander Ye
# Triplet preprocessing 
# Usage: python -m strats.preprocess

from common.io import get_patient_files, load_outcomes
from strats.utils import build_triplet_dataset, save_triplet_data
from config import DATA_TRIPLET, STATIC_VARS, TIME_SERIES_VARS, N_EXAMPLE

def main():
    """Triplet-based preprocessing"""
    train_files, test_files = get_patient_files()
    train_dict, test_dict = load_outcomes()

    print("\nProcessing training data (triplet format)...")
    triplets_train, y_train, feature_to_id = build_triplet_dataset(
        train_files[:N_EXAMPLE], train_dict, STATIC_VARS, TIME_SERIES_VARS
    )
    
    # Process test data
    print("\nProcessing test data (triplet format)...")
    triplets_test, y_test, _ = build_triplet_dataset(
        test_files[:N_EXAMPLE], test_dict, STATIC_VARS, TIME_SERIES_VARS
    )
    
    # Save (which will save triplets, y, feature_to_id for both train and test)
    save_triplet_data(triplets_train, y_train, feature_to_id, DATA_TRIPLET, split_name="train")
    save_triplet_data(triplets_test, y_test, feature_to_id, DATA_TRIPLET, split_name="test")


if __name__ == "__main__":
    main()
    