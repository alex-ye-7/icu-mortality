# Alexander Ye
# Input output

import torch
from pathlib import Path

def save_processed_data(X_train, X_test, y_train, y_test, save_dir):
    """
    Save processed tensors to disk
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(X_train, save_dir / "X_train.pt")
    torch.save(X_test, save_dir / "X_test.pt")
    torch.save(y_train, save_dir / "y_train.pt")
    torch.save(y_test, save_dir / "y_test.pt")
    print(f"Saved processed data to {save_dir}")


def load_processed_data(load_dir):
    """
    Load processed tensors from disk
    """
    load_dir = Path(load_dir)
    return (
        torch.load(load_dir / "X_train.pt"),
        torch.load(load_dir / "X_test.pt"),
        torch.load(load_dir / "y_train.pt"),
        torch.load(load_dir / "y_test.pt")
    )


def save_triplet_data(triplets_list, y, feature_to_id, save_dir, split_name="train"):
    """
    Save triplet dataset to disk.
    
    Params:
        triplets_list: List of variable-length tensors
        y: Outcome labels tensor
        feature_to_id: Feature mapping dictionary
        save_dir: Directory to save files
        split_name: 'train' or 'test'
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(triplets_list, save_dir / f"triplets_{split_name}.pt")
    torch.save(y, save_dir / f"y_{split_name}.pt")
    torch.save(feature_to_id, save_dir / f"feature_to_id_{split_name}.pt")
    print(f"Saved {len(triplets_list)} patients from {split_name} triplet data to {save_dir}")


def load_triplet_data(load_dir, split_name="train"):
    """
    Load triplet dataset from disk.

    Returns:
        triplets_list: List of variable-length tensors
        y: Outcome labels
        feature_to_id: Feature mapping
    """
    load_dir = Path(load_dir)

    triplets_list = torch.load(load_dir / f"triplets_{split_name}.pt")
    y = torch.load(load_dir / f"y_{split_name}.pt")
    feature_to_id = torch.load(load_dir / f"feature_to_id_{split_name}.pt")

    return triplets_list, y, feature_to_id


def save_value_stats(stats, save_dir):
    """Save normalization stats to disk."""
    save_dir = Path(save_dir)
    torch.save(stats, save_dir / "value_stats.pt")


def load_value_stats(load_dir):
    """Load normalization stats from disk."""
    return torch.load(Path(load_dir) / "value_stats.pt", weights_only=False)
