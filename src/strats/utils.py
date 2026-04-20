# Alexander Ye

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pandas.errors import EmptyDataError
from config import DEMO_FEATURES, DEMO_NORMALIZERS

def create_triplets(file_path, feature_to_id):
    """
    Create triplet representation directly from raw patient CSV file
    Params: file_path to csv, feature_to_id dictionary mapping features to ID
    Returns: torch.Tensor of shape (num_observations, 3)
    """
    try:
        df = pd.read_csv(file_path)
    except EmptyDataError: # one of the files is empty for some reason
        return torch.zeros((0, 3), dtype=torch.float32)

    triplets = []
    df['Time_min'] = df['Time'].apply(lambda x: int(x[:2]) * 60 + int(x[3:])) # Convert time to minutes 
    df = df.sort_values("Time_min")
    
    filter = df['Parameter'].isin(feature_to_id) & df['Value'].notna() & (df['Value'] >= 0)
    filtered = df[filter]
    feature_ids = filtered['Parameter'].map(feature_to_id)
    triplets = torch.tensor(
       np.column_stack((filtered['Time_min'].values, feature_ids.values, filtered['Value'].values)),
       dtype=torch.float32)
    return triplets


def build_triplet_dataset(file_list, outcomes_dict, static_vars, time_series_vars):
    """
    Build dataset in triplet format directly from raw CSV files
    
    Params: 
        file_list of patient paths
        outcomes_dict mapping RecordID to 0 or 1
        static_vars: List of static variable names
        time_series_vars: List of time-series variable names
    
    Returns:
        triplets_list: List of variable-length tensors, each shape (num_obs, 3)
        y: Tensor of outcomes
        feature_to_id: Mapping from feature name to feature ID
    """
    all_features = static_vars + time_series_vars
    feature_to_id = {name: idx for idx, name in enumerate(all_features)}
    triplets_list, y_list = [], []
    for file_path in tqdm(file_list):
        record_id = int(Path(file_path).stem)
        if record_id not in outcomes_dict: continue

        triplets = create_triplets(file_path, feature_to_id)

        if len(triplets) > 0:
            triplets_list.append(triplets)
            y_list.append(outcomes_dict[record_id])

    y = torch.tensor(y_list, dtype=torch.float32)
    return triplets_list, y, feature_to_id


def compute_value_stats(triplets_list):
    """
    Compute per-variable mean and std from triplet data (call on training set only).
    Params: triplets_list: List of tensors, each shape (num_obs, 3)
    Returns: Dict mapping feature_id -> (mean, std)
    """
    stats = {}
    all_triplets = torch.cat([t for t in triplets_list if len(t) > 0], dim=0)
    fids = all_triplets[:, 1].unique().int().tolist()
    for fid in fids: 
        vals = all_triplets[all_triplets[:, 1] == fid, 2]
        stats[fid] = (vals.mean().item(), vals.std().item() + 1e-08)
    return stats


def normalize_demographics(triplets_list, feature_to_id):
    """
    Extract demographic features from triplet data and normalize.
    
    Args: triplets_list: List of tensors with shape (num_obs, 3), feature_to_id dictionary 
    Returns: Normalized demographics (n_samples, D)
    """
    n_samples = len(triplets_list)
    D = len(DEMO_FEATURES)
    demo_array = np.zeros((n_samples, D))
    
    for idx, triplet in enumerate(triplets_list):
        if len(triplet) == 0: continue
        
        # Extract all feature values for this patient
        for feature_idx, feature_name in enumerate(DEMO_FEATURES):
            if feature_name not in feature_to_id: continue
                
            feature_id = feature_to_id[feature_name]

            mask = triplet[:, 1] == feature_id
            if mask.any():
                value = triplet[mask, 2][0].item()  # Get first occurrence
                
                # Normalize to [0, 1]
                min_val, max_val = DEMO_NORMALIZERS[feature_name]
                normalized = (value - min_val) / (max_val - min_val)
                normalized = np.clip(normalized, 0, 1)
                demo_array[idx, feature_idx] = normalized

    return demo_array


def normalize_triplet_values(triplets_list, stats):
    """
    Z-score normalize the value column (col 2) of each triplet in-place.
    Uses lookup tensors for vectorized normalization (no inner loops).
    Params:
        triplets_list: List of tensors, each shape (num_obs, 3)
        stats: Dict from compute_value_stats, mapping feature_id -> (mean, std)
    """
    max_fid = max(stats.keys()) + 1
    means = torch.zeros(max_fid)
    stds = torch.ones(max_fid)
    for fid, (m, s) in stats.items():
        means[fid] = m
        stds[fid] = s

    for triplet in triplets_list:
        if len(triplet) == 0:
            continue
        fids = triplet[:, 1].long()
        triplet[:, 2] = (triplet[:, 2] - means[fids]) / stds[fids]


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
