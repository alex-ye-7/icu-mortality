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
    # Convert time to minutes 
    df['Time_min'] = df['Time'].apply(lambda x: int(x[:2]) * 60 + int(x[3:]))
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
    
    triplets_list = []
    y_list = []
    
    for file_path in tqdm(file_list):
        record_id = int(Path(file_path).stem)
        
        if record_id not in outcomes_dict:
            continue

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
    values_per_var = {}
    for triplet in triplets_list:
        if len(triplet) == 0:
            continue
        for i in range(len(triplet)):
            fid = int(triplet[i, 1].item())
            if fid not in values_per_var:
                values_per_var[fid] = []
            values_per_var[fid].append(triplet[i, 2].item())
    stats = {}
    for fid, vals in values_per_var.items():
        stats[fid] = (np.mean(vals), np.std(vals) + 1e-8)
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

            mask = triplet[:, 1] == feature_id # Find first occurrence
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
    Params:
        triplets_list: List of tensors, each shape (num_obs, 3)
        stats: Dict from compute_value_stats, mapping feature_id -> (mean, std)
    """
    for triplet in triplets_list:
        if len(triplet) == 0:
            continue
        for i in range(len(triplet)):
            fid = int(triplet[i, 1].item())
            if fid in stats:
                mean, std = stats[fid]
                triplet[i, 2] = (triplet[i, 2] - mean) / std
