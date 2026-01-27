# Alexander Ye
# Preprocessing functions

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from pandas.errors import EmptyDataError


def parse_time_to_hours(time_str):
  """
  Convert time string to integer hour 

  Params: A string of time from dataset (HH:MM)
  """
  split = time_str.split(':')
  hours = int(split[0])
  return hours

def read_patient_file(file_path, static_vars, time_series_vars):
  '''
  Read a single patient file and return a DataFrame with one row per hour

  Params: Path to the patient data file
  Returns: Patient DataFrame by hour and all variables
  '''
  try:
    df = pd.read_csv(file_path) # Read the file

    df['Hour'] = df['Time'].apply(parse_time_to_hours) # Convert time string to hourly

    all_hours = sorted(df['Hour'].unique()) # All unique hours of data
    # study_hours = np.arange(13,37)
    rows = [] # One row for every hour

    # Extract static variables once
    static_data = {}
    for var in static_vars:
      if var in df['Parameter'].values:
        static_data[var] = df[df['Parameter'] == var]['Value'].iloc[0]
      else:
        static_data[var] = np.nan

    for hour in all_hours:
      row = {'Hour': hour }
      hour_data = df[df['Hour'] == hour] # Select measurements within this hour

      # Add static variables (same for every hour)
      for var in static_vars:
        row[var] = static_data.get(var, np.nan)
      # Except change if weight changes
      if hour_data[hour_data['Parameter'] == 'Weight']['Value'].any():
        row['Weight'] = hour_data[hour_data['Parameter'] == 'Weight']['Value'].mean()

      # For time-varying variabels
      for var in time_series_vars:
        var_measurements = hour_data[hour_data['Parameter'] == var]['Value']
        if len(var_measurements) > 0:
          row[var] = var_measurements.mean() # mean if there's multiple
        else:
          row[var] = np.nan

      rows.append(row)

    patient_df = pd.DataFrame(rows) # Now make the DataFrame

    # Turn categorical variables into ints
    for var in ['RecordID', 'Age', 'Gender', 'ICUType']:
      patient_df[var] = patient_df[var].astype('Int64')

    patient_id = patient_df['RecordID'].iloc[0] # To be used for matching to outcome

    return patient_df, patient_id

  except EmptyDataError: # ran into empty file
    print(f"Error: No columns to parse from file {file_path}. Returning empty Dataframe")
    return pd.DataFrame(columns=["RecordID"]), 0
  
def impute(input_df):
  """
  Perform median/mode imputation for non observed hours 

  Params: Loaded in Dataframe from read_patient_file
  Returns: Imputed Dataframe
  """
  imputed_df = input_df.copy()
  categorical_cols = ['MechVent']
  continuous_cols = [
      'Age', 'Height', 'Weight', 'BUN', 'Creatinine', 'DiasABP', 'FiO2', 'GCS',
      'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'Na', 
      'NISysABP', 'NIDiasABP', 'NIMAP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 
      'SaO2', 'SysABP', 'Temp', 'Urine', 'WBC'
  ]

  # Impute continuous columns with median
  for col in continuous_cols:
      if col in imputed_df.columns:
          median_val = imputed_df[col].median()
          imputed_df.fillna({col: median_val}, inplace=True)

  # Impute categorical columns with mode
  for col in categorical_cols:
      if col in imputed_df.columns:
          mode_series = imputed_df[col].mode()
          if not mode_series.empty:
            mode_val = mode_series.iloc[0]
            imputed_df.fillna({col: mode_val}, inplace=True)

  return imputed_df

def create_patient_sequence(input_df):
  '''
  Converts DataFrame into 

  Params: Imputed DataFrame from impute function
  Returns: A list of lists for each patient  
  '''
  input_df = input_df.sort_values("Hour")
  study_hours = list(range(13, 37))
  # Reindex
  df_full = (
      input_df
      .set_index("Hour")
      .reindex(study_hours)
      .ffill()
      .reset_index()
  )
  df_full["Hour"] = df_full["Hour"].astype(float)

  # Drop non-feature identifiers
  feature_cols = [col for col in df_full.columns if col != "RecordID"]

  return df_full[feature_cols].to_numpy(dtype=np.float32)

def build_dataset(file_list, outcomes_dict, static_vars, time_series_vars):
    """
    Build X and y tensors from list of files
    """
    X, y = [], []
    for file_path in tqdm(file_list):
        patient_df, record_id = read_patient_file(file_path, static_vars, time_series_vars)
        if patient_df.empty:
            continue
        imputed_df = impute(patient_df)
        patient_seq = create_patient_sequence(imputed_df)
        patient_seq = np.nan_to_num(patient_seq, nan=-1.0)
        X.append(torch.tensor(patient_seq, dtype=torch.float32))
        y.append(outcomes_dict[record_id])
    
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)

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

def create_triplets(file_path, feature_to_id):
    """
    Create triplet representation directly from raw patient CSV file
    
    Params: file_path to csv, feature_to_id dictionary mapping features to ID
    Returns: torch.Tensor of shape (num_observations, 3), recordID of type int
    """
    try:
        df = pd.read_csv(file_path)
    except EmptyDataError: # one of the files is empty for some reason
        return torch.zeros((0, 3), dtype=torch.float32), 0
    
    patient_id = int(df['Value'][df['Parameter']=='RecordID'].iloc[0]) # To be used for matching to outcome

    triplets = []
    # Convert time to minutes 
    df['Time_min'] = df['Time'].apply(lambda x: int(x[:2]) * 60 + int(x[3:]))
    df = df.sort_values("Time_min")
    
    # Extract all observations in triplet format
    for _, row in df.iterrows():
        time_min = row['Time_min']
        feature_name = row['Parameter']
        value = row['Value']
        
        # Skip unknown parameters, na values, and -1's
        if feature_name in feature_to_id and not pd.isna(value) and value>=0:
            feature_id = feature_to_id[feature_name]
            triplets.append([time_min, feature_id, float(value)])

    return torch.tensor(triplets, dtype=torch.float32), patient_id


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
        id_to_feature: Mapping from feature ID to feature name
    """
    all_features = static_vars + time_series_vars
    feature_to_id = {name: idx for idx, name in enumerate(all_features)}
    id_to_feature = {idx: name for name, idx in feature_to_id.items()}
    
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
    
    return triplets_list, y, feature_to_id, id_to_feature


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
    
    # Save triplets as list (preserves variable lengths)
    torch.save(triplets_list, save_dir / f"triplets_{split_name}.pt")
    # Save outcomes
    torch.save(y, save_dir / f"y_{split_name}.pt")
    # Save feature mapping
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