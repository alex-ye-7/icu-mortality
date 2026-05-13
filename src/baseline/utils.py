# Alexander Ye
# Baseline functions for reading patient files for imputation + fixed-length sequencing

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pandas.errors import EmptyDataError
from config import STUDY_HOURS


def read_patient_file(file_path, static_vars, time_series_vars):
  ''' Read single patient file and return DataFrame with one row per hour '''
  try:
    df = pd.read_csv(file_path)
  except EmptyDataError: 
    print(f"Error: No columns to parse from file {file_path}. Returning empty Dataframe")
    return pd.DataFrame(columns=["RecordID"]), 0

  df['Hour'] = df['Time'].str.split(':').str[0].astype(int) # Vectorized hour conversion

  # Extract static variables once (first occurance)
  static_lookup = df.drop_duplicates('Parameter').set_index('Parameter')['Value']
  static_data = {v: static_lookup.get(v, np.nan) for v in static_vars}
  record_id = int(static_data.get('RecordID', 0))

  # Pivot time-series: one row per hour, one column per parameter
  pivot = (df.groupby(['Hour', 'Parameter'])['Value']
            .mean()
            .unstack('Parameter'))
  pivot = pivot.reindex(columns=time_series_vars) # Create space for missing 

  for var in static_vars:
    pivot[var] = static_data[var] # Broadcast static data
  
  if 'Weight' in pivot.columns and 'Weight' in static_data: # Weight change may or may not be recorded 
    pivot['Weight'] = pivot['Weight'].fillna(static_data['Weight'])

  patient_df = pivot.reset_index() # Turn hour index into its own column

  for var in ['RecordID', 'Age', 'Gender', 'ICUType']:
    patient_df[var] = patient_df[var].astype(int) # Turn categorical variables into ints

  return patient_df, record_id
  
def impute(input_df):
  """Perform median/mode imputation for non observed hours """
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
  """Create fixed-length sequence from imputed DataFrame"""
  input_df = input_df.sort_values("Hour")
  study_hours = list(range(STUDY_HOURS[0], STUDY_HOURS[1]))
  # Reindex
  df_full = (
      input_df
      .set_index("Hour")
      .reindex(study_hours)
      .ffill()
      .reset_index()
  )
  df_full["Hour"] = df_full["Hour"].astype(float)
  feature_cols = [col for col in df_full.columns if col != "RecordID"]  # Drop non-feature identifiers
  return df_full[feature_cols].to_numpy(dtype=np.float32)


def build_dataset(file_list, outcomes_dict, static_vars, time_series_vars):
    """ Build X and y tensors from list of files """
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
    """ Save processed tensors to disk """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(X_train, save_dir / "X_train.pt")
    torch.save(X_test, save_dir / "X_test.pt")
    torch.save(y_train, save_dir / "y_train.pt")
    torch.save(y_test, save_dir / "y_test.pt")
    print(f"Saved processed data to {save_dir}")


def load_processed_data(load_dir):
    """ Load processed tensors from disk """
    load_dir = Path(load_dir)
    return (
        torch.load(load_dir / "X_train.pt"),
        torch.load(load_dir / "X_test.pt"),
        torch.load(load_dir / "y_train.pt"),
        torch.load(load_dir / "y_test.pt")
    )
