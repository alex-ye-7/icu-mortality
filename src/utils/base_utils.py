# Alexander Ye

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.errors import EmptyDataError
from config import STUDY_HOURS

def parse_time_to_hours(time_str):
  """Convert time string to integer hour"""
  split = time_str.split(':')
  hours = int(split[0])
  return hours

def read_patient_file(file_path, static_vars, time_series_vars):
  ''' Read single patient file and return DataFrame with one row per hour '''
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

    patient_df = pd.DataFrame(rows)

    # Turn categorical variables into ints
    for var in ['RecordID', 'Age', 'Gender', 'ICUType']:
      patient_df[var] = patient_df[var].astype('Int64')

    patient_id = patient_df['RecordID'].iloc[0] # To be used for matching to outcome
    return patient_df, patient_id
  except EmptyDataError: # ran into empty file in the dataset
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
  Params: Imputed DataFrame from impute function
  Returns: A list of lists for each patient  
  '''
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
