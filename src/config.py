# src/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "_processed"

# Variables extracted from https://physionet.org/content/challenge-2012/1.0.0/
# Then narrowed down to 33 relevent variables upon data exploration
STATIC_VARS = [
    'RecordID',    # Unique integer for each ICU stay
    'Age',         # Years
    'Gender',      # 0: female, 1: male
    'Height',      # cm
    'ICUType',     # 1: CCU, 2: CSRU, 3: Medical ICU, 4: Surgical ICU
    'Weight'       # kg (can vary over time)
]

# Time Series Variables (observed at least 3 times in over 40% of patients)
TIME_SERIES_VARS = [
    'BUN',         # Blood urea nitrogen (mg/dL)
    'Creatinine',  # Serum creatinine (mg/dL)
    'DiasABP',     # Invasive diastolic arterial blood pressure (mmHg)
    'FiO2',        # Fractional inspired O2 (0-1)
    'GCS',         # Glasgow Coma Score (3-15)
    'Glucose',     # Serum glucose (mg/dL)
    'HCO3',        # Serum bicarbonate (mmol/L)
    'HCT',         # Hematocrit (%)
    'HR',          # Heart rate (bpm)
    'K',           # Serum potassium (mEq/L)
    'Lactate',     # mmol/L
    'Mg',          # Serum magnesium (mmol/L)
    'MAP',         # Invasive mean arterial blood pressure (mmHg)
    'MechVent',    # Mechanical ventilation (0: false, 1: true)
    'Na',          # Serum sodium (mEq/L)
    'NISysABP',    # Non-invasive systolic arterial blood pressure (mmHg)
    'NIDiasABP',   # Non-invasive diastolic arterial blood pressure (mmHg)
    'NIMAP',       # Non-invasive mean arterial blood pressure (mmHg)
    'PaCO2',       # Partial pressure of arterial CO2 (mmHg)
    'PaO2',        # Partial pressure of arterial O2 (mmHg)
    'pH',          # Arterial pH (0-14)
    'Platelets',   # cells/nL
    'SaO2',        # O2 saturation in hemoglobin (%)
    'SysABP',      # Invasive systolic arterial blood pressure (mmHg)
    'Temp',        # Temperature
    'Urine',       # Urine output (mL)
    'WBC',         # White blood cell count (cells/nL)
]

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
D_MODEL = 64
N_HEAD = 4
DIM_FF = 128

# Study parameters
STUDY_HOURS = (13, 37)