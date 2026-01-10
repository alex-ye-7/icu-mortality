# Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012
## Alexander Ye

This repo contains implementations of deep learning models that ingests multivariate clinical time-series data to predict mortality of ICU patients. 

### Dataset 

Download PhysioNet2012 dataset from https://physionet.org/content/challenge-2012/1.0.0/.

### Preprocessing 

Fixed 33 variables based on data explopration
Aggregate time stamps by hour, studying the 13-36 hour period 
Median and mode imputation for NA values not recorded within the study period (ie: Urine recorded at hour 13 only)
-1 imputation for NA values not recorded at all for the patient across entire stay (ie: Glucose not recorded at all for patient)
No standardization of values

### Models

LSTM
GRU?
Transformer-based?
STraTS? - will require different different preprocessing