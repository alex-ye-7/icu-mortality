# Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012
## Alexander Ye

This repo contains implementations of deep learning models that ingests irregularly-sampled multivariate time-series data to predict mortality of ICU patients. 

### Dataset 

Download PhysioNet2012 dataset from https://physionet.org/content/challenge-2012/1.0.0/.

Each patient has a CSV file that looks like:

<img src="data_sample.png" alt="Alt text" width="150" height="250">

I treated set-a as the training set and set-b as the test set.

### Establishing Baseline

To establish a baseline for predictions, I pursued a standard hourly-aggregation and imputation approach. That is, fixing an hours by features matrix to capture the data. To preprocess,

- Fixed 33 variables based on data explopration
- Aggregate time stamps by hour, studying the 13-36 hour period 
- Median and mode imputation for NA values not recorded within the study period (ie: Urine recorded at hour 13 only)
- -1 imputation for NA values not recorded at all for the patient across entire stay (ie: Glucose not recorded at all for patient)
- No standardization of values 

Model performance was evaluated using AUROC and AUPRC, which are threshold-independent metrics. Given the class imbalance (~14% mortality), AUPRC was emphasized as it is more informative than AUROC in imbalanced settings.

|  | GRU | LSTM | Transformer|
|---   |---  |---   | ---  |
|AUROC | 0.7750 | 0.7846 | 0.7915 |
|AUPRC | 0.3840 | 0.4073 | 0.3779 |

### The Triplet Representation

The triplet representation inspired from STraTS and TransEHR eliminates the need for time discretization and missing value imputation. The observation triplet is defined as the triple (t, f, v) where t is the time, f is feature or variable of interest, and v is the value of the observation. 

- Feature embeddings are obtained from a simple lookup table similar to word embeddings. 
- Values and times were embedded using continous value embedding technique using one-to-many Feed-Forward Network (FFN) with learnable parameters. The FFNs have one input neuron and $d$ output neurons and a single hidden layer with $\sqrt{d}$ neurons and $\tanh(.)$ activation. (Can also experiment with sinusoidal encodings to embed time)