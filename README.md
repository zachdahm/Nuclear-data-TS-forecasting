This repository contains normalized datasets, trained models, and the code used for the paper "One-Class Anomaly Detection in Nuclear Reactor Time-Series"

DATASETS - All datasets are provided as .csv titles with column headers and are loaded into the code as dataframes.
real_dataset_normalized.csv - Normalized full reactor power cycle dataset, 200,000 rows of the 8 features used in the paper plus an index used for identifying different operation periods. This dataset is split into the training, validation, and testing datasets in the example code.
transient_normal.csv - Normalized transient dataset, subset of testing dataset only containing data where the neutron change rate exceeds 0.1%/sec
scrams_normal.csv - Normalized scrams dataset, contains data from 11 scrams of PUR-1
FDI1_normal.csv - Normalized FDI #1 dataset, with control rod positions and active states falsified
FDI2_normal.csv - Normalized FDI #2 dataset, with control rod positions and active states, and neutron count change rate falsified
MODELS
GRU_Final.pt - State dictionary of trained GRU network, in order to use must have a GRU_Model class defined like the one in the example code
forecast_journal_scaler.pt - Min-max scaler fitted to only the training dataset, unneccesary for the example code because all of the provided data is normalized

CODE
Forecasting_upload.py - Python code for running the project, includes all data preprocessing, option for hyperparameter tuning and model selection, training, and evaluation using preloaded model.
