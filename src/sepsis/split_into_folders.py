import os
from tqdm import tqdm
import pandas as pd
# TODO: this has to be removed and made a part of sepsis processing scrtip/notebook
data_directory = 'data/challenge-2019/processed_samples'  # Replace with the actual path to your data directory

for filename in tqdm(os.listdir(data_directory), desc="Loading data"):
    if filename.endswith('.csv'):
        patient_sample = pd.read_csv(os.path.join(data_directory, filename))
        last_row = patient_sample.iloc[-1]
        sepsis_label = last_row['SepsisLabel']
        if sepsis_label == 0:
            patient_sample.to_csv(os.path.join(data_directory, "no-sepsis", filename))
        if sepsis_label == 1:
            patient_sample.to_csv(os.path.join(data_directory, "sepsis", filename))
