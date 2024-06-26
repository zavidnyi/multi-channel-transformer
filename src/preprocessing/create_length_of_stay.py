import argparse
import os
import random

import numpy as np
import pandas as pd

random.seed(49297)
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Create data for length of stay prediction task."
)
parser.add_argument(
    "root_path", type=str, help="Path to root folder containing train and test sets."
)
parser.add_argument(
    "output_path", type=str, help="Directory where the created data should be stored."
)
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

sample_rate = 1.0
length = 24.0
eps = 1e-6
output_dir = os.path.join(args.output_path)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

xty_triples = []
patients: list[str] = list(
    filter(str.isdigit, os.listdir(os.path.join(args.root_path)))
)
for patient in tqdm(patients, desc="Iterating over patients in."):
    patient_folder = os.path.join(args.root_path, patient)
    patient_ts_files = list(
        filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder))
    )

    for ts_filename in patient_ts_files:
        with open(os.path.join(patient_folder, ts_filename)) as tsfile:
            lb_filename = ts_filename.replace("_timeseries", "")
            label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

            # empty label file
            if label_df.shape[0] == 0:
                print("\n\t(empty label file)", patient, ts_filename)
                continue

            los = 24.0 * label_df.iloc[0]["Length of Stay"]  # in hours
            if pd.isnull(los):
                print("\n\t(length of stay is missing)", patient, ts_filename)
                continue

            if los < length - eps:
                continue

            ts_lines = tsfile.readlines()
            header = ts_lines[0]
            ts_lines = ts_lines[1:]
            event_times = [float(line.split(",")[0]) for line in ts_lines]

            ts_lines = [
                line for (line, t) in zip(ts_lines, event_times) if -eps < t < length + eps
            ]
            event_times = [t for t in event_times if -eps < t < length + eps]

            # no measurements in ICU
            if len(ts_lines) == 0:
                print("\n\t(no events in ICU) ", patient, ts_filename)
                continue

            output_ts_filename = patient + "_" + ts_filename
            with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                outfile.write(header)
                for line in ts_lines:
                    outfile.write(line)

            xty_triples.append(
                (
                    output_ts_filename,
                    24,
                    np.digitize(
                        los - 24, [24, 48, 72, 96, 120, 144, 168, 336, np.inf]
                    ),
                )
            )

print("Number of created samples:", len(xty_triples))
labels, counts = np.unique([y for x, t, y in xty_triples], return_counts=True)
for value, count in zip(labels, counts):
    print(f"Value: {value}, Count: {count}")

with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
    listfile.write("stay,period_length,y_true\n")
    for x, t, y in xty_triples:
        listfile.write("{},{:.6f},{:.6f}\n".format(x, t, y))
