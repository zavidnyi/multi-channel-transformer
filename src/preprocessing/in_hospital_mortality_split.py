import os.path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

np.random.seed(32)

parse = ArgumentParser()
parse.add_argument("--data_path", type=str, default="../data/in_hospital_mortality/")
args = parse.parse_args()

labels_file = pd.read_csv(os.path.join(args.data_path, "listfile.csv"))

negative_samples = labels_file[labels_file["y_true"] == 0]
positive_samples = labels_file[labels_file["y_true"] == 1]


def split(data: pd.DataFrame, ratio: float) -> (pd.DataFrame, pd.DataFrame):
    split_index = int(np.floor(len(data) * ratio))
    return data[0:split_index], data[split_index:]


(test_negative_samples, train_negative_samples) = split(negative_samples, 0.15)
(test_positive_samples, train_positive_samples) = split(positive_samples, 0.15)

(test_negative_samples, val_negative_samples) = split(test_negative_samples, 0.8)
(test_positive_samples, val_positive_samples) = split(test_positive_samples, 0.8)

test_samples = pd.concat([test_negative_samples, test_positive_samples])
test_samples = test_samples.iloc[
    np.random.permutation(np.arange(len(test_samples)))
]  # shuffle
train_samples = pd.concat([train_negative_samples, train_positive_samples])
train_samples = train_samples.iloc[
    np.random.permutation(np.arange(len(train_samples)))
]  # shuffle
val_samples = pd.concat([val_negative_samples, val_positive_samples])
val_samples = val_samples.iloc[
    np.random.permutation(np.arange(len(val_samples)))
]  # shuffle

print("TEST: positive samples: ", len(test_positive_samples))
print("TEST: negative samples: ", len(test_negative_samples))
print(
    "TEST: ratio of positive samples: ",
    len(test_positive_samples)
    / (len(test_positive_samples) + len(test_negative_samples)),
)

print("TRAIN: positive samples: ", len(train_positive_samples))
print("TRAIN: negative samples: ", len(train_negative_samples))
print(
    "TRAIN: ratio of positive samples: ",
    len(train_positive_samples)
    / (len(train_positive_samples) + len(train_negative_samples)),
)

print("VAL: positive samples: ", len(val_positive_samples))
print("VAL: negative samples: ", len(val_negative_samples))
print(
    "VAL: ratio of positive samples: ",
    len(val_positive_samples) / (len(val_positive_samples) + len(val_negative_samples)),
)

test_samples.to_csv("../data/in_hospital_mortality/test_listfile.csv", index=False)
train_samples.to_csv("../data/in_hospital_mortality/train_listfile.csv", index=False)
val_samples.to_csv("../data/in_hospital_mortality/val_listfile.csv", index=False)
