import os

import numpy as np
import pandas as pd
import torch.utils.data

from src.mimic.prepare_data import prepare_data


class MimicTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        data_in_dir_processed: bool,
        list_file_path: str,
        max_seq_len: int,
        one_hot: bool,
        normalize: bool,
        discretize: bool,
        small: bool,
    ):
        self.discretize = discretize
        self.data_dir = data_dir
        self.data_in_dir_processed = data_in_dir_processed
        self.one_hot = one_hot
        self.normalize = normalize
        self.max_seq_len = max_seq_len
        self.small = small
        listfile = np.loadtxt(
            list_file_path,
            delimiter=",",
            skiprows=1,
            dtype=str,
        )
        self.data_files = listfile[:, 0]
        self.cache = {}
        self.targets = listfile[:, -1]

    def __len__(self):
        if self.small:
            return len(self.data_files) // 10

        return len(self.data_files)

    def __getitem__(self, idx):
        return self.process_data(idx)

    def process_data(self, index: int):
        if index in self.cache:
            data = self.cache[index]
        else:
            data = pd.read_csv(
                os.path.join(self.data_dir, self.data_files[index]),
            )
            if self.data_in_dir_processed is None:
                data = prepare_data(
                    data,
                    self.max_seq_len,
                    self.discretize,
                    self.normalize,
                    self.one_hot,
                )
            self.cache[index] = data

        t = torch.int if self.discretize else torch.float
        return (
            torch.tensor(data.values, dtype=t),
            torch.tensor(float(self.targets[index]), dtype=torch.long),
        )
