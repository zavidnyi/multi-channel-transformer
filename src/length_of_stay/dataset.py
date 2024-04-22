import os

import multiprocessing as mp
import numpy as np
import pandas as pd
import torch.utils.data


class LengthOfStayDataset(torch.utils.data.Dataset):
    cache = {}

    def __init__(
        self,
        data_dir: str,
        list_file_path: str,
        one_hot: bool,
        normalize: bool,
        discretize: bool,
    ):
        self.discretize = discretize
        self.data_dir = data_dir
        self.one_hot = one_hot
        self.normalize = normalize
        listfile = np.loadtxt(
            os.path.join(self.data_dir, list_file_path),
            delimiter=",",
            skiprows=1,
            dtype=str,
        )
        self.data_files = listfile[:, 0]
        self.lengths = np.array(listfile[:, 1]).astype(float).astype(int)
        self.targets = np.array(listfile[:, 2]).astype(float)

    def __len__(self):
        return max(700, len(self.data_files) // 100)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]

        if file_name in LengthOfStayDataset.cache:
            data = LengthOfStayDataset.cache[file_name]
        else:
            data = np.loadtxt(
                os.path.join(self.data_dir, "prepared",  file_name), delimiter=",", skiprows=1
            )
            LengthOfStayDataset.cache[file_name] = data

        return data[: self.lengths[idx]], self.targets[idx]
