import os.path

import lightning as L
import numpy as np
import pandas as pd
import torch.utils.data
from numpy import ndarray, dtype
from torch.utils.data import DataLoader


class InHospitalMortalityDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        self.train_data = self.load_for_listfile("train_listfile.csv")
        self.val_data = self.load_for_listfile("val_listfile.csv")
        self.test_data = self.load_for_listfile("test_listfile.csv")

    def load_for_listfile(self, list_file_path):
        loaded_list_file: ndarray[str, dtype[str]] = np.loadtxt(
            os.path.join(self.data_dir, list_file_path),
            delimiter=",",
            skiprows=1,
            dtype=str,
        )
        data = [
            pd.read_csv(
                os.path.join(self.data_dir, episode_file),
            )
            for episode_file in loaded_list_file[:753, 0]
        ]
        gcs_eye_mapping = {
            "No Response": 1,
            "To Pain": 2,
            "To Speech": 3,
            "Spontaneously": 4,
        }
        gcs_verbal_mapping = {
            "No Response": 1,
            "Incomprehensible sounds": 2,
            "Inappropriate Words": 3,
            "Confused": 4,
            "Oriented": 5,
        }
        gcs_motor_mapping = {
            "No Response": 1,
            "Abnormal extension": 2,
            "Abnormal Flexion": 3,
            "Flex-withdraws": 4,
            "Localizes Pain": 5,
            "Obeys Commands": 6,
        }

        def sliding_window_iter(series, size):
            """series is a column of a dataframe"""
            for start_row in range(0, len(series), size):
                if (start_row + size) >= len(series) - 1:
                    left_index = len(series) - 1 - size
                    assert left_index + size == len(series) - 1
                    yield series[left_index : left_index + size]
                    break
                else:
                    yield series[start_row : start_row + size]

        def one_hot_encode(df, column, num_classes):
            one_hot_encoded_tensor = torch.nn.functional.one_hot(
                torch.tensor(df[column].values).to(torch.int64), num_classes=num_classes
            )
            one_hot_encoded_df = pd.DataFrame(
                one_hot_encoded_tensor.numpy(),
                columns=[f"{column}_{i}" for i in range(num_classes)],
            )
            df.drop(column, axis=1, inplace=True)
            df = df.join(one_hot_encoded_df)
            return df

        samples = None
        labels = None
        for ep_index, episode in enumerate(data):
            episode = episode.drop("Hours", axis=1)
            episode["Glascow coma scale eye opening"] = episode[
                "Glascow coma scale eye opening"
            ].map(gcs_eye_mapping)
            episode["Glascow coma scale motor response"] = episode[
                "Glascow coma scale motor response"
            ].map(gcs_motor_mapping)
            episode["Glascow coma scale verbal response"] = episode[
                "Glascow coma scale verbal response"
            ].map(gcs_verbal_mapping)

            episode["Glascow coma scale total"] = (
                episode["Glascow coma scale eye opening"]
                + episode["Glascow coma scale motor response"]
                + episode["Glascow coma scale verbal response"]
            )

            episode = episode.fillna(0)

            episode = one_hot_encode(episode, "Capillary refill rate", 2)
            episode = one_hot_encode(episode, "Glascow coma scale eye opening", 5)
            episode = one_hot_encode(episode, "Glascow coma scale motor response", 7)
            episode = one_hot_encode(episode, "Glascow coma scale verbal response", 6)
            episode = one_hot_encode(episode, "Glascow coma scale total", 16)

            # if torch.tensor(episode.values).isnan().any():
            #     print(episode)

            windows = [window for window in sliding_window_iter(episode, 48)]

            for index, window in enumerate(windows):
                if len(window) < 48:
                    empty_measurements = pd.DataFrame(
                        0, index=range(48 - len(window)), columns=window.columns
                    )
                    windows[index] = pd.concat([empty_measurements, window])
            if samples is None:
                samples = np.array([w.values for w in windows])
                labels = np.repeat(int(loaded_list_file[ep_index][1]), len(windows))
            else:
                samples = np.concatenate(
                    [samples, np.array([w.values for w in windows])]
                )
                labels = np.concatenate(
                    [
                        labels,
                        np.repeat(int(loaded_list_file[ep_index][1]), len(windows)),
                    ]
                )

        return torch.utils.data.TensorDataset(
            torch.tensor(samples, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float),
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
