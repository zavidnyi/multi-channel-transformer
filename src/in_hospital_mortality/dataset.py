import os

import numpy as np
import pandas as pd
import torch.utils.data


class InHospitalMortalityDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, list_file_path: str):
        self.data_dir = data_dir
        listfile = np.loadtxt(
            os.path.join(self.data_dir, list_file_path),
            delimiter=",",
            skiprows=1,
            dtype=str,
        )
        self.data_files = listfile[:, 0]
        self.targets = listfile[:, 1]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return self.process_data(idx)

    def process_data(self, index: int):
        episode = pd.read_csv(
            os.path.join(self.data_dir, self.data_files[index]),
        )

        episode = episode[: min(48, len(episode))]

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

        if len(episode) < 48:
            empty_measurements = pd.DataFrame(
                0, index=range(48 - len(episode)), columns=episode.columns
            )
            episode = pd.concat([empty_measurements, episode])

        return (
            torch.tensor(episode.values, dtype=torch.float),
            torch.tensor(int(self.targets[index]), dtype=torch.float),
        )


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
