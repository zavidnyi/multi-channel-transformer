import os

import numpy as np
import pandas as pd
import torch.utils.data


class InHospitalMortalityDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, list_file_path: str, one_hot: bool):
        self.data_dir = data_dir
        self.one_hot = one_hot
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

        # group measurements which occurred in the same hour
        episode["Hours"] = np.floor(episode["Hours"]).astype(int)

        continuous_column_names = [
            column
            for column in episode.columns
            if column != "Capillary refill rate" and "Glascow coma scale" not in column
        ]

        categorical_column_names = [
            column
            for column in episode.columns
            if column not in continuous_column_names
        ]

        aggregation_operations = {
            column: "mean" for column in continuous_column_names
        } | {column: "max" for column in categorical_column_names}

        episode = (
            episode.groupby("Hours").agg(aggregation_operations).reset_index(drop=True)
        )

        full_df = pd.DataFrame(index=range(48))

        # Merge full_df with episode on "Hours"
        episode = pd.merge(
            full_df, episode, left_index=True, right_on="Hours", how="left"
        )

        # Set "Hours" as index again
        episode.set_index("Hours", inplace=True)

        episode = episode.fillna(0)
        if self.one_hot:
            episode = one_hot_encode(episode, "Capillary refill rate", 2)
            episode = one_hot_encode(episode, "Glascow coma scale eye opening", 5)
            episode = one_hot_encode(episode, "Glascow coma scale motor response", 7)
            episode = one_hot_encode(episode, "Glascow coma scale verbal response", 6)
            episode = one_hot_encode(episode, "Glascow coma scale total", 16)

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
