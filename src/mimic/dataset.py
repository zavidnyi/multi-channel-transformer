import os

import numpy as np
import pandas as pd
import torch.utils.data


class InHospitalMortalityDataset(torch.utils.data.Dataset):
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
        self.cache = {}
        self.targets = listfile[:, 1]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return self.process_data(idx)

    def process_data(self, index: int):
        if index in self.cache:
            return self.cache[index]
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
            (
                episode["Glascow coma scale eye opening"]
                + episode["Glascow coma scale motor response"]
                + episode["Glascow coma scale verbal response"]
            )
            .map(gcs_total_mapping)
            .fillna(0)
        )

        episode["Glascow coma scale eye opening"] = (
            episode["Glascow coma scale eye opening"].fillna(0).astype(int)
        )
        episode["Glascow coma scale motor response"] = (
            episode["Glascow coma scale motor response"].fillna(0).astype(int)
        )
        episode["Glascow coma scale verbal response"] = (
            episode["Glascow coma scale verbal response"].fillna(0).astype(int)
        )

        episode["Capillary refill rate"] = (
            episode["Capillary refill rate"]
            .map(cap_refill_rate_mapping)
            .fillna(0)
            .astype(int)
        )

        if self.discretize:
            for column, bin_range in bin_ranges.items():
                episode[column] = np.digitize(episode[column].fillna(-1), bin_range)

        # group measurements which occurred in the same hour
        episode["Hours"] = np.floor(episode["Hours"]).astype(int)

        if self.discretize:

            def agg(column):
                count = np.bincount(column[column != 0])
                if count.size == 0:
                    return 0
                return np.argmax(count)

            # aggregation operations to take most occured class which is not zero
            aggregation_operations = agg
        else:
            continuous_column_names = [
                column
                for column in episode.columns
                if column != "Capillary refill rate"
                and "Glascow coma scale" not in column
            ]

            categorical_column_names = [
                column
                for column in episode.columns
                if column not in continuous_column_names
            ]

            aggregation_operations = {
                column: "mean" for column in continuous_column_names
            } | {column: "max" for column in categorical_column_names}

        episode = episode.groupby("Hours", as_index=False).agg(aggregation_operations)

        full_df = pd.DataFrame(index=range(48))

        # Merge full_df with episode on "Hours"
        episode = pd.merge(
            full_df, episode, left_index=True, right_on="Hours", how="left"
        )

        # Set "Hours" as index again
        episode.set_index("Hours", inplace=True)

        if self.normalize and not self.discretize:
            for column in episode.columns:
                if column in means and column in stds:
                    episode[column] = (episode[column] - means[column]) / stds[column]

        episode = episode.fillna(0)

        if self.one_hot and not self.discretize:
            episode = one_hot_encode(episode, "Capillary refill rate", 2)
            episode = one_hot_encode(episode, "Glascow coma scale eye opening", 5)
            episode = one_hot_encode(episode, "Glascow coma scale motor response", 7)
            episode = one_hot_encode(episode, "Glascow coma scale verbal response", 6)
            episode = one_hot_encode(episode, "Glascow coma scale total", 16)

        if self.discretize:
            counter = 0
            for column in episode.columns:
                episode[column] += counter
                counter += classes_per_column(column)
        
        t = torch.int if self.discretize else torch.float
        data = (
            torch.tensor(episode.values, dtype=t),
            torch.tensor(int(self.targets[index]), dtype=torch.int64),
        )
        self.cache[index] = data
        return data
