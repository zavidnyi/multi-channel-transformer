import os

import numpy as np
import pandas as pd
import torch.utils.data


class InHospitalMortalityDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, list_file_path: str, one_hot: bool, normalize: bool):
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
        return 753

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

        if self.normalize:
            for column in episode.columns:
                if column in means and column in stds:
                    episode[column] = (episode[column] - means[column]) / stds[column]

        episode = episode.fillna(0)
        if self.one_hot:
            episode = one_hot_encode(episode, "Capillary refill rate", 2)
            episode = one_hot_encode(episode, "Glascow coma scale eye opening", 5)
            episode = one_hot_encode(episode, "Glascow coma scale motor response", 7)
            episode = one_hot_encode(episode, "Glascow coma scale verbal response", 6)
            episode = one_hot_encode(episode, "Glascow coma scale total", 16)

        data = (
            torch.tensor(episode.values, dtype=torch.float),
            torch.tensor(int(self.targets[index]), dtype=torch.float),
        )
        self.cache[index] = data
        return data


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

stds = dict(
    [
        ("Hours", 14.396261767527724),
        ("Diastolic blood pressure", 285.80064177699705),
        ("Fraction inspired oxygen", 0.1961013042470289),
        ("Glucose", 9190.367721597377),
        ("Heart Rate", 132.41586088485442),
        ("Height", 12.332785645604897),
        ("Mean blood pressure", 266.45492092726295),
        ("Oxygen saturation", 2094.753594800329),
        ("Respiratory rate", 2025.1666030044469),
        ("Systolic blood pressure", 882.396478974552),
        ("Temperature", 12.879852903644485),
        ("Weight", 95.5778654729231),
        ("pH", 11110.745176079576),
    ]
)

means = dict(
    [
        ("Hours", 22.028152722731797),
        ("Diastolic blood pressure", 63.40139620838688),
        ("Fraction inspired oxygen", 0.5220774309673805),
        ("Glucose", 233.5193111471457),
        ("Heart Rate", 86.05173178993036),
        ("Height", 169.33463796477494),
        ("Mean blood pressure", 78.61474847093386),
        ("Oxygen saturation", 100.99360210904216),
        ("Respiratory rate", 21.34307497701275),
        ("Systolic blood pressure", 118.69927129942835),
        ("Temperature", 36.96791995122653),
        ("Weight", 84.91834694253167),
        ("pH", 130.70163154775614),
    ]
)

# first value in bin is the cutout values that we treat as missing value
bin_ranges = {
    # https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
    # <0 941,842 instances
    # 0-119 775,786 instnaces
    # 120-129 204,999 instances
    # 130-139 151,336 instances
    # 140-179 205,956 instances
    # >=180 12,336 instances
    "Systolic blood pressure": [0, 120, 130, 140, 180],
    # <0 942,208 instances
    # 0-79 1,188,103 instances
    # 80-89 95,648 instances
    # 90-119 61,476 instances
    # >=120 4,840 instances
    "Diastolic blood pressure": [0, 80, 90, 120],
    # Oxygen-enriched air has a higher FIO2 than 0.21; up to 1.00 which means 100% oxygen.
    # FIO2 is typically maintained below 0.5 even with mechanical ventilation,
    # to avoid oxygen toxicity, but there are applications when up to 100% is routinely used.
    # https://en.wikipedia.org/wiki/Fraction_of_inspired_oxygen
    # <0 2,124,492 instances
    # <0.21 824 instances
    # 0.21<0.3 880 instances
    # 0.3<0.4 17,687 instances
    # 0.4<0.5 55,093 instances
    # 0.5<0.6 49,645 instances
    # 0.6<0.7 13,784 instances
    # 0.7<0.8 8,000 instances
    # 0.8<0.9 4,652 instances
    # 0.9<1.0 2,083 instances
    # >=1.0 15,135 instances
    "Fraction inspired oxygen": [
        0,
        0.21,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    # https://www.cdc.gov/diabetes/basics/getting-tested.html
    # <0 1,984,546 instances
    # 0<90 27,612 instances
    # 90<100 23,040 instances
    # 100<125 79,210 instances
    # 125<140 43,496 instances
    # 140<200 888,566 instances
    # >=200 45,805 instances
    "Glucose": [0, 90, 100, 125, 140, 200],
     # <0 941,580
    # 0<50 14,732
    # 50<60 61,650
    # 60<70 177,829
    # 70<80 271,464
    # 80<90 303,199
    # 90<100 223,888
    # 100<110 230,482
    # 110<120 39,889
    # 120<130 16,968,
    # 130<140 6,335
    # 140<150 2,603
    # 150<160 987
    # 160<170 377
    # 170<180 149
    # 180<190 73
    # 190<200 70
    "Heart Rate": [
        0,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
    ],
    # <0 2,291,764
    # 0<150 9
    # 150<160 83
    # 160<170 148
    # 170<180 168
    # 180<190 96
    # >=190 7
    "Height": [0, 150, 160, 170, 180, 190],
    # https://en.wikipedia.org/wiki/Mean_arterial_pressure
    'Mean blood pressure': [90, 92, 96],
    'Oxygen saturation': [92, 93, 94, 95, 96],
    'Respiratory rate': [12, 18, 25],
    'Temperature': [35, 36.5, 37.5, 38.3, 40, 41],
    'Body weight': [60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    'pH': [7.35, 7.4, 7.45, 7.5],
}

