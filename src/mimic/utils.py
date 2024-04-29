import pandas as pd
import torch

from src.mimic.constants import bin_ranges


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


def classes_per_column(column: str) -> int:
    if column == "Capillary refill rate":
        return 3
    if column == "Glascow coma scale eye opening":
        return 5
    if column == "Glascow coma scale motor response":
        return 7
    if column == "Glascow coma scale verbal response":
        return 6
    if column == "Glascow coma scale total":
        return 14
    return len(bin_ranges[column]) + 1
