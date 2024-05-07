import argparse
import os

import numpy as np
import pandas as pd
from pqdm.threads import pqdm

from src.mimic.constants import (
    gcs_eye_mapping,
    gcs_motor_mapping,
    gcs_verbal_mapping,
    gcs_total_mapping,
    cap_refill_rate_mapping,
    bin_ranges,
    means,
    stds,
)
from src.mimic.utils import one_hot_encode, classes_per_column


def prepare_data(
    episode, max_seq_len, discretize=False, normalize=False, one_hot=False
):
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

    if discretize:
        for column, bin_range in bin_ranges.items():
            episode[column] = np.digitize(episode[column].fillna(-1), bin_range)

    if max_seq_len != -1:
        # group measurements which occurred in the same hour
        episode["Hours"] = np.floor(episode["Hours"]).astype(int)

        if discretize:

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

        full_df = pd.DataFrame(index=range(max_seq_len))

        episode = episode.groupby("Hours", as_index=False).agg(aggregation_operations)

        # Merge full_df with episode on "Hours"
        episode = pd.merge(
            full_df, episode, left_index=True, right_on="Hours", how="left"
        )

    # Set "Hours" as index again
    episode.drop("Hours", axis=1, inplace=True)

    if normalize and not discretize:
        for column in episode.columns:
            if column in means and column in stds:
                episode[column] = (episode[column] - means[column]) / stds[column]

    episode = episode.fillna(0)

    if one_hot and not discretize:
        episode = one_hot_encode(episode, "Capillary refill rate", 2)
        episode = one_hot_encode(episode, "Glascow coma scale eye opening", 5)
        episode = one_hot_encode(episode, "Glascow coma scale motor response", 7)
        episode = one_hot_encode(episode, "Glascow coma scale verbal response", 6)
        episode = one_hot_encode(episode, "Glascow coma scale total", 16)

    episode = episode.fillna(0)


    if discretize:
        counter = 0
        for column in episode.columns:
            episode[column] += counter
            counter += classes_per_column(column)

    return episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--discretize", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--one_hot", action="store_true")
    args = parser.parse_args()

    def do(file):
        if file.endswith("timeseries.csv"):
            episode = pd.read_csv(os.path.join(args.data, file))
            episode = prepare_data(
                episode, args.max_seq_len, args.discretize, args.normalize, args.one_hot
            )
            episode.to_csv(os.path.join(args.output, file), index=False)

    pqdm(os.listdir(args.data), do, n_jobs=args.threads)
