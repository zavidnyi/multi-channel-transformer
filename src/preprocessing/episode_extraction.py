import argparse
import sys

from tqdm import tqdm

from src.preprocessing.episode_extraction_util import *

parser = argparse.ArgumentParser(description="Extract episodes from per-subject data.")
parser.add_argument(
    "subjects_root_path", type=str, help="Directory containing subject sub-directories."
)
parser.add_argument(
    "--variable_map_file",
    type=str,
    default=os.path.join(
        os.path.dirname(__file__), "resources/itemid_to_variable_map.csv"
    ),
    help="CSV containing ITEMID-to-VARIABLE map.",
)
parser.add_argument(
    "--reference_range_file",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "resources/variable_ranges.csv"),
    help="CSV containing reference ranges for VARIABLEs.",
)
args, _ = parser.parse_known_args()

item_id_to_var = read_itemid_to_variable_map(args.variable_map_file)
variables = item_id_to_var.VARIABLE.unique()
events_total = 0
for subject_dir in tqdm(
    os.listdir(args.subjects_root_path), desc="Iterating over subjects"
):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except Exception as e:
        sys.stderr.write(str(e))
        sys.stderr.write("Error reading from disk for subject: {}\n".format(subject_id))
        continue

    episodic_data = assemble_episodic_data(stays, diagnoses)

    # cleaning and converting to time series
    events = events.merge(item_id_to_var, left_on="itemid", right_index=True)
    events = clean_events(events)
    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    events_total += events.shape[0]
    timeseries = convert_events_to_timeseries(events, variables=variables)

    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue

        episode = (
            add_hours_elpased_to_events(episode, intime)
            .set_index("HOURS")
            .sort_index(axis=0)
        )
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, "Weight"] = get_first_valid_from_timeseries(
                episode, "Weight"
            )
            episodic_data.loc[stay_id, "Height"] = get_first_valid_from_timeseries(
                episode, "Height"
            )
        episodic_data.loc[episodic_data.index == stay_id].to_csv(
            os.path.join(
                args.subjects_root_path, subject_dir, "episode{}.csv".format(i + 1)
            ),
            index_label="Icustay",
        )
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(
            os.path.join(
                args.subjects_root_path,
                subject_dir,
                "episode{}_timeseries.csv".format(i + 1),
            ),
            index_label="Hours",
        )

print("Total number of events: ", events_total)
