import argparse

import yaml

from src.preprocessing.extract_patients_utils import *

parser = argparse.ArgumentParser(
    description="Extract per-subject data from MIMIC-IV CSV files."
)
parser.add_argument(
    "mimic4_path", type=str, help="Directory containing MIMIC-IV CSV files."
)
parser.add_argument(
    "output_path", type=str, help="Directory where per-subject data should be written."
)
parser.add_argument(
    "--event_tables",
    "-e",
    type=str,
    nargs="+",
    help="Tables from which to read events.",
    default=["icu/chartevents", "hosp/labevents", "icu/outputevents"],
)
parser.add_argument(
    "--phenotype_definitions",
    "-p",
    type=str,
    default=os.path.join(
        os.path.dirname(__file__), "resources/hcup_ccs_2015_definitions.yaml"
    ),
    help="YAML file with phenotype definitions.",
)
parser.add_argument(
    "--itemids_file", "-i", type=str, help="CSV containing list of ITEMIDs to keep."
)
parser.add_argument(
    "--verbose", "-v", dest="verbose", action="store_true", help="Verbosity in output"
)
parser.add_argument(
    "--quiet",
    "-q",
    dest="verbose",
    action="store_false",
    help="Suspend printing of details",
)
parser.set_defaults(verbose=True)
parser.add_argument(
    "--test",
    action="store_true",
    help="TEST MODE: process only 1000 subjects, 1000000 events.",
)
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

patients = read_patients_table(os.path.join(args.mimic4_path, "hosp/patients.csv"))
admissions = read_admissions_table(
    os.path.join(args.mimic4_path, "hosp/admissions.csv")
)
stays = read_icustays_table(os.path.join(args.mimic4_path, "icu/icustays.csv"))

if args.verbose:
    print(
        f"START:\n\t"
        f"stay_ids: {stays.stay_id.unique().shape[0]}\n\t"
        f"hadm_ids: {stays.hadm_id.unique().shape[0]}\n\t"
        f"subject_ids: {stays.subject_id.unique().shape[0]}"
    )

stays = remove_icustays_with_transfers(stays)
if args.verbose:
    print(
        f"REMOVE ICU STAYS WITH TRANSFERS:\n\t"
        f"stay_ids: {stays.stay_id.unique().shape[0]}\n\t"
        f"hadm_ids: {stays.hadm_id.unique().shape[0]}\n\t"
        f"subject_ids: {stays.subject_id.unique().shape[0]}"
    )

stays = stays.merge(
    admissions,
    how="inner",
    left_on=["subject_id", "hadm_id"],
    right_on=["subject_id", "hadm_id"],
)
stays = stays.merge(
    patients, how="inner", left_on=["subject_id"], right_on=["subject_id"]
)
stays = filter_admissions_on_number_of_icustays(stays)
if args.verbose:
    print(
        f"REMOVE MULTIPLE STAYS PER ADMISSION:\n\t"
        f"stay_ids: {stays.stay_id.unique().shape[0]}\n\t"
        f"hadm_ids: {stays.hadm_id.unique().shape[0]}\n\t"
        f"subject_ids: {stays.subject_id.unique().shape[0]}"
    )

stays["age"] = stays["anchor_age"]
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
# does nothing, since neonates have been removed from the dataset in MIMIC-IV 2.0
# https://mimic.mit.edu/docs/iv/about/changelog/#mimic-iv-v20
# stays = filter_icustays_on_age(stays)
# if args.verbose:
#     print(
#         f"REMOVE PATIENTS AGE < 18:\n\t"
#         f"stay_ids: {stays.stay_id.unique().shape[0]}\n\t"
#         f"hadm_ids: {stays.hadm_id.unique().shape[0]}\n\t"
#         f"subject_ids: {stays.subject_id.unique().shape[0]}"
#     )

stays.to_csv(os.path.join(args.output_path, "all_stays.csv"), index=False)

diagnoses = read_icd_diagnoses_table(
    d_icd_diagnoses_path=os.path.join(args.mimic4_path, "hosp/d_icd_diagnoses.csv"),
    diagnoses_icd_path=os.path.join(args.mimic4_path, "hosp/diagnoses_icd.csv"),
)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)

diagnoses.to_csv(os.path.join(args.output_path, "all_diagnoses.csv"), index=False)

count_icd_codes(diagnoses).to_csv(
    os.path.join(args.output_path, "diagnosis_counts.csv"), index_label="icd_code"
)

phenotypes = add_hcup_ccs_2015_groups(
    diagnoses, yaml.safe_load(open(args.phenotype_definitions, "r"))
)

make_phenotype_label_matrix(phenotypes, stays).to_csv(
    os.path.join(args.output_path, "phenotype_labels.csv"),
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(
        patients[["subject_id"]], left_on="subject_id", right_on="subject_id"
    )
    args.event_tables = [args.event_tables[0]]
    print("Using only", stays.shape[0], "stays and only", args.event_tables[0], "table")

subjects = stays.subject_id.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
items_to_keep = (
    set(
        [
            int(itemid)
            for itemid in pd.read_csv(args.itemids_file, header=0, index_col=None)[
                "ITEMID"
            ].unique()
        ]
    )
    if args.itemids_file
    else None
)
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(
        args.mimic4_path,
        table,
        args.output_path,
        items_to_keep=items_to_keep,
        subjects_to_keep=subjects,
    )
