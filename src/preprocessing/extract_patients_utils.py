import csv
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_patients_table(path: str) -> pd.DataFrame:
    patients = pd.read_csv(path, header=0, index_col=None)
    patients = patients[["subject_id", "gender", "anchor_age", "dod"]]
    patients.dod = pd.to_datetime(patients.dod)
    return patients


def read_admissions_table(path: str) -> pd.DataFrame:
    admissions = pd.read_csv(path, header=0, index_col=None)
    admissions = admissions[
        ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race"]
    ]
    admissions.admittime = pd.to_datetime(admissions.admittime)
    admissions.dischtime = pd.to_datetime(admissions.dischtime)
    admissions.deathtime = pd.to_datetime(admissions.deathtime)
    return admissions


def read_icustays_table(path: str) -> pd.DataFrame:
    stays = pd.read_csv(path, header=0, index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays


def read_icd_diagnoses_table(
    d_icd_diagnoses_path: str, diagnoses_icd_path: str
) -> pd.DataFrame:
    """
    :param d_icd_diagnoses_path: icd code to long title mapping
    :param diagnoses_icd_path: subject_id to icd code mapping
    """
    codes = pd.read_csv(d_icd_diagnoses_path, header=0, index_col=None)
    codes = codes[["icd_code", "long_title"]]
    diagnoses = pd.read_csv(diagnoses_icd_path, header=0, index_col=None)
    diagnoses = diagnoses.merge(
        codes, how="inner", left_on="icd_code", right_on="icd_code"
    )
    diagnoses[["subject_id", "hadm_id", "seq_num"]] = diagnoses[
        ["subject_id", "hadm_id", "seq_num"]
    ].astype(int)
    return diagnoses


def read_events_table_by_row(mimic4_path, table):
    nb_rows = {
        "icu/chartevents": 330712484,
        "hosp/labevents": 118171368,
        "icu/outputevents": 4349219,
    }
    reader = csv.DictReader(open(os.path.join(mimic4_path, table + ".csv"), "r"))
    for i, row in enumerate(reader):
        if "stay_id" not in row:
            row["stay_id"] = ""
        yield row, i, nb_rows[table.lower()]


def count_icd_codes(diagnoses: pd.DataFrame) -> pd.DataFrame:
    code_counts = (
        diagnoses[["icd_code", "long_title"]].drop_duplicates().set_index("icd_code")
    )
    code_counts["stay_count"] = diagnoses.groupby("icd_code")["stay_id"].count()
    code_counts.stay_count = code_counts.stay_count.fillna(0).astype(int)
    code_counts = code_counts[code_counts.stay_count > 0]
    return code_counts


def remove_icustays_with_transfers(stays: pd.DataFrame) -> pd.DataFrame:
    stays = stays[stays.first_careunit == stays.last_careunit]
    return stays[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ]
    ]


def add_inhospital_mortality_to_icustays(stays: pd.DataFrame) -> pd.DataFrame:
    mortality = stays.dod.notnull() & (
        (stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod)
    )
    mortality = mortality | (
        stays.deathtime.notnull()
        & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime))
    )
    stays["mortality"] = mortality.astype(int)
    stays["mortality_inhospital"] = stays["mortality"]
    return stays


def add_inunit_mortality_to_icustays(stays: pd.DataFrame) -> pd.DataFrame:
    mortality = stays.dod.notnull() & (
        (stays.intime <= stays.dod) & (stays.outtime >= stays.dod)
    )
    mortality = mortality | (
        stays.deathtime.notnull()
        & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime))
    )
    stays["mortality_inunit"] = mortality.astype(int)
    return stays


def filter_admissions_on_number_of_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby("hadm_id").count()[["stay_id"]].reset_index()
    to_keep = to_keep[
        (to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)
    ][["hadm_id"]]
    stays = stays.merge(to_keep, how="inner", left_on="hadm_id", right_on="hadm_id")
    return stays


def filter_icustays_on_age(
    stays: pd.DataFrame, min_age: int = 18, max_age: int = 100
) -> pd.DataFrame:
    stays = stays[(stays.age >= min_age) & (stays.age <= max_age)]
    return stays


def filter_diagnoses_on_stays(
    diagnoses: pd.DataFrame, stays: pd.DataFrame
) -> pd.DataFrame:
    return diagnoses.merge(
        stays[["subject_id", "hadm_id", "stay_id"]].drop_duplicates(),
        how="inner",
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "hadm_id"],
    )


def break_up_stays_by_subject(
    stays: pd.DataFrame, output_path: str, subjects: np.ndarray = None
) -> None:
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(
        subjects, total=nb_subjects, desc="Breaking up stays by subjects"
    ):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.subject_id == subject_id].sort_values(by="intime").to_csv(
            os.path.join(dn, "stays.csv"), index=False
        )


def break_up_diagnoses_by_subject(
    diagnoses: pd.DataFrame, output_path: str, subjects: np.ndarray = None
) -> None:
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(
        subjects, total=nb_subjects, desc="Breaking up diagnoses by subjects"
    ):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.subject_id == subject_id].sort_values(
            by=["stay_id", "seq_num"]
        ).to_csv(os.path.join(dn, "diagnoses.csv"), index=False)


def add_hcup_ccs_2015_groups(
    diagnoses: pd.DataFrame, definitions: pd.DataFrame
) -> pd.DataFrame:
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]["codes"]:
            def_map[code] = (dx, definitions[dx]["use_in_benchmark"])
    diagnoses["hcup_ccs_2015"] = diagnoses.icd_code.apply(
        lambda c: def_map[c][0] if c in def_map else None
    )
    diagnoses["use_in_benchmark"] = diagnoses.icd_code.apply(
        lambda c: int(def_map[c][1]) if c in def_map else None
    )
    return diagnoses


def make_phenotype_label_matrix(phenotypes, stays=None):
    phenotypes = (
        phenotypes[["stay_id", "hcup_ccs_2015"]]
        .loc[phenotypes.use_in_benchmark > 0]
        .drop_duplicates()
    )
    phenotypes["value"] = 1
    phenotypes = phenotypes.pivot(
        index="stay_id", columns="hcup_ccs_2015", values="value"
    )
    if stays is not None:
        phenotypes = phenotypes.reindex(stays.stay_id.sort_values())
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)


def read_events_table_and_break_up_by_subject(
    mimic4_path, table, output_path, items_to_keep=None, subjects_to_keep=None
):
    obs_header = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "value",
        "valueuom",
    ]
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ""
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, "events.csv")
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, "w")
            f.write(",".join(obs_header) + "\n")
            f.close()
        w = csv.DictWriter(
            open(fn, "a"), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL
        )
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    nb_rows_dict = {
        "icu/chartevents": 330712484,
        "hosp/labevents": 118171368,
        "icu/outputevents": 4349219,
    }
    nb_rows = nb_rows_dict[table.lower()]

    for row, row_no, _ in tqdm(
        read_events_table_by_row(mimic4_path, table),
        total=nb_rows,
        desc="Processing {} table".format(table),
    ):

        if (subjects_to_keep is not None) and (
            row["subject_id"] not in subjects_to_keep
        ):
            continue
        if (items_to_keep is not None) and (row["itemid"] not in items_to_keep):
            continue

        row_out = {
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "stay_id": "" if "stay_id" not in row else row["stay_id"],
            "charttime": row["charttime"],
            "itemid": row["itemid"],
            "value": row["value"],
            "valueuom": row["valueuom"],
        }
        if (
            data_stats.curr_subject_id != ""
            and data_stats.curr_subject_id != row["subject_id"]
        ):
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row["subject_id"]

    if data_stats.curr_subject_id != "":
        write_current_observations()
