import os
import re

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

###############################
# Non-time series preprocessing
###############################

g_map = {"F": 1, "M": 2, "OTHER": 3, "": 0}


def transform_gender(gender_series):
    global g_map
    return {
        "Gender": gender_series.fillna("").apply(
            lambda s: g_map[s] if s in g_map else g_map["OTHER"]
        )
    }


e_map = {
    "ASIAN": 1,
    "BLACK": 2,
    "CARIBBEAN ISLAND": 2,
    "HISPANIC": 3,
    "SOUTH AMERICAN": 3,
    "WHITE": 4,
    "MIDDLE EASTERN": 4,
    "PORTUGUESE": 4,
    "AMERICAN INDIAN": 0,
    "NATIVE HAWAIIAN": 0,
    "UNABLE TO OBTAIN": 0,
    "PATIENT DECLINED TO ANSWER": 0,
    "UNKNOWN": 0,
    "OTHER": 0,
    "": 0,
}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(" OR ", "/").split(" - ")[0].split("/")[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {
        "Ethnicity": ethnicity_series.fillna("").apply(
            lambda s: e_map[s] if s in e_map else e_map["OTHER"]
        )
    }


def assemble_episodic_data(stays, diagnoses):
    data = {
        "Icustay": stays.stay_id,
        "Age": stays.age,
        "Length of Stay": stays.los,
        "Mortality": stays.mortality,
    }
    data.update(transform_gender(stays.gender))
    data.update(transform_ethnicity(stays.race))
    data["Height"] = np.nan
    data["Weight"] = np.nan
    data = DataFrame(data).set_index("Icustay")
    data = data[
        [
            "Ethnicity",
            "Gender",
            "Age",
            "Height",
            "Weight",
            "Length of Stay",
            "Mortality",
        ]
    ]
    return data.merge(
        extract_diagnosis_labels(diagnoses), left_index=True, right_index=True
    )


diagnosis_labels = [
    "4019",
    "4280",
    "41401",
    "42731",
    "25000",
    "5849",
    "2724",
    "51881",
    "53081",
    "5990",
    "2720",
    "2859",
    "2449",
    "486",
    "2762",
    "2851",
    "496",
    "V5861",
    "99592",
    "311",
    "0389",
    "5859",
    "5070",
    "40390",
    "3051",
    "412",
    "V4581",
    "2761",
    "41071",
    "2875",
    "4240",
    "V1582",
    "V4582",
    "V5867",
    "4241",
    "40391",
    "78552",
    "5119",
    "42789",
    "32723",
    "49390",
    "9971",
    "2767",
    "2760",
    "2749",
    "4168",
    "5180",
    "45829",
    "4589",
    "73300",
    "5845",
    "78039",
    "5856",
    "4271",
    "4254",
    "4111",
    "V1251",
    "30000",
    "3572",
    "60000",
    "27800",
    "41400",
    "2768",
    "4439",
    "27651",
    "V4501",
    "27652",
    "99811",
    "431",
    "28521",
    "2930",
    "7907",
    "E8798",
    "5789",
    "79902",
    "V4986",
    "V103",
    "42832",
    "E8788",
    "00845",
    "5715",
    "99591",
    "07054",
    "42833",
    "4275",
    "49121",
    "V1046",
    "2948",
    "70703",
    "2809",
    "5712",
    "27801",
    "42732",
    "99812",
    "4139",
    "3004",
    "2639",
    "42822",
    "25060",
    "V1254",
    "42823",
    "28529",
    "E8782",
    "30500",
    "78791",
    "78551",
    "E8889",
    "78820",
    "34590",
    "2800",
    "99859",
    "V667",
    "E8497",
    "79092",
    "5723",
    "3485",
    "5601",
    "25040",
    "570",
    "71590",
    "2869",
    "2763",
    "5770",
    "V5865",
    "99662",
    "28860",
    "36201",
    "56210",
]


def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses["value"] = 1
    labels = (
        diagnoses[["stay_id", "icd_code", "value"]]
        .drop_duplicates()
        .pivot(index="stay_id", columns="icd_code", values="value")
        .fillna(0)
        .astype(int)
    )

    # Create a DataFrame of zeros with columns from diagnosis_labels
    df_zeros = pd.DataFrame(
        columns=[l for l in diagnosis_labels if l not in labels.columns]
    )

    # Concatenate labels with df_zeros. This will add missing columns with zeros in labels
    labels = pd.concat([labels, df_zeros], axis=1)

    # Reorder the columns according to diagnosis_labels
    labels = labels[diagnosis_labels]

    return labels.rename(
        dict(zip(diagnosis_labels, ["Diagnosis " + d for d in diagnosis_labels])),
        axis=1,
    )


###################################
# Time series preprocessing
###################################


def read_itemid_to_variable_map(fn, variable_column="LEVEL2"):
    var_map = pd.read_csv(fn, header=0, index_col=None).fillna("").astype(str)
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != "") & (var_map.COUNT > 0)]
    var_map = var_map[(var_map.STATUS == "ready")]
    var_map.itemid = var_map.itemid.astype(int)
    var_map = var_map[[variable_column, "itemid", "MIMIC LABEL"]].set_index("itemid")
    return var_map.rename(
        {variable_column: "VARIABLE", "MIMIC LABEL": "MIMIC_LABEL"}, axis=1
    )


def read_variable_ranges(fn, variable_column="LEVEL2"):
    columns = [
        variable_column,
        "OUTLIER LOW",
        "VALID LOW",
        "IMPUTE",
        "VALID HIGH",
        "OUTLIER HIGH",
    ]
    to_rename = dict(zip(columns, [c.replace(" ", "_") for c in columns]))
    to_rename[variable_column] = "VARIABLE"
    var_ranges = pd.read_csv(fn, header=0, index_col=None)
    # var_ranges = var_ranges[variable_column].apply(lambda s: s.lower())
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset="VARIABLE", keep="first")
    var_ranges.set_index("VARIABLE", inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]


def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = events.VARIABLE == variable
    v = events.value[idx].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, "value"] = v
    return events


# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match("^(\d+)/(\d+)$", s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match("^(\d+)/(\d+)$", s).group(2))
    return v.astype(float)


# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.value is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.value.astype(str)

    v.loc[(df_value_str == "Normal <3 secs") | (df_value_str == "Brisk")] = 0
    v.loc[(df_value_str == "Abnormal >3 secs") | (df_value_str == "Delayed")] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.value.astype(float).copy()

    """ The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    """
    # idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    """ The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    """
    # idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.value > 1.0)

    """ The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    """
    is_str = np.array(map(lambda x: type(x) == str, list(df.value)), dtype=bool)
    idx = df.valueuom.fillna("").apply(lambda s: "torr" not in s.lower()) & (
        is_str | (~is_str & (v > 1.0))
    )

    v.loc[idx] = v[idx] / 100.0
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match("^(\d+(\.\d*)?|\.\d+)$", s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match("^(\d+(\.\d*)?|\.\d+)$", s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = v <= 1
    v.loc[idx] = v[idx] * 100.0
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.value.astype(float).copy()
    idx = (
        df.valueuom.fillna("").apply(lambda s: "F" in s.lower())
        | df.MIMIC_LABEL.apply(lambda s: "F" in s.lower())
        | (v >= 79)
    )
    v.loc[idx] = (v[idx] - 32) * 5.0 / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.value.astype(float).copy()
    # ounces
    idx = df.valueuom.fillna("").apply(
        lambda s: "oz" in s.lower()
    ) | df.MIMIC_LABEL.apply(lambda s: "oz" in s.lower())
    v.loc[idx] = v[idx] / 16.0
    # pounds
    idx = (
        idx
        | df.valueuom.fillna("").apply(lambda s: "lb" in s.lower())
        | df.MIMIC_LABEL.apply(lambda s: "lb" in s.lower())
    )
    v.loc[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.value.astype(float).copy()
    idx = df.valueuom.fillna("").apply(
        lambda s: "in" in s.lower()
    ) | df.MIMIC_LABEL.apply(lambda s: "in" in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
clean_fns = {
    "Capillary refill rate": clean_crr,
    "Diastolic blood pressure": clean_dbp,
    "Systolic blood pressure": clean_sbp,
    "Fraction inspired oxygen": clean_fio2,
    "Oxygen saturation": clean_o2sat,
    "Glucose": clean_lab,
    "pH": clean_lab,
    "Temperature": clean_temperature,
    "Weight": clean_weight,
    "Height": clean_height,
}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = events.VARIABLE == var_name
        try:
            events.loc[idx, "value"] = clean_fn(events[idx])
        except Exception as e:
            import traceback

            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            exit()
    return events.loc[events.value.notnull()]


def read_stays(subject_path):
    path = os.path.join(subject_path, "stays.csv")
    stays = pd.read_csv(path, header=0, index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    stays.sort_values(by=["intime", "outtime"], inplace=True)
    return stays


def read_diagnoses(subject_path):
    path = os.path.join(subject_path, "diagnoses.csv")
    return pd.read_csv(path, header=0, index_col=None)


def read_events(subject_path, remove_null=True):
    path = os.path.join(subject_path, "events.csv")
    events = pd.read_csv(path, header=0, index_col=None)
    if remove_null:
        events = events[events.value.notnull()]
    events.charttime = pd.to_datetime(events.charttime)
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)
    events.stay_id = events.stay_id.fillna(value=-1).astype(int)
    events.valueuom = events.valueuom.fillna("").astype(str)
    # events.sort_values(by=['charttime', 'ITEMID', 'stay_id'], inplace=True)
    return events


def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = events.stay_id == icustayid
    if intime is not None and outtime is not None:
        idx = idx | ((events.charttime >= intime) & (events.charttime <= outtime))
    events = events[idx]
    del events["stay_id"]
    return events


def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events["HOURS"] = (
        (events.charttime - dt).apply(lambda s: s / np.timedelta64(1, "s")) / 60.0 / 60
    )
    if remove_charttime:
        del events["charttime"]
    return events


def convert_events_to_timeseries(events, variable_column="VARIABLE", variables=[]):
    metadata = (
        events[["charttime", "stay_id"]]
        .sort_values(by=["charttime", "stay_id"])
        .drop_duplicates(keep="first")
        .set_index("charttime")
    )
    timeseries = (
        events[["charttime", variable_column, "value"]]
        .sort_values(by=["charttime", variable_column, "value"], axis=0)
        .drop_duplicates(subset=["charttime", variable_column], keep="last")
    )
    timeseries = (
        timeseries.pivot(index="charttime", columns=variable_column, values="value")
        .merge(metadata, left_index=True, right_index=True)
        .sort_index(axis=0)
        .reset_index()
    )
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
