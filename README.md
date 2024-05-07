# Transformer Architectures for Multi-Channel Data

## Getting MIMIC-IV

The MIMIC-IV dataset is available from PhysioNet. You can request access to the
dataset [here](https://physionet.org/content/mimiciv/2.2/).
Note that accessing a dataset requires both a verified PhysioNet account and passing an online course on handling
sensitive data.
Once you have access, you can download the dataset using the following command:

```bash
wget -r -N -c -np --user <your_username> --ask-password https://physionet.org/files/mimiciv/2.2/
```

Note that the download may take a while, due to the slow upload speed of the server, where dataset is stored.
For me it was 4 hours.
See [this](https://github.com/MIT-LCP/mimic-code/issues/600) GitHub discussion.

## Setup

Create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing

Here are the required steps to build the benchmark.
All the commands are run from the repository root.

1. For each patient we create a directory `data/mimic/<patient_id>/`, where we aggregate and store information about
   their stay `stay.csv`, events `events.csv` and diagnoses `diagnoses.csv`.
   Since aggregation happens from many different tables, this step will take quite some time.
   Path to mimic should be of this form `<dir_where_downloaded>/physionet.org/files/mimiciv/2.2/`.

```bash
python -m src.preprocessing.extract_patients <path_to_mimic> data/mimic/
```

2. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing
   information.

```bash
python -m src.preprocessing.validate_events data/mimic/
```

3. For each ICU stay of the patient we create a seprate file with events occured during their stay.

```bash
python -m src.preprocessing.episode_extraction data/mimic/
```

5. Create data samples for the in-hospital mortality prediction:

```bash
 python -m src.preprocessing.create_in_hospital_mortality data/mimic/ data/in-hospital-mortality/
```

6. Create data samples for the length of stay prediction:

```bash
python -m src.preprocessing.create_length_of_stay data/mimic/ data/length-of-stay/
```

For reproducibility, we share a listfiles, describing episodes used for training `train_listfile.csv`,
validation `val_listfile.csv` and test `test_listfile.csv`.
Listfiles are stored in `data/in-hospital-mortality/` and `data/length-of-stay/` directories.