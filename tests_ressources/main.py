from dimlpfidex.fidex import fidexGloRules
from dimlpfidex.dimlp import dimlpBT
import src.data_helper as dh
from src.patient import *
from src.utils import *
from src.rule import *
import pandas as pd
import os


def write_patient() -> Patient:
    trial_data = pd.read_excel(
        "input/PRE-ACT-01_Flow2_20241115_HES-SO.xlsx", index_col=False
    ).iloc[:, :5]
    clinical_data = dh.obtain_data(
        "input/PRE-ACT-01_Flow2_20241115_HES-SO.xlsx", training=False
    )
    clinical_data = clinical_data.assign(
        BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3)
    )

    p = Patient(trial_data, clinical_data)

    # one hotting classes (useless but must be done)
    clinical_data = clinical_data.assign(Lymphodema_NO=lambda x: 1)
    clinical_data = clinical_data.assign(Lymphodema_YES=lambda x: 0)

    clinical_data.to_csv(
        "input/test_sample_data.csv", header=False, index=False, sep=","
    )

    return p

# this is for testing puroposes only
def write_samples_file(n: int) -> Patient:
    patient_metadata = pd.read_excel(
        "input/PRE-ACT-01_Flow2_20241115_HES-SO.xlsx", index_col=False
    ).iloc[:, :5]
    data = pd.read_csv("temp/train_data.csv", sep=" ", header=None)

    sample_data = data.sample(n)

    abspath = os.path.dirname(__file__)

    patients = []
    for i in range(n):
        patient_metadata.iloc[0, 3] = f"FRA_TEST_00{i+1}" # FOR TESTING PURPOSES ONLY
        patients.append(Patient(patient_metadata, sample_data.iloc[i, :], abspath))

    return patients


def write_attributes_file(attributes: list[str]) -> list[str]:
    attributes = attributes + ["LYMPHODEMA_NO", "LYMPHODEMA_YES"]

    with open("temp/attributes.txt", "w") as f:
        for attribute in attributes:
            f.write(attribute + "\n")

    return attributes


def write_train_data(data: pd.DataFrame, labels: pd.Series) -> None:
    labels = pd.get_dummies(labels).astype("uint")

    data.to_csv("temp/train_data.csv", sep=",", header=False, index=False)
    labels.to_csv("temp/train_classes.csv", sep=",", header=False, index=False)

def write_results(patients: list[Patient], attributes: list[str]) -> None:
    abspath = os.path.abspath(os.path.dirname(__file__))
    write_path = os.path.join(abspath, "output", f"results_{datetime.today().strftime('%Y%m%d')}.csv")

    unicancer_headers = ["VISIT", "STUDY_ID", "SITE_NAME", "SITENAME", "SUBJECT_ID"]
    rule_headers = ["RISK", "LOW_INTERVAL", "HIGH_INTERVAL", "RULE_ID"]
    headers = unicancer_headers + rule_headers + attributes

    string = ",".join(headers) + "\n"

    for patient in patients:
        string += patient.format_results()

    with open(write_path,"w") as fp:
        fp.write(string)

if __name__ == "__main__":
    args = init_args()
    abspath = os.path.abspath(os.path.dirname(__file__))

    if args.cleanall:
        clean_dir("temp")
        clean_dir("logs")
        args.train = True

    if args.clean:
        clean_dir("temp")

    data, labels = dh.obtain_data("dataset/clinical_complete_rev1.csv")
    data, labels = dh.filter_clinical(data, labels)
    # added BMI column
    data = data.assign(BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3))

    nb_features = data.shape[1]
    nb_classes = 2

    attributes = write_attributes_file(data.columns.to_list())
    write_train_data(data, labels)
    update_config_files(abspath, nb_features, nb_classes)
    # patient = write_patient()
    patients = write_samples_file(3)

    # normalization("--json_config_file config/normalization.json")

    if args.train:
        dimlpBT("--json_config_file config/dimlpbt.json")
        fidexGloRules("--json_config_file config/fidexglorules.json")
        exit()

    global_rules = GlobalRules.from_json_file("temp/global_rules_denormalized.json")

    for patient in patients:
        patient.extract_rules(global_rules)

    write_results(patients, attributes)