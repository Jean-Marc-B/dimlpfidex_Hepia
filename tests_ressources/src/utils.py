import src.constants as constants
import src.data_helper as dh
import pandas as pd
import argparse
import json
import csv
import os


def load_clinical_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    datas, labels = dh.obtain_data(path)
    datas, labels = dh.filter_clinical(datas, labels)

    # adding BMI column
    datas = datas.assign(BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3))

    return datas, labels


def get_most_recent_input_file(absolute_path: str) -> str:
    # get all filepaths inside input/ folder
    list_filepaths = [
        os.path.join(absolute_path, filename) for filename in os.listdir(absolute_path)
    ]

    # if none found
    if len(list_filepaths) < 1:
        return ""

    # get most recent by comparing UNIX timestamps
    return max(list_filepaths, key=lambda filepath: os.path.getctime(filepath))


def reorder_data_columns(data: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(data[1:], columns=data[0], dtype="string")
    df = df.loc[
        :,
        [
            "STUDYID",
            "SITEIDN",
            "SITENAME",
            "VISIT",
            "SUBJID",
            "RULE_ID",
            "RISK",
            "LOW_INTERVAL",
            "HIGH_INTERVAL",
            "COVERING",
            "WEIGHT",
            "TYPE_SURGERY_UNKNOWN",
            "TYPE_SURGERY_MASTECTOMY",
            "TYPE_SURGERY_BREAST_CONSERVING_SURGERY",
            "TREATED_BREAST_RADIO_RIGHT",
            "TREATED_BREAST_RADIO_UNKNOWN",
            "TREATED_BREAST_RADIO_LEFT",
            "SMOKER_CURRENT",
            "SMOKER_FORMER",
            "SMOKER_NEVER",
            "SMOKER_UNKNOWN",
            "SMOKER_FORMER",
            "SIDE_OF_PRIMARY_RIGHT",
            "SIDE_OF_PRIMARY_UNKNOWN",
            "SIDE_OF_PRIMARY_LEFT",
            "SENTINEL_NODE_BIOPSY",
            "PR_STATUS_POSITIVE",
            "PR_STATUS_NEGATIVE",
            "PR_STATUS_UNKNOWN",
            "PLANNED_AXILLARY_DISSECTION",
            "PATHOLOGICAL_TUMOUR_SIZE",
            "NUMBER_OF_FRACTIONS",
            "NODES_INVOLVED",
            "NODES_REMOVED",
            "NEOADJUVANT_CHEMOTHERAPY_YES",
            "NEOADJUVANT_CHEMOTHERAPY_NO",
            "NEOADJUVANT_CHEMOTHERAPY_UNKNOWN",
            "MENOPAUSAL_PRE_MENOPAUSAL",
            "MENOPAUSAL_POST_MENOPAUSAL",
            "MENOPAUSAL_NOT_MENOPAUSAL",
            "MENOPAUSAL_UNKNOWN",
            "KI_67_STATUS",
            "IMRT_YES",
            "IMRT_NO",
            "IMRT_UNKNOWN",
            "HISTOLOGICAL_TYPE_OTHER",
            "HISTOLOGICAL_TYPE_UNKNOWN",
            "HISTOLOGICAL_TYPE_LOBULAR",
            "HISTOLOGICAL_TYPE_DUCTAL/INVASIVE_CARCINOMA_NST",
            "HER_2_STATUS_POSITIVE",
            "HER_2_STATUS_NEGATIVE",
            "HER_2_STATUS_UNKNOWN",
            "HEIGHT",
            "ER_STATUS_POSITIVE",
            "ER_STATUS_NEGATIVE",
            "ER_STATUS_UNKNOWN",
            "DIABETIES_YES",
            "DIABETIES_NO",
            "DIABETIES_UNKNOWN",
            "CLINICAL_T_STAGE_TX",
            "CLINICAL_T_STAGE_TIS",
            "CLINICAL_T_STAGE_T4D",
            "CLINICAL_T_STAGE_T4B",
            "CLINICAL_T_STAGE_T4A",
            "CLINICAL_T_STAGE_T4",
            "CLINICAL_T_STAGE_T3",
            "CLINICAL_T_STAGE_T2",
            "CLINICAL_T_STAGE_T1",
            "CLINICAL_T_STAGE_T0",
            "CLINICAL_T_STAGE_UNKNOWN",
            "CLINICAL_N_STAGE_NX",
            "CLINICAL_N_STAGE_UNKNOWN",
            "CLINICAL_N_STAGE_N3",
            "CLINICAL_N_STAGE_N2",
            "CLINICAL_N_STAGE_N1",
            "CLINICAL_N_STAGE_N0",
            "BOOST_YES",
            "BOOST_NO",
            "BOOST_UNKNOWN",
            "BASELINE_ARM_LYMPHEDEMA_YES",
            "BASELINE_ARM_LYMPHEDEMA_NO",
            "BASELINE_ARM_LYMPHEDEMA_UNKNOWN",
            "AGE",
            "ADJUVANT_CHEMOTHERAPY_YES",
            "ADJUVANT_CHEMOTHERAPY_NO",
            "ADJUVANT_CHEMOTHERAPY_UNKNOWN",
            "3D_CRT_YES",
            "3D_CRT_NO",
            "3D_CRT_UNKNOWN",
            "BMI",
        ],
    ]

    return df


def write_attributes_file(abspath: str, attributes: list[str]) -> list[str]:
    file_path = os.path.join(abspath, constants.MODEL_DIRNAME, "attributes.txt")
    attributes = attributes + ["LYMPHODEMA_NO", "LYMPHODEMA_YES"]

    with open(file_path, "w") as f:
        for attribute in attributes:
            f.write(attribute + "\n")

    return attributes


def read_attributes_file(abspath: str) -> list[str]:
    file_path = os.path.join(abspath, constants.MODEL_DIRNAME, "attributes.txt")

    with open(file_path, "r") as f:
        return f.read().splitlines()


def write_train_data(
    abspath: str, data: pd.DataFrame, labels: pd.Series, split: float = 0.0
) -> None:
    labels = pd.get_dummies(labels).astype("uint")
    train_data_file = os.path.join(abspath, constants.MODEL_DIRNAME, "train_data.csv")
    train_labels_file = os.path.join(
        abspath, constants.MODEL_DIRNAME, "train_classes.csv"
    )
    test_data_file = os.path.join(abspath, constants.MODEL_DIRNAME, "test_data.csv")
    test_labels_file = os.path.join(
        abspath, constants.MODEL_DIRNAME, "test_classes.csv"
    )

    train_data = data
    train_labels = labels

    if split > 0.0:
        split = 0.5 if split > 0.5 else split
        split_idx = data.shape[0] - int(data.shape[0] * split)

        train_data = data.iloc[:split_idx]
        train_labels = labels.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        test_labels = labels.iloc[split_idx:]

        test_data.to_csv(test_data_file, sep=",", header=False, index=False)
        test_labels.to_csv(test_labels_file, sep=",", header=False, index=False)

    train_data.to_csv(train_data_file, sep=",", header=False, index=False)
    train_labels.to_csv(train_labels_file, sep=",", header=False, index=False)


def read_json_file(path: str) -> dict:
    with open(path) as fp:
        return json.load(fp)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", type=int)

    return parser.parse_args()
