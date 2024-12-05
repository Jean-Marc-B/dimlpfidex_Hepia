import src.data_helper as dh
import pandas as pd
import argparse
import json
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
    # get most recent by comparing UNIX timestamps
    return max(list_filepaths, key=lambda filepath: os.path.getctime(filepath))


def write_attributes_file(abspath: str, attributes: list[str]) -> list[str]:
    file_path = os.path.join(abspath, "temp", "attributes.txt")
    attributes = attributes + ["LYMPHODEMA_NO", "LYMPHODEMA_YES"]

    with open(file_path, "w") as f:
        for attribute in attributes:
            f.write(attribute + "\n")

    return attributes


def read_attributes_file(abspath: str) -> list[str]:
    file_path = os.path.join(abspath, "temp", "attributes.txt")

    with open(file_path, "r") as f:
        return f.read().splitlines()


def write_train_data(
    abspath: str, data: pd.DataFrame, labels: pd.Series, split: float = 0.0
) -> None:
    labels = pd.get_dummies(labels).astype("uint")
    train_data_file = os.path.join(abspath, "temp", "train_data.csv")
    train_labels_file = os.path.join(abspath, "temp", "train_classes.csv")
    test_data_file = os.path.join(abspath, "temp", "test_data.csv")
    test_labels_file = os.path.join(abspath, "temp", "test_classes.csv")

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
