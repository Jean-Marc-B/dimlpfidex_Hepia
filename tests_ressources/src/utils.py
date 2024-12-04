from datetime import datetime
import pandas as pd
import argparse
import json
import os


def get_most_recent_input_file(absolute_path: str) -> str:
    # get all filepaths inside input/ folder
    list_filepaths = [
        os.path.join(absolute_path, filename) for filename in os.listdir(absolute_path)
    ]
    # get most recent by comparing UNIX timestamps
    return max(list_filepaths, key=lambda filepath: os.path.getctime(filepath))


def update_config_file(filename: str, params: dict) -> None:
    with open(filename, "r+") as f:
        config = json.load(f)
        for key, param in params.items():
            config[key] = param

        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()


def write_attributes_file(abspath: str, attributes: list[str]) -> list[str]:
    file_path = os.path.join(abspath, "temp", "attributes.txt")
    attributes = attributes + ["LYMPHODEMA_NO", "LYMPHODEMA_YES"]

    with open(file_path, "w") as f:
        for attribute in attributes:
            f.write(attribute + "\n")

    return attributes


def write_train_data(abspath: str, data: pd.DataFrame, labels: pd.Series) -> None:
    labels = pd.get_dummies(labels).astype("uint")

    data_file = os.path.join(abspath, "temp", "train_data.csv")
    labels_file = os.path.join(abspath, "temp", "train_classes.csv")

    data.to_csv(data_file, sep=",", header=False, index=False)
    labels.to_csv(labels_file, sep=",", header=False, index=False)


def update_config_files(root_folder: str, nb_features: int, nb_classes: int):
    confpath = os.path.join(root_folder, "config")
    logpath = "logs/"

    programs = ["dimlpbt", "denscls", "fidexglo", "fidexglorules"]

    for program in programs:
        config_filename = os.path.join(confpath, f"{program}.json")
        config = dict()
        config["root_folder"] = root_folder
        config["nb_attributes"] = nb_features
        config["nb_classes"] = nb_classes
        config["console_file"] = (
            f"{logpath + datetime.today().strftime('%Y%m%d%H%M')}_{program}.log"
        )

        update_config_file(config_filename, config)

    update_config_file(
        os.path.join(confpath, "train_normalization.json"),
        {"root_folder": root_folder, "nb_attributes": nb_features},
    )
    update_config_file(
        os.path.join(confpath, "train_denormalization.json"),
        {"root_folder": root_folder, "nb_attributes": nb_features},
    )


def read_json_file(path: str) -> dict:
    with open(path) as fp:
        return json.load(fp)


def write_json_file(path: str, data: dict, mode: str = "w") -> None:
    with open(path, mode) as fp:
        json.dump(data, fp, indent=4)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", type=int)

    return parser.parse_args()
