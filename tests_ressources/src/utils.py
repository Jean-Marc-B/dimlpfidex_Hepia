import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

def get_most_recent_file(absolute_path: str) -> str:
    # get all filepaths inside input/ folder 
    list_filepaths = [os.path.join(absolute_path, filename) for filename in os.listdir(absolute_path)]
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


def write_csv(path: str, data: list[list]) -> None:
    with open(path, "w") as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerows(data)


def check_working_subdirs(abspath: str) -> None:
    dirs = ["temp", "logs"]

    for dir in dirs:
        path = os.path.join(abspath, dir)
        Path(path).mkdir(exist_ok=True)


def clean_dir(abspath: str, dirname: str):
    nremoved = 0
    temppath = os.path.join(
        abspath,
        dirname,
    )

    print(f"Cleaning {temppath} directory...")

    for file in os.listdir(temppath):
        filepath = os.path.join(temppath, file)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
                nremoved += 1
        except OSError as ose:
            print(f"Error occured while trying to delete {file}. Error: {ose}")

    print(f"{temppath} cleaned, {nremoved} files deleted.")


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-c", "--clean", action="store_true")
    parser.add_argument("--cleanall", action="store_true")

    return parser.parse_args()
