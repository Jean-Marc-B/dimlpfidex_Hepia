from dimlpfidex.fidex import fidexGlo, fidexGloRules
from dimlpfidex.dimlp import dimlpBT, dimlpPred
from trainings import normalization
from datetime import datetime
import data_helper as dh
import pandas as pd
import json
import csv
import os


def update_config_file(filename: str, params: dict) -> None:
    with open(filename, "r+") as f:
        config = json.load(f)
        for key, param in params.items():
            config[key] = param

        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()


def update_config_files(root_folder, nb_features, nb_classes):
    confpath = "config/"
    logpath = "output/logs/"

    dimlpbt_config_filename = confpath + "dimlpbt.json"
    dimlpbt_config = dict()
    dimlpbt_config["root_folder"] = root_folder
    dimlpbt_config["nb_attributes"] = nb_features
    dimlpbt_config["nb_classes"] = nb_classes
    dimlpbt_config["console_file"] = (
        f"{logpath + datetime.today().strftime('%Y%m%d%H%M%S')}_dimlpbt.log"
    )

    dimlppred_config_filename = confpath + "dimlppred.json"
    dimlppred_config = dict()
    dimlppred_config["root_folder"] = root_folder
    dimlppred_config["nb_attributes"] = nb_features
    dimlppred_config["nb_classes"] = nb_classes
    dimlppred_config["console_file"] = (
        f"{logpath + datetime.today().strftime('%Y%m%d%H%M%S')}_dimlppred.log"
    )

    fidexglorules_config_filename = confpath + "fidexglorules.json"
    fidexglorules_config = dict()
    fidexglorules_config["root_folder"] = root_folder
    fidexglorules_config["nb_attributes"] = nb_features
    fidexglorules_config["nb_classes"] = nb_classes
    fidexglorules_config["console_file"] = (
        f"{logpath + datetime.today().strftime('%Y%m%d%H%M%S')}_fidexglorules.log"
    )

    fidexglo_config_filename = confpath + "fidexglo.json"
    fidexglo_config = dict()
    fidexglo_config["root_folder"] = root_folder
    fidexglo_config["nb_attributes"] = nb_features
    fidexglo_config["nb_classes"] = nb_classes
    fidexglo_config["console_file"] = (
        f"{logpath + datetime.today().strftime('%Y%m%d%H%M%S')}_fidexglo.log"
    )

    update_config_file(dimlppred_config_filename, dimlppred_config)
    update_config_file(dimlpbt_config_filename, dimlpbt_config)
    update_config_file(fidexglorules_config_filename, fidexglorules_config)
    update_config_file(fidexglo_config_filename, fidexglo_config)


def write_single_sample(data: pd.DataFrame, labels: pd.DataFrame, n: int) -> None:
    sample_data = (
        data.iloc[n].to_frame().T
    )  # to_frame.T to avoid sample being written vertically

    sample_class = labels.iloc[n]

    sample_data.to_csv("input/test_sample_data.csv", header=False, index=False)

    with open("input/test_sample_class.csv", "w") as fp:
        fp.write(str(sample_class))


def preprocess_data(
    data: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # enlève les données où les labels sont indéterminés
    labels = labels[labels != 2]

    # filtre les données selon les critères envoyés par Guido
    data = data[
        ((data["Clinical T-Stage_nan"] != 1) & (data["Clinical N-Stage_nan"] != 1))
        & ~(
            (data["Clinical N-Stage_N0"] == 1)
            & (data["Clinical N-Stage_N1"] == 0)
            & (data["Clinical T-Stage_T0"] == 1)
            & (data["Clinical T-Stage_Tis"] == 0)
        )
        & (
            (data["Planned axillary dissection"] == 1)
            | (data["Sentinel node biopsy"] == 1)
        )
    ]

    # enlève les lignes correspondantes aux labels indéterminés
    data = data.loc[data.index.isin(labels.index)]

    # harmonise en ne gardant que les indexs contenus dans Data
    labels = labels.loc[labels.index.isin(data.index)]

    return data, labels


def write_train_data(data: pd.DataFrame, labels: pd.Series) -> None:
    data.to_csv("temp/train_data.csv", sep=",", header=False, index=False)
    labels.to_csv("temp/train_classes.csv", sep=",", header=False, index=False)


def get_inequality(b: bool) -> str:
    return ">=" if b else "<"


def write_results(labels: pd.DataFrame, data: dict, nb_features: int) -> None:
    data = []

    for sample in sampleRules["samples"]:
        for rule in sample["rules"]:
            line = [""] * nb_features
            line[0] = labels.index[sample["sampleId"]]
            line[1] = "idrule"  # TODO: define rule IDs
            line[2] = "risk"  # TODO: probability given by dimlpBT 1st neuron
            line[3] = (
                "confidence interval"  # TODO: std. dev. of dimlpCls (ask JM) (each dimlpBT network is used in dimlpCls)
            )
            line[4] = rule["coveringSize"]
            for antecedant in rule["antecedents"]:
                line[antecedant["attribute"] + 4] = get_inequality(
                    antecedant["inequality"]
                ) + str(antecedant["value"])

            data.append(line)

    write_csv(data)


def write_csv(data: list[list]) -> None:
    with open("output/result.csv", "w") as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerows(data)


if __name__ == "__main__":
    abspath = os.path.abspath(os.path.dirname(__file__)) + "/"
    data, labels = dh.obtain_data("dataset/clinical_complete_rev1.csv")

    data, labels = preprocess_data(
        data, labels
    )  # This should ensure data are well shaped (according to Guido's directives)

    nb_classes = 2
    nb_features = data.shape[1]

    write_train_data(data, labels)
    write_single_sample(data, labels, 10)
    update_config_files(abspath, nb_features, nb_classes)

    # normalization("--json_config_file config/normalization.json")
    # dimlpBT("--json_config_file config/dimlpbt.json")
    # fidexGloRules("--json_config_file config/fidexglorules.json")
    # dimlpPred("--json_config_file config/dimlppred.json")
    # fidexGlo("--json_config_file config/fidexglo.json")

    with open("temp/explanation.json") as fp:
        sampleRules = json.load(fp)

    with open("temp/global_rules.json") as fp:
        globalRules = json.load(fp)

    write_results(labels, sampleRules, nb_features)

    # globalRulesSet = set(globalRules.values())

    # TODO normalization file to be generated (ask JM) (for BT and GloRules)
