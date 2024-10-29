from dimlpfidex.fidex import fidexGlo, fidexGloRules
from dimlpfidex.dimlp import dimlpBT, densCls
from trainings import normalization
from datetime import datetime
import data_helper as dh
import pandas as pd
import hashlib as hash
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


def update_config_files(root_folder: str, nb_features: int, nb_classes: int):
    confpath = os.path.join(root_folder, "config")
    logpath = "output/logs"

    programs = ["dimlpbt", "denscls", "fidexglo", "fidexglorules"]

    for program in programs:
        config_filename = os.path.join(confpath, f"{program}.json")
        config = dict()
        config["root_folder"] = root_folder
        config["nb_attributes"] = nb_features
        config["nb_classes"] = nb_classes
        config["console_file"] = (
            f"{logpath + datetime.today().strftime('%Y%m%d%H%M%S')}_{program}.log"
        )

        update_config_file(config_filename, config)

    update_config_file(
        os.path.join(confpath, "normalization.json"), {"nb_attributes": nb_features}
    )
    update_config_file(
        os.path.join(confpath, "denormalization.json"), {"nb_attributes": nb_features}
    )


# This has a random behaviour yet until we get real data
def write_test_samples(nsamples: int) -> None:
    data = pd.read_csv("temp/train_data_normalized.csv")

    samples_data = data.sample(
        nsamples
    )  # to_frame.T to avoid sample being written vertically

    # one hotting classes (useless but must be done, )
    samples_data = samples_data.assign(Lymphodema_NO=lambda x: 1)
    samples_data = samples_data.assign(Lymphodema_YES=lambda x: 0)

    samples_data.to_csv("input/test_sample_data.csv", header=False, index=False)

    return samples_data.index


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

    # Ajoute le BMI
    data = data.assign(BMI=lambda x: round(x.weight / (x.height / 100.0) ** 2, 3))

    return data, labels


def write_train_data(data: pd.DataFrame, labels: pd.Series) -> None:
    labels = pd.get_dummies(labels).astype("uint")
    data.to_csv("temp/train_data.csv", sep=",", header=False, index=False)
    labels.to_csv("temp/train_classes.csv", sep=",", header=False, index=False)


def read_json_file(path: str) -> dict:
    with open(path) as fp:
        return json.load(fp)


def get_inequality(b: bool) -> str:
    return ">=" if b else "<"


def write_results(
    used_rules_id: list[int], sample_ids: list[str], data: dict, nb_features: int
) -> None:
    res = []

    for sample in data["samples"]:
        for i, rule in enumerate(sample["rules"]):
            line = [""] * (nb_features + 5)
            line[0] = sample_ids[sample["sampleId"]]
            line[1] = used_rules_id[i]
            line[2] = "risk"  # TODO: probability given by dimlpBT 1st neuron
            line[3] = (
                "confidence interval"  # TODO: std. dev. of dimlpCls (ask JM) (each dimlpBT network is used in dimlpCls)
            )
            line[4] = rule["coveringSize"]
            for antecedant in rule["antecedents"]:
                line[antecedant["attribute"] + 5] = get_inequality(
                    antecedant["inequality"]
                ) + str(antecedant["value"])

            res.append(line)

    write_csv(res)


def write_csv(data: list[list]) -> None:
    with open("output/result.csv", "w") as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerows(data)


def print_rule(rule):
    print(
        f"""Rule:
        - accuracy: {rule["accuracy"]}
        - antecedants: {rule["antecedents"]}
        - confidence: {rule["confidence"]}
        - samples covered: {rule["coveringSize"]}
        - fidelity: {rule["fidelity"]}"""
    )


def global_rule_id(rule):
    global_rules = read_json_file("temp/global_rules.json")

    print(len(global_rules))
    print("Searing for: ")
    print_rule(rule)
    print()

    for i, global_rule in enumerate(global_rules["rules"]):
        print("Trying to match with: ")
        print_rule(global_rule)
        if global_rule == rule:
            print(f"ID = {i}")
            print("*" * 30)
            return i

    return -1


if __name__ == "__main__":
    abspath = os.path.abspath(os.path.dirname(__file__))
    data, labels = dh.obtain_data("dataset/clinical_complete_rev1.csv")

    data, labels = preprocess_data(data, labels)
    nb_classes = 2
    nb_features = data.shape[1]

    write_train_data(data, labels)
    update_config_files(abspath, nb_features, nb_classes)
    normalization("--json_config_file config/normalization.json")

    # TODO: there will be missing values in the real samples
    samples_id = write_test_samples(10)

    # dimlpBT("--json_config_file config/dimlpbt.json")
    # fidexGloRules("--json_config_file config/fidexglorules.json")
    # densCls("--json_config_file config/denscls.json")
    # fidexGlo("--json_config_file config/fidexglo.json")

    samples_rules = read_json_file("temp/explanation.json")

    # TODO: do we add generated rules with the global rules ? 
    used_rules_id = [
        global_rule_id(rule)
        for sample in samples_rules["samples"]
        for rule in sample["rules"]
    ]

    normalization("--json_config_file config/denormalization.json")

    write_results(used_rules_id, samples_id, samples_rules, nb_features)

    print("OK")
