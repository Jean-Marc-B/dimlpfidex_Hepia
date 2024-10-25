from dimlpfidex.fidex import fidexGlo, fidexGloRules
from dimlpfidex.dimlp import dimlpBT, densCls
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
    programs = ["dimlpbt", "denscls", "fidexglo", "fidexglorules"]
    

    for program in programs:
        config_filename = confpath + f"{program}.json"
        config = dict()
        config["root_folder"] = root_folder
        config["nb_attributes"] = nb_features
        config["nb_classes"] = nb_classes
        # config["train_data_file"] = "train_data"
        config["console_file"] = (
            f"{logpath + datetime.today().strftime('%Y%m%d%H%M%S')}_{program}.log"
        )

        update_config_file(config_filename, config)


def write_single_sample(data: pd.DataFrame, labels: pd.DataFrame, n: int) -> None:
    sample_data = (
        data.iloc[n].to_frame().T
    )  # to_frame.T to avoid sample being written vertically

    sample_data=sample_data.assign(Lymphodema_NO=[1])
    sample_data=sample_data.assign(Lymphodema_YES=[0])
    sample_data.to_csv("input/test_sample_data.csv", header=False, index=False)
    
    return labels.index[n]


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
    labels = pd.get_dummies(labels).astype('uint')
    data.to_csv("temp/train_data.csv", sep=",", header=False, index=False)
    labels.to_csv("temp/train_classes.csv", sep=",", header=False, index=False)


def read_json_file(path: str) -> dict:
    with open(path) as fp:
        return json.load(fp)
    

def get_inequality(b: bool) -> str:
    return ">=" if b else "<"


def write_results(sample_ids: list[str], data: dict, nb_features: int) -> None:
    res = []

    for sample in data["samples"]:
        for rule in sample["rules"]:
            line = [""] * (nb_features + 5)
            line[0] = sample_ids[sample["sampleId"]]
            line[1] = "idrule"  # TODO: define rule IDs
            line[2] = "risk"  # TODO: probability given by dimlpBT 1st neuron
            line[3] = "confidence interval"  # TODO: std. dev. of dimlpCls (ask JM) (each dimlpBT network is used in dimlpCls)
            line[4] = rule["coveringSize"]
            for antecedant in rule["antecedents"]:
                line[antecedant["attribute"] + 5] = get_inequality(antecedant["inequality"]) + str(antecedant["value"])

            res.append(line)

    write_csv(res)


def write_csv(data: list[list]) -> None:
    with open("output/result.csv", "w") as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerows(data)

def global_rule_id(rule):
    global_rules = read_json_file("temp/global_rules_denormalized.json")

    for i,global_rule in enumerate(global_rules["rules"]):
        if global_rule == rule:
            print("match")
            return i
        
    return -1

if __name__ == "__main__":
    abspath = os.path.abspath(os.path.dirname(__file__)) + "/"
    data, labels = dh.obtain_data("dataset/clinical_complete_rev1.csv")
    # TODO: compute and add BMI (height & weight, beware of units)

    data, labels = preprocess_data(
        data, labels
    )  # This should ensure data are well shaped (according to Guido's directives)
    
    write_train_data(data, labels)
    normalization("--json_config_file config/normalization.json")

    nb_classes = 2
    nb_features = data.shape[1]

    # TODO: there will be missing values in the real samples
    sample_id = write_single_sample(data, labels, 10)
    update_config_files(abspath, nb_features, nb_classes)
    

    # dimlpBT("--json_config_file config/dimlpbt.json") # TODO: re-add 10 for the second hidden layer
    # fidexGloRules("--json_config_file config/fidexglorules.json")
    # densCls("--json_config_file config/denscls.json") 
    # fidexGlo("--json_config_file config/fidexglo.json")

    normalization("--json_config_file config/denormalization.json")
    
    #TODO denormalize rules
    samples_rules = read_json_file("temp/explanation.json")

    for sample in samples_rules["samples"]:
        for rule in sample["rules"]:
            global_rule_id(rule)

    # write_results([sample_id], samples_rules, nb_features)

    print('OK')