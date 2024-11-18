from dimlpfidex.fidex import fidexGlo, fidexGloRules
from dimlpfidex.dimlp import dimlpBT, densCls
from trainings import normalization
import tests_ressources.src.data_helper as dh
from tests_ressources.src.utils import *
from tests_ressources.src.rule import *
import pandas as pd
import numpy as np
import math
import os


# This get random samples until we get real data
def write_test_samples(nsamples: int) -> None:
    data = pd.read_csv("temp/train_data_normalized.csv", sep=" ")

    samples_data = data.sample(nsamples)

    # one hotting classes (useless but must be done)
    samples_data = samples_data.assign(Lymphodema_NO=lambda x: 1)
    samples_data = samples_data.assign(Lymphodema_YES=lambda x: 0)

    samples_data.to_csv(
        "input/test_sample_data.csv", header=False, index=False, sep=","
    )

    return samples_data.index


def write_train_data(data: pd.DataFrame, labels: pd.Series) -> None:
    labels = pd.get_dummies(labels).astype("uint")

    with open("temp/attributes.txt", "w") as f:
        f.writelines(data.columns + "\n")

    data.to_csv("temp/train_data.csv", sep=",", header=False, index=False)
    labels.to_csv("temp/train_classes.csv", sep=",", header=False, index=False)


def write_results(
    used_rules_id: list[int], sample_ids: list[str], nb_features: int
) -> None:
    res = []
    g_rules = read_json_file("temp/global_rules_denormalized.json")

    lower_interval, upper_interval = get_metrics()
    risks = get_risk()

    # add STUDYID, SITEIDN, SITENAME, SUBJID, VISIT

    for sample, risk in zip(data["samples"], risks):
        for i, rule in enumerate(sample["rules"]):
            line = [""] * (nb_features + 5)
            line[0] = sample_ids[sample["sampleId"]]
            line[1] = used_rules_id[i]
            line[2] = f"{risk * 100.0:.3f}"
            line[3] = f"{lower_interval: .3f}"
            line[4] = f"{upper_interval: .3f}"
            line[5] = rule["coveringSize"]
            for antecedant in rule["antecedents"]:
                line[antecedant["attribute"] + 5] = get_inequality(
                    antecedant["inequality"]
                ) + str(antecedant["value"])

            res.append(line)

    write_csv(res)


def get_risk() -> list[float]:
    abspath = os.path.abspath(os.path.dirname(__file__))
    preds_filepath = os.path.join(abspath, "input", "test_sample_pred.csv")

    return np.loadtxt(preds_filepath)[:, 1]


def get_metrics() -> tuple[float, float]:
    data = read_json_file(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "temp", "stds-avg.json")
    )
    nb_nets = data["nbNets"]

    lower_conf_interval = (
        data["avgs"][1] - (1.96 / math.sqrt(nb_nets)) * data["stds"][1]
    )
    upper_conf_interval = (
        data["avgs"][1] + (1.96 / math.sqrt(nb_nets)) * data["stds"][1]
    )

    nb_decimals = 4
    return (
        round(lower_conf_interval, nb_decimals),
        round(upper_conf_interval, nb_decimals),
    )



def pretty_print_results(used_rules_id: list[int], sample_ids: list[str], data: dict):
    lower_interval, upper_interval = get_metrics()
    risks = get_risk()

    for sample, risk in zip(data["samples"], risks):
        print(f"Sample {sample_ids[sample['sampleId']]} activated rules:")
        for i, rule in enumerate(sample["rules"]):
            print(f"Rule ID {used_rules_id[i]}")
            print(
                f"Risk {risk} \n  lower interval: {lower_interval}\n  upper interval: {upper_interval}"
            )
            # print_rule(rule)
            print("\n")




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

    write_train_data(data, labels)
    update_config_files(abspath, nb_features, nb_classes)
    normalization("--json_config_file config/normalization.json")

    # TODO: there will be missing values in the real samples
    samples_ids = write_test_samples(5)

    if args.train:
        dimlpBT("--json_config_file config/dimlpbt.json")
        fidexGloRules("--json_config_file config/fidexglorules.json")

    densCls("--json_config_file config/denscls.json")
    fidexGlo("--json_config_file config/fidexglo.json")
    normalization("--json_config_file config/denormalization.json")

    global_rules = GlobalRules.from_json_file("temp/global_rules_denormalized.json")
    patient_rules = read_json_file("temp/explanation.json")



    # TODO: add generated rules with the global rules
    # selected_rules = 

    write_results(selected_rules_id, samples_ids, samples_rules, nb_features)
