from __future__ import annotations
from dimlpfidex.fidex import fidexGlo
from dimlpfidex.dimlp import densCls
from src.utils import read_json_file
from trainings import normalization
from datetime import datetime
import src.data_helper as dh
from pathlib import Path
from src.utils import *
from src.rule import *
import pandas as pd
import numpy as np
import math
import os


class Patient:
    def __init__(
        self, metadata: pd.Series, clinical_data: pd.Series, abspath: str
    ) -> None:
        self.study_id = metadata.loc["STUDYID"]
        self.site_idn = metadata.loc["SITEIDN"]
        self.site_name = metadata.loc["SITENAME"]
        self.subj_id = metadata.loc["SUBJID"]
        self.visit = metadata.loc["VISIT"]
        self.clinical_data = clinical_data

        self.project_abspath = abspath
        self.reldir = os.path.join("patients_data", self.subj_id)
        self.absdir = os.path.join(abspath, self.reldir)
        Path(self.absdir).mkdir(exist_ok=True)

        # properties to be set during rule extraction
        self.risk = -1
        self.low_interval = -1
        self.high_interval = -1
        self.selected_rules = []

    def to_str_list(self) -> list[list[str]]:
        res = []
        row = [
            str(self.study_id),
            str(self.site_idn),
            str(self.site_name),
            str(self.subj_id),
            str(self.visit),
            str(round(self.risk * 100, 4)),
            str(round(self.low_interval * 100, 4)),
            str(round(self.high_interval * 100, 4)),
        ]

        for rule in self.selected_rules:
            res.append(row + rule.to_str_list())

        return res

    def __prepare_input_for_extraction(self) -> str:
        input_filepath = os.path.join(self.absdir, "input_data.csv")

        # one hotting classes (useless but must be done in order to work with densCls)
        to_write = self.clinical_data.to_frame().T.copy()
        to_write = to_write.assign(Lymphodema_NO=lambda x: 1)
        to_write = to_write.assign(Lymphodema_YES=lambda x: 0)
        to_write.to_csv(input_filepath, sep=" ", header=False, index=False)

        normalized_file_path = f"{self.reldir}/input_data_normalized.csv"
        normalization(self.__get_normalization_config())

        # one hotting classes (useless but must be done in order to work with densCls)
        to_write = pd.read_csv(normalized_file_path, sep=" ", header=None)
        to_write = to_write.assign(Lymphodema_NO=lambda x: 1)
        to_write = to_write.assign(Lymphodema_YES=lambda x: 0)
        to_write.to_csv(normalized_file_path, sep=" ", header=False, index=False)

    def extract_rules(
        self, global_rules: GlobalRules, normalize: bool = True
    ) -> GlobalRules:
        if normalize:
            self.__prepare_input_for_extraction()

        densCls(self.__get_denscls_config())
        fidexGlo(self.__get_fidexglo_config())

        self.__set_metrics()
        self.__set_risk()
        self.__rewrite_extracted_rules_file()

        if normalize:
            normalization(self.__get_denormalization_config())
            selected_rules_dict = read_json_file(
                f"{self.absdir}/extracted_rules_denormalized.json"
            )
        else:
            selected_rules_dict = read_json_file(f"{self.absdir}/extracted_rules.json")

        self.selected_rules = Rule.list_from_dict(selected_rules_dict)
        
        # return updated global rules
        return self.__process_selected_rules(global_rules)

    def __get_normalization_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            f"--data_files [{self.reldir}/input_data.csv] "
            f"--output_normalization_file {self.absdir}/normalization_stats.txt "
            "--normalization_file temp/normalization_stats.txt "
            "--nb_attributes 79 "
            "--nb_classes 2 "
            "--attributes_file temp/attributes.txt "
            "--missing_values NaN"
        )

    def __get_denormalization_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            f"--rule_files {self.reldir}/extracted_rules.json "
            "--normalization_file temp/normalization_stats.txt "
            "--nb_attributes 79 "
            "--attributes_file temp/attributes.txt"
        )

    def __get_fidexglo_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            "--train_data_file temp/train_data_normalized.csv "
            "--train_class_file temp/train_classes.csv "
            "--train_pred_file temp/dimlpBTTrain.out "
            f"--test_data_file {self.reldir}/input_data_normalized.csv "
            f"--test_pred_file {self.reldir}/prediction.csv "
            "--weights_file temp/dimlpBT.wts "
            "--global_rules_file temp/global_rules.json "
            f"--console_file {self.reldir}/202412031114.log "
            "--nb_attributes 79 "
            "--nb_classes 2 "
            f"--explanation_file {self.reldir}/extracted_rules.json "
            "--with_minimal_version true "
            "--with_fidex true "
            "--min_fidelity 1.0 "
            "--lowest_min_fidelity 1.0 "
            "--min_covering 6 "
            "--max_iterations 20 "
            "--nb_fidex_rules 1 "
            "--dropout_dim 0.5 "
            "--dropout_hyp 0.5"
        )

    def __get_denscls_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            "--train_data_file temp/train_data_normalized.csv "
            "--train_class_file temp/train_classes.csv "
            f"--test_data_file {self.reldir}/input_data_normalized.csv "
            "--train_pred_outfile temp/train_pred_out.csv "
            "--weights_file temp/dimlpBT.wts "
            f"--stats_file {self.reldir}/densCls_stats.txt "
            "--hidden_layers_file temp/hidden_layers.out "
            f"--metrics_file {self.reldir}/densClsMetrics.json "
            "--nb_attributes 79 "
            "--nb_classes 2 "
            f"--test_pred_outfile {self.reldir}/prediction.csv "
            f"--console_file {self.reldir}/202412031114.log"
        )

    def __process_selected_rules(self, global_rules: GlobalRules) -> GlobalRules:
        updated_selected_rules = []

        for rule in self.selected_rules:
            id = global_rules.get_rule_id(rule)
            if id == -1:
                id == len(global_rules.rules)
                global_rules.rules.append(rule)  # save generated rule if not found

            updated_selected_rules.append(rule.set_id(id))

        self.selected_rules = updated_selected_rules
        return global_rules

    def __rewrite_extracted_rules_file(self) -> None:
        rules_file_path = os.path.join(self.reldir, "extracted_rules.json")
        selected_rules = read_json_file(rules_file_path)["samples"][0]["rules"]

        with open(rules_file_path, "w") as fp:
            json.dump({"rules": selected_rules}, fp, indent=4)

    def __set_metrics(self):
        data = read_json_file(os.path.join(self.absdir, "densClsMetrics.json"))
        nb_nets = data["nbNets"]

        self.low_interval = (
            data["avgs"][1] - (1.96 / math.sqrt(nb_nets)) * data["stds"][1]
        )
        self.high_interval = (
            data["avgs"][1] + (1.96 / math.sqrt(nb_nets)) * data["stds"][1]
        )

    def __set_risk(self):
        preds_filepath = os.path.join(self.absdir, "prediction.csv")
        self.risk = np.loadtxt(preds_filepath)[1]

    def format_results(self) -> str:
        string = ""

        for row in self.to_str_list():
            string += ",".join(row) + "\n"

        return string

    def write_results(self, attributes: list[str]) -> None:
        unicancer_headers = ["VISIT", "STUDY_ID", "SITE_NAME", "SITENAME", "SUBJECT_ID"]
        rule_headers = ["RISK", "LOW_INTERVAL", "HIGH_INTERVAL", "RULE_ID"]
        headers = unicancer_headers + rule_headers + attributes

        with open(
            f"{self.project_abspath}/output/{self.subj_id}_results_{datetime.today().strftime('%Y%m%d%H%M')}.csv",
            "w",
        ) as fp:
            fp.write(",".join(headers) + "\n")
            for row in self.to_str_list():
                fp.write(",".join(row) + "\n")

    def __repr__(self) -> str:
        if len(self.selected_rules) < 1:
            rules_str = "None"
        else:
            rules_str = "\n".join(
                [rule.__repr__() + " " for rule in self.selected_rules]
            )

        return f"""Patient:
Study ID: {self.study_id}
Site IDN: {self.site_idn}
Site name: {self.site_name}
Subject ID: {self.subj_id}
Visit: {self.visit}
Risk: {self.risk}
Low conf. int.: {self.low_interval}
High conf. int.: {self.high_interval}
Patient's data: 
{self.clinical_data}
Selected rules: {rules_str}"""

    def pretty_repr(self, attributes: list[str]) -> str:
        rules_str = (
            "".join(
                [rule.pretty_repr(attributes) + "\n" for rule in self.selected_rules]
            )
            + "\n"
        )

        return f"""Patient:
Study ID: {self.study_id}
Site IDN: {self.site_idn}
Site name: {self.site_name}
Subject ID: {self.subj_id}
Visit: {self.visit}
Risk: {self.risk}
Low conf. int.: {self.low_interval}
High conf. int.: {self.high_interval}
Patient's data: 
{self.clinical_data}
Selected rules: 
{rules_str}
        """


def write_patients(abspath: str) -> list[Patient]:
    input_file_path = get_most_recent_input_file(abspath)

    # TODO: check if read_excel is adapted
    metadata = pd.read_excel(input_file_path, index_col=False).iloc[:, :5]
    clinical_data = dh.obtain_data(input_file_path, training=False)
    clinical_data = clinical_data.assign(
        BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3)
    )

    patients = []
    for i in range(clinical_data.shape[0]):
        patients.append(Patient(metadata.iloc[i], clinical_data.iloc[i], abspath))

    return patients


# this is for testing puroposes only
def write_samples_file(abspath: str, n: int) -> list[Patient]:
    metadata_file = os.path.join(
        abspath, "input", "PRE-ACT-01_Flow2_20241115_HES-SO.xlsx"
    )
    data_file = os.path.join(abspath, "temp", "train_data.csv")

    metadata = pd.read_excel(metadata_file, index_col=False).iloc[0, :5]
    data = pd.read_csv(data_file, sep=" ", header=None)

    sample_data = data.sample(n)

    patients = []
    for i in range(n):
        metadata.iloc[3] = f"FRA_TEST_00{i+1}"
        patients.append(Patient(metadata, sample_data.iloc[i, :], abspath))

    return patients


def write_results(abspath: str, patients: list[Patient], attributes: list[str]) -> None:
    today = datetime.today().strftime("%Y_%m_%d")
    write_path = os.path.join(abspath, "output", f"results_{today}.csv")

    unicancer_headers = ["STUDYID", "SITEIDN", "SITENAME", "SUBJID", "VISIT"]
    rule_headers = ["RISK", "LOW_INTERVAL", "HIGH_INTERVAL", "RULE_ID"]
    headers = unicancer_headers + rule_headers + attributes

    string = ",".join(headers) + "\n"

    for patient in patients:
        string += patient.format_results()

    with open(write_path, "w") as fp:
        fp.write(string)
