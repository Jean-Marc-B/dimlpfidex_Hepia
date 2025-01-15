from __future__ import annotations
from src.global_rules import GlobalRules
from dimlpfidex.fidex import fidexGlo
from dimlpfidex.dimlp import densCls
from src.utils import read_json_file
from trainings import normalization
import src.constants as constants
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
        self.reldir = os.path.join(constants.PATIENTS_DATA_DIRNAME, self.subj_id)
        self.absdir = os.path.join(abspath, self.reldir)

        # if a patient ID is a duplicate, then a copy of its folder is created inside patients_data folder
        i = 2
        suffix = ""
        while not create_folder(self.absdir + suffix):
            suffix = f"_{i}"
            i += 1
            print(f"Creating a copy with this name instead: {self.reldir + suffix}")

        self.reldir += suffix
        self.absdir += suffix

        # properties to be set during rule extraction
        self.risk = -1
        self.low_interval = -1
        self.high_interval = -1
        self.selected_rules = []

    def to_str_list(self) -> list[list[str]]:
        res = []

        for rule in self.selected_rules:
            rule = rule.postprocess()

            row = [
                str(self.study_id),
                str(self.site_idn),
                str(self.site_name),
                str(self.subj_id),
                str(self.visit),
                str(rule.id),
                str(round(self.risk * 100, 4)),
                str(round(self.low_interval * 100, 4)),
                str(round(self.high_interval * 100, 4)),
                str(rule.covering),
            ] + rule.to_str_list()

            res.append(row)

        return res

    def __prepare_input_for_extraction(self) -> str:
        input_filepath = os.path.join(self.absdir, "input_data.csv")

        # one hotting classes (useless but must be done in order to work with densCls)
        to_write = self.clinical_data.to_frame().T.copy()
        to_write = to_write.assign(Lymphodema_NO=lambda x: 1)
        to_write = to_write.assign(Lymphodema_YES=lambda x: 0)
        to_write.to_csv(input_filepath, sep=" ", header=False, index=False)

        normalized_file_path = f"{self.reldir}/input_data_normalized.csv"
        self.exec_normalization()

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

        self.exec_prediction()
        self.exec_rule_extraction()

        self.__set_metrics()
        self.__set_risk()
        self.__rewrite_extracted_rules_file()

        if normalize:
            self.exec_denormalization()
            selected_rules_dict = read_json_file(
                f"{self.absdir}/extracted_rules_denormalized.json"
            )
        else:
            selected_rules_dict = read_json_file(f"{self.absdir}/extracted_rules.json")

        attributes = read_attributes_file(self.project_abspath)
        self.selected_rules = Rule.list_from_dict(selected_rules_dict, attributes)

        # return updated global rules
        updated_global_rules = self.__process_selected_rules(global_rules)

        # write human readable file
        human_readable_filepath = os.path.join(self.absdir, "readable_output.txt")
        with open(human_readable_filepath, "w") as f:
            f.write(self.pretty_repr(attributes))

        return updated_global_rules

    def exec_prediction(self) -> None:
        config = self.__get_denscls_config()
        status = densCls(config)

        if status != 0:
            print(
                "-" * 20
                + "AN ERROR OCCURED"
                + "-" * 20
                + f"\nStopping model prediction process for patient ID {self.subj_id} for the reason above."
            )
            exit(1)

    def exec_rule_extraction(self) -> None:
        config = self.__get_fidexglo_config()
        status = fidexGlo(config)

        if status != 0:
            print(
                "-" * 20
                + "AN ERROR OCCURED"
                + "-" * 20
                + f"\nStopping rule extraction process for patient ID {self.subj_id} for the reason above."
            )
            exit(1)

    def exec_normalization(self) -> None:
        config = self.__get_normalization_config()
        status = normalization(config)

        if status != 0:
            print(
                "-" * 20
                + "AN ERROR OCCURED"
                + "-" * 20
                + f"\nStopping data normalization process for patient ID {self.subj_id} for the reason above."
            )
            exit(1)

    def exec_denormalization(self) -> None:
        config = self.__get_denormalization_config()
        status = normalization(config)

        if status != 0:
            print(
                "-" * 20
                + "AN ERROR OCCURED"
                + "-" * 20
                + f"\nStopping data denormalization process for patient ID {self.subj_id} for the reason above."
            )
            exit(1)

    def __get_normalization_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            f"--data_files [{self.reldir}/input_data.csv] "
            f"--output_normalization_file {self.absdir}/normalization_stats.txt "
            f"--normalization_file {constants.MODEL_DIRNAME}/normalization_stats.txt "
            "--nb_attributes 79 "
            "--nb_classes 2 "
            f"--attributes_file {constants.MODEL_DIRNAME}/attributes.txt "
            "--missing_values NaN"
        )

    def __get_denormalization_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            f"--rule_files {self.reldir}/extracted_rules.json "
            f"--normalization_file {constants.MODEL_DIRNAME}/normalization_stats.txt "
            "--nb_attributes 79 "
            f"--attributes_file {constants.MODEL_DIRNAME}/attributes.txt"
        )

    def __get_fidexglo_config(self) -> str:
        today = datetime.today().strftime("%Y_%m_%d")
        return (
            f"--root_folder {self.project_abspath} "
            f"--train_data_file {constants.MODEL_DIRNAME}/train_data_normalized.csv "
            f"--train_class_file {constants.MODEL_DIRNAME}/train_classes.csv "
            f"--train_pred_file {constants.MODEL_DIRNAME}/dimlpbt_train_predictions.out "
            f"--test_data_file {self.reldir}/input_data_normalized.csv "
            f"--test_pred_file {self.reldir}/prediction.csv "
            f"--weights_file {constants.MODEL_DIRNAME}/dimlpBT.wts "
            f"--attributes_file {constants.MODEL_DIRNAME}/attributes.txt "
            f"--global_rules_file {constants.MODEL_DIRNAME}/global_rules_normalized.json "
            f"--console_file {self.reldir}/fidexglo_{today}.log "
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
        today = datetime.today().strftime("%Y_%m_%d")
        return (
            f"--root_folder {self.project_abspath} "
            f"--train_data_file {constants.MODEL_DIRNAME}/train_data_normalized.csv "
            f"--train_class_file {constants.MODEL_DIRNAME}/train_classes.csv "
            f"--test_data_file {self.reldir}/input_data_normalized.csv "
            f"--train_pred_outfile {constants.MODEL_DIRNAME}/train_pred_out.csv "
            f"--attributes_file {constants.MODEL_DIRNAME}/attributes.txt "
            f"--weights_file {constants.MODEL_DIRNAME}/dimlpBT.wts "
            f"--stats_file {self.reldir}/densCls_stats.txt "
            f"--hidden_layers_file {constants.MODEL_DIRNAME}/hidden_layers.out "
            f"--metrics_file {self.reldir}/densClsMetrics.json "
            "--nb_attributes 79 "
            "--nb_classes 2 "
            f"--test_pred_outfile {self.reldir}/prediction.csv "
            f"--console_file {self.reldir}/denscls_{today}.log"
        )

    def __process_selected_rules(self, global_rules: GlobalRules) -> GlobalRules:
        for i in range(len(self.selected_rules)):
            rule = self.selected_rules[i]
            id = global_rules.get_rule_id(rule)

            if id == -1:
                id = len(global_rules.rules)
                global_rules.rules.append(rule)  # save generated rule if not found

            rule.set_id(id)

        return global_rules

    def __rewrite_extracted_rules_file(self) -> None:
        rules_file_path = os.path.join(self.reldir, "extracted_rules.json")
        selected_rules = read_json_file(rules_file_path)["samples"][0]["rules"]

        with open(rules_file_path, "w") as fp:
            json.dump({"rules": selected_rules}, fp, indent=4)

    def __set_metrics(self):
        data = read_json_file(os.path.join(self.absdir, "densClsMetrics.json"))
        preds_filepath = os.path.join(self.absdir, "prediction.csv")

        # setting risk (output of 2nd output neuron)
        self.risk = np.loadtxt(preds_filepath)[1]

        # setting upper and lower confidence interval
        interval = (1.96 / math.sqrt(data["nbNets"])) * data["stds"][1]
        self.low_interval = data["avgs"][1] - interval
        self.high_interval = data["avgs"][1] + interval

    def __set_risk(self):
        preds_filepath = os.path.join(self.absdir, "prediction.csv")
        self.risk = np.loadtxt(preds_filepath)[1]

    def format_results(self) -> list[list[str]]:
        return [row for row in self.to_str_list()]

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

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            clinical_data_str = self.clinical_data.__repr__()

        return f"""
Study ID: {self.study_id}
Site IDN: {self.site_idn}
Site name: {self.site_name}
Subject ID: {self.subj_id}
Visit: {self.visit}
Risk: {self.risk}
Low conf. int.: {self.low_interval}
High conf. int.: {self.high_interval}

Patient's initial data: 
{clinical_data_str}

Selected rules: 
{rules_str}
        """


def write_patients(abspath: str) -> list[Patient]:
    input_dirpath = os.path.join(abspath, constants.INPUT_DIRNAME)
    input_filepath = get_most_recent_input_file(input_dirpath)

    if input_filepath == "":
        print(f"ERROR: No input file inside {input_dirpath} was found")
        exit()

    ext = os.path.splitext(input_filepath)[1].lower()

    if ext in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:
        metadata = pd.read_excel(input_filepath, index_col=False).iloc[:, :5]

    elif ext == ".csv":
        metadata = pd.read_csv(input_filepath, index_col=False).iloc[:, :5]
    else:
        raise NotImplementedError(
            f"Support for {ext} extension in {input_filepath} file is not implemented."
        )

    clinical_data = dh.obtain_data(input_filepath, training=False)
    clinical_data = clinical_data.assign(
        BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3)
    )

    patients = []
    for i in range(clinical_data.shape[0]):
        patients.append(Patient(metadata.iloc[i], clinical_data.iloc[i], abspath))

    return patients


# this is for testing puroposes only
def write_samples_file(abspath: str, n: int) -> list[Patient]:
    data_file = os.path.join(abspath, constants.MODEL_DIRNAME, "test_data.csv")

    metadata = pd.Series(
        data={
            "STUDYID": "PRE-ACT-01-DRAFT",
            "SITEIDN": "FRA-98",
            "SITENAME": "UNICANCER_TEST",
            "SUBJID": "FRA-98-002",
            "VISIT": "BASELINE",
        }
    )
    data = pd.read_csv(data_file, sep=" ", header=None)
    max_test_samples = data.shape[0]

    if max_test_samples < n:
        raise ValueError(
            f"The number of test samples specified cannot be greater than {max_test_samples}"
        )

    sample_data = data.sample(n)

    patients = []
    for i in range(n):
        metadata.iloc[3] = f"FRA_TEST_00{i+1}"
        patients.append(Patient(metadata, sample_data.iloc[i, :], abspath))

    return patients


def write_results(abspath: str, patients: list[Patient]) -> None:
    today = datetime.today().strftime("%Y_%m_%d")
    write_path = os.path.join(abspath, constants.OUTPUT_DIRNAME, f"results_{today}.csv")

    attributes = read_attributes_file(abspath)
    unicancer_headers = ["STUDYID", "SITEIDN", "SITENAME", "SUBJID", "VISIT"]
    rule_headers = ["RULE_ID", "RISK", "LOW_INTERVAL", "HIGH_INTERVAL", "COVERING"]

    headers = unicancer_headers + rule_headers + attributes[:-2]

    data = [headers]

    for patient in patients:
        for record in patient.format_results():
            data.append(record)

    reorder_data_columns(data).to_csv(write_path, index=False)
