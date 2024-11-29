from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
from src.rule import *
from pathlib import Path
from datetime import datetime
from trainings import normalization
from dimlpfidex.dimlp import densCls
from dimlpfidex.fidex import fidexGlo
from src.utils import update_config_file, read_json_file


class Patient:
    def __init__(
        self, patient_data: pd.DataFrame, clinical_data: pd.DataFrame, abspath: str
    ) -> None:
        self.study_id = patient_data.iloc[0, 0]
        self.site_idn = patient_data.iloc[0, 1]
        self.site_name = patient_data.iloc[0, 2]
        self.subj_id = patient_data.iloc[0, 3]
        self.visit = patient_data.iloc[0, 4]
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
        normalization_conf_file = os.path.join(
            self.project_abspath, "config", "normalization.json"
        )

        # one hotting classes (useless but must be done in order to work with densCls)
        to_write = self.clinical_data.to_frame().T.copy()
        to_write = to_write.assign(Lymphodema_NO=lambda x: 1)
        to_write = to_write.assign(Lymphodema_YES=lambda x: 0)
        to_write.to_csv(input_filepath, sep=" ", header=False, index=False)

        update_config_file(
            normalization_conf_file,
            {
                "data_files": f"[{self.reldir}/input_data.csv]",
                "output_normalization_file": f"{self.reldir}/normalization_stats.txt",
            },
        )
        # TODO: normalization("--json_config_file config/normalization.json") 

    def extract_rules(self):
        denscls_conf_file = os.path.join(self.project_abspath, "config", "denscls.json")
        fidexglo_conf_file = os.path.join(
            self.project_abspath, "config", "fidexglo.json"
        )

        update_config_file(
            denscls_conf_file,
            {
                "test_data_file": f"{self.reldir}/input_data.csv",
                "test_pred_outfile": f"{self.reldir}/prediction.csv",
                "metrics_file": f"{self.reldir}/densClsMetrics.json",
                "console_file":  f"{self.reldir}/{datetime.today().strftime('%Y%m%d%H%M')}.log"
            },
        )

        update_config_file(
            fidexglo_conf_file,
            {
                "test_data_file": f"{self.reldir}/input_data.csv",
                "test_pred_file": f"{self.reldir}/prediction.csv",
                "explanation_file": f"{self.reldir}/extracted_rules.json",
                "console_file":  f"{self.reldir}/{datetime.today().strftime('%Y%m%d%H%M')}.log"
            },
        )

        self.__prepare_input_for_extraction()
        densCls(f"--json_config_file {denscls_conf_file}")
        fidexGlo(f"--json_config_file {fidexglo_conf_file}")

        self.__set_metrics()
        self.__set_risk()

        selected_rules = read_json_file(f"{self.absdir}/extracted_rules.json")[
            "samples"
        ][0]
        self.selected_rules = Rule.list_from_dict(selected_rules)

        # normalization("--json_config_file config/denormalization.json")

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
        rules_str = (
            "".join([rule.__repr__() + " " for rule in self.selected_rules]) + "\n"
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
