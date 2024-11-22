from __future__ import annotations
import os
import pandas as pd
from src.rule import *


class Patient:
    def __init__(self, input_data: pd.DataFrame) -> None:
        # add STUDYID, SITEIDN, SITENAME, SUBJID, VISIT
        if input_data.shape[0] != 1:
            raise ValueError(
                "Error while creating a patient: Cannot create a patient from more/less than one row of a pandas DataFrame."
            )

        self.study_id = input_data.iloc[0, 0]
        self.site_idn = input_data.iloc[0, 1]
        self.site_name = input_data.iloc[0, 2]
        self.subj_id = input_data.iloc[0, 3]
        self.visit = input_data.iloc[0, 4]
        self.input_data = input_data.iloc[0, 5:]

        # properties to be set during rule extraction
        self.risk = -1
        self.low_interval = -1
        self.high_interval = -1

        self.selected_rules = []

    def set_selected_rules(self, selected_rules: list[Rule]):
        self.selected_rules = selected_rules

    @staticmethod
    def create_test(input_data: pd.DataFrame) -> Patient:
        data = input_data.copy()

        data.insert(0, "VISIT", ["VISIT_000"])
        data.insert(0, "STUDY_ID", ["STUDYID_000"])
        data.insert(0, "SITE_NAME", ["SITE_000"])
        data.insert(0, "SITENAME", ["SITENAME_000"])
        data.insert(0, "SUBJECT_ID", ["SUBJID_000"])

        return Patient(data)

    def set_metrics(
        self, risk: float, low_interval: float, high_interval: float
    ) -> None:
        self.risk = risk
        self.low_interval = low_interval
        self.high_interval = high_interval

    def to_str_list(self) -> list[list[str]]:
        res = []
        row = [
            str(self.study_id),
            str(self.site_idn),
            str(self.site_name),
            str(self.subj_id),
            str(self.visit),
            str(round(self.risk*100,4)),
            str(self.low_interval),
            str(self.high_interval),
        ]

        for rule in self.selected_rules:
            res.append(row + rule.to_str_list())

        return res

    def prepare_input_for_extraction(self, path: str) -> str:
        filepath = os.path.join(path, f"input_{self.id}.csv")

        # one hotting classes (useless but must be done in order to work with densCls)
        to_write = self.input_data.copy(deep=True).to_frame().T
        to_write = to_write.assign(Lymphodema_NO=lambda x: 1)
        to_write = to_write.assign(Lymphodema_YES=lambda x: 0)

        to_write.to_csv(filepath, sep=",", header=False, index=False)

        return filepath

    def write_results(self)-> None:
        with open(f"output/results_{self.subj_id}.csv", "w") as fp:
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
{self.input_data}
Selected rules: 
{rules_str}
        """
