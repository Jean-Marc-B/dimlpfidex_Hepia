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

        self.selected_rules = []

    def set_selected_rules(self, selected_rules: list[Rule]):
        self.selected_rules = selected_rules

    def to_str_list(self) -> list[list[str]]:
        res = []
        row = [
            self.id,
            self.study_id,
            self.site_idn,
            self.site_name,
            self.subj_id,
            self.visit,
        ]

        for rule in self.selected_rules:
            row += rule.to_str_list()

        return res

    def prepare_input_for_extraction(self, path: str) -> str:
        filepath = os.path.join(path, f"input_{self.id}.csv")

        # one hotting classes (useless but must be done in order to work with densCls)
        to_write = self.input_data.copy(deep=True).to_frame().T
        to_write = to_write.assign(Lymphodema_NO=lambda x: 1)
        to_write = to_write.assign(Lymphodema_YES=lambda x: 0)

        to_write.to_csv(filepath, sep=",", header=False, index=False)

        return filepath

    def __repr__(self) -> str:
        rules_str = (
            "".join([rule.__repr__() + " " for rule in self.selected_rules]) + "\n"
        )

        return f"""Patient:
ID: {self.id}
Study ID: {self.study_id}
Site IDN: {self.site_idn}
Site name: {self.site_name}
Subject ID: {self.subj_id}
Visit: {self.visit}
Patient's data: 
{self.input_data}
Selected rules: 
{rules_str}
        """
