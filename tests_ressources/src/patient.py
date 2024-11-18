from tests_ressources.src.rule import *


class Patient:

    def __init__(self, id, study_id, site_idn, site_name, subj_id, visit) -> None:
        # add STUDYID, SITEIDN, SITENAME, SUBJID, VISIT
        self.id = id
        self.study_id = study_id
        self.site_idn = site_idn
        self.site_name = site_name
        self.subj_id = subj_id
        self.visit = visit

        self.selected_rules = []

    def set_selected_rules(self, selected_rules: list[Rule]):
        self.selected_rules = selected_rules

    # def to_str_list(self, attributes: list[str] = None) -> list[list[str]]:
    #     res = []

    #     for rule in self.selected_rules:

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
        Selected rules: 
        {rules_str}
        """
