from __future__ import annotations
from src.antecedant import Antecedant
import json
import math
import copy


class Rule:
    def __init__(self, json_data: dict, attributes: list[str]) -> None:
        try:
            self.id = -1
            self.covering = json_data["coveringSize"]
            self.covered_samples = json_data["coveredSamples"]
            self.fidelity = json_data["fidelity"]
            self.confidence = json_data["confidence"]
            self.accuracy = json_data["accuracy"]
            self.output = json_data["outputClass"]
            self.antecedants = []

            for antecedant in json_data["antecedents"]:
                attribute_id = antecedant["attribute"]
                inequality = antecedant["inequality"]
                value = antecedant["value"]

                self.antecedants.append(
                    Antecedant(attribute_id, inequality, value, attributes)
                )

        except KeyError as e:
            print(f"Rule creation from JSON error: {e}")

    def __repr__(self) -> str:
        antecedants_str = "".join(
            [antecedant.__repr__() + " " for antecedant in self.antecedants]
        )
        return f"""
Covering: {self.covering}
Fidelity: {self.fidelity:.3f}
Accuracy: {self.accuracy:.3f}
Antecedants: {antecedants_str}
Output: {self.output}"""

    def __eq__(self, other: Rule) -> bool:
        eps = 1e-4

        if not math.isclose(self.accuracy, other.accuracy, rel_tol=eps):
            return False
        if not math.isclose(self.covering, other.covering, rel_tol=eps):
            return False
        if not math.isclose(self.confidence, other.confidence, rel_tol=eps):
            return False
        if not math.isclose(self.fidelity, other.fidelity, rel_tol=eps):
            return False
        if not math.isclose(self.output, other.output, rel_tol=eps):
            return False
        if not len(self.antecedants) == len(other.antecedants):
            return False

        for self_antecedant, other_antecedant in zip(
            self.antecedants, other.antecedants
        ):
            if self_antecedant != other_antecedant:
                return False

        return True

    def set_id(self, id: int) -> Rule:
        self.id = id
        return self

    # designed for unicancer format
    def to_str_list(self):
        list_str = [""] * 79
        for antecedant in self.antecedants:
            list_str[antecedant.attribute_id] = antecedant.to_string()

        return list_str

    def pretty_repr(self, attributes: list[str]) -> str:
        labels = attributes[-2:]
        string = f"""
ID: {self.id}
Covering: {self.covering}
Fidelity: {self.fidelity:.3f}
Accuracy: {self.accuracy:.3f}   
Antecedants:\n\t"""
        for antecedant in self.antecedants:
            string += antecedant.pretty_repr() + " "
        string += f"\nOutput: {labels[self.output]}"

        return string

    def postprocess(self) -> Rule:
        # TODO: sentinel_node_biopsy exludes planned_axillary_dissection (do not touch yet)
        # TODO: get rid of double negated unknown features
        #
        # for age, weight, height, nodes_involved, tumor_size, KI_67, fractions, nodes_removed:
        # => if < then floor() elif >= then ceil()
        # => floor is used, then change < inequality symbol to <=

        new_rule = copy.deepcopy(self)

        antecedants_to_round = [
            "NODES_INVOLVED",
            "AGE",
            "WEIGHT",
            "HEIGHT",
            "TUMOR_SIZE",
            "KI_67",
            "FRACTIONS",
        ]

        antecedants_to_ignore_if_negative = [
            "TYPE_SURGERY_UNKNOWN",
            "TREATED_BREAST_RADIO_UNKNOWN",
            "SMOKER_UNKNOWN",
            "SIDE_OF_PRIMARY_UNKNOWN",
            "PR_STATUS_UNKNOWN",
            "NEOADJUVANT_CHEMOTHERAPY_UNKNOWN",
            "MENOPAUSAL_UNKNOWN",
            "IMRT_UNKNOWN",
            "HISTOLOGICAL_TYPE_UNKNOWN",
            "HER_2_STATUS_UNKNOWN",
            "ER_STATUS_UNKNOWN",
            "DIABETIES_UNKNOWN",
            "CLINICAL_T_STAGE_UNKNOWN",
            "CLINICAL_N_STAGE_UNKNOWN",
            "BOOST_UNKNOWN",
            "BASELINE_ARM_LYMPHEDEMA_UNKNOWN",
            "ADJUVANT_CHEMOTHERAPY_UNKNOWN",
            "3D_CRT_UNKNOWN",
        ]

        new_antecedants = []

        for antecedant in new_rule.antecedants:
            if antecedant.attribute_name in antecedants_to_ignore_if_negative:
                if antecedant.value < 0.5 and antecedant.inequality == False:
                    continue

            if antecedant.attribute_name in antecedants_to_round:
                # nodes_examined = nodes_removed
                # if new_rule.antecedants[i].attribute_name == "NODES_INVOLVED":
                #     new_rule.antecedants[i].attribute_name = "NODES_REMOVED"
                antecedant = antecedant.round()

            new_antecedants.append(antecedant)

        new_rule.antecedants = new_antecedants

        return new_rule

    @staticmethod
    def from_json_file(path: str, attributes: list[str]) -> list[Rule]:
        with open(path, "r") as fp:
            data = json.load(fp)

        return Rule.list_from_dict(data, attributes)

    @staticmethod
    def list_to_dict(rule_list: list[Rule]) -> dict:
        return [rule.to_dict() for rule in rule_list]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "antecedents": Antecedant.list_to_dict(self.antecedants),
            "confidence": self.confidence,
            "coveredSamples": self.covered_samples,
            "coveringSize": self.covering,
            "fidelity": self.fidelity,
            "outputClass": self.output,
        }

    def filter_redundancies(self) -> Rule:
        new_rule = copy.deepcopy(self)

        non_redundant_antecedants = []
    
        for antecedant in new_rule.antecedants:
            if not non_redundant_antecedants:
                non_redundant_antecedants.append(antecedant)
            else:
                last_antecedant = non_redundant_antecedants[-1]

                if (antecedant.attribute_name == last_antecedant.attribute_name and 
                    antecedant.inequality == last_antecedant.inequality):
                    if antecedant.inequality and antecedant.value <= last_antecedant.value:
                        continue 
                    if not antecedant.inequality and antecedant.value >= last_antecedant.value:
                        continue
                    
               
                non_redundant_antecedants.append(antecedant)

        new_rule.antecedants = non_redundant_antecedants
        
        return new_rule 

    @staticmethod
    def list_from_dict(data: dict, attributes: list[str]) -> list[Rule]:
        return [Rule(rule_data, attributes) for rule_data in data["rules"]]
