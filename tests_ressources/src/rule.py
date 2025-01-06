from __future__ import annotations
from src.antecedant import Antecedant
import json
import math

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

    def postprocess(self):
        # nodes_examined = nodes_removed
        # TODO: sentinel_node_biopsy exlue planned_axillary_dissection (do not touch yet)

        # TODO: get rid of double negated unknown features 
        # TODO: age, weight, height, nodes_involved, tumor_size, KI_67, fractions, nodes_removed
        # for these, if < then floor() elif >= then ceil()
        # TODO: if floor is used, change < to <=
        pass

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

    @staticmethod
    def list_from_dict(data: dict, attributes: list[str]) -> list[Rule]:
        return [Rule(rule_data, attributes) for rule_data in data["rules"]]
