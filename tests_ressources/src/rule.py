from __future__ import annotations
import src.constants as constants
from math import ceil, floor
import json
import math
import copy
import os


class Antecedant:
    def __init__(
        self, attribute_id: dict, inequality: str, value: float, attributes: list[str]
    ) -> None:
        self.attribute_id = attribute_id
        self.attribute_name = attributes[attribute_id]
        self.inequality = inequality
        self.value = value

    def __repr__(self) -> str:
        ineq_str = ">=" if self.inequality else "<"
        return f"{self.attribute_id} {ineq_str} {self.value:.4f}"

    def __eq__(self, other: Antecedant) -> bool:
        eps = 1e-4

        if not self.attribute_id == other.attribute_id:
            return False
        if not self.inequality == other.inequality:
            return False
        if not math.isclose(self.value, other.value, rel_tol=eps):
            return False

        return True

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute_id,
            "inequality": self.inequality,
            "value": self.value,
        }

    def pretty_repr(self) -> str:
        ineq_str = ">=" if self.inequality else "<"

        return f"{self.attribute_name} {ineq_str} {self.value:.4f}"

    # designed for unicancer format
    def to_string(self) -> str:
        ineq_str = ">=" if self.inequality else "<"
        return f"{ineq_str}{self.value:.4f}"

    @staticmethod
    def list_to_dict(antecedants_list: list[Antecedant]) -> dict:
        return [antecedant.to_dict() for antecedant in antecedants_list]

    def round(self) -> Antecedant:
        res = copy.deepcopy(self)

        if res.inequality:
            res.value = ceil(res.value)
        else:
            res.value = floor(res.value)

        return res


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
    def from_json_file(path: str) -> list[Rule]:
        with open(path, "r") as fp:
            data = json.load(fp)

        return Rule.list_from_dict(data["rules"])

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


class GlobalRules:
    def __init__(
        self,
        rules: list[Rule],
        positive_index_class: int = None,
        threshold: float = None,
    ) -> None:
        # setting rules ID
        self.rules = [rule.set_id(id) for id, rule in enumerate(rules)]

        if positive_index_class == None and threshold != None:
            raise ValueError(
                "Error while creating Global Rule list: the threshold argument must be specified IF the positive class index argument is specified."
            )
        elif positive_index_class != None and threshold == None:
            raise ValueError(
                "Error while creating Global Rule list: the positive class index argument must be specified IF the threshold argument is specified."
            )

        self.positive_index_class = positive_index_class
        self.threshold = threshold

    def __len__(self):
        return len(self.rules)

    @staticmethod
    def from_json_file(path: str, attributes: list[str]):
        with open(path, "r") as fp:
            data = json.load(fp)

        return GlobalRules(
            rules=Rule.list_from_dict(data, attributes),
            positive_index_class=data.get("positive index class", None),
            threshold=data.get("threshold", None),
        )

    def save(self, abspath: str):
        global_rules_path = os.path.join(
            abspath, constants.MODEL_DIRNAME, "global_rules_denormalized.json"
        )

        self.__to_json_file(global_rules_path)

    def __to_json_file(
        self,
        path: str,
    ) -> None:
        if self.positive_index_class != None and self.threshold != None:
            json_data = {
                "positive index class": self.positive_index_class,
                "rules": Rule.list_to_dict(self.rules),
                "threshold": self.threshold,
            }
        else:
            json_data = {
                "rules": Rule.list_to_dict(self.rules),
            }

        with open(path, "w") as fp:
            json.dump(json_data, fp, indent=4)

    def get_rule_id(self, target: Rule):
        for rule in self.rules:
            if rule == target:
                return rule.id

        return -1

    def __repr__(self) -> str:
        res = f"""Global rule set:
Positive index class: {self.positive_index_class}
Threshold: {self.threshold}
Rule set size: {len(self.rules)}
Rules:"""

        for i, rule in enumerate(self.rules):
            res += f"Rule #{i+1}:" + rule.__repr__() + "\n\n"

        return res

    def pretty_repr(self, attributes: list[str]) -> str:
        string = f"""Global rule set:
Positive index class: {self.positive_index_class}
Threshold: {self.threshold}
Rule set size: {len(self.rules)}
Rules:"""

        for i, rule in enumerate(self.rules):
            string += f"\n\nRule #{i+1}:"
            string += rule.pretty_repr(attributes)

        return string
