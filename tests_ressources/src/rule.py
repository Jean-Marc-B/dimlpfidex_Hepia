from __future__ import annotations
import json


class Antecedant:
    def __init__(self, attribute_id: dict, inequality: str, value: float) -> None:
        self.attribute_id = attribute_id
        self.inequality = inequality
        self.value = value

    def __repr__(self) -> str:
        # nodes_examined = nodes_removed
        # TODO: sentinel_node_biopsy exlue planned_axillary_dissection
        # TODO: age, weight, height, nodes_involved, tumor_size, KI_67, fractions, nodes_removed
        # for these, if < then floor()
        # elif >= then ceil()
        ineq_str = ">=" if self.inequality else "<"
        return f"{self.attribute_id} {ineq_str} {self.value:.4f}"

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute_id,
            "inequality": self.inequality,
            "value": self.value,
        }

    def pretty_print(self, attributes: list[str]) -> None:
        ineq_str = ">=" if self.inequality else "<"
        attribute = attributes[self.attribute_id]

        print(f"{attribute} {ineq_str} {self.value:.4f}", end=" ")

    # designed for unicancer format
    def to_string(self) -> str:
        ineq_str = ">=" if self.inequality else "<"
        return f"{ineq_str}{self.value:.4f}"

    @staticmethod
    def list_to_dict(antecedants_list: list[Antecedant]) -> dict:
        return [antecedant.to_dict() for antecedant in antecedants_list]


class Rule:
    def __init__(self, json_data: dict) -> None:
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

                self.antecedants.append(Antecedant(attribute_id, inequality, value))

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

    # designed for unicancer format
    def to_str_list(self):
        base_rule = [
            str(self.id),
        ]

        list_str = [""] * 79
        for antecedant in self.antecedants:
            list_str[antecedant.attribute_id] = antecedant.to_string()
            
        return base_rule + list_str

    def pretty_print(self, attributes: list[str]) -> None:
        print(
            f"""
Covering: {self.covering}
Fidelity: {self.fidelity:.3f}
Accuracy: {self.accuracy:.3f}   
Antecedants:"""
        )
        for antecedant in self.antecedants:
            antecedant.pretty_print(attributes)

        print(f"Output: {self.output}" "")

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
    def list_from_dict(data: dict) -> list[Rule]:
        return [Rule(rule_data) for rule_data in data["rules"]]


class GlobalRules:
    def __init__(
        self,
        rules: list[Rule],
        positive_index_class: int = None,
        threshold: float = None,
    ) -> None:
        self.rules = rules

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

    @staticmethod
    def from_json_file(path: str):
        with open(path, "r") as fp:
            data = json.load(fp)

        return GlobalRules(
            rules=Rule.list_from_dict(data),
            positive_index_class=data.get("positive_index_class", None),
            threshold=data.get("positive_index_class", None),
        )

    def to_json_file(
        self,
        path: str,
    ) -> None:
        if self.positive_index_class != None and self.threshold != None:
            json_data = {
                "positive index class": self.positive_index_class,
                "rules": Rule.list_to_dict(self.rules),
                "thresold": self.threshold,
            }
        else:
            json_data = {
                "rules": Rule.list_to_dict(self.rules),
            }

        with open(path, "w") as fp:
            json.dump(json_data, fp, indent=4)

    def set_rules_id(self):
        for i, rule in enumerate(self.rules):
            rule.id = i  # TODO check if this works

    def get_rule_id(self, target: Rule):
        for i, rule in enumerate(self.rules):
            if rule == target:
                return i

        return -1

    def add_rule(self, rule: Rule) -> int:
        self.rules.append(rule)
        return len(self.rules)

    def __repr__(self) -> str:
        res = f"""Global rule set:
Positive index class: {self.positive_index_class}
Threshold: {self.threshold}
Rule set size: {len(self.rules)}
Rules:"""

        for i, rule in enumerate(self.rules):
            res += f"Rule #{i+1}:" + rule.__repr__() + "\n\n"

        return res

    def pretty_print(self, attributes: list[str]) -> None:
        print(
            f"""Global rule set:
Positive index class: {self.positive_index_class}
Threshold: {self.threshold}
Rule set size: {len(self.rules)}
Rules:"""
        )

        for i, rule in enumerate(self.rules):
            print(f"Rule #{i+1}:")
            rule.pretty_print(attributes)

    def extract_selected_rules(self, list_rules: list[Rule]) -> dict[int, Rule]:
        res = {}

        for selected_rule in list_rules:
            id = self.get_rule_id(selected_rule)
            if id == -1:
                id = self.add_rule(selected_rule)

            res[id] = selected_rule

        return res
