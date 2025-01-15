import src.constants as constants
from src.rule import Rule
import json
import os

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

        self.to_json_file(global_rules_path)

    def to_json_file(
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
