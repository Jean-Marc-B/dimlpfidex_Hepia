from __future__ import annotations
from src.antecedant import Antecedant
import json
import math
import copy


class Rule:
    """Rule class represents a fidex rule. A rule contains various data:
    - ID
    - Covering size (amount of samples covered by this rule)
    - List of covered samples
    - Fidelity (a value representing a proportion of matches between the rule output and its own output)
    - Confidence
    - Accuracy
    - Output class 
    - A list of Antecedants objects (see antecedants.py)
    """
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

    def to_str_list(self) -> list[str]:
        """Returns a Rule into a list of string. Format designed for UNICANCER.

        Returns:
            list[str]: Rule data converted into list
        """
        list_str = [""] * 79
        for antecedant in self.antecedants:
            list_str[antecedant.attribute_id] = antecedant.to_string()

        return list_str

    def pretty_repr(self, attributes: list[str]) -> str:
        """Returns a prettier (at least readable) representation of a rule

        Args:
            attributes (list[str]): path leading to the file.

        Returns:
            list[str]: Rule data converted into list
        """
        labels = attributes[-2:]
        string = ""

        for antecedant in self.antecedants:
            string += antecedant.pretty_repr() + " "

        string += f" -> {labels[self.output]}"

        string += f"""
Train Covering Size : {self.covering}
Train Fidelity : {self.fidelity:.6f}
Train Accuracy : {self.accuracy:.6f}   
Train confidence : {self.confidence:.6f}

"""
        return string

    def postprocess(self) -> Rule:
        """Processes the rule's antecedants in order to apply theses changes:
        - Ignore attributes "(...)_UNKNOWN" that are < 0.5 (meaning not unknown) 
        - Round specific values (if < then the value is floor(), elif >= then ceil() is used. 
        Also, if floor is used, then change < inequality symbol to <=).
        # TODO: sentinel_node_biopsy exludes planned_axillary_dissection (do not touch yet)
        # TODO: get rid of double negated unknown features
        # TODO: NODES_REMOVED = NODES_EXAMINED ?
        
        Returns:
            Rule: a new rule containing applied changes
        """

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
        """Create a list of rule from a JSON rule file

        Args:
            path (str): file to read
            attributes (list[str]): list of attributes to be used when creating rules

        Returns:
            list[Rule]: List containing created rules
        """
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
        """Checks and removes if there is redundancies in the rule's antecedants.
        Example: NODES_EXAMINED > 6.5 and NODES_EXAMINED > 17.2 (in this case, only the second one is kept) 

        Returns:
            Rule: a new rule without redundancies
        """
        new_rule = copy.deepcopy(self)

        for i in range(len(self.antecedants)):
            current = self.antecedants[i]
            for j in range(i+1, len(self.antecedants)):
                to_compare = self.antecedants[j]

                if current.attribute_name != to_compare.attribute_name: 
                    continue

                if current.inequality != to_compare.inequality:
                    continue

                inequality = current.inequality

                if inequality: # >=
                    if current.value >= to_compare.value:
                        new_rule.antecedants[j] = None
                    else:
                        new_rule.antecedants[i] = None
                else: # <
                    if current.value < to_compare.value:
                        new_rule.antecedants[j] = None
                    else:
                        new_rule.antecedants[i] = None

        new_rule.antecedants = [a for a in new_rule.antecedants if a is not None]
                    
        return new_rule 

    @staticmethod
    def list_from_dict(data: dict, attributes: list[str]) -> list[Rule]:
        return [Rule(rule_data, attributes) for rule_data in data["rules"]]
