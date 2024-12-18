from __future__ import annotations
import math
import copy


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
            res.value = math.ceil(res.value)
        else:
            res.value = math.floor(res.value)

        return res