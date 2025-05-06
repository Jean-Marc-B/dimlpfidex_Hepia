from src.global_rules import GlobalRules
from src.utils import read_attributes_file

if __name__ == "__main__":
    attributes = read_attributes_file(".")
    rules = GlobalRules.from_json_file("model/global_rules_denormalized.json", attributes)
    print(rules.pretty_repr(attributes))