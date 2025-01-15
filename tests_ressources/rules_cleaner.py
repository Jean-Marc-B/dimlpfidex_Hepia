from src.global_rules import GlobalRules
import src.constants as constants
from datetime import datetime
import src.utils as utils
import os

if __name__ == "__main__":
    today = datetime.today().strftime("%Y_%m_%d")
    abspath = os.path.abspath(os.path.dirname(__file__))

    attributes = utils.read_attributes_file(abspath)
    global_rules_path = os.path.join(abspath, constants.MODEL_DIRNAME, "global_rules_denormalized.json")
    global_rules_json_out_path = os.path.join(abspath, constants.MODEL_DIRNAME, f"global_rules_cleaned_{today}.json")
    global_rules_readable_out_path = os.path.join(abspath, constants.MODEL_DIRNAME, f"global_rules_cleaned_{today}.txt")

    global_rules = GlobalRules.from_json_file(global_rules_path, attributes)

    cleaned_global_rules = []
    for rule in global_rules.rules:
        cleaned_global_rules.append(rule.filter_redundancies())

    global_rules.rules = cleaned_global_rules

    global_rules.to_json_file(global_rules_json_out_path)

    with open(global_rules_readable_out_path, "w") as f:
        f.write(global_rules.pretty_repr(attributes))
