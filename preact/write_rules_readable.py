import sys
from src.global_rules import GlobalRules


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 write_rules_readable.py <json_filepath_to_translate> <attribute_file_path>")
        exit(1)

    with open(sys.argv[2], "r") as f:
        attributes = f.read().splitlines()
        rules = GlobalRules.from_json_file(sys.argv[1], attributes)
        print(rules.pretty_repr(attributes))