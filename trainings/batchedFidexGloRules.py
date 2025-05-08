from argparse import ArgumentParser
from dimlpfidex import fidex 
import multiprocessing
import os


def init_args() -> ArgumentParser:
    parser = ArgumentParser(prog="batchedFidexGloRules")
    
    parser.add_argument("--train_data_file", help="File path containing training datas.", required=True, type=lambda path: is_valid_file(parser, path))
    parser.add_argument("--train_class_file", help="File path containing true class datas.", type=lambda path: is_valid_file(parser, path))
    parser.add_argument("--train_pred_file", help="File path containing predictions data.", required=True, type=lambda path: is_valid_file(parser, path))
    parser.add_argument("--keep_tmp_files", help="Wether temporary files should be kept or not.", action="store_true", default=False)
    parser.add_argument("--output_filename", help="Name of generated file that will contain final Global Rules.", type=str, default="fidexGloRules.json")
    parser.add_argument("--json_config_file", help="JSON file containing FidexGloRules configuration.", type=str)
    
    parser.parse_args()

    return parser


def is_valid_file(parser: ArgumentParser, arg: str):
    if not os.path.exists(arg):
        parser.error(f"The path '{arg}' given is not leading to a existing file.")


def batched_fidexglorules():
    pass


def merge_results():
    pass


if __name__ == "__main__":
    args = init_args()

    print(args)
