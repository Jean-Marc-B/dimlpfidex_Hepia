from argparse import ArgumentParser
from datetime import datetime
from dimlpfidex import fidex 
import multiprocessing
import os


class TemporaryWorkspace:

    def __init__(self):
        absolute_path = os.path(__file__, "batched_tmp")
        data_dir_path =  os.path(absolute_path, "datas")
        class_dir_path =   os.path(absolute_path, "classes")
        pred_data_dir_path =   os.path(absolute_path, "preds")


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


def is_valid_file(parser: ArgumentParser, arg: str)-> None:
    if not os.path.exists(arg):
        parser.error(f"The path '{arg}' given is not leading to a existing file.")



def create_tmp_directories() -> None: 
    os.mkdir()
    pass



def split_file(src_path: str, n_splits: int) -> None:
    pass



def batched_fidexglorules():
    pass


def merge_results():
    pass


def clean_tmp_directories() -> None:
    pass



if __name__ == "__main__":
    args = init_args()

    print(args)
