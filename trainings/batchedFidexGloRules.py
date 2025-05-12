from argparse import ArgumentParser, Namespace
from datetime import datetime
from dimlpfidex import fidex 
# import multiprocessing
from enum import Enum
import pandas as pd
import numpy as np
import os

class FileType(Enum):
        TRAIN=0
        CLASS=1
        PRED=2

class TemporaryWorkspace:

    def __init__(self, has_classes: bool = True):
        script_absolute_path = os.path.abspath(os.path.dirname(__file__))
        self.absolute_path = os.path.join(script_absolute_path, f"batched_tmp_{today_str()}")
        self.data_dir_path = os.path.join(self.absolute_path, "datas")
        self.class_dir_path = os.path.join(self.absolute_path, "classes")
        self.pred_dir_path = os.path.join(self.absolute_path, "preds")

        os.makedirs(self.absolute_path, exist_ok=True)
        os.makedirs(self.data_dir_path, exist_ok=True)
        os.makedirs(self.class_dir_path, exist_ok=True)
        os.makedirs(self.pred_dir_path, exist_ok=True)
    

    def split_file_into(self, src_path: str, file_type: FileType, n_splits: int) -> None:
        data = pd.read_csv(src_path,sep=',', header=None,index_col=None)
        filename = os.path.basename(src_path)

        chunk_size = data.shape[0] // n_splits

        if (rem := data.shape[0] % n_splits) != 0:
            print(f"The number of splits is not optimal, {rem} datas are added to the last file.") 

        for i in range(n_splits):
            subfilename = filename.split('.')
            subfilename.insert(1, f"_{i+1}.")
            subfilename = "".join(subfilename)

            self.__write_into(subfilename, data.iloc[i*chunk_size : (i+1) * chunk_size, :], file_type)



    def __write_into(self, filename: str, data: pd.DataFrame, file_type: FileType) -> None:
        dst_path = ""

        if file_type == FileType.TRAIN:
            dst_path = os.path.join(self.data_dir_path, filename)
        elif file_type == FileType.CLASS:
            dst_path = os.path.join(self.class_dir_path, filename)
        else:
            dst_path = os.path.join(self.pred_dir_path, filename)

        data.to_csv(dst_path, sep=',', header=None, index=None)


    def __repr__(self):
        fmt_paths = "{!r},{!r},{!r}".format(self.data_dir_path,self.class_dir_path,self.pred_dir_path)
        return f"TemporaryWorkspace({fmt_paths})"


def today_str() -> str:
    return datetime.today().strftime('%Y%m%d-%H%M')


def init_args() -> Namespace:
    parser = ArgumentParser(prog="batchedFidexGloRules")
    
    parser.add_argument("--train_data_file", help="File path containing training datas.", required=True, type=lambda path: is_valid_file(parser, path))
    parser.add_argument("--train_class_file", help="File path containing true class datas.", type=lambda path: is_valid_file(parser, path))
    parser.add_argument("--train_pred_file", help="File path containing predictions data.", required=True, type=lambda path: is_valid_file(parser, path))
    parser.add_argument("--keep_tmp_files", help="Wether temporary files should be kept or not.", action="store_true", default=False)
    parser.add_argument("--output_filename", help="Name of generated file that will contain final Global Rules.", type=str, default="fidexGloRules.json")
    parser.add_argument("--json_config_file", help="JSON file containing FidexGloRules configuration.", type=str)
    
    args = parser.parse_args()

    return args


def is_valid_file(parser: ArgumentParser, arg: str) -> str:
    if not os.path.exists(arg):
        parser.error(f"The path '{arg}' given is not leading to a existing file.")
    else:
        return arg


def batched_fidexglorules():
    pass


def merge_results():
    pass


if __name__ == "__main__":
    args = init_args()

    tw = TemporaryWorkspace(has_classes=True)
    tw.split_file_into(args.train_data_file, FileType.TRAIN, 10)
    tw.split_file_into(args.train_class_file, FileType.CLASS, 10)
    tw.split_file_into(args.train_pred_file, FileType.PRED, 10)
