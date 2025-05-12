from multiprocessing import Process
from datetime import datetime
from dimlpfidex import fidex 
from enum import Enum
import pandas as pd
import argparse
import json
import os

class FileType(Enum):
        TRAIN=0
        CLASS=1
        PRED=2

class BatchedFidexGloRules:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.nb_processes = self.args.nb_processes
        script_absolute_path = os.path.abspath(os.path.dirname(__file__))
        self.absolute_path = os.path.join(script_absolute_path, f"batched_tmp_{today_str()}")
        self.data_dir_path = os.path.join(self.absolute_path, "datas")
        self.class_dir_path = os.path.join(self.absolute_path, "classes")
        self.pred_dir_path = os.path.join(self.absolute_path, "preds")

        os.makedirs(self.absolute_path, exist_ok=True)
        os.makedirs(self.data_dir_path, exist_ok=True)
        os.makedirs(self.class_dir_path, exist_ok=True)
        os.makedirs(self.pred_dir_path, exist_ok=True)

        self.__split_file_into(self.args.train_data_file, FileType.TRAIN)
        self.__split_file_into(self.args.train_class_file, FileType.CLASS)
        self.__split_file_into(self.args.train_pred_file, FileType.PRED)
    

    def __split_file_into(self, src_path: str, file_type: FileType) -> None:
        data = pd.read_csv(src_path, sep=',', header=None,index_col=None)
        filename = os.path.basename(src_path)

        chunk_size = data.shape[0] // self.nb_processes

        if (rem := data.shape[0] % self.nb_processes) != 0:
            print(f"The number of splits is not optimal, {rem} datas are added to the last file.") 

        for i in range(self.nb_processes):
            subfilename = filename.split('.')
            subfilename = f"{subfilename[0]}_{i+1}.{subfilename[1]}"

            if i == (self.nb_processes - 1):
                self.__write_into(subfilename, data.iloc[i*chunk_size : (i+1) * chunk_size + rem, :], file_type)
            else:
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

    def __call__(self, *args, **kwds):
        processes = []

        for _ in range(self.nb_processes):
            processes.append(Process(target=fidex.fidexGloRules))

        for process in processes:
            process.start()

        for process in processes:
            process.join()
        
    def __repr__(self):
        fmt_paths = "{!r},{!r},{!r}".format(self.data_dir_path,self.class_dir_path,self.pred_dir_path)
        return f"{self.__class__.__name__}({fmt_paths})"
    
    def __merge_results(self):
        ...

def today_str() -> str:
    return datetime.today().strftime('%Y%m%d-%H%M')

def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FidexGlo Rule Extraction Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--train_data_file', type=lambda x: is_valid_file(parser, x), help="Path to training data file")
    parser.add_argument('--train_class_file', type=lambda x: is_valid_file(parser, x), help="Path to the file containing the train true classes of the dataset, not mandatory if classes are specified in train data file")
    parser.add_argument('--train_pred_file', type=lambda x: is_valid_file(parser, x), help="Path to predictions on training data")
    parser.add_argument('--weights_file', type=lambda x: is_valid_file(parser, x), help="Path to the file containing the trained weights of the model (not mandatory if a rules file is given with --rules_file)")
    parser.add_argument('--rules_file', type=str, help="Path to the file containing the trained rules to be converted to hyperlocus (not mandatory if a weights file is given with --weights_file)")
    parser.add_argument('--global_rules_outfile', type=str, help="Path to the file where the output rule(s) will be stored. If a .json extension is given, rules are saved in JSON format")
    parser.add_argument('--heuristic', type=int, choices=[1, 2, 3],  help="Heuristic 1: optimal fidexGlo, 2: fast fidexGlo 3: very fast fidexGlo. (Faster algorithms are less efficient)")
    parser.add_argument('--nb_attributes', type=int, help="Number of attributes")
    parser.add_argument('--nb_classes', type=int, help="Number of classes")

    # Optional arguments
    parser.add_argument('--json_config_file', type=str, help="Path to a JSON file containing configuration parameters")
    parser.add_argument('--root_folder', type=str, help="Root folder for input/output files")
    parser.add_argument("--keep_tmp_files", help="Wether temporary files should be kept or not.", action="store_true", default=False)
    parser.add_argument("--nb_processes", help="Number of processes the data will be divided into", type=int, default=1)
    parser.add_argument('--attributes_file', type=lambda x: is_valid_file(parser, x), help="Path to attributes file")
    parser.add_argument('--console_file', type=str, help="Redirect terminal output to this file")
    parser.add_argument('--max_iterations', type=int, default=10, help="Max rule antecedents/iterations")
    parser.add_argument('--min_covering', type=int, default=2, help="Minimum number of samples covered by rule")
    parser.add_argument('--covering_strategy', type=bool, default=True, help="Use covering strategy")
    parser.add_argument('--max_failed_attempts', type=int, default=30, help="Max failed attempts for covering=1")
    parser.add_argument('--min_fidelity', type=float, default=1.0, help="Minimum rule fidelity")
    parser.add_argument('--lowest_min_fidelity', type=float, default=0.75, help="Lowest accepted min fidelity")
    parser.add_argument('--dropout_dim', type=float, default=0.0, help="Dropout probability for dimensions")
    parser.add_argument('--dropout_hyp', type=float, default=0.0, help="Dropout probability for hyperplanes")
    parser.add_argument('--decision_threshold', type=float, help="Threshold used for predictions")
    parser.add_argument('--positive_class_index', type=int, help="Index of positive class")
    parser.add_argument('--nb_quant_levels', type=int, default=50, help="Number of stairs in the staircase activation function (default: 50)")
    parser.add_argument('--normalization_file', type=str, help="Normalization file (mean/std)")
    parser.add_argument('--mus', type=float, nargs='+', help="List of means per attribute")
    parser.add_argument('--sigmas', type=float, nargs='+', help="List of std deviations per attribute")
    parser.add_argument('--normalization_indices', type=int, nargs='+', help="Indices of normalized attributes")
    parser.add_argument('--nb_threads', type=int, default=1, help="Number of threads")
    parser.add_argument('--seed', type=int, default=0, help="Random seed (0 for random)")
    
    args = parser.parse_args()

    #If JSON config is given, insert values
    if args.json_config_file:
        with open(args.json_config_file, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' in JSON file")

    return args

def is_valid_file(parser: argparse.ArgumentParser, arg: str) -> str:
    if not os.path.exists(arg):
        parser.error(f"The path '{arg}' given is not leading to a existing file.")
    else:
        return arg


if __name__ == "__main__":
    args = init_args()
    bfgr = BatchedFidexGloRules(args)

    for arg in dir(args):
        print(arg)
    # bfgr()