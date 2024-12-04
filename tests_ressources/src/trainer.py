from dimlpfidex.fidex import fidexGloRules
from src.patient import write_train_data
from dimlpfidex.dimlp import dimlpBT
from trainings import normalization
from datetime import datetime
import pandas as pd
import os


class Trainer:
    def __init__(
        self,
        abspath: str,
        datas: pd.DataFrame,
        labels: pd.Series,
    ):
        self.project_abspath = abspath
        self.reldir = "temp"
        self.absdir = os.path.join(self.project_abspath, self.reldir)
        self.nb_features = datas.shape[1]
        self.labels = labels
        self.nb_classes = 2
        self.datas = datas
        self.nb_nets = 30
        self.epochs = 300

    def train(self, normalize: bool = True, split: float = 0.0):
        write_train_data(self.project_abspath, self.datas, self.labels, split)

        if normalize:
            print("Normalizing datas...")
            normalization(self.__get_train_normalization_config(split > 0.0))

        print("Training...")
        dimlpBT(self.__get_train_dimlpbt_config())
        print("Training done")
        print("Extracting global rules...")
        fidexGloRules(self.__get_train_fidexglorules_config())
        print("Global rules extraction done")

        if normalize:
            print("Denormalizing datas...")
            normalization(self.__get_train_denormalization_config())

    def __get_train_normalization_config(self, has_test: bool = False) -> str:
        test_file = f",{self.reldir}/test_data.csv" if has_test else ""

        return (
            f"--root_folder {self.project_abspath} "
            f"--data_files [{self.reldir}/train_data.csv{test_file}] "
            f"--output_normalization_file {self.reldir}/normalization_stats.txt "
            f"--nb_attributes {self.nb_features} "
            f"--nb_classes {self.nb_classes} "
            f"--attributes_file {self.reldir}/attributes.txt "
            "--missing_values NaN"
        )

    def __get_train_denormalization_config(self) -> str:
        return (
            f"--root_folder {self.project_abspath} "
            f"--rule_files {self.reldir}/global_rules.json "
            f"--normalization_file {self.reldir}/normalization_stats.txt "
            f"--nb_attributes {self.nb_features} "
            f"--attributes_file {self.reldir}/attributes.txt"
        )

    def __get_train_dimlpbt_config(self, has_test: bool = False) -> str:
        today = datetime.today().strftime("%Y_%m_%d")
        test_datas = ""

        if has_test:
            test_datas = (
                f"--test_data_file {self.reldir}/test_data_normalized.csv "
                f"--test_class_file {self.reldir}/test_classes.csv "
                f"--train_pred_outfile {self.reldir}/dimlpbt_test_predictions.out "
            )

        return (
            f"--root_folder {self.project_abspath} "
            f"--train_data_file {self.reldir}/train_data_normalized.csv "
            f"--train_class_file {self.reldir}/train_classes.csv "
            f"--train_pred_outfile {self.reldir}/dimlpbt_train_predictions.out "
            + test_datas
            + f"--weights_outfile {self.reldir}/dimlpBT.wts "
            f"--console_file {self.reldir}/dimlpbt_{today}.log "
            f"--hidden_layers_outfile {self.reldir}/hidden_layers.out "
            f"--stats_file {self.reldir}/dimlpBT_stats.out "
            f"--metrics_file {self.reldir}/metrics.json "
            f"--nb_attributes {self.nb_features} "
            f"--nb_classes {self.nb_classes} "
            f"--nb_dimlp_nets {self.nb_nets} "
            f"--hidden_layers [10] "
            f"--nb_epochs {self.epochs}"
        )

    def __get_train_fidexglorules_config(self) -> str:
        today = datetime.today().strftime("%Y_%m_%d")

        return (
            f"--root_folder {self.project_abspath} "
            f"--train_data_file {self.reldir}/train_data_normalized.csv "
            f"--train_pred_file {self.reldir}/dimlpbt_train_predictions.out "
            f"--train_class_file {self.reldir}/train_classes.csv "
            f"--console_file {self.reldir}/fidexglorules_{today}.log "
            f"--nb_attributes {self.nb_features} "
            f"--nb_classes {self.nb_classes} "
            f"--weights_file {self.reldir}/dimlpBT.wts "
            f"--global_rules_outfile {self.reldir}/global_rules.json "
            "--heuristic 1 "
            "--nb_threads 4 "
            "--min_covering 10 "
            "--max_iterations 20 "
            "--min_fidelity 1.0 "
            "--lowest_min_fidelity 1.0 "
            "--dropout_dim 0.7 "
            "--dropout_hyp 0.7 "
            "--decision_threshold 0.06 "
            "--positive_class_index 1"
        )
