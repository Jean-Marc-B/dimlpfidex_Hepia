# generate_rules.py
import time
from dimlpfidex import fidex
from utils.config import *

def generate_rules(cfg, args, nb_attributes = None):
    start_time_global_rules = time.time()
    if nb_attributes is None:
        nb_attributes = cfg["nb_stats_attributes"]
    if args.statistic == "convDimlpFilter":
        train_data_file=cfg["train_feature_map_file"]
        test_data_file=cfg["test_feature_map_file"]
    else:
        train_data_file=cfg["train_stats_file"]
        test_data_file=cfg["test_stats_file"]

    # 1) Generate global rules
    command = (
        f'--train_data_file {train_data_file} '
        f'--train_pred_file {cfg["second_model_train_pred"]} '
        f'--train_class_file {cfg["train_class_file"]} '
        f'--nb_classes {cfg["nb_classes"]} '
        f'--global_rules_outfile {cfg["global_rules_file"]} '
        f'--nb_attributes {nb_attributes} '
        f'--heuristic 1 '
        f'--nb_threads 8 '
        f'--max_iterations 25 '
        f'--nb_quant_levels {NB_QUANT_LEVELS} '
        f'--dropout_dim {DROPOUT_DIM} '
        f'--dropout_hyp {DROPOUT_HYP} '
    )
    if args.statistic == "histogram":
        command += f'--attributes_file {cfg["attributes_file"]} '
    if cfg["using_decision_tree_model"]:
        command += f'--rules_file {cfg["second_model_output_weights"]} '
    else:
        command += f'--weights_file {cfg["second_model_output_weights"]} '

    print("\nComputing global rules...\n")
    status = fidex.fidexGloRules(command)
    if status != -1:
        print("\nGlobal rules computed.")

    # 2) Generate statistics of the ruleset
    command = (
        f'--test_data_file {test_data_file} '
        f'--test_pred_file {cfg["second_model_test_pred"]} '
        f'--test_class_file {cfg["test_class_file"]} '
        f'--nb_classes {cfg["nb_classes"]} '
        f'--global_rules_file {cfg["global_rules_file"]} '
        f'--nb_attributes {nb_attributes} '
        f'--global_rules_outfile {cfg["global_rules_with_test_stats"]} '
        f'--stats_file {cfg["global_rules_stats"]}'
    )

    print("\nComputing statistics on global rules...\n")
    status = fidex.fidexGloStats(command)
    if status != -1:
        print("\nStatistics computed.")

    end_time_global_rules = time.time()
    full_time_global_rules = end_time_global_rules - start_time_global_rules
    print(f"\nGlobal rules time = {full_time_global_rules:.2f} sec")
