from src.global_rules import GlobalRules
import src.constants as constants
from src.utils import init_args
from src.trainer import Trainer
from src.patient import *
import time
import os


def normalize_global_rules(abspath: str) -> None:
    normalization(
        f"--root_folder {abspath} "
        f"--rule_files {constants.MODEL_DIRNAME}/global_rules_denormalized.json "
        f"--output_rule_files {constants.MODEL_DIRNAME}/global_rules_normalized.json "
        f"--normalization_file {constants.MODEL_DIRNAME}/normalization_stats.txt "
        "--nb_attributes 79 "
        "--normalize_rules True "
        f"--attributes_file {constants.MODEL_DIRNAME}/attributes.txt"
    )


if __name__ == "__main__":
    args = init_args()
    abspath = os.path.abspath(os.path.dirname(__file__))
    constants.check_directories_existance(abspath)


    if args.train is not None:
        if args.train < 0.0 or args.train > 1.0:
            raise ValueError("Train argument value cannot be outside of [0.0,1.0].")

        trainer = Trainer(abspath)
        normalize = True
        trainer.train(normalize, args.train)
        exit(0)

    elif args.test:
        patients = write_samples_file(abspath, args.test)

    else:
        patients = write_patients(abspath)

    print("Loading global rules...")
    global_rules = GlobalRules.from_json_file(
        os.path.join(constants.MODEL_DIRNAME, "global_rules_denormalized.json"),
        read_attributes_file(abspath),
    )

    nb_global_rules = len(global_rules)
    nb_patients = len(patients)
    total_elapsed = 0.0

    print(f"Rule extraction is going to be performed on {nb_patients} patients")
    for i, patient in enumerate(patients):
        print(f"Extracting rules for patient {i+1}/{nb_patients}...")

        start = time.time()
        global_rules = patient.extract_rules(global_rules)
        end = time.time()
        elapsed = end - start
        total_elapsed += elapsed

        print(
            f"Extraction done, {len(patient.selected_rules)} rules found in {elapsed:.3f} seconds"
        )

    print(
        f"Rule extraction done for {nb_patients} patients in {total_elapsed:.3f} seconds\nWriting UNICANCER results file..."
    )
    write_results(abspath, patients)
    print(f"File successfully written")

    # save potentially generated rules if new ones appeared
    if len(global_rules) > nb_global_rules:
        difference = len(global_rules) - nb_global_rules
        print(f"{difference} new rule(s) have been generated, saving...")
        global_rules.save(abspath)
        normalize_global_rules(abspath) 

    print("Rule extraction program done")
