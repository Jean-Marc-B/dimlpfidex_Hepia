from src.utils import init_args
from src.trainer import Trainer
from src.patient import *
from src.rule import *
import os

if __name__ == "__main__":
    args = init_args()
    abspath = os.path.abspath(os.path.dirname(__file__))

    if args.train:
        trainer = Trainer(abspath)
        trainer.train(True, 0.1)
        exit()

    elif args.test:
        # write_train_data(abspath, data, labels)
        patients = write_samples_file(abspath, args.test)

    else:
        # write_train_data(abspath, data, labels)
        patients = write_patients(abspath)

    print("Loading global rules...")
    global_rules = GlobalRules.from_json_file("temp/global_rules_denormalized.json")
    nb_global_rules = len(global_rules)

    npatients = len(patients)
    print(f"Rule extraction is going to be performed on {npatients} patients")
    for i, patient in enumerate(patients):
        print(f"Extracting rules for patient {i+1}/{npatients}...")
        global_rules = patient.extract_rules(global_rules)
        print(f"Extraction done, {len(patient.selected_rules)} rules found")

    print(f"Rule extraction done for {npatients} patients")
    print(f"Writing UNICANCER results file...")
    write_results(abspath, patients)
    print(f"File successfully written")

    # save potentially generated rules if new ones appeared
    if len(global_rules) > nb_global_rules:
        difference = len(global_rules) - nb_global_rules
        print(f"{difference} new rule(s) have been generated, saving...")
        global_rules.save(abspath)

    print("Rule extraction program done")
