#!/usr/bin/env python3
"""
Simple cross-validation runner for ../data/datanorm.

Edit the configuration block below, then run:

    python crossVal_datanorm_dimlp.py

For each fold, the script writes train/test files, runs:
  1. dimlp.dimlpTrn
  2. fidex.fidexGloRules
  3. fidex.fidexGloStats

The final mean(std) summary is written in OUTPUT_ROOT.
"""

from pathlib import Path
import json
import math
import re
import shlex
import time

import numpy as np
from dimlpfidex import dimlp, fidex


# =============================================================================
# Configuration to edit by hand
# =============================================================================

N_FOLDS = 10
SEED = None  # None = generated once when the script starts

DATASET_DIR = Path("../data/datanorm")
OUTPUT_ROOT = DATASET_DIR / "testCrossVal"

TRAIN_DATA_FILE = DATASET_DIR / "train_data.txt"
TRAIN_CLASS_FILE = DATASET_DIR / "train_class.txt"
TEST_DATA_FILE = DATASET_DIR / "test_data.txt"
TEST_CLASS_FILE = DATASET_DIR / "test_class.txt"

# Inferred from files when set to None.
NB_ATTRIBUTES = None
NB_CLASSES = None
FIDEX_VERSION = "fidexEarlyStopping"  # "fidexEarlyStopping" or "fidexFull"
ZERO_FIDELITY_RATIO = 0.06
DROPOUT_DIM = 0.8
DROPOUT_HYP = 0.8


# Command options. Keep paths out of these strings; fold-specific paths are added
# by the script. These are intentionally plain strings to make experiments easy.
DIMLP_TRN_OPTIONS = (
    "--hidden_layers 5 "
    "--nb_epochs 1500 "
    "--nb_quant_levels 50 "
    "--seed 0 "
)

FIDEX_GLO_RULES_OPTIONS = (
    "--heuristic 1 "
    "--nb_threads 4 "
    "--max_iterations 25 "
    "--nb_quant_levels 50 "
    f"--dropout_dim {DROPOUT_DIM} "
    f"--dropout_hyp {DROPOUT_HYP} "
    f"--fidexVersion {FIDEX_VERSION} "
    f"--zeroFidelityRatio {ZERO_FIDELITY_RATIO} "
    "--verbose 3 "
)

FIDEX_GLO_STATS_OPTIONS = ""


STATS_FILES = (
    "statsDimlpTrn.txt",
    "global_rules_stats.txt",
)

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
KEY_VALUE_RE = re.compile(
    rf"^\s*(?P<key>[^:=]+?)\s*[:=]\s*(?P<value>{NUMBER_RE.pattern})\s*$"
)
REQUESTED_METRICS = (
    ("statsDimlpTrn.txt", ("Accuracy on training set",), "DIMLP train accuracy"),
    ("statsDimlpTrn.txt", ("Accuracy on testing set", "Accuracy on test set"), "DIMLP test accuracy"),
    ("global_rules_stats.txt", ("Number of rules",), "Number of rules"),
    ("global_rules_stats.txt", ("mean sample covering number per rule",), "Mean sample covering number per rule"),
    ("global_rules_stats.txt", ("mean number of antecedents per rule",), "Mean number of antecedents per rule"),
)


def main():
    seed = SEED if SEED is not None else time.time_ns() % (2**32)
    print(f"Cross-validation seed: {seed}")

    data, classes = load_full_dataset()
    nb_attributes = NB_ATTRIBUTES or data.shape[1]
    nb_classes = NB_CLASSES or classes.shape[1]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_indices = build_folds(len(data), N_FOLDS, seed)

    command_log = []
    for fold_id, test_indices in enumerate(fold_indices, start=1):
        fold_dir = OUTPUT_ROOT / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Fold {fold_id}/{N_FOLDS}: {fold_dir}")
        print("=" * 80)

        train_indices = train_indices_from_test_indices(len(data), test_indices)
        write_fold_files(fold_dir, data, classes, train_indices, test_indices)

        commands = build_commands(fold_dir, nb_attributes, nb_classes)
        for name, command, runner in commands:
            command_record = {"fold": fold_id, "step": name, "command": command}
            command_log.append(command_record)
            print(f"\n[{name}] {command}\n")
            step_start = time.perf_counter()
            status = runner(command)
            elapsed_seconds = time.perf_counter() - step_start
            command_record["elapsed_seconds"] = elapsed_seconds
            print(f"[{name}] execution time: {elapsed_seconds:.2f} sec")
            if status == -1:
                raise RuntimeError(f"{name} failed on fold {fold_id}.")

    write_command_log(command_log)
    metrics, missing_files = collect_metrics()
    write_summary(seed, nb_attributes, nb_classes, metrics, missing_files, command_log)

    summary_file = OUTPUT_ROOT / "crossval_stats.txt"
    print(f"\nCross-validation summary written to {summary_file}")
    print("\n" + "=" * 80)
    print(summary_file.read_text())


def load_full_dataset():
    """Load original train/test files and concatenate them into one dataset."""
    train_data = np.atleast_2d(np.loadtxt(TRAIN_DATA_FILE))
    test_data = np.atleast_2d(np.loadtxt(TEST_DATA_FILE))
    train_classes = np.atleast_2d(np.loadtxt(TRAIN_CLASS_FILE)).astype(int)
    test_classes = np.atleast_2d(np.loadtxt(TEST_CLASS_FILE)).astype(int)

    data = np.concatenate((train_data, test_data), axis=0)
    classes = np.concatenate((train_classes, test_classes), axis=0)
    return data, classes


def build_folds(nb_samples, n_folds, seed):
    """Shuffle the whole dataset once and rotate the test fold."""
    if n_folds < 2:
        raise ValueError("N_FOLDS must be at least 2.")
    if nb_samples < n_folds:
        raise ValueError(f"Dataset has only {nb_samples} samples for {n_folds} folds.")

    rng = np.random.default_rng(seed)
    indices = np.arange(nb_samples)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def train_indices_from_test_indices(nb_samples, test_indices):
    """Return every index not used by the current test fold."""
    train_mask = np.ones(nb_samples, dtype=bool)
    train_mask[test_indices] = False
    return np.where(train_mask)[0]


def write_fold_files(fold_dir, data, classes, train_indices, test_indices):
    """Write the fold train/test files consumed by the DIMLP/Fidex bindings."""
    np.savetxt(fold_dir / "train_data.txt", data[train_indices], fmt="%.18g")
    np.savetxt(fold_dir / "test_data.txt", data[test_indices], fmt="%.18g")
    np.savetxt(fold_dir / "train_class.txt", classes[train_indices], fmt="%d")
    np.savetxt(fold_dir / "test_class.txt", classes[test_indices], fmt="%d")
    np.savetxt(fold_dir / "train_indices.txt", train_indices, fmt="%d")
    np.savetxt(fold_dir / "test_indices.txt", test_indices, fmt="%d")


def build_commands(fold_dir, nb_attributes, nb_classes):
    """Build fold-specific binding commands."""
    root_folder = shlex.quote(str(fold_dir.resolve()))

    dimlp_command = (
        f"--root_folder {root_folder} "
        f"--train_data_file train_data.txt "
        f"--train_class_file train_class.txt "
        f"--test_data_file test_data.txt "
        f"--test_class_file test_class.txt "
        f"--nb_attributes {nb_attributes} "
        f"--nb_classes {nb_classes} "
        f"--weights_outfile weights.wts "
        f"--stats_file statsDimlpTrn.txt "
        f"--train_pred_outfile predTrain.out "
        f"--test_pred_outfile predTest.out "
        f"{DIMLP_TRN_OPTIONS}"
    )

    rules_command = (
        f"--root_folder {root_folder} "
        f"--train_data_file train_data.txt "
        f"--train_pred_file predTrain.out "
        f"--train_class_file train_class.txt "
        f"--nb_attributes {nb_attributes} "
        f"--nb_classes {nb_classes} "
        f"--weights_file weights.wts "
        f"--global_rules_outfile globalRules.rls "
        f"{FIDEX_GLO_RULES_OPTIONS}"
    )

    stats_command = (
        f"--root_folder {root_folder} "
        f"--test_data_file test_data.txt "
        f"--test_pred_file predTest.out "
        f"--test_class_file test_class.txt "
        f"--nb_attributes {nb_attributes} "
        f"--nb_classes {nb_classes} "
        f"--global_rules_file globalRules.rls "
        f"--global_rules_outfile globalRulesWithStats.rls "
        f"--stats_file global_rules_stats.txt "
        f"{FIDEX_GLO_STATS_OPTIONS}"
    )

    return (
        ("dimlpTrn", dimlp_command, dimlp.dimlpTrn),
        ("fidexGloRules", rules_command, fidex.fidexGloRules),
        ("fidexGloStats", stats_command, fidex.fidexGloStats),
    )


def collect_metrics():
    """Parse fold stat files and collect numeric 'name : value' metrics."""
    metrics = {}
    missing_files = []

    for fold_id in range(1, N_FOLDS + 1):
        fold_dir = OUTPUT_ROOT / f"fold_{fold_id:02d}"
        for file_name in STATS_FILES:
            file_path = fold_dir / file_name
            if not file_path.exists():
                missing_files.append({"fold": fold_id, "file": str(file_path)})
                continue
            for metric_name, value in parse_stats_file(file_path).items():
                metrics.setdefault(file_name, {}).setdefault(metric_name, []).append(
                    {"fold": fold_id, "value": value}
                )

    return metrics, missing_files


def parse_stats_file(file_path):
    """Extract numeric metrics from `key : value`, `key = value`, and comma-separated summaries."""
    stats = {}
    for line in file_path.read_text(errors="replace").splitlines():
        for part in line.split(","):
            match = KEY_VALUE_RE.match(part)
            if match:
                stats[match.group("key").strip()] = float(match.group("value"))
    return stats


def write_summary(seed, nb_attributes, nb_classes, metrics, missing_files, command_log):
    """Write text and JSON summaries with mean(std) across folds."""
    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_root": str(OUTPUT_ROOT),
        "n_folds": N_FOLDS,
        "seed": seed,
        "nb_attributes": nb_attributes,
        "nb_classes": nb_classes,
        "fidex_version": FIDEX_VERSION,
        "zero_fidelity_ratio": ZERO_FIDELITY_RATIO,
        "metrics": {},
        "missing_files": missing_files,
        "commands": command_log,
    }
    if FIDEX_VERSION == "fidexFull":
        summary["dropout_dim"] = DROPOUT_DIM
        summary["dropout_hyp"] = DROPOUT_HYP

    lines = [
        "DIMLP/Fidex cross-validation statistics",
        f"Dataset: {DATASET_DIR}",
        f"Folds: {N_FOLDS}",
        f"Seed: {seed}",
        f"Number of attributes: {nb_attributes}",
        f"Number of classes: {nb_classes}",
        f"Fidex version: {FIDEX_VERSION}",
        f"Zero fidelity ratio: {ZERO_FIDELITY_RATIO}",
    ]
    if FIDEX_VERSION == "fidexFull":
        lines.extend([
            f"Dropout dim: {DROPOUT_DIM}",
            f"Dropout hyp: {DROPOUT_HYP}",
        ])
    lines.extend([
        "",
        "Values are reported as mean (std).",
        "",
    ])

    append_requested_metrics(lines, summary, metrics, command_log)

    for file_name in STATS_FILES:
        file_metrics = metrics.get(file_name, {})
        if not file_metrics:
            continue
        lines.append(file_name)
        summary["metrics"][file_name] = {}
        for metric_name in sorted(file_metrics):
            values = [entry["value"] for entry in file_metrics[metric_name]]
            mean, std = mean_std(values)
            folds = [entry["fold"] for entry in file_metrics[metric_name]]
            lines.append(f"  {metric_name} : {mean:.10g} ({std:.10g}) [n={len(values)} folds={folds}]")
            summary["metrics"][file_name][metric_name] = {
                "mean": mean,
                "std": std,
                "n": len(values),
                "values": file_metrics[metric_name],
            }
        lines.append("")

    if missing_files:
        lines.append("Missing files")
        for missing in missing_files:
            lines.append(f"  fold {missing['fold']} : {missing['file']}")
        lines.append("")

    (OUTPUT_ROOT / "crossval_stats.txt").write_text("\n".join(lines))
    (OUTPUT_ROOT / "crossval_stats.json").write_text(json.dumps(summary, indent=2))


def append_requested_metrics(lines, summary, metrics, command_log):
    """Add the high-level metrics most commonly compared across cross-validation folds."""
    selected = {}
    selected_lines = []

    for file_name, candidate_names, display_name in REQUESTED_METRICS:
        metric_name, entries = find_metric(metrics, file_name, candidate_names)
        if not entries:
            continue

        values = [entry["value"] for entry in entries]
        mean, std = mean_std(values)
        folds = [entry["fold"] for entry in entries]
        selected_lines.append(f"  {display_name} : {mean:.10g} ({std:.10g}) [n={len(values)} folds={folds}]")
        selected[display_name] = {
            "source_file": file_name,
            "source_metric": metric_name,
            "mean": mean,
            "std": std,
            "n": len(values),
            "values": entries,
        }

    timing_entries = [
        {"fold": record["fold"], "value": record["elapsed_seconds"]}
        for record in command_log
        if record["step"] == "fidexGloRules" and "elapsed_seconds" in record
    ]
    if timing_entries:
        values = [entry["value"] for entry in timing_entries]
        mean, std = mean_std(values)
        folds = [entry["fold"] for entry in timing_entries]
        display_name = "fidexGloRules execution time (s)"
        selected_lines.append(f"  {display_name} : {mean:.10g} ({std:.10g}) [n={len(values)} folds={folds}]")
        selected[display_name] = {
            "source": "command_log",
            "mean": mean,
            "std": std,
            "n": len(values),
            "values": timing_entries,
        }

    if not selected_lines:
        return

    summary["selected_metrics"] = selected
    lines.append("Selected metrics")
    lines.extend(selected_lines)
    lines.append("")


def find_metric(metrics, file_name, candidate_names):
    """Return the first available metric among aliases for one source file."""
    file_metrics = metrics.get(file_name, {})
    for name in candidate_names:
        if name in file_metrics:
            return name, file_metrics[name]
    return None, []


def write_command_log(command_log):
    """Write all commands executed by the bindings for reproducibility."""
    lines = []
    for record in command_log:
        lines.append(f"fold {record['fold']} [{record['step']}]")
        lines.append(record["command"])
        if "elapsed_seconds" in record:
            lines.append(f"elapsed_seconds: {record['elapsed_seconds']:.6f}")
        lines.append("")
    (OUTPUT_ROOT / "crossval_commands.txt").write_text("\n".join(lines))


def mean_std(values):
    """Population mean/std used for cross-validation summaries."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


if __name__ == "__main__":
    main()
