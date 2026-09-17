#!/usr/bin/env python3
"""
Tabular cross-validation runner for DIMLP/Fidex.

This file:
  - loads a prepared tabular dataset in DIMLP/Fidex text format;
  - rebuilds stratified cross-validation folds from the concatenated data;
  - trains DIMLP once per fold;
  - runs Fidex rule extraction and rule statistics for each requested
    parameter configuration;
  - writes per-fold outputs and global text/JSON summaries.

The original train/test files are concatenated once, then stratified folds are
rebuilt from the class file. For each fold, DIMLP is trained once and Fidex rule
generation/statistics are run for every parameter combination.

Expected dataset files:
  <dataset_dir>/train_data.txt
  <dataset_dir>/train_class.txt
  <dataset_dir>/test_data.txt
  <dataset_dir>/test_class.txt
  <dataset_dir>/attributes.txt     optional

Example:
  python scripts/crossVal_tabular.py \
    --dataset breastCancer \
    --n_folds 10 \
    --fidexVersion fidexEarlyStopping fidexFull \
    --zeroFidelityRatio 0.01 0.02 \
    --threshold_decay_function Linear FastExponential \
    --dropout_dim 0.8 \
    --dropout_hyp 0.8 \
    --fidelity_importance 0.6 \
    --threshold_fidelity_only 0.6 \
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Metrics parsing (to generate cross-validation summary)
# =============================================================================

MODEL_STATS_FILE = "model/statsDimlpTrn.txt"
RULE_STATS_FILE = "global_rules_stats.txt"

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
STAT_ENTRY_RE = re.compile(
    rf"(?:^|,\s*)(?P<key>.+?)\s*[:=]\s*(?P<value>{NUMBER_RE.pattern})(?=,|$)"
)

# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class Dataset:
    """Container for a prepared tabular dataset used in cross-validation.

    Args:
        data: Concatenated input samples from original train and test files.
        classes: Concatenated one-hot class matrix.
        labels: One-dimensional class indices derived from classes.
        attributes_file: Optional attribute-name file copied into each fold.
    """

    data: np.ndarray
    classes: np.ndarray
    labels: np.ndarray
    attributes_file: Path | None

    @property
    def nb_attributes(self) -> int:
        """Return the number of input attributes in the dataset."""
        return int(self.data.shape[1])

    @property
    def nb_classes(self) -> int:
        """Return the number of one-hot output classes in the dataset."""
        return int(self.classes.shape[1])


@dataclass(frozen=True)
class FidexConfig:
    """Container for one Fidex parameter configuration.

    Args:
        fidex_version: Fidex algorithm version, or None to use the C++ default.
        zero_fidelity_ratio: Early-stopping zero-fidelity ratio, if provided.
        threshold_decay_function: Early-stopping threshold decay function, if provided.
        fidelity_importance: Candidate scoring fidelity weight, if provided.
        threshold_fidelity_only: Fidelity-only switch threshold, if provided.
        dropout_dim: Full Fidex dimension dropout, if provided.
        dropout_hyp: Full Fidex hyperplane dropout, if provided.
        max_iterations: Fidex maximum iterations, if provided.
        min_covering: Minimum rule covering, if provided.
    """

    fidex_version: str | None
    zero_fidelity_ratio: float | None
    threshold_decay_function: str | None
    fidelity_importance: float | None
    threshold_fidelity_only: float | None
    dropout_dim: float | None
    dropout_hyp: float | None
    max_iterations: int | None
    min_covering: int | None

    @property
    def name(self) -> str:
        """Build a stable directory name from the explicitly configured parameters."""
        parts = [self.fidex_version or "fidexDefault"]
        if self.fidelity_importance is not None:
            parts.append(f"fi{token(self.fidelity_importance)}")
        if self.threshold_fidelity_only is not None:
            parts.append(f"thr{token(self.threshold_fidelity_only)}")
        if self.max_iterations is not None:
            parts.append(f"it{self.max_iterations}")
        if self.fidex_version == "fidexFull":
            if self.dropout_dim is not None:
                parts.append(f"dd{token(self.dropout_dim)}")
            if self.dropout_hyp is not None:
                parts.append(f"dh{token(self.dropout_hyp)}")
        else:
            if self.zero_fidelity_ratio is not None:
                parts.append(f"zfr{token(self.zero_fidelity_ratio)}")
            if self.threshold_decay_function is not None:
                parts.append(self.threshold_decay_function)
        if self.min_covering is not None:
            parts.append(f"mc{self.min_covering}")
        return "_".join(parts)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line options and return the resulting namespace."""
    parser = argparse.ArgumentParser(
        description="Run stratified cross-validation experiments on a prepared tabular dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    dataset = parser.add_argument_group("Dataset")
    dataset.add_argument("--dataset", required=True, help="Dataset name or path.")
    dataset.add_argument("--train_data_file", default="train_data.txt")
    dataset.add_argument("--train_class_file", default="train_class.txt")
    dataset.add_argument("--test_data_file", default="test_data.txt")
    dataset.add_argument("--test_class_file", default="test_class.txt")
    dataset.add_argument("--attributes_file", default="attributes.txt")

    cv = parser.add_argument_group("Cross-validation")
    cv.add_argument("--n_folds", "--n_trials", dest="n_folds", type=int, default=10)
    cv.add_argument("--start_fold", type=int, default=1, help="First fold to run, using one-based numbering.")
    cv.add_argument("--end_fold", type=int, default=None, help="Last fold to run, using one-based numbering.")
    cv.add_argument("--seed", "--crossval_seed", dest="seed", type=int, default=None)
    cv.add_argument("--output_root", default=None, help="Directory where fold outputs and summaries are written.")
    cv.add_argument("--dry_run", action="store_true", help="Write fold files and print commands without running DIMLP/Fidex.")
    cv.add_argument("--summary_only", action="store_true", help="Only aggregate existing fold outputs into summary files.")
    cv.add_argument("--skip_train", action="store_true", help="Reuse existing DIMLP outputs.")
    cv.add_argument("--skip_rules", action="store_true", help="Only run DIMLP training.")
    cv.add_argument("--keep_going", action="store_true", help="Continue with later steps/folds after a failed command.")

    dimlp = parser.add_argument_group("DIMLP")
    dimlp.add_argument("--hidden_layers", default="5")
    dimlp.add_argument("--nb_epochs", type=int, default=1500)
    dimlp.add_argument("--nb_quant_levels", type=int, default=None)
    dimlp.add_argument("--dimlp_seed", type=int, default=0)
    dimlp.add_argument("--dimlp_options", default="", help="Extra raw command options appended to every dimlpTrn call.")

    fidex = parser.add_argument_group("Fidex grid")
    fidex.add_argument(
        "--fidexVersion",
        nargs="+",
        choices=("fidexEarlyStopping", "fidexFull"),
        default=None,
    )
    fidex.add_argument("--zeroFidelityRatio", nargs="+", type=float, default=None)
    fidex.add_argument(
        "--threshold_decay_function",
        nargs="+",
        choices=(
            "Linear",
            "FastPower",
            "SlowPower",
            "VeryFastPower",
            "VerySlowPower",
            "FastExponential",
            "SlowExponential",
        ),
        default=None,
    )
    fidex.add_argument("--fidelity_importance", nargs="+", type=float, default=None)
    fidex.add_argument("--threshold_fidelity_only", nargs="+", type=float, default=None)
    fidex.add_argument("--dropout_dim", nargs="+", type=float, default=None)
    fidex.add_argument("--dropout_hyp", nargs="+", type=float, default=None)
    fidex.add_argument("--max_iterations", nargs="+", type=int, default=None)
    fidex.add_argument("--min_covering", nargs="+", type=int, default=None)
    fidex.add_argument("--heuristic", type=int, default=None)
    fidex.add_argument("--nb_threads", type=int, default=None)
    fidex.add_argument("--fidex_seed", type=int, default=None)
    fidex.add_argument("--verbose", type=int, default=None)
    fidex.add_argument("--rules_extension", choices=("rls", "json"), default="rls")
    fidex.add_argument("--fidex_options", default="", help="Extra raw command options appended to every fidexGloRules call.")
    fidex.add_argument("--fidex_stats_options", default="", help="Extra raw command options appended to every fidexGloStats call.")

    args = parser.parse_args()
    if args.end_fold is None:
        args.end_fold = args.n_folds
    validate_arguments(parser, args)
    return args


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate command-line arguments.

    Args:
        parser: Argument parser used to report validation errors.
        args: Parsed command-line arguments.
    """
    if args.n_folds < 2:
        parser.error("--n_folds must be at least 2.")
    if args.start_fold < 1 or args.end_fold > args.n_folds or args.start_fold > args.end_fold:
        parser.error("--start_fold and --end_fold must be inside 1..--n_folds.")
    if args.skip_train and args.skip_rules:
        parser.error("--skip_train and --skip_rules leave nothing to run.")

    for option in (
        "zeroFidelityRatio",
        "fidelity_importance",
        "threshold_fidelity_only",
        "dropout_dim",
        "dropout_hyp",
    ):
        values = getattr(args, option)
        if values is not None and any(value < 0.0 or value > 1.0 for value in values):
            parser.error(f"--{option} values must be between 0 and 1.")


# =============================================================================
# Main flow
# =============================================================================

def main() -> None:
    """Run the full cross-validation workflow or summary-only aggregation."""
    args = parse_arguments()
    dataset_dir = resolve_dataset_dir(args.dataset)
    output_root = Path(args.output_root).resolve() if args.output_root else dataset_dir / "CrossVal_tabular"
    seed = args.seed if args.seed is not None else time.time_ns() % (2**32)
    configs = build_fidex_configs(args)
    command_log: list[dict] = []

    print(f"Cross-validation seed: {seed}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output root: {output_root}")
    print(f"Fidex configurations: {len(configs)}")

    if not args.summary_only:
        dataset = load_dataset(args, dataset_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        write_metadata(output_root, args, dataset, seed, configs)
        run_folds(args, dataset, output_root, seed, configs, command_log)

    if args.dry_run:
        print("\nDry-run complete. No DIMLP/Fidex binding was executed.")
        return

    metrics, missing_files = collect_metrics(args, output_root, configs)
    write_summary(output_root, args, seed, configs, metrics, missing_files, command_log)
    print(f"\nCross-validation summary written to {output_root / 'crossval_stats.txt'}")


# =============================================================================
# Dataset loading and folds
# =============================================================================

def resolve_dataset_dir(dataset: str) -> Path:
    """Resolve a dataset name or path to a dataset directory.

    Args:
        dataset: Dataset name or path passed through the CLI.
    """
    dataset_path = Path(dataset).expanduser()
    candidates = [
        dataset_path,
        SCRIPT_DIR / dataset_path,
        SCRIPT_DIR / "data" / dataset_path,
        REPO_ROOT / dataset_path,
        REPO_ROOT / "data" / dataset_path,
        REPO_ROOT.parent / "data" / dataset_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return (REPO_ROOT / "data" / dataset_path).resolve()


def load_dataset(args: argparse.Namespace, dataset_dir: Path) -> Dataset:
    """Load and concatenate the prepared train/test dataset files.

    Args:
        args: Parsed command-line arguments containing file names.
        dataset_dir: Directory containing the prepared dataset files.
    """
    train_data = load_matrix(dataset_dir / args.train_data_file)
    test_data = load_matrix(dataset_dir / args.test_data_file)
    train_classes = load_matrix(dataset_dir / args.train_class_file)
    test_classes = load_matrix(dataset_dir / args.test_class_file)

    data = np.concatenate((train_data, test_data), axis=0)
    classes = np.concatenate((train_classes, test_classes), axis=0).astype(int)
    if data.shape[0] != classes.shape[0]:
        raise ValueError("Data and class files do not contain the same number of samples.")
    if classes.ndim != 2 or classes.shape[1] < 2:
        raise ValueError("Class files must be one-hot matrices with at least two classes.")

    attributes_file = dataset_dir / args.attributes_file
    return Dataset(
        data=data,
        classes=classes,
        labels=np.argmax(classes, axis=1),
        attributes_file=attributes_file if attributes_file.exists() else None,
    )


def load_matrix(path: Path) -> np.ndarray:
    """Load a numeric text file as a two-dimensional NumPy array.

    Args:
        path: Path to the text file to load.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return np.atleast_2d(np.loadtxt(path))


def build_stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Build stratified test indices for each fold.

    Args:
        labels: One-dimensional class labels for all samples.
        n_folds: Number of folds to create.
        seed: Random seed used to shuffle indices inside each class.
    """
    class_counts = np.bincount(labels)
    non_empty_counts = class_counts[class_counts > 0]
    if len(non_empty_counts) and int(non_empty_counts.min()) < n_folds:
        raise ValueError("Every class must contain at least n_folds samples.")

    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_folds)]
    for class_id in sorted(np.unique(labels)):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        for fold_id, part in enumerate(np.array_split(indices, n_folds)):
            fold_indices[fold_id].extend(part.tolist())

    return [np.array(sorted(indices), dtype=int) for indices in fold_indices]


def train_indices_from_test_indices(nb_samples: int, test_indices: np.ndarray) -> np.ndarray:
    """Return all sample indices that are not part of the test fold.

    Args:
        nb_samples: Total number of samples in the full dataset.
        test_indices: Indices assigned to the test fold.
    """
    mask = np.ones(nb_samples, dtype=bool)
    mask[test_indices] = False
    return np.where(mask)[0]


def write_fold_files(fold_dir: Path, dataset: Dataset, train_indices: np.ndarray, test_indices: np.ndarray) -> None:
    """Write train/test files consumed by DIMLP and Fidex for one fold.

    Args:
        fold_dir: Output directory for the current fold.
        dataset: Full loaded dataset.
        train_indices: Sample indices used for training.
        test_indices: Sample indices used for testing.
    """
    (fold_dir / "model").mkdir(parents=True, exist_ok=True)
    np.savetxt(fold_dir / "train_data.txt", dataset.data[train_indices], fmt="%.18g")
    np.savetxt(fold_dir / "test_data.txt", dataset.data[test_indices], fmt="%.18g")
    np.savetxt(fold_dir / "train_class.txt", dataset.classes[train_indices], fmt="%d")
    np.savetxt(fold_dir / "test_class.txt", dataset.classes[test_indices], fmt="%d")
    np.savetxt(fold_dir / "train_indices.txt", train_indices, fmt="%d")
    np.savetxt(fold_dir / "test_indices.txt", test_indices, fmt="%d")
    if dataset.attributes_file is not None:
        (fold_dir / "attributes.txt").write_text(dataset.attributes_file.read_text(errors="replace"))


# =============================================================================
# Execution
# =============================================================================

def run_folds(
    args: argparse.Namespace,
    dataset: Dataset,
    output_root: Path,
    seed: int,
    configs: list[FidexConfig],
    command_log: list[dict],
) -> None:
    """Run selected folds, including training and rule generation.

    Args:
        args: Parsed command-line arguments.
        dataset: Full loaded dataset.
        output_root: Root directory for cross-validation outputs.
        seed: Seed used to build stratified folds.
        configs: Fidex configurations to run for each fold.
        command_log: Mutable list collecting executed command records.
    """
    folds = build_stratified_folds(dataset.labels, args.n_folds, seed)
    runners = dry_run_runners() if args.dry_run else load_runners()
    width = max(2, len(str(args.n_folds)))

    for fold in range(args.start_fold, args.end_fold + 1):
        fold_dir = output_root / f"fold_{fold:0{width}d}"
        test_indices = folds[fold - 1]
        train_indices = train_indices_from_test_indices(len(dataset.data), test_indices)

        print("\n" + "=" * 80)
        print(f"Fold {fold}/{args.n_folds}: {fold_dir}")
        print("=" * 80)

        fold_dir.mkdir(parents=True, exist_ok=True)
        write_fold_files(fold_dir, dataset, train_indices, test_indices)

        if not args.skip_train:
            run_step(command_log, fold, "dimlpTrn", None, build_dimlp_command(args, fold_dir, dataset), runners["dimlpTrn"], args)

        if args.skip_rules:
            continue

        for config in configs:
            (fold_dir / "rules" / config.name).mkdir(parents=True, exist_ok=True)
            run_step(command_log, fold, "fidexGloRules", config, build_rules_command(args, fold_dir, dataset, config), runners["fidexGloRules"], args)
            run_step(command_log, fold, "fidexGloStats", config, build_stats_command(args, fold_dir, dataset, config), runners["fidexGloStats"], args)


def load_runners() -> dict[str, Callable[[str], int]]:
    """Import and return the real DIMLP/Fidex Python binding functions."""
    from dimlpfidex import dimlp, fidex

    return {
        "dimlpTrn": dimlp.dimlpTrn,
        "fidexGloRules": fidex.fidexGloRules,
        "fidexGloStats": fidex.fidexGloStats,
    }


def dry_run_runners() -> dict[str, Callable[[str], int]]:
    """Return no-op runners used when commands should only be printed."""
    return {
        "dimlpTrn": lambda _command: 0,
        "fidexGloRules": lambda _command: 0,
        "fidexGloStats": lambda _command: 0,
    }


def run_step(
    command_log: list[dict],
    fold: int,
    step: str,
    config: FidexConfig | None,
    command: str,
    runner: Callable[[str], int],
    args: argparse.Namespace,
) -> None:
    """Run one command-producing step and record its status.

    Args:
        command_log: Mutable list collecting command records.
        fold: One-based fold number.
        step: Step name, such as dimlpTrn or fidexGloRules.
        config: Fidex configuration, or None for DIMLP training.
        command: Command string passed to the runner.
        runner: Python binding function that executes the command.
        args: Parsed command-line arguments controlling dry-run and errors.
    """
    record = {"fold": fold, "step": step, "config": config.name if config else None, "command": command}
    command_log.append(record)
    print(f"\n[{step}{' / ' + config.name if config else ''}] {command}\n")
    if args.dry_run:
        return

    start = time.perf_counter()
    status = runner(command)
    elapsed = time.perf_counter() - start
    record["elapsed_seconds"] = elapsed
    print(f"[{step}] execution time: {elapsed:.2f} sec")

    if status == -1:
        record["failed"] = True
        message = f"{step} failed on fold {fold}"
        if config is not None:
            message += f" ({config.name})"
        if args.keep_going:
            print(message)
            return
        raise RuntimeError(message)


# =============================================================================
# Command builders
# =============================================================================

def build_dimlp_command(args: argparse.Namespace, fold_dir: Path, dataset: Dataset) -> str:
    """Build the dimlpTrn command for one fold.

    Args:
        args: Parsed command-line arguments.
        fold_dir: Directory containing fold files and receiving outputs.
        dataset: Full loaded dataset, used for dimensions.
    """
    parts = [
        "--root_folder", fold_dir.resolve(),
        "--train_data_file", "train_data.txt",
        "--train_class_file", "train_class.txt",
        "--test_data_file", "test_data.txt",
        "--test_class_file", "test_class.txt",
        "--nb_attributes", dataset.nb_attributes,
        "--nb_classes", dataset.nb_classes,
        "--weights_outfile", "model/weights.wts",
        "--stats_file", MODEL_STATS_FILE,
        "--train_pred_outfile", "model/predTrain.out",
        "--test_pred_outfile", "model/predTest.out",
        "--hidden_layers", args.hidden_layers,
        "--nb_epochs", args.nb_epochs,
        "--seed", args.dimlp_seed,
    ]
    if args.nb_quant_levels is not None:
        parts.extend(["--nb_quant_levels", args.nb_quant_levels])
    return command_string(parts, args.dimlp_options)


def build_rules_command(args: argparse.Namespace, fold_dir: Path, dataset: Dataset, config: FidexConfig) -> str:
    """Build the fidexGloRules command for one fold and one configuration.

    Args:
        args: Parsed command-line arguments.
        fold_dir: Directory containing fold files and model outputs.
        dataset: Full loaded dataset, used for dimensions.
        config: Fidex parameter configuration to run.
    """
    parts = [
        "--root_folder", fold_dir.resolve(),
        "--train_data_file", "train_data.txt",
        "--train_pred_file", "model/predTrain.out",
        "--train_class_file", "train_class.txt",
        "--nb_attributes", dataset.nb_attributes,
        "--nb_classes", dataset.nb_classes,
        "--weights_file", "model/weights.wts",
        "--global_rules_outfile", f"rules/{config.name}/globalRules.{args.rules_extension}",
    ]
    if args.nb_quant_levels is not None:
        parts.extend(["--nb_quant_levels", args.nb_quant_levels])
    if args.heuristic is not None:
        parts.extend(["--heuristic", args.heuristic])
    if args.nb_threads is not None:
        parts.extend(["--nb_threads", args.nb_threads])
    if args.verbose is not None:
        parts.extend(["--verbose", args.verbose])
    if config.max_iterations is not None:
        parts.extend(["--max_iterations", config.max_iterations])
    if config.fidex_version is not None:
        parts.extend(["--fidexVersion", config.fidex_version])
    if config.fidelity_importance is not None:
        parts.extend(["--fidelity_importance", config.fidelity_importance])
    if config.threshold_fidelity_only is not None:
        parts.extend(["--threshold_fidelity_only", config.threshold_fidelity_only])
    if config.fidex_version == "fidexFull":
        if config.dropout_dim is not None:
            parts.extend(["--dropout_dim", config.dropout_dim])
        if config.dropout_hyp is not None:
            parts.extend(["--dropout_hyp", config.dropout_hyp])
    else:
        if config.zero_fidelity_ratio is not None:
            parts.extend(["--zeroFidelityRatio", config.zero_fidelity_ratio])
        if config.threshold_decay_function is not None:
            parts.extend(["--threshold_decay_function", config.threshold_decay_function])
    if (fold_dir / "attributes.txt").exists():
        parts.extend(["--attributes_file", "attributes.txt"])
    if config.min_covering is not None:
        parts.extend(["--min_covering", config.min_covering])
    if args.fidex_seed is not None:
        parts.extend(["--seed", args.fidex_seed])
    return command_string(parts, args.fidex_options)


def build_stats_command(args: argparse.Namespace, fold_dir: Path, dataset: Dataset, config: FidexConfig) -> str:
    """Build the fidexGloStats command for one fold and one configuration.

    Args:
        args: Parsed command-line arguments.
        fold_dir: Directory containing fold files and rule outputs.
        dataset: Full loaded dataset, used for dimensions.
        config: Fidex parameter configuration to evaluate.
    """
    rules_file = f"rules/{config.name}/globalRules.{args.rules_extension}"
    parts = [
        "--root_folder", fold_dir.resolve(),
        "--test_data_file", "test_data.txt",
        "--test_pred_file", "model/predTest.out",
        "--test_class_file", "test_class.txt",
        "--nb_attributes", dataset.nb_attributes,
        "--nb_classes", dataset.nb_classes,
        "--global_rules_file", rules_file,
        "--global_rules_outfile", f"rules/{config.name}/globalRulesWithStats.{args.rules_extension}",
        "--stats_file", f"rules/{config.name}/{RULE_STATS_FILE}",
    ]
    if (fold_dir / "attributes.txt").exists():
        parts.extend(["--attributes_file", "attributes.txt"])
    return command_string(parts, args.fidex_stats_options)


def command_string(parts: list, raw_options: str = "") -> str:
    """Join command arguments into a shell-like string.

    Args:
        parts: Structured command tokens to quote safely.
        raw_options: Optional raw string appended unchanged at the end.
    """
    command = " ".join(shlex.quote(str(part)) for part in parts)
    return f"{command} {raw_options.strip()}".strip()


def build_fidex_configs(args: argparse.Namespace) -> list[FidexConfig]:
    """Build all requested Fidex parameter configurations.

    Args:
        args: Parsed command-line arguments containing optional parameter lists.
    """
    versions = values_or_default(args.fidexVersion)
    min_covering_values = values_or_default(args.min_covering)
    configs = []
    for version in versions:
        common_values = itertools.product(
            values_or_default(args.fidelity_importance),
            values_or_default(args.threshold_fidelity_only),
            values_or_default(args.max_iterations),
            min_covering_values,
        )
        for fidelity_importance, threshold_fidelity_only, max_iterations, min_covering in common_values:
            if version == "fidexFull":
                full_values = itertools.product(
                    values_or_default(args.dropout_dim),
                    values_or_default(args.dropout_hyp),
                )
                for dropout_dim, dropout_hyp in full_values:
                    configs.append(FidexConfig(
                        version,
                        None,
                        None,
                        fidelity_importance,
                        threshold_fidelity_only,
                        dropout_dim,
                        dropout_hyp,
                        max_iterations,
                        min_covering,
                    ))
            else:
                early_values = itertools.product(
                    values_or_default(args.zeroFidelityRatio),
                    values_or_default(args.threshold_decay_function),
                )
                for zero_fidelity_ratio, threshold_decay_function in early_values:
                    configs.append(FidexConfig(
                        version,
                        zero_fidelity_ratio,
                        threshold_decay_function,
                        fidelity_importance,
                        threshold_fidelity_only,
                        None,
                        None,
                        max_iterations,
                        min_covering,
                    ))
    return configs


def values_or_default(values: list | None) -> list:
    """Return CLI values or a single None to represent the C++ default.

    Args:
        values: Optional list of values parsed from the CLI.
    """
    return values if values is not None else [None]


# =============================================================================
# Summary
# =============================================================================

def collect_metrics(args: argparse.Namespace, output_root: Path, configs: list[FidexConfig]) -> tuple[dict, list[dict]]:
    """Collect numeric metrics from all expected fold output files.

    Args:
        args: Parsed command-line arguments defining fold range.
        output_root: Root directory containing fold outputs.
        configs: Fidex configurations expected under each fold.
    """
    metrics = {"model": {}, "rules": {}}
    missing_files: list[dict] = []
    width = max(2, len(str(args.n_folds)))

    for fold in range(args.start_fold, args.end_fold + 1):
        fold_dir = output_root / f"fold_{fold:0{width}d}"
        model_file = fold_dir / MODEL_STATS_FILE
        if model_file.exists():
            add_file_metrics(metrics["model"], model_file, fold)
        elif not args.skip_train:
            missing_files.append({"fold": fold, "config": None, "file": str(model_file)})

        if args.skip_rules:
            continue

        for config in configs:
            rules_file = fold_dir / "rules" / config.name / RULE_STATS_FILE
            if rules_file.exists():
                add_file_metrics(metrics["rules"].setdefault(config.name, {}), rules_file, fold)
            else:
                missing_files.append({"fold": fold, "config": config.name, "file": str(rules_file)})

    return metrics, missing_files


def add_file_metrics(target: dict, file_path: Path, fold: int) -> None:
    """Add all metrics from one stats file into a metrics dictionary.

    Args:
        target: Metrics dictionary to mutate.
        file_path: Stats file to parse.
        fold: One-based fold number associated with the file.
    """
    for metric, value in parse_stats_file(file_path).items():
        target.setdefault(metric, []).append({"fold": fold, "value": value})


def parse_stats_file(file_path: Path) -> dict[str, float]:
    """Parse numeric key/value metrics from a stats text file.

    Args:
        file_path: Path to a DIMLP or Fidex stats file.
    """
    stats: dict[str, float] = {}
    for line in file_path.read_text(errors="replace").splitlines():
        for match in STAT_ENTRY_RE.finditer(line):
            stats[match.group("key").strip()] = float(match.group("value"))
    return stats


def write_summary(
    output_root: Path,
    args: argparse.Namespace,
    seed: int,
    configs: list[FidexConfig],
    metrics: dict,
    missing_files: list[dict],
    command_log: list[dict],
) -> None:
    """Write text and JSON summaries for the cross-validation run.

    Args:
        output_root: Root directory where summary files are written.
        args: Parsed command-line arguments.
        seed: Seed used for fold generation.
        configs: Fidex configurations included in the summary.
        metrics: Collected model and rule metrics.
        missing_files: Expected stats files that were not found.
        command_log: Commands executed during the current run.
    """
    timings = collect_timings(command_log)
    summary = {
        "dataset": args.dataset,
        "output_root": str(output_root),
        "n_folds": args.n_folds,
        "start_fold": args.start_fold,
        "end_fold": args.end_fold,
        "seed": seed,
        "model": build_model_summary(metrics["model"], timings.get("model", {})),
        "fidex_configs": build_fidex_summaries(args, configs, metrics["rules"], timings),
        "missing_files": missing_files,
        "commands": command_log,
    }
    lines = [
        "Tabular DIMLP/Fidex cross-validation statistics",
        f"Dataset: {args.dataset}",
        f"Output root: {output_root}",
        f"Folds: {args.start_fold}..{args.end_fold} / {args.n_folds}",
        f"Seed: {seed}",
        "",
        "Values are reported as mean (std).",
        "",
    ]

    append_model_report(lines, summary["model"])
    if not args.skip_rules:
        append_fidex_reports(lines, summary["fidex_configs"])
    append_missing_files(lines, missing_files)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "crossval_stats.txt").write_text("\n".join(lines))
    (output_root / "crossval_stats.json").write_text(json.dumps(summary, indent=2))
    write_command_log(output_root, command_log)


def build_model_summary(metrics: dict, timings: dict) -> dict:
    """Build the complete model report payload.

    Args:
        metrics: Model metrics collected from DIMLP stats files.
        timings: Command timings collected for model steps.
    """
    return {
        "source_file": MODEL_STATS_FILE,
        "timings": summarize_timings(timings),
        "metrics": summarize_metrics(metrics),
    }


def build_fidex_summaries(
    args: argparse.Namespace,
    configs: list[FidexConfig],
    metrics: dict,
    timings: dict,
) -> dict:
    """Build complete report payloads for every Fidex configuration.

    Args:
        args: Parsed command-line arguments containing cross-validation settings.
        configs: Fidex configurations to report.
        metrics: Rule metrics indexed by configuration name.
        timings: Command timings indexed by configuration name.
    """
    summaries = {}
    for config in configs:
        summaries[config.name] = {
            "parameters": asdict(config),
            "parameters_text": format_config_parameters(args, config),
            "source_file": f"rules/{config.name}/{RULE_STATS_FILE}",
            "timings": summarize_timings(timings.get(config.name, {})),
            "metrics": summarize_metrics(metrics.get(config.name, {})),
        }
    return summaries


def append_model_report(lines: list[str], model_summary: dict) -> None:
    """Append the complete model report to the text summary.

    Args:
        lines: Text summary lines to mutate.
        model_summary: Complete model report payload.
    """
    if not model_summary["metrics"] and not model_summary["timings"]:
        return
    lines.append("Model")
    lines.append(f"  source file: {model_summary['source_file']}")
    append_timing_block(lines, model_summary["timings"])
    append_metric_block(lines, model_summary["metrics"])
    lines.append("")


def append_fidex_reports(lines: list[str], fidex_summaries: dict) -> None:
    """Append complete Fidex configuration reports to the text summary.

    Args:
        lines: Text summary lines to mutate.
        fidex_summaries: Complete Fidex report payloads indexed by config name.
    """
    lines.append("Fidex configurations")
    for index, (name, config_summary) in enumerate(fidex_summaries.items(), start=1):
        lines.append(f"{index}. {name}")
        lines.append(f"  parameters: {config_summary['parameters_text']}")
        lines.append(f"  source file: {config_summary['source_file']}")
        append_timing_block(lines, config_summary["timings"])
        append_metric_block(lines, config_summary["metrics"])
    lines.append("")


def format_config_parameters(args: argparse.Namespace, config: FidexConfig) -> str:
    """Format the main experiment parameters for one Fidex configuration.

    Args:
        args: Parsed command-line arguments containing cross-validation settings.
        config: Fidex configuration to describe.
    """
    parameters = [
        ("method", config.fidex_version or "C++ default"),
        ("nb_trials", args.n_folds),
    ]
    if config.fidelity_importance is not None:
        parameters.append(("fidelity_importance", config.fidelity_importance))
    if config.threshold_fidelity_only is not None:
        parameters.append(("threshold_fidelity_only", config.threshold_fidelity_only))
    if config.max_iterations is not None:
        parameters.append(("max_iterations", config.max_iterations))
    if config.min_covering is not None:
        parameters.append(("min_covering", config.min_covering))

    if config.fidex_version == "fidexFull":
        if config.dropout_dim is not None:
            parameters.append(("dropout_dim", config.dropout_dim))
        if config.dropout_hyp is not None:
            parameters.append(("dropout_hyp", config.dropout_hyp))
    else:
        if config.zero_fidelity_ratio is not None:
            parameters.append(("zeroFidelityRatio", config.zero_fidelity_ratio))
        if config.threshold_decay_function is not None:
            parameters.append(("threshold_decay_function", config.threshold_decay_function))

    return ", ".join(f"{name}={value}" for name, value in parameters)


def append_timing_block(lines: list[str], timings: dict) -> None:
    """Append timing summaries to the text report.

    Args:
        lines: Text summary lines to mutate.
        timings: Timing summaries indexed by display name.
    """
    for timing_name in sorted(timings):
        lines.append("  " + summary_line(timing_name, timings[timing_name]))


def append_metric_block(lines: list[str], metrics: dict) -> None:
    """Append metric summaries to the text report.

    Args:
        lines: Text summary lines to mutate.
        metrics: Metric summaries indexed by source metric name.
    """
    for metric_name in sorted(metrics):
        lines.append("  " + summary_line(metric_name, metrics[metric_name]))


def append_missing_files(lines: list[str], missing_files: list[dict]) -> None:
    """Append missing output files to the text summary.

    Args:
        lines: Text summary lines to mutate.
        missing_files: Missing file records collected during aggregation.
    """
    if not missing_files:
        return
    lines.append("Missing files")
    for missing in missing_files:
        config = f" config={missing['config']}" if missing.get("config") else ""
        lines.append(f"  fold {missing['fold']}{config}: {missing['file']}")
    lines.append("")


def summarize_metrics(metrics: dict) -> dict:
    """Summarize every collected metric for JSON and text reports.

    Args:
        metrics: Raw metric entries indexed by metric name.
    """
    return {metric_name: summarize_entries(entries) for metric_name, entries in metrics.items()}


def summarize_timings(timings: dict) -> dict:
    """Summarize every collected command timing for JSON and text reports.

    Args:
        timings: Raw timing entries indexed by command step.
    """
    return {
        f"{step} execution time (s)": summarize_entries(entries, source="command_log")
        for step, entries in timings.items()
    }


def summarize_entries(entries: list[dict], source: str | None = None) -> dict:
    """Compute mean/std and build a JSON summary for one metric.

    Args:
        entries: Fold/value entries for the metric.
        source: Optional source label for generated metrics such as timings.
    """
    values = [entry["value"] for entry in entries]
    mean, std = mean_std(values)
    payload = {"mean": mean, "std": std, "n": len(values), "values": entries}
    if source is not None:
        payload["source"] = source
    return payload


def summary_line(name: str, payload: dict) -> str:
    """Format one summarized metric or timing for the text report.

    Args:
        name: Display name of the metric or timing.
        payload: Summary payload produced by summarize_entries.
    """
    folds = [entry["fold"] for entry in payload["values"]]
    return f"{name} : {payload['mean']:.10g} ({payload['std']:.10g}) [n={payload['n']} folds={folds}]"


def collect_timings(command_log: list[dict]) -> dict:
    """Collect elapsed command times from the command log.

    Args:
        command_log: Command records produced during the current run.
    """
    timings: dict = {}
    for record in command_log:
        if "elapsed_seconds" not in record:
            continue
        config = record["config"] or "model"
        timings.setdefault(config, {}).setdefault(record["step"], []).append(
            {"fold": record["fold"], "value": record["elapsed_seconds"]}
        )
    return timings


def write_command_log(output_root: Path, command_log: list[dict]) -> None:
    """Write executed commands and timings to a reproducibility log.

    Args:
        output_root: Directory where the command log is written.
        command_log: Command records produced during the current run.
    """
    if not command_log:
        return
    lines = []
    for record in command_log:
        config = f" {record['config']}" if record.get("config") else ""
        lines.append(f"fold {record['fold']} [{record['step']}{config}]")
        lines.append(record["command"])
        if "elapsed_seconds" in record:
            lines.append(f"elapsed_seconds: {record['elapsed_seconds']:.6f}")
        if record.get("failed"):
            lines.append("failed: true")
        lines.append("")
    (output_root / "crossval_commands.txt").write_text("\n".join(lines))


def write_metadata(output_root: Path, args: argparse.Namespace, dataset: Dataset, seed: int, configs: list[FidexConfig]) -> None:
    """Write dataset and run metadata to JSON.

    Args:
        output_root: Directory where metadata is written.
        args: Parsed command-line arguments.
        dataset: Full loaded dataset.
        seed: Seed used for fold generation.
        configs: Fidex configurations built for the run.
    """
    metadata = {
        "dataset": args.dataset,
        "nb_samples": int(dataset.data.shape[0]),
        "nb_attributes": dataset.nb_attributes,
        "nb_classes": dataset.nb_classes,
        "seed": seed,
        "args": vars(args),
        "fidex_configs": {config.name: asdict(config) for config in configs},
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))


# =============================================================================
# Small helpers
# =============================================================================

def mean_std(values: list[float]) -> tuple[float, float]:
    """Compute population mean and standard deviation.

    Args:
        values: Numeric values to summarize.
    """
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def token(value: object) -> str:
    """Format a value for compact and filesystem-friendly names.

    Args:
        value: Value to convert into a name token.
    """
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
