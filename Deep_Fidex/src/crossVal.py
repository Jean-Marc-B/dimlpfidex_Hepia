#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

"""
Exemple pour lancer :
python crossVal.py --n_folds 3   --dataset Cifar   --statistic patch_impact_and_image   --train_with_patches False   --train --stats --second_train --rules   --folder_sufix _fidImportance06_test --test


"""

VALUE_OPTIONS = {
    "--dataset",
    "--statistic",
    "--train_with_patches",
    "--folder_sufix",
    "--images",
    "--dct_patch_size",
    "--crossval_n_folds",
    "--crossval_fold",
    "--crossval_seed",
    "--crossval_output_folder",
    "--zeroFidelityRatio",
    "--fidexVersion",
    "--fidelity_importance",
    "--threshold_fidelity_only",
    "--gpu",
    "--alternative_folder",
}

CROSSVAL_OPTIONS = {
    "--crossval_n_folds",
    "--crossval_fold",
    "--crossval_seed",
    "--crossval_output_folder",
}

TRAIN_PHASE_FLAGS = {"--train", "--stats", "--second_train", "--heatmap"}
RULES_INCOMPATIBLE_FLAGS = {"--train", "--stats", "--second_train", "--images", "--heatmap"}
IMAGES_INCOMPATIBLE_FLAGS = {"--train", "--stats", "--second_train", "--rules", "--heatmap"}

STATS_FILES = (
    "stats_model.txt",
    "second_model_stats.txt",
    "global_rules_stats.txt",
)

STATISTIC_FOLDERS = {
    "histogram": "Histograms",
    "activation_layer": "Activations_Sum",
    "probability_multi_nets": "Probability_Multi_Nets_Images",
    "probability": "Probability_Images",
    "probability_and_image": "Probability_and_image",
    "probability_multi_nets_and_image": "Probability_Multi_Nets_and_image",
    "probability_multi_nets_and_image_in_one": "Probability_Multi_Nets_and_image_in_one",
    "convDimlpFilter": "Conv_DIMLP_Filter",
    "HOG_and_image": "HOG_and_image",
    "SHAP_and_image": "SHAP_and_image",
    "LBP_and_image": "LBP_and_image",
    "DCT_and_image": "DCT_and_image",
    "stats_and_image": "stats_and_image",
    "HOG": "HOG",
    "patch_impact_and_image": "patch_impact_and_image",
    "patch_impact_and_stats": "patch_impact_and_stats",
}

DATASET_FOLDERS = {
    "Mnist": "Mnist",
    "Cifar": "Cifar",
    "Happy": "Happy",
    "HAM10000": "HAM10000",
    "Pneumonia": "Pneumonia",
}

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
STAT_ENTRY_RE = re.compile(
    rf"(?:^|,\s*)(?P<key>.+?)\s*:\s*(?P<value>{NUMBER_RE.pattern})(?=,|$)"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run imageScan.py on several cross-validation folds, split rules "
            "generation from GPU stages, and aggregate numeric stats."
        )
    )
    parser.add_argument("--n_folds", "--n_trials", dest="n_folds", type=int, default=10)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=None)
    parser.add_argument("--crossval_seed", type=int, default=None)
    parser.add_argument("--keep_going", action="store_true", help="Continue with next folds after a failed command")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    parser.add_argument("--summary_only", action="store_true", help="Only aggregate existing fold statistics without running imageScan.py")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch imageScan.py")

    args, image_scan_args = parser.parse_known_args()
    if args.end_fold is None:
        args.end_fold = args.n_folds
    if args.n_folds < 2:
        parser.error("--n_folds must be at least 2.")
    if args.start_fold < 1 or args.end_fold > args.n_folds or args.start_fold > args.end_fold:
        parser.error("--start_fold and --end_fold must define a valid range inside 1..--n_folds.")

    for option in ("--dataset", "--statistic", "--train_with_patches"):
        if _get_option(image_scan_args, option) is None:
            parser.error(f"{option} must be provided for imageScan.py.")

    return args, image_scan_args


def main():
    args, image_scan_args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    image_scan_py = script_dir / "imageScan.py"
    if args.crossval_seed is None:
        args.crossval_seed = time.time_ns() % (2**32)
    # One seed is shared by every subprocess so train/stats and rules rebuild identical folds.
    print(f"Cross-validation seed: {args.crossval_seed}")

    base_suffix = _get_option(image_scan_args, "--folder_sufix", "") or ""
    root_name = _crossval_root_name(image_scan_args, base_suffix)
    width = max(2, len(str(args.n_folds)))
    command_records = []

    if not args.summary_only:
        for fold in range(args.start_fold, args.end_fold + 1):
            fold_name = f"fold_{fold:0{width}d}"
            fold_output_folder = f"{root_name}/{fold_name}"
            fold_args = _with_fold_options(image_scan_args, fold_output_folder, args.n_folds, fold, args.crossval_seed)
            fold_args = _with_fold_placeholders(fold_args, fold, fold_name)

            print("\n" + "=" * 80)
            print(f"Cross-validation fold {fold}/{args.n_folds} -> {fold_output_folder}")
            print("=" * 80)

            for phase, phase_args in _build_phase_args(fold_args):
                command = [args.python, str(image_scan_py)] + phase_args
                command_records.append({"fold": fold, "phase": phase, "command": command})
                print(f"\n[{phase}] {shlex.join(command)}\n")
                if args.dry_run:
                    continue
                status = subprocess.run(command, cwd=script_dir)
                if status.returncode != 0:
                    print(f"Command failed with status {status.returncode}: {shlex.join(command)}")
                    if not args.keep_going:
                        raise SystemExit(status.returncode)

    if args.dry_run:
        return

    summary_dir = _summary_dir(script_dir, image_scan_args, base_suffix)
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        _write_command_log(summary_dir, command_records)
    metrics, missing_files = _collect_metrics(script_dir, image_scan_args, base_suffix, args)
    _write_summary(summary_dir, args, image_scan_args, metrics, missing_files)

    print(f"\nCross-validation summary written to {summary_dir / 'crossval_stats.txt'}")


def _build_phase_args(fold_args):
    run_rules = _has_flag(fold_args, "--rules")
    run_images = _get_option(fold_args, "--images") is not None
    phases = []

    train_args = _remove_options(fold_args, {"--rules", "--images"})
    if _has_any_flag(train_args, TRAIN_PHASE_FLAGS):
        phases.append(("train_stats_second_train", train_args))

    if run_rules:
        rules_args = _remove_options(fold_args, RULES_INCOMPATIBLE_FLAGS)
        if not _has_flag(rules_args, "--rules"):
            rules_args.append("--rules")
        phases.append(("rules", rules_args))

    if run_images:
        images_args = _remove_options(fold_args, IMAGES_INCOMPATIBLE_FLAGS)
        phases.append(("images", images_args))

    if not phases:
        phases.append(("imageScan", fold_args))

    return phases


def _with_fold_options(image_scan_args, fold_output_folder, n_folds, fold, seed):
    args = _remove_options(image_scan_args, CROSSVAL_OPTIONS | {"--folder_sufix"})
    args.extend(
        [
            "--crossval_n_folds",
            str(n_folds),
            "--crossval_fold",
            str(fold),
            "--crossval_seed",
            str(seed),
            "--crossval_output_folder",
            fold_output_folder,
        ]
    )
    return args

# Replace placeholders in args with fold-specific values
def _with_fold_placeholders(args, fold, fold_name): 
    result = []
    for token in args:
        result.append(token.replace("{fold}", str(fold)).replace("{fold_name}", fold_name))
    return result


def _collect_metrics(script_dir, image_scan_args, base_suffix, args):
    metrics = {}
    missing_files = []
    width = max(2, len(str(args.n_folds)))

    for fold in range(args.start_fold, args.end_fold + 1):
        fold_name = f"fold_{fold:0{width}d}"
        files_dir = _summary_dir(script_dir, image_scan_args, base_suffix) / fold_name / "files"
        for file_name in STATS_FILES:
            file_path = files_dir / file_name
            if not file_path.exists():
                missing_files.append({"fold": fold, "file": str(file_path)})
                continue
            for metric_name, value in _parse_stats_file(file_path).items():
                metrics.setdefault(file_name, {}).setdefault(metric_name, []).append(
                    {"fold": fold, "value": value}
                )

    return metrics, missing_files


def _parse_stats_file(file_path):
    stats = {}
    for line in file_path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        # Some lines pack several "key : value" pairs separated by commas.
        # Metric names can also contain commas, so match complete key/value pairs.
        for match in STAT_ENTRY_RE.finditer(line):
            stats[match.group("key").strip()] = float(match.group("value"))
    return stats


def _write_summary(summary_dir, args, image_scan_args, metrics, missing_files):
    summary = {
        "n_folds": args.n_folds,
        "start_fold": args.start_fold,
        "end_fold": args.end_fold,
        "crossval_seed": args.crossval_seed,
        "image_scan_args": image_scan_args,
        "metrics": {},
        "missing_files": missing_files,
    }

    lines = [
        "Cross-validation statistics",
        f"Folds: {args.start_fold}..{args.end_fold} / {args.n_folds}",
        f"Seed: {args.crossval_seed}",
        f"Base imageScan args: {shlex.join(image_scan_args)}",
        "",
        "Values are reported as mean (std).",
        "",
    ]

    for file_name in STATS_FILES:
        file_metrics = metrics.get(file_name, {})
        if not file_metrics:
            continue
        lines.append(file_name)
        summary["metrics"][file_name] = {}
        for metric_name in sorted(file_metrics):
            values = [entry["value"] for entry in file_metrics[metric_name]]
            mean, std = _mean_std(values)
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

    (summary_dir / "crossval_stats.txt").write_text("\n".join(lines))
    (summary_dir / "crossval_stats.json").write_text(json.dumps(summary, indent=2))


def _write_command_log(summary_dir, command_records):
    lines = []
    for record in command_records:
        lines.append(f"fold {record['fold']} [{record['phase']}]")
        lines.append(shlex.join(record["command"]))
        lines.append("")
    (summary_dir / "crossval_commands.txt").write_text("\n".join(lines))


def _mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _summary_dir(script_dir, image_scan_args, base_suffix):
    dataset = _get_option(image_scan_args, "--dataset")
    test_mode = _has_flag(image_scan_args, "--test")
    scan_root = _scan_root(script_dir, dataset, test_mode)
    return scan_root / _crossval_root_name(image_scan_args, base_suffix)


def _crossval_root_name(image_scan_args, base_suffix):
    statistic = _get_option(image_scan_args, "--statistic")
    train_with_patches = _str_to_bool(_get_option(image_scan_args, "--train_with_patches"))
    statistic_folder = _statistic_folder(statistic, train_with_patches, base_suffix)
    return f"CrossVal_{statistic_folder}"


def _scan_root(script_dir, dataset, test_mode):
    if dataset == "testDataset":
        base_folder = script_dir / "Test"
    else:
        dataset_folder = DATASET_FOLDERS.get(dataset, dataset)
        base_folder = (script_dir.parent / "../../data" / dataset_folder).resolve()
    scan_folder = "Scan" if test_mode else "ScanFull"
    return base_folder / "evaluation" / scan_folder


def _statistic_folder(statistic, train_with_patches, folder_suffix):
    if statistic not in STATISTIC_FOLDERS:
        raise ValueError(f"Unknown statistic: {statistic}")
    patches_suffix = "_train_patches" if train_with_patches else ""
    return f"{STATISTIC_FOLDERS[statistic]}{patches_suffix}{folder_suffix}"


def _remove_options(args, options):
    result = []
    i = 0
    while i < len(args):
        token = args[i]
        option = _option_name(token)
        if option in options:
            if "=" not in token and option in VALUE_OPTIONS and i + 1 < len(args):
                i += 2
            else:
                i += 1
            continue
        result.append(token)
        i += 1
    return result


def _get_option(args, option, default=None):
    for i, token in enumerate(args):
        if token == option:
            if i + 1 >= len(args):
                return default
            return args[i + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return default


def _has_flag(args, option):
    return any(token == option for token in args)


def _has_any_flag(args, options):
    return any(_has_flag(args, option) for option in options)


def _option_name(token):
    if not token.startswith("--"):
        return None
    return token.split("=", 1)[0]


def _str_to_bool(value):
    return str(value).lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
