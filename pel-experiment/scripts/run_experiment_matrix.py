#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pel.config import DATA_FRACTIONS, METHODS, RERUN_DATASETS, SEEDS, CHECKPOINT_DIR


def parse_csv_arg(value, valid_values=None, cast=str):
    if value is None:
        return None
    items = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if valid_values is not None:
        invalid = [item for item in items if item not in valid_values]
        if invalid:
            raise ValueError(f"Invalid values {invalid}; valid values are {valid_values}")
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="Run the downstream rerun matrix across the configured datasets, fractions, methods, and seeds.")
    parser.add_argument("--modalities", default=None, help="Comma-separated subset, e.g. audio,text")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset subset")
    parser.add_argument("--methods", default=None, help="Comma-separated subset: scratch,finetune,pel_frozen")
    parser.add_argument("--fractions", default=None, help="Comma-separated subset: 0.01,0.1,1.0")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = parse_csv_arg(args.modalities, list(RERUN_DATASETS.keys())) or list(RERUN_DATASETS.keys())
    methods = parse_csv_arg(args.methods, METHODS) or METHODS
    fractions = parse_csv_arg(args.fractions, DATA_FRACTIONS, float) or DATA_FRACTIONS
    seeds = parse_csv_arg(args.seeds, None, int) or SEEDS
    dataset_filter = set(parse_csv_arg(args.datasets) or [])

    py = sys.executable
    script = os.path.join("scripts", "run_experiment_cell.py")

    for seed in seeds:
        for modality in modalities:
            for dataset in RERUN_DATASETS[modality]:
                if dataset_filter and dataset not in dataset_filter:
                    continue
                for fraction in fractions:
                    # The plan prioritises the pretrained comparison. Keep this order.
                    for method in methods:
                        ckpt_name = f"{modality}_{dataset}_{method}_{int(round(fraction * 100))}pct_seed{seed}.pth"
                        ckpt_path = os.path.join(CHECKPOINT_DIR, "rerun", ckpt_name)
                        if os.path.exists(ckpt_path):
                            print(f"\n>> Skipping already completed run: {ckpt_name}", flush=True)
                            continue

                        cmd = [
                            py,
                            script,
                            "--modality",
                            modality,
                            "--dataset",
                            dataset,
                            "--method",
                            method,
                            "--fraction",
                            str(fraction),
                            "--seed",
                            str(seed),
                        ]
                        if args.batch_size is not None:
                            cmd.extend(["--batch-size", str(args.batch_size)])
                        print("\n>>", " ".join(cmd), flush=True)
                        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
