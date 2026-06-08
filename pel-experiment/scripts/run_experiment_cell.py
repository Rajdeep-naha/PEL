#!/usr/bin/env python3
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pel.config import DATA_FRACTIONS, METHODS, MODALITY_BATCH_SIZE, RERUN_DATASETS, SEEDS
from pel.data_loader import get_dataset_bundle
from pel.protocol import run_downstream_once


def parse_args():
    parser = argparse.ArgumentParser(description="Run one validation-safe PEL rerun cell.")
    parser.add_argument("--modality", required=True, choices=sorted(RERUN_DATASETS.keys()))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--fraction", type=float, required=True, choices=DATA_FRACTIONS)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset or RERUN_DATASETS[args.modality][0]
    batch_size = args.batch_size or MODALITY_BATCH_SIZE.get(args.modality, 64)
    bundle = get_dataset_bundle(args.modality, dataset, batch_size=batch_size)
    row, diagnostic_row = run_downstream_once(bundle, args.method, args.fraction, args.seed)
    print(
        "DONE "
        f"{row['modality']}/{row['dataset']} {row['method']} "
        f"{int(row['label_fraction'] * 100)}% seed={row['seed']} "
        f"val={row['best_validation_accuracy']:.4f} test={row['final_test_accuracy']:.4f}"
    )
    if diagnostic_row is not None:
        print(
            "DIAGNOSTICS "
            f"GV={diagnostic_row['gradient_variance']:.6g} "
            f"FD={diagnostic_row['feature_drift']:.6g} "
            f"probe_pre={diagnostic_row['offline_probe_pretrained_test_accuracy']:.4f} "
            f"probe_ft={diagnostic_row['offline_probe_finetuned_test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
