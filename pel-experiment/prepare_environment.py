import importlib.util
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pel.config import CHECKPOINT_DIR, DATA_DIR, PLOTS_DIR, RERUN_DATASETS, RESULTS_DIR
from pel.data_loader import get_dataset_bundle, prepare_dataset


DIRS_TO_CREATE = [
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    os.path.join(DATA_DIR, "audio"),
    os.path.join(DATA_DIR, "text"),
    os.path.join(DATA_DIR, "vision"),
]


def setup_directories():
    for directory in DIRS_TO_CREATE:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure initialized.")


def check_hardware():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Use the server run for the full matrix.")


def ensure_setup_dependencies():
    if importlib.util.find_spec("datasets") is not None:
        return
    print("Installing optional setup dependency: datasets")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])


def validate_imagenet100_files():
    bad_files = []
    for split in ("train", "val"):
        split_dir = os.path.join(DATA_DIR, "vision", split)
        for root, _, filenames in os.walk(split_dir):
            for filename in filenames:
                path = os.path.join(root, filename)
                try:
                    if os.path.getsize(path) == 0:
                        bad_files.append(path)
                except OSError:
                    bad_files.append(path)
                if len(bad_files) >= 20:
                    sample = ", ".join(bad_files[:5])
                    raise FileNotFoundError(
                        f"ImageNet-100 contains zero-byte or unreadable files. "
                        f"Found at least {len(bad_files)}; sample: {sample}"
                    )
    if bad_files:
        sample = ", ".join(bad_files[:5])
        raise FileNotFoundError(
            f"ImageNet-100 contains {len(bad_files)} zero-byte or unreadable files. Sample: {sample}"
        )


def verify_or_download_datasets():
    print("\n--- Dataset Verification ---")
    all_ok = True
    for modality, datasets in RERUN_DATASETS.items():
        for dataset in datasets:
            print(f"Checking {modality}/{dataset}...")
            try:
                # Attempt download/repair before validating availability.
                prepare_dataset(modality, dataset)
                if modality == "vision" and dataset == "imagenet100":
                    validate_imagenet100_files()
                bundle = get_dataset_bundle(modality, dataset)
                print(
                    f"  ✅ Found: train={len(bundle.train_pool)}, "
                    f"test={len(bundle.test)}, classes={bundle.num_classes}"
                )
            except Exception as exc:
                print(f"  ❌ Missing or failed: {exc}")
                if dataset == "imagenet100":
                    print("     Note: ImageNet-100 must be uploaded manually to data/vision/train and data/vision/val")
                elif dataset == "speechcommands":
                    print("     Note: The SpeechCommands folder exists, but the split size is far below a usable dataset.")
                elif dataset == "fsdkaggle2018":
                    print("     Note: Expected Zenodo audio_train/meta contents for FSDKaggle2018 were not found or are incomplete.")
                elif dataset == "dbpedia":
                    print("     Note: DBpedia will be fetched from Hugging Face if `datasets` is installed; otherwise install it first.")
                all_ok = False
    return all_ok


def check_pretrained_checkpoints():
    print("\n--- Pretrained Checkpoints Check ---")
    all_ok = True
    from pel.config import PRETRAINED_CHECKPOINTS
    for modality, filename in PRETRAINED_CHECKPOINTS.items():
        path = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(path):
            print(f"  ✅ {modality}: Found {filename}")
        else:
            print(f"  ⚠️  {modality}: Missing {filename} (Required for FineTune/PeLFrozen)")
            all_ok = False
    return all_ok


def main():
    setup_directories()
    ensure_setup_dependencies()
    check_hardware()
    ds_ok = verify_or_download_datasets()
    ckpt_ok = check_pretrained_checkpoints()

    print("\n====================================================")
    if ds_ok and ckpt_ok:
        print("  🎉 Setup complete! Ready for run_experiment_matrix.py")
    else:
        print("  ⚠️  Setup finished with warnings (see above).")
        print("  Please address missing data/checkpoints before running.")
    print("====================================================")


if __name__ == "__main__":
    main()
