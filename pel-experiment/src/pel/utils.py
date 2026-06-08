import csv
import os
import random
import time

import numpy as np
import torch

from .config import CHECKPOINT_DIR, RESULTS_CSV


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def append_dict_csv(path, row, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def log_experiment(track, modality, stage, metric, value, notes=""):
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Track", "Modality", "Stage", "Metric", "Value", "Notes"])
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                track,
                modality,
                stage,
                metric,
                value,
                notes,
            ]
        )
    try:
        printable = f"{float(value):.4f}"
    except Exception:
        printable = str(value)
    print(f"Logged: {modality} | {stage} -> {metric}: {printable}")


def save_checkpoint(model, filename, epoch=None, optimizer=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dict = raw_model.state_dict()
    if epoch is not None and optimizer is not None:
        payload = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }
    else:
        payload = state_dict
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")


def _clean_state_dict_for_encoder(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("encoder."):
            key = key[len("encoder.") :]
        if key.startswith("module.encoder."):
            key = key[len("module.encoder.") :]
        if key.startswith("net."):
            continue
        cleaned[key] = value
    return cleaned


def load_checkpoint(model, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return False

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = _clean_state_dict_for_encoder(state_dict)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    real_missing = [key for key in missing if "fc" not in key]
    if real_missing:
        print(f"Partial checkpoint load, missing keys: {real_missing[:5]}")
    if unexpected:
        print(f"Partial checkpoint load, unexpected keys: {unexpected[:5]}")
    print(f"Loaded weights from: {path}")
    return True


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())
