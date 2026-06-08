import copy
import math
import os
import csv
import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset, TensorDataset

from .config import *
from .data_loader import get_dataset_labels, worker_seed_init
from .models import M11, TextPeLEncoder, get_resnet_encoder, pool_encoder_output
from .utils import append_dict_csv, count_trainable_parameters, load_checkpoint, set_seed


RESULT_FIELDS = [
    "timestamp",
    "modality",
    "dataset",
    "method",
    "seed",
    "label_fraction",
    "train_size",
    "validation_size",
    "test_size",
    "best_validation_epoch",
    "best_validation_accuracy",
    "final_test_accuracy",
    "number_of_trainable_parameters",
    "optimiser",
    "encoder_learning_rate",
    "head_learning_rate",
    "early_stopping_patience",
    "split_strategy",
    "checkpoint_path",
]


DIAGNOSTIC_FIELDS = [
    "timestamp",
    "modality",
    "dataset",
    "method",
    "seed",
    "label_fraction",
    "gradient_variance",
    "feature_drift",
    "probe_source",
    "probe_size",
    "pretrained_representation_silhouette",
    "finetuned_representation_silhouette",
    "offline_probe_pretrained_test_accuracy",
    "offline_probe_finetuned_test_accuracy",
]


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_encoder(modality, num_classes):
    if modality == "vision":
        encoder, feature_dim = get_resnet_encoder(modality="vision")
    elif modality == "audio":
        encoder = M11(n_input=1, n_output=num_classes)
        feature_dim = AUDIO_FEAT_DIM
    elif modality == "text":
        encoder = TextPeLEncoder(pad_idx=1)
        feature_dim = TEXT_EMBED_DIM
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return encoder, feature_dim


def load_pretrained_encoder(encoder, modality):
    ckpt_name = PRETRAINED_CHECKPOINTS[modality]
    loaded = load_checkpoint(encoder, ckpt_name)
    if not loaded:
        raise FileNotFoundError(
            f"Required pretrained checkpoint is missing: {os.path.join(CHECKPOINT_DIR, ckpt_name)}"
        )


def _group_indices_by_label(labels):
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)
    return groups


def _sample_labelled_subset(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    n_total = len(labels)
    target_size = n_total if fraction >= 1.0 else max(1, int(round(n_total * fraction)))
    if target_size >= n_total:
        return np.arange(n_total, dtype=np.int64)

    groups = _group_indices_by_label(labels)
    selected = []
    remainders = []
    for label, indices in groups.items():
        exact = len(indices) * fraction
        take = int(math.floor(exact))
        remainders.append((exact - take, label))
        if take > 0:
            selected.extend(rng.choice(indices, size=take, replace=False).tolist())

    remaining = target_size - len(selected)
    selected_set = set(selected)
    if remaining > 0:
        for _, label in sorted(remainders, reverse=True):
            candidates = [idx for idx in groups[label] if idx not in selected_set]
            if not candidates:
                continue
            pick = int(rng.choice(candidates))
            selected.append(pick)
            selected_set.add(pick)
            remaining -= 1
            if remaining == 0:
                break

    if remaining > 0:
        candidates = [idx for idx in range(n_total) if idx not in selected_set]
        selected.extend(rng.choice(candidates, size=remaining, replace=False).tolist())

    rng.shuffle(selected)
    return np.array(selected, dtype=np.int64)


def _stratified_train_val_split(labels, labelled_indices, seed):
    rng = np.random.default_rng(seed + 17)
    labelled_labels = [int(labels[idx]) for idx in labelled_indices]
    groups = _group_indices_by_label(labelled_labels)

    def split_with_fraction(val_fraction):
        train, val = [], []
        omitted_from_val = False
        fewer_than_two_val = False
        for local_label, local_positions in groups.items():
            shuffled = np.array(local_positions, dtype=np.int64)
            rng.shuffle(shuffled)
            class_count = len(shuffled)
            if class_count == 1:
                train.append(int(labelled_indices[shuffled[0]]))
                omitted_from_val = True
                fewer_than_two_val = True
                continue

            val_count = max(1, int(round(class_count * val_fraction)))
            val_count = min(val_count, class_count - 1)
            if val_count < 2:
                fewer_than_two_val = True
            val.extend([int(labelled_indices[pos]) for pos in shuffled[:val_count]])
            train.extend([int(labelled_indices[pos]) for pos in shuffled[val_count:]])
        return train, val, omitted_from_val, fewer_than_two_val

    train, val, omitted, few = split_with_fraction(VALIDATION_FRACTION)
    strategy = "stratified_80_20"
    if omitted or few:
        train, val, omitted, few = split_with_fraction(FALLBACK_VALIDATION_FRACTION)
        strategy = "stratified_90_10_fallback"

    if len(val) == 0 and len(train) > 1:
        val.append(train.pop())
        strategy += "_single_val_repair"

    rng.shuffle(train)
    rng.shuffle(val)
    if omitted:
        strategy += "_class_val_infeasible"
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64), strategy


def make_protocol_splits(labels, fraction, seed, probe_size=PROBE_SET_SIZE):
    labels = [int(label) for label in labels]
    labelled = _sample_labelled_subset(labels, fraction, seed)
    train_idx, val_idx, strategy = _stratified_train_val_split(labels, labelled, seed)

    used = set(labelled.tolist())
    remaining = np.array([idx for idx in range(len(labels)) if idx not in used], dtype=np.int64)
    rng = np.random.default_rng(seed + 31)
    if len(remaining) > 0:
        rng.shuffle(remaining)
        probe_idx = remaining[: min(probe_size, len(remaining))]
        probe_source = "heldout_train_pool"
    else:
        probe_idx = val_idx[: min(probe_size, len(val_idx))]
        probe_source = "validation_fallback"

    return {
        "labelled": labelled,
        "train": train_idx,
        "val": val_idx,
        "probe": np.array(probe_idx, dtype=np.int64),
        "probe_source": probe_source,
        "strategy": strategy,
    }


def make_loader(dataset, indices=None, batch_size=DOWNSTREAM_BATCH_SIZE, shuffle=False, collate_fn=None, modality=None):
    ds = dataset if indices is None else Subset(dataset, [int(idx) for idx in indices])
    workers = 0 if modality == "audio" else NUM_WORKERS
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_fn,
        worker_init_fn=worker_seed_init if workers > 0 else None,
    )


def forward_logits(encoder, head, x):
    return head(pool_encoder_output(encoder(x)))


def evaluate_accuracy(encoder, head, loader):
    encoder.eval()
    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).long()
            with autocast_context():
                logits = forward_logits(encoder, head, x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def get_gradient_parameters(encoder, modality):
    if modality == "vision":
        for module in reversed(list(encoder.modules())):
            params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            if params:
                return params
    if modality == "audio":
        params = []
        for name, param in encoder.named_parameters():
            if name.startswith("conv4") or name.startswith("bn4"):
                if param.requires_grad:
                    params.append(param)
        return params
    if modality == "text":
        return [p for p in encoder.transformer.layers[-1].parameters() if p.requires_grad]
    return [p for p in encoder.parameters() if p.requires_grad]


def flatten_gradients(params):
    grads = []
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.detach().float().flatten().cpu())
    if not grads:
        return None
    return torch.cat(grads)


def normalised_gradient_variance(grad_vectors):
    if len(grad_vectors) < 2:
        return float("nan")
    grads = torch.stack(grad_vectors).float()
    mean_grad = grads.mean(dim=0)
    centered = grads - mean_grad
    trace = centered.pow(2).sum(dim=1).mean()
    denom = mean_grad.pow(2).sum().clamp_min(1e-12)
    return float((trace / denom).item())


def extract_features(encoder, dataset, indices, collate_fn, modality, batch_size=FEATURE_BATCH_SIZE):
    loader = make_loader(dataset, indices, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, modality=modality)
    encoder.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with autocast_context():
                features = pool_encoder_output(encoder(x))
            all_features.append(features.float().cpu())
            all_labels.append(y.long().cpu())
    if not all_features:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_features), torch.cat(all_labels)


def cosine_feature_drift(pretrained_features, finetuned_features):
    if pretrained_features.numel() == 0 or finetuned_features.numel() == 0:
        return float("nan")
    pretrained = nn.functional.normalize(pretrained_features.float(), dim=1)
    finetuned = nn.functional.normalize(finetuned_features.float(), dim=1)
    return float((1.0 - (pretrained * finetuned).sum(dim=1)).mean().item())


def representation_silhouette(features, labels, max_points=2000):
    if features.numel() == 0:
        return float("nan")
    labels_np = labels.numpy()
    unique = np.unique(labels_np)
    if len(unique) < 2:
        return float("nan")
    features_np = features.numpy()
    if len(labels_np) > max_points:
        rng = np.random.default_rng(123)
        keep = rng.choice(len(labels_np), size=max_points, replace=False)
        features_np = features_np[keep]
        labels_np = labels_np[keep]
        if len(np.unique(labels_np)) < 2:
            return float("nan")
    try:
        return float(silhouette_score(features_np, labels_np))
    except Exception:
        return float("nan")


def train_offline_probe(train_features, train_labels, val_features, val_labels, test_features, test_labels, num_classes):
    if train_features.numel() == 0 or val_features.numel() == 0 or test_features.numel() == 0:
        return float("nan")

    feature_dim = train_features.shape[1]
    probe = nn.Linear(feature_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=HEAD_LR["pel_frozen"], weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(probe.state_dict())
    best_val = -1.0
    epochs_no_improve = 0

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    for _ in range(OFFLINE_PROBE_EPOCHS):
        probe.train()
        for feats, labels in train_loader:
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE).long()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(feats), labels)
            loss.backward()
            optimizer.step()

        val_acc = _probe_accuracy(probe, val_features, val_labels)
        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            best_state = copy.deepcopy(probe.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= OFFLINE_PROBE_PATIENCE:
                break

    probe.load_state_dict(best_state)
    return _probe_accuracy(probe, test_features, test_labels)


def _probe_accuracy(probe, features, labels):
    probe.eval()
    with torch.no_grad():
        logits = probe(features.to(DEVICE))
        preds = logits.argmax(dim=1).cpu()
    return float((preds == labels.long()).float().mean().item())


def run_downstream_once(bundle, method, fraction, seed):
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    # Check if result already exists to allow resuming a cancelled matrix run
    if os.path.exists(RERUN_RESULTS_CSV):
        try:
            with open(RERUN_RESULTS_CSV, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get("modality") == bundle.modality and
                        row.get("dataset") == bundle.dataset and
                        row.get("method") == method and
                        math.isclose(float(row.get("label_fraction", 0)), fraction) and
                        int(row.get("seed", -1)) == seed):
                        # Cast numeric values back to ensure consistency with return expectations
                        row["label_fraction"] = float(row["label_fraction"])
                        row["seed"] = int(row["seed"])
                        row["best_validation_accuracy"] = float(row["best_validation_accuracy"])
                        row["final_test_accuracy"] = float(row["final_test_accuracy"])
                        return row, None
        except (ValueError, KeyError, IOError):
            pass

    set_seed(seed)
    labels = get_dataset_labels(bundle.train_pool)
    splits = make_protocol_splits(labels, fraction, seed)

    train_loader = make_loader(
        bundle.train_pool,
        splits["train"],
        batch_size=bundle.batch_size,
        shuffle=True,
        collate_fn=bundle.collate_fn,
        modality=bundle.modality,
    )
    val_loader = make_loader(
        bundle.train_pool,
        splits["val"],
        batch_size=bundle.batch_size,
        shuffle=False,
        collate_fn=bundle.collate_fn,
        modality=bundle.modality,
    )
    test_loader = make_loader(
        bundle.test,
        None,
        batch_size=bundle.batch_size,
        shuffle=False,
        collate_fn=bundle.collate_fn,
        modality=bundle.modality,
    )

    encoder, feature_dim = build_encoder(bundle.modality, bundle.num_classes)
    encoder = encoder.to(DEVICE)
    if method in ("finetune", "pel_frozen"):
        load_pretrained_encoder(encoder, bundle.modality)

    pretrained_encoder = None
    diagnostics_enabled = method == "finetune" and fraction in DIAGNOSTIC_FRACTIONS
    if diagnostics_enabled:
        pretrained_encoder = copy.deepcopy(encoder).to(DEVICE)
        pretrained_encoder.eval()

    if method == "pel_frozen":
        for param in encoder.parameters():
            param.requires_grad = False

    head = nn.Linear(feature_dim, bundle.num_classes).to(DEVICE)
    trainable_params = count_trainable_parameters(encoder) + count_trainable_parameters(head)
    params = [{"params": head.parameters(), "lr": HEAD_LR[method]}]
    if method != "pel_frozen":
        params.insert(0, {"params": encoder.parameters(), "lr": ENCODER_LR[method]})
    optimizer = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    gradient_params = get_gradient_parameters(encoder, bundle.modality) if diagnostics_enabled else []
    grad_vectors = []
    total_steps = 0
    best_val_acc = -1.0
    best_epoch = 0
    best_encoder_state = copy.deepcopy(encoder.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())
    epochs_no_improve = 0

    # Calculate accumulation steps to reach effective batch size
    effective_bs = MODALITY_EFFECTIVE_BATCH_SIZE.get(bundle.modality, bundle.batch_size)
    accumulation_steps = max(1, effective_bs // bundle.batch_size)

    for epoch in range(1, MAX_EPOCHS[method] + 1):
        encoder.train(method != "pel_frozen")
        head.train()
        optimizer.zero_grad(set_to_none=True)
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).long()
            with autocast_context():
                logits = forward_logits(encoder, head, x)
                loss = criterion(logits, y) / accumulation_steps
            
            loss.backward()

            if diagnostics_enabled and total_steps < GRADIENT_VARIANCE_STEPS:
                grad_vector = flatten_gradients(gradient_params)
                if grad_vector is not None:
                    grad_vectors.append(grad_vector)

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_steps += 1

        val_acc = evaluate_accuracy(encoder, head, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE[method]:
                break

    encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)
    final_test_acc = evaluate_accuracy(encoder, head, test_loader)

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, "rerun")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_name = (
        f"{bundle.modality}_{bundle.dataset}_{method}_"
        f"{int(round(fraction * 100))}pct_seed{seed}.pth"
    )
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "best_validation_accuracy": best_val_acc,
            "final_test_accuracy": final_test_acc,
            "best_validation_epoch": best_epoch,
            "split_indices": {key: value.tolist() for key, value in splits.items() if hasattr(value, "tolist")},
            "split_strategy": splits["strategy"],
        },
        ckpt_path,
    )

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modality": bundle.modality,
        "dataset": bundle.dataset,
        "method": method,
        "seed": seed,
        "label_fraction": fraction,
        "train_size": len(splits["train"]),
        "validation_size": len(splits["val"]),
        "test_size": len(bundle.test),
        "best_validation_epoch": best_epoch,
        "best_validation_accuracy": best_val_acc,
        "final_test_accuracy": final_test_acc,
        "number_of_trainable_parameters": trainable_params,
        "optimiser": "AdamW",
        "encoder_learning_rate": ENCODER_LR[method],
        "head_learning_rate": HEAD_LR[method],
        "early_stopping_patience": EARLY_STOP_PATIENCE[method],
        "split_strategy": splits["strategy"],
        "checkpoint_path": ckpt_path,
    }
    append_dict_csv(RERUN_RESULTS_CSV, row, RESULT_FIELDS)

    diagnostic_row = None
    if diagnostics_enabled:
        diagnostic_row = run_finetune_diagnostics(
            bundle=bundle,
            pretrained_encoder=pretrained_encoder,
            finetuned_encoder=encoder,
            splits=splits,
            grad_vectors=grad_vectors,
            fraction=fraction,
            seed=seed,
        )

    return row, diagnostic_row


def run_finetune_diagnostics(bundle, pretrained_encoder, finetuned_encoder, splits, grad_vectors, fraction, seed):
    probe_indices = splits["probe"]
    pre_probe, _ = extract_features(
        pretrained_encoder, bundle.train_pool, probe_indices, bundle.collate_fn, bundle.modality
    )
    ft_probe, _ = extract_features(
        finetuned_encoder, bundle.train_pool, probe_indices, bundle.collate_fn, bundle.modality
    )

    train_pre, train_labels = extract_features(
        pretrained_encoder, bundle.train_pool, splits["train"], bundle.collate_fn, bundle.modality
    )
    val_pre, val_labels = extract_features(
        pretrained_encoder, bundle.train_pool, splits["val"], bundle.collate_fn, bundle.modality
    )
    test_pre, test_labels = extract_features(
        pretrained_encoder, bundle.test, None, bundle.collate_fn, bundle.modality
    )

    train_ft, _ = extract_features(
        finetuned_encoder, bundle.train_pool, splits["train"], bundle.collate_fn, bundle.modality
    )
    val_ft, _ = extract_features(
        finetuned_encoder, bundle.train_pool, splits["val"], bundle.collate_fn, bundle.modality
    )
    test_ft, _ = extract_features(
        finetuned_encoder, bundle.test, None, bundle.collate_fn, bundle.modality
    )

    diagnostic_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modality": bundle.modality,
        "dataset": bundle.dataset,
        "method": "finetune",
        "seed": seed,
        "label_fraction": fraction,
        "gradient_variance": normalised_gradient_variance(grad_vectors),
        "feature_drift": cosine_feature_drift(pre_probe, ft_probe),
        "probe_source": splits["probe_source"],
        "probe_size": len(probe_indices),
        "pretrained_representation_silhouette": representation_silhouette(test_pre, test_labels),
        "finetuned_representation_silhouette": representation_silhouette(test_ft, test_labels),
        "offline_probe_pretrained_test_accuracy": train_offline_probe(
            train_pre, train_labels, val_pre, val_labels, test_pre, test_labels, bundle.num_classes
        ),
        "offline_probe_finetuned_test_accuracy": train_offline_probe(
            train_ft, train_labels, val_ft, val_labels, test_ft, test_labels, bundle.num_classes
        ),
    }
    append_dict_csv(RERUN_DIAGNOSTICS_CSV, diagnostic_row, DIAGNOSTIC_FIELDS)
    return diagnostic_row
