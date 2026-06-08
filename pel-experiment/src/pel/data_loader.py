import csv
import os
import random
import requests
import shutil
import json
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import *


def _download_enabled():
    return os.environ.get("PEL_DOWNLOAD_DATA", "1") != "0"


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _download_file(url, path):
    if os.path.exists(path):
        return
    if not _download_enabled():
        raise FileNotFoundError(
            f"{path} is missing and PEL_DOWNLOAD_DATA=0 disables downloads."
        )
    _ensure_parent(path)
    
    # Use requests with a User-Agent to handle redirects and auth
    print(f"Downloading {url} to {path}...")
    headers = {"User-Agent": "curl/8.5.0", "Accept": "*/*"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Automatically extract if it's a tarball
    if path.endswith(".tar.gz"):
        print(f"Extracting {path}...")
        import tarfile
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(path))


def patch_torchaudio_soundfile():
    """Use soundfile for local Windows compatibility when torchaudio backends fail."""
    try:
        import soundfile as sf
    except ImportError:
        return

    def soundfile_load(filepath, **kwargs):
        data, sample_rate = sf.read(filepath)
        tensor = torch.as_tensor(data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.t()
        return tensor, sample_rate

    torchaudio.load = soundfile_load


patch_torchaudio_soundfile()


@dataclass
class DatasetBundle:
    modality: str
    dataset: str
    train_pool: Dataset
    test: Dataset
    num_classes: int
    feature_dim: int
    collate_fn: object = None
    batch_size: int = DOWNSTREAM_BATCH_SIZE


def get_dataset_labels(dataset):
    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    raise ValueError(f"Dataset {type(dataset).__name__} does not expose labels.")


# Vision
def _vision_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(VISION_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _vision_pretrain_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )


def get_vision_dataset_bundle(dataset="imagenet100", batch_size=None):
    bs = batch_size or MODALITY_BATCH_SIZE["vision"]
    transform = _vision_eval_transform()

    if dataset == "imagenet100":
        train_pool = datasets.ImageFolder(VISION_TRAIN_DIR, transform=transform)
        test = datasets.ImageFolder(VISION_VAL_DIR, transform=transform)
        num_classes = VISION_DATASET_NUM_CLASSES[dataset]
    elif dataset == "cifar100":
        root = os.path.join(DATA_DIR, "vision", "cifar100")
        train_pool = datasets.CIFAR100(root=root, train=True, transform=transform, download=_download_enabled())
        test = datasets.CIFAR100(root=root, train=False, transform=transform, download=_download_enabled())
        num_classes = VISION_DATASET_NUM_CLASSES[dataset]
    else:
        raise ValueError(f"Unsupported vision dataset: {dataset}")

    return DatasetBundle("vision", dataset, train_pool, test, num_classes, VISION_FEAT_DIM, None, bs)


def get_vision_loaders(stage="eval", batch_size=None, dataset="imagenet100"):
    bs = batch_size or VISION_BATCH_SIZE
    if stage == "pretrain":
        train_ds = datasets.ImageFolder(VISION_TRAIN_DIR, transform=_vision_pretrain_transform())
        test_ds = datasets.ImageFolder(VISION_VAL_DIR, transform=_vision_eval_transform())
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, drop_last=True),
            DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS),
        )

    bundle = get_vision_dataset_bundle(dataset=dataset, batch_size=batch_size)
    return bundle.train_pool, DataLoader(
        bundle.test,
        batch_size=bundle.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=bundle.collate_fn,
    )


# Audio
SPEECHCOMMANDS_LABELS = sorted(
    [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero",
    ]
)


def _standardize_waveform(waveform, sample_rate, target_rate=AUDIO_SAMPLE_RATE, target_len=None):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_rate:
        waveform = AF.resample(waveform, sample_rate, target_rate)
    if target_len is not None:
        if waveform.shape[1] < target_len:
            waveform = nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
    return waveform


class SpeechCommandsDataset(Dataset):
    MIN_SAMPLES = {
        "training": 1000,
        "testing": 1000,
        "validation": 500,
    }

    def __init__(self, subset="training"):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=AUDIO_ROOT,
            subset=subset,
            download=_download_enabled(),
        )
        self.label_to_idx = {label: i for i, label in enumerate(SPEECHCOMMANDS_LABELS)}
        if hasattr(self.dataset, "_walker"):
            label_names = [os.path.basename(os.path.dirname(path)) for path in self.dataset._walker]
            self.labels = [self.label_to_idx[label] for label in label_names]
        else:
            self.labels = [self.label_to_idx[self.dataset[idx][2]] for idx in range(len(self.dataset))]
        self.target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
        min_samples = self.MIN_SAMPLES.get(subset)
        if min_samples is not None and len(self.dataset) < min_samples:
            raise FileNotFoundError(
                f"SpeechCommands {subset} split is incomplete at {self.dataset._path}. "
                f"Found only {len(self.dataset)} samples; expected at least {min_samples}."
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, _, _ = self.dataset[idx]
        waveform = _standardize_waveform(waveform, sample_rate, target_len=self.target_len)
        return waveform, self.label_to_idx[label]


def _repair_speechcommands_if_partial():
    dataset_root = os.path.join(AUDIO_ROOT, "SpeechCommands", "speech_commands_v0.02")
    if not os.path.isdir(dataset_root):
        return
    wav_count = 0
    for _, _, filenames in os.walk(dataset_root):
        wav_count += sum(name.endswith(".wav") for name in filenames)
        if wav_count >= SpeechCommandsDataset.MIN_SAMPLES["training"]:
            return
    _remove_path(os.path.join(AUDIO_ROOT, "SpeechCommands"))
    _remove_path(os.path.join(AUDIO_ROOT, "speech_commands_v0.02.tar.gz"))


class ESC50Dataset(Dataset):
    URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

    def __init__(self, split="train"):
        self.root = os.path.join(AUDIO_ROOT, "esc50")
        self.extract_dir = os.path.join(self.root, "ESC-50-master")
        self.archive_path = os.path.join(self.root, "esc50-master.zip")
        self._ensure_data()

        meta_path = os.path.join(self.extract_dir, "meta", "esc50.csv")
        rows = []
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold = int(row["fold"])
                is_test = fold == 5
                if (split == "test" and is_test) or (split == "train" and not is_test):
                    rows.append(row)

        self.items = rows
        self.labels = [int(row["target"]) for row in rows]

    def _ensure_data(self):
        meta_path = os.path.join(self.extract_dir, "meta", "esc50.csv")
        audio_dir = os.path.join(self.extract_dir, "audio")
        if os.path.exists(meta_path) and os.path.isdir(audio_dir) and os.listdir(audio_dir):
            return
        if os.path.exists(self.extract_dir):
            raise FileNotFoundError(
                "ESC-50 is partially present at "
                f"{self.extract_dir}. Expected meta/esc50.csv and audio/*.wav."
            )
        os.makedirs(self.root, exist_ok=True)
        _download_file(self.URL, self.archive_path)
        with zipfile.ZipFile(self.archive_path, "r") as zf:
            zf.extractall(self.root)
        if not os.path.exists(meta_path) or not os.path.isdir(audio_dir) or not os.listdir(audio_dir):
            raise FileNotFoundError(
                "ESC-50 download/extract completed but the dataset is incomplete. "
                f"Expected files under {self.extract_dir}/meta and {self.extract_dir}/audio."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        path = os.path.join(self.extract_dir, "audio", row["filename"])
        waveform, sample_rate = torchaudio.load(path)
        waveform = _standardize_waveform(waveform, sample_rate, target_len=None)
        return waveform, int(row["target"])


def _repair_esc50_if_partial():
    root = os.path.join(AUDIO_ROOT, "esc50")
    extract_dir = os.path.join(root, "ESC-50-master")
    meta_path = os.path.join(extract_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(extract_dir, "audio")
    if os.path.exists(meta_path) and os.path.isdir(audio_dir) and os.listdir(audio_dir):
        return
    if os.path.exists(extract_dir):
        _remove_path(extract_dir)
    archive_path = os.path.join(root, "esc50-master.zip")
    if os.path.exists(archive_path) and not zipfile.is_zipfile(archive_path):
        _remove_path(archive_path)


class FSDKaggle2018Dataset(Dataset):
    AUDIO_ARCHIVE = "FSDKaggle2018.audio_train.zip"
    TEST_AUDIO_ARCHIVE = "FSDKaggle2018.audio_test.zip"
    META_ARCHIVE = "FSDKaggle2018.meta.zip"
    AUDIO_URL = "https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip?download=1"
    TEST_AUDIO_URL = "https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_test.zip?download=1"
    META_URL = "https://zenodo.org/records/2552860/files/FSDKaggle2018.meta.zip?download=1"
    TRAIN_CSV_CANDIDATES = [
        "train_post_competition.csv",
        "train.csv",
    ]
    TEST_CSV_CANDIDATES = [
        "test_post_competition_scoring_clips.csv",
        "test.csv",
    ]

    def __init__(self, split="train"):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported FSDKaggle2018 split: {split}")

        self.root = os.path.join(AUDIO_ROOT, "fsdkaggle2018")
        self.train_audio_root = self._resolve_dir("FSDKaggle2018.audio_train", "audio_train")
        self.test_audio_root = self._resolve_dir("FSDKaggle2018.audio_test", "audio_test")
        self.meta_root = self._resolve_dir("FSDKaggle2018.meta", "meta")
        self._ensure_data()

        train_csv = self._find_first_existing(self.TRAIN_CSV_CANDIDATES)
        test_csv = self._find_first_existing(self.TEST_CSV_CANDIDATES)
        label_to_idx = self._build_label_index(train_csv)
        csv_path = train_csv if split == "train" else test_csv
        audio_root = self.train_audio_root if split == "train" else self.test_audio_root

        self.items = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("fname") or row.get("filename")
                label = row.get("label")
                if not filename or not label:
                    continue
                audio_path = os.path.join(audio_root, filename)
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Missing FSDKaggle2018 audio file: {audio_path}")
                self.items.append((audio_path, label_to_idx[label]))

        self.labels = [label for _, label in self.items]

    def _find_first_existing(self, candidates):
        for name in candidates:
            path = os.path.join(self.meta_root, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Missing FSDKaggle2018 metadata under {self.meta_root}. Checked: {candidates}"
        )

    def _build_label_index(self, train_csv):
        labels = []
        with open(train_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label")
                if label:
                    labels.append(label)
        classes = sorted(set(labels))
        if len(classes) != AUDIO_DATASET_NUM_CLASSES["fsdkaggle2018"]:
            raise ValueError(
                "Unexpected FSDKaggle2018 class count. "
                f"Expected {AUDIO_DATASET_NUM_CLASSES['fsdkaggle2018']}, found {len(classes)}."
            )
        return {label: idx for idx, label in enumerate(classes)}

    def _resolve_dir(self, primary, fallback):
        primary_path = os.path.join(self.root, primary)
        fallback_path = os.path.join(self.root, fallback)
        if os.path.isdir(primary_path):
            return primary_path
        if os.path.isdir(fallback_path):
            return fallback_path
        return primary_path

    def _ensure_data(self):
        if (
            os.path.isdir(self.train_audio_root)
            and os.path.isdir(self.test_audio_root)
            and os.path.isdir(self.meta_root)
        ):
            return

        os.makedirs(self.root, exist_ok=True)
        audio_archive = os.path.join(self.root, self.AUDIO_ARCHIVE)
        test_audio_archive = os.path.join(self.root, self.TEST_AUDIO_ARCHIVE)
        meta_archive = os.path.join(self.root, self.META_ARCHIVE)
        _download_file(self.AUDIO_URL, audio_archive)
        _download_file(self.TEST_AUDIO_URL, test_audio_archive)
        _download_file(self.META_URL, meta_archive)

        for archive_path in (audio_archive, test_audio_archive, meta_archive):
            if not zipfile.is_zipfile(archive_path):
                raise FileNotFoundError(f"Invalid FSDKaggle2018 archive: {archive_path}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(self.root)

        self.train_audio_root = self._resolve_dir("FSDKaggle2018.audio_train", "audio_train")
        self.test_audio_root = self._resolve_dir("FSDKaggle2018.audio_test", "audio_test")
        self.meta_root = self._resolve_dir("FSDKaggle2018.meta", "meta")
        if (
            not os.path.isdir(self.train_audio_root)
            or not os.path.isdir(self.test_audio_root)
            or not os.path.isdir(self.meta_root)
        ):
            raise FileNotFoundError(
                "FSDKaggle2018 download/extract completed but expected audio_train/, audio_test/, and meta/ "
                f"under {self.root} were not found."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform, sample_rate = torchaudio.load(path)
        waveform = _standardize_waveform(waveform, sample_rate, target_len=None)
        return waveform, label


def _repair_fsdkaggle2018_if_partial():
    root = os.path.join(AUDIO_ROOT, "fsdkaggle2018")
    train_audio_roots = [os.path.join(root, "FSDKaggle2018.audio_train"), os.path.join(root, "audio_train")]
    test_audio_roots = [os.path.join(root, "FSDKaggle2018.audio_test"), os.path.join(root, "audio_test")]
    meta_roots = [os.path.join(root, "FSDKaggle2018.meta"), os.path.join(root, "meta")]
    if any(os.path.isdir(path) for path in train_audio_roots) and any(os.path.isdir(path) for path in test_audio_roots) and any(os.path.isdir(path) for path in meta_roots):
        return
    for path in [*train_audio_roots, *test_audio_roots, *meta_roots]:
        if os.path.exists(path):
            _remove_path(path)
    for archive_name in (
        FSDKaggle2018Dataset.AUDIO_ARCHIVE,
        FSDKaggle2018Dataset.TEST_AUDIO_ARCHIVE,
        FSDKaggle2018Dataset.META_ARCHIVE,
    ):
        archive_path = os.path.join(root, archive_name)
        if os.path.exists(archive_path) and not zipfile.is_zipfile(archive_path):
            _remove_path(archive_path)


def audio_collate(batch):
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[-1] for w in waveforms)
    padded = []
    for waveform in waveforms:
        if waveform.shape[-1] < max_len:
            waveform = nn.functional.pad(waveform, (0, max_len - waveform.shape[-1]))
        padded.append(waveform)
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


def get_audio_dataset_bundle(dataset="speechcommands", batch_size=None):
    bs = batch_size or MODALITY_BATCH_SIZE["audio"]
    if dataset == "speechcommands":
        train_pool = SpeechCommandsDataset("training")
        test = SpeechCommandsDataset("testing")
    elif dataset == "fsdkaggle2018":
        train_pool = FSDKaggle2018Dataset("train")
        test = FSDKaggle2018Dataset("test")
    else:
        raise ValueError(f"Unsupported audio dataset: {dataset}")

    return DatasetBundle(
        "audio",
        dataset,
        train_pool,
        test,
        AUDIO_DATASET_NUM_CLASSES[dataset],
        AUDIO_FEAT_DIM,
        audio_collate,
        bs,
    )


def get_audio_loaders(stage="eval", batch_size=None, dataset="speechcommands"):
    bs = batch_size or AUDIO_BATCH_SIZE
    if stage == "pretrain":
        train_ds = SpeechCommandsDataset("training")
        test_ds = SpeechCommandsDataset("testing")
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=audio_collate, num_workers=NUM_WORKERS),
            DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=audio_collate, num_workers=NUM_WORKERS),
        )

    bundle = get_audio_dataset_bundle(dataset=dataset, batch_size=batch_size)
    return bundle.train_pool, DataLoader(
        bundle.test,
        batch_size=bundle.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=bundle.collate_fn,
    )


# Text
class SimpleVocab:
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.pad_idx = token_to_idx.get("<pad>", 1)
        self.mask_idx = token_to_idx.get("<mask>", 3)
        self.unk_idx = token_to_idx.get("<unk>", 0)

    def __len__(self):
        return len(self.token_to_idx)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk_idx)


TEXT_PRETRAIN_VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "pel_text_vocab.json")
TEXT_PRETRAIN_DATASET = "ag_news"


def _tokenize(text):
    return text.lower().split()[:TEXT_MAX_LEN]


def _encode_text(text, vocab):
    tokens = _tokenize(text)
    ids = torch.full((TEXT_MAX_LEN,), vocab.pad_idx, dtype=torch.long)
    encoded = [vocab[token] for token in tokens]
    if encoded:
        ids[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    return ids


class SimCLRTextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = texts
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def hard_augment(self, tokens):
        x = tokens.clone()
        non_pad = (x != self.vocab.pad_idx).nonzero(as_tuple=True)[0]
        n = len(non_pad)
        if n < 2:
            return x
        mask_count = int(n * 0.50)
        if mask_count > 0:
            x[non_pad[torch.randperm(n)[:mask_count]]] = self.vocab.mask_idx
        if torch.rand(1) > 0.6 and n > 5:
            delete_idx = int(torch.randint(0, n, (1,)).item())
            kept = torch.cat([x[non_pad[:delete_idx]], x[non_pad[delete_idx + 1 :]]])
            x = torch.full_like(tokens, self.vocab.pad_idx)
            x[: len(kept)] = kept
        return x

    def __getitem__(self, idx):
        ids = _encode_text(self.texts[idx], self.vocab)
        return self.hard_augment(ids), self.hard_augment(ids)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = [int(label) for label in labels]
        self.vocab = vocab
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _encode_text(self.texts[idx], self.vocab), torch.tensor(self.labels[idx], dtype=torch.long)


TEXT_DATASET_SPECS = {
    "ag_news": {
        "dir": "ag_news_csv",
        "archive": "ag_news_csv.tar.gz",
        "urls": [
            "https://dl.fbaipublicfiles.com/fasttext/datasets/ag_news_csv.tar.gz",
        ],
        "local_dirs": [
            "AG_NEWS",
            os.path.join("datasets", "AG_NEWS"),
            "ag_news_csv",
        ],
    },
    "dbpedia": {
        "dir": "dbpedia_csv",
        "archive": "dbpedia_csv.tar.gz",
        "urls": [
            "https://dl.fbaipublicfiles.com/fasttext/datasets/dbpedia_csv.tar.gz",
        ],
        "local_dirs": [
            "DBPEDIA",
            os.path.join("datasets", "DBPEDIA"),
            "dbpedia_csv",
        ],
    },
}


def _export_hf_text_dataset(dataset, output_dir):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Downloading text datasets requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc

    hf_name = {
        "ag_news": "ag_news",
        "dbpedia": "dbpedia_14",
    }[dataset]
    ds = load_dataset(hf_name)
    os.makedirs(output_dir, exist_ok=True)
    split_specs = {"train": os.path.join(output_dir, "train.csv"), "test": os.path.join(output_dir, "test.csv")}
    for split_name, output_path in split_specs.items():
        split = ds[split_name]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in split:
                label = int(row["label"]) + 1
                if dataset == "ag_news":
                    writer.writerow([label, row.get("title", ""), row.get("description", "")])
                elif dataset == "dbpedia":
                    writer.writerow([label, row.get("title", ""), row.get("content", "")])


def _candidate_text_csv_pairs(dataset):
    spec = TEXT_DATASET_SPECS[dataset]
    candidate_roots = [os.path.join(TEXT_ROOT, spec["dir"])]
    candidate_roots.extend(os.path.join(TEXT_ROOT, rel_dir) for rel_dir in spec.get("local_dirs", []))
    seen = set()
    for root in candidate_roots:
        if root in seen:
            continue
        seen.add(root)
        yield os.path.join(root, "train.csv"), os.path.join(root, "test.csv")


def _ensure_text_csv(dataset):
    spec = TEXT_DATASET_SPECS[dataset]
    for train_csv, test_csv in _candidate_text_csv_pairs(dataset):
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            return train_csv, test_csv

    if not _download_enabled():
        raise FileNotFoundError(
            f"{dataset} is missing. Expected train/test CSVs under {os.path.join(TEXT_ROOT, spec['dir'])} "
            f"or one of {spec.get('local_dirs', [])}, and PEL_DOWNLOAD_DATA=0 disables downloads."
        )

    archive_path = os.path.join(TEXT_ROOT, spec["archive"])
    last_error = None
    for url in spec["urls"]:
        try:
            _download_file(url, archive_path)
            break
        except Exception as exc:
            last_error = exc

    if not os.path.exists(archive_path):
        try:
            _export_hf_text_dataset(dataset, os.path.join(TEXT_ROOT, spec["dir"]))
        except Exception as exc:
            if last_error is None:
                last_error = exc

    for train_csv, test_csv in _candidate_text_csv_pairs(dataset):
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            return train_csv, test_csv

    raise FileNotFoundError(
        f"{dataset} is unavailable locally after checking known paths and download URLs. "
        f"Last download error: {last_error}"
    )


def _read_text_csv(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            labels.append(int(row[0]) - 1)
            texts.append(" ".join(row[1:]))
    return texts, labels


def _build_vocab(texts):
    specials = ["<unk>", "<pad>", "<cls>", "<mask>"]
    token_to_idx = {token: idx for idx, token in enumerate(specials)}

    wiki_path = os.path.join(TEXT_ROOT, "wikitext-2", "wiki.train.tokens")
    corpus = []
    if os.path.exists(wiki_path):
        with open(wiki_path, encoding="utf-8") as f:
            corpus.extend(f.read().replace("\n", " <eos> ").split())
    for text in texts:
        corpus.extend(_tokenize(text))

    for token, _ in Counter(corpus).most_common(TEXT_VOCAB_SIZE - len(specials)):
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    return SimpleVocab(token_to_idx)


def _save_vocab(vocab, path=TEXT_PRETRAIN_VOCAB_PATH):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.token_to_idx, f, ensure_ascii=True, sort_keys=False)


def _load_vocab(path=TEXT_PRETRAIN_VOCAB_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        token_to_idx = json.load(f)
    return SimpleVocab(token_to_idx)


def get_text_pretraining_vocab():
    vocab = _load_vocab()
    if vocab is not None:
        return vocab

    train_csv, _ = _ensure_text_csv(TEXT_PRETRAIN_DATASET)
    train_texts, _ = _read_text_csv(train_csv)
    vocab = _build_vocab(train_texts)
    _save_vocab(vocab)
    return vocab


def get_text_dataset_bundle(dataset="ag_news", batch_size=None):
    bs = batch_size or MODALITY_BATCH_SIZE["text"]
    if dataset not in TEXT_DATASET_SPECS:
        raise ValueError(f"Unsupported text dataset: {dataset}")

    train_csv, test_csv = _ensure_text_csv(dataset)
    train_texts, train_labels = _read_text_csv(train_csv)
    test_texts, test_labels = _read_text_csv(test_csv)
    vocab = get_text_pretraining_vocab()

    train_pool = TextClassificationDataset(train_texts, train_labels, vocab)
    test = TextClassificationDataset(test_texts, test_labels, vocab)
    return DatasetBundle("text", dataset, train_pool, test, TEXT_DATASET_NUM_CLASSES[dataset], TEXT_EMBED_DIM, None, bs)


def get_text_loaders(stage="eval", batch_size=None, dataset="ag_news"):
    bs = batch_size or TEXT_BATCH_SIZE
    train_csv, test_csv = _ensure_text_csv(dataset)
    train_texts, train_labels = _read_text_csv(train_csv)
    if stage == "pretrain":
        vocab = _build_vocab(train_texts)
        _save_vocab(vocab)
    else:
        vocab = get_text_pretraining_vocab()

    if stage == "pretrain":
        return DataLoader(SimCLRTextDataset(train_texts, vocab), batch_size=bs, shuffle=True), None

    test_texts, test_labels = _read_text_csv(test_csv)
    train_ds = TextClassificationDataset(train_texts, train_labels, vocab)
    test_ds = TextClassificationDataset(test_texts, test_labels, vocab)
    return train_ds, DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)


def get_dataset_bundle(modality, dataset, batch_size=None):
    if modality == "vision":
        return get_vision_dataset_bundle(dataset=dataset, batch_size=batch_size)
    if modality == "audio":
        return get_audio_dataset_bundle(dataset=dataset, batch_size=batch_size)
    if modality == "text":
        return get_text_dataset_bundle(dataset=dataset, batch_size=batch_size)
    raise ValueError(f"Unsupported modality: {modality}")


def prepare_dataset(modality, dataset):
    if modality == "vision":
        if dataset == "cifar100":
            root = os.path.join(DATA_DIR, "vision", "cifar100")
            datasets.CIFAR100(root=root, train=True, download=_download_enabled())
            datasets.CIFAR100(root=root, train=False, download=_download_enabled())
        return

    if modality == "audio":
        if dataset == "speechcommands":
            _repair_speechcommands_if_partial()
            torchaudio.datasets.SPEECHCOMMANDS(root=AUDIO_ROOT, subset="training", download=_download_enabled())
            torchaudio.datasets.SPEECHCOMMANDS(root=AUDIO_ROOT, subset="testing", download=_download_enabled())
            return
        if dataset == "fsdkaggle2018":
            _repair_fsdkaggle2018_if_partial()
            FSDKaggle2018Dataset("train")
            FSDKaggle2018Dataset("test")
            return

    if modality == "text":
        _ensure_text_csv(dataset)


def worker_seed_init(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
