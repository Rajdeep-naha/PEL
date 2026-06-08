import os
import torch

# System and hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(os.environ.get("PEL_NUM_WORKERS", "16"))
SEED = 42
SEEDS = [42, 43, 44, 45, 46]

# Paths
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESULTS_CSV = os.path.join(RESULTS_DIR, "experiment_log.csv")
RERUN_RESULTS_CSV = os.path.join(RESULTS_DIR, "rerun_results.csv")
RERUN_DIAGNOSTICS_CSV = os.path.join(RESULTS_DIR, "rerun_diagnostics.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Rerun protocol
METHODS = ["scratch", "finetune", "pel_frozen"]
DATA_FRACTIONS = [0.01, 0.10, 1.0]
VALIDATION_FRACTION = 0.20
FALLBACK_VALIDATION_FRACTION = 0.10
PROBE_SET_SIZE = int(os.environ.get("PEL_PROBE_SET_SIZE", "2048"))
DIAGNOSTIC_FRACTIONS = [0.01, 0.10]
GRADIENT_VARIANCE_STEPS = 50

MAX_EPOCHS = {
    "scratch": 50,
    "finetune": 50,
    "pel_frozen": 100,
}
EARLY_STOP_PATIENCE = {
    "scratch": 15,
    "finetune": 15,
    "pel_frozen": 20,
}

ENCODER_LR = {
    "scratch": 1e-4,
    "finetune": 1e-4,
    "pel_frozen": 0.0,
}
HEAD_LR = {
    "scratch": 1e-3,
    "finetune": 1e-3,
    "pel_frozen": 1e-3,
}
WEIGHT_DECAY = 1e-2
DOWNSTREAM_BATCH_SIZE = 64
MODALITY_BATCH_SIZE = {
    "vision": 2048,
    "audio": 256,
    "text": 2048,
}
MODALITY_EFFECTIVE_BATCH_SIZE = {
    "vision": 2048,
    "audio": 2048,
    "text": 2048,
}
FEATURE_BATCH_SIZE = 128
OFFLINE_PROBE_EPOCHS = 100
OFFLINE_PROBE_PATIENCE = 15

# Dataset matrix from rerun plan.
# ImageNet-100 uses the existing val folder as the held-out final test split.
# CIFAR-100 is the lightweight second vision task selected for the VTAB-style
# transfer slot in this codebase. FSDKaggle2018 uses the official train/test
# split from the Zenodo release. DBpedia is used as the second text
# classification dataset.
RERUN_DATASETS = {
    "vision": ["imagenet100", "cifar100"],
    "audio": ["speechcommands", "fsdkaggle2018"],
    "text": ["ag_news", "dbpedia"],
}

# Pretraining / encoder settings
PRETRAINED_CHECKPOINTS = {
    "vision": "pel_vision_full.pth",
    "audio": "pel_audio_full.pth",
    "text": "pel_text_full.pth",
}

# SimCLR / contrastive pretraining
SIMCLR_TEMPERATURE = 0.1
SIMCLR_PROJECTION_DIM = 128

# Vision
VISION_TRAIN_DIR = os.path.join(DATA_DIR, "vision", "train")
VISION_VAL_DIR = os.path.join(DATA_DIR, "vision", "val")
VISION_IMG_SIZE = 224
VISION_INPUT_SHAPE = (3, 224, 224)
VISION_BATCH_SIZE = 1024
VISION_FEAT_DIM = 512
VISION_LR_PRETRAIN = 1e-3
VISION_LR_PROBE = 1e-3
VISION_EPOCHS_PRETRAIN = 200
VISION_EPOCHS_LINEAR = 200
VISION_DATASET_NUM_CLASSES = {
    "imagenet100": 100,
    "cifar100": 100,
}
VISION_NUM_CLASSES = VISION_DATASET_NUM_CLASSES["imagenet100"]

# Audio
AUDIO_ROOT = os.path.join(DATA_DIR, "audio")
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
AUDIO_N_FFT = 1024
AUDIO_N_MELS = 32
AUDIO_HOP_LENGTH = 160
AUDIO_INPUT_SHAPE = (1, int(AUDIO_SAMPLE_RATE * AUDIO_DURATION))
AUDIO_FEAT_DIM = 128
AUDIO_LR_PRETRAIN = 1e-3
AUDIO_BATCH_SIZE = 512
AUDIO_EPOCHS_PRETRAIN = 200
AUDIO_EPOCHS_LINEAR = 1000
AUDIO_DATASET_NUM_CLASSES = {
    "speechcommands": 35,
    "fsdkaggle2018": 41,
}
AUDIO_NUM_CLASSES = AUDIO_DATASET_NUM_CLASSES["speechcommands"]

# Text
TEXT_ROOT = os.path.join(DATA_DIR, "text")
TEXT_MAX_LEN = 128
TEXT_VOCAB_SIZE = 30522
TEXT_EMBED_DIM = 256
TEXT_LAYERS = 4
TEXT_HEADS = 4
TEXT_BATCH_SIZE = 512
TEXT_EPOCHS_PRETRAIN = 100
TEXT_DATASET_NUM_CLASSES = {
    "ag_news": 4,
    "dbpedia": 14,
}
TEXT_NUM_CLASSES = TEXT_DATASET_NUM_CLASSES["ag_news"]

# Robustness is appendix-only in the rerun plan.
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
FORCE_RETRAIN = False
DEBUG_MODE = False
