# Perception Learning (PeL): Reproducibility Repository

Official reproduction code for:

**When to Freeze: Testing Perception-Decision Separation Across Modalities and Label Budgets**

This repository contains the code required to reproduce the experiments, diagnostics, and synthetic crossover analyses reported in the paper. The implementation includes PeL encoder pretraining, downstream transfer evaluation, and the full experimental protocol used throughout the study.

---

## Repository Structure

```text
.
├── src/pel/
│   ├── config.py
│   ├── data_loader.py
│   ├── models.py
│   ├── protocol.py
│   └── utils.py
│
├── scripts/
│   ├── pretrain_pel.py
│   ├── run_experiment_cell.py
│   ├── run_experiment_matrix.py
│   ├── figure1_theoretical_crossover.py
│   └── slurm_run_experiment_matrix_fullgpu.sh
│
├── prepare_environment.py
├── requirements.txt
└── README.md
```

---

## Components

### Pretraining

```text
scripts/pretrain_pel.py
```

Pretrains perception encoders using a SimCLR-style self-supervised objective for:

* Vision
* Audio
* Text

The resulting checkpoints are used by both the Fine-Tuning and PeL-Frozen adaptation regimes.

### Downstream Evaluation

```text
scripts/run_experiment_cell.py
```

Runs a single experiment configuration defined by:

* Dataset
* Adaptation method
* Label budget
* Random seed

```text
scripts/run_experiment_matrix.py
```

Executes the complete experimental matrix used in the paper.

### Synthetic Crossover Analysis

```text
scripts/figure1_theoretical_crossover.py
```

Generates the controlled synthetic experiment used to illustrate the bias-variance crossover mechanism described in Section 3.

### Environment Validation

```text
prepare_environment.py
```

Creates required directories, verifies datasets, and checks for pretrained checkpoints before running experiments.

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Datasets

The benchmark consists of six downstream datasets spanning three modalities.

| Modality | Dataset        |
| -------- | -------------- |
| Vision   | ImageNet-100   |
| Vision   | CIFAR-100      |
| Audio    | SpeechCommands |
| Audio    | FSDKaggle2018  |
| Text     | AG News        |
| Text     | DBpedia        |

Most datasets are downloaded automatically during setup.

### ImageNet-100

ImageNet-100 must be provided manually and placed in:

```text
data/vision/train/
data/vision/val/
```

---

## Pretrained Checkpoints

Fine-Tuning and PeL-Frozen experiments require pretrained encoder checkpoints:

```text
results/checkpoints/pel_vision_full.pth
results/checkpoints/pel_audio_full.pth
results/checkpoints/pel_text_full.pth
```

To regenerate these checkpoints:

```bash
python scripts/pretrain_pel.py --modality vision
python scripts/pretrain_pel.py --modality audio
python scripts/pretrain_pel.py --modality text
```

---

## Reproducing the Experiments

### Single Experiment

Example:

```bash
python scripts/run_experiment_cell.py \
    --modality audio \
    --dataset speechcommands \
    --method finetune \
    --fraction 0.01 \
    --seed 42
```

### Full Experimental Matrix

```bash
python scripts/run_experiment_matrix.py
```

### Slurm Execution

```bash
sbatch scripts/slurm_run_experiment_matrix_fullgpu.sh
```

---

## Experimental Matrix

### Adaptation Regimes

* `scratch`
* `finetune`
* `pel_frozen`

### Label Budgets

* `1%`
* `10%`
* `100%`

### Random Seeds

```text
42, 43, 44, 45, 46
```

---

## Outputs

Results are written to:

```text
results/rerun_results.csv
results/rerun_diagnostics.csv
results/checkpoints/rerun/
```

The diagnostics file contains the quantities reported in the representation-analysis section of the paper, including feature drift, gradient variance, and offline probe metrics.

---

## Hardware

The experiments reported in the paper were executed on NVIDIA H100 GPUs. Smaller subsets of the benchmark can be reproduced on consumer GPUs by running individual experiment cells.

---

## Citation

If you find this repository useful, please cite:

```bibtex
[To be added after publication]
```
