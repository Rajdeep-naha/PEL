#!/bin/bash
#SBATCH --job-name=pel_rerun_fullgpu
#SBATCH --output=/localstorage/home/f20221218/PEL/results/slurm/pel_rerun_fullgpu_%j.out
#SBATCH --error=/localstorage/home/f20221218/PEL/results/slurm/pel_rerun_fullgpu_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=h100-full
#SBATCH --account=professors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h100_nvl:1

set -euo pipefail

PROJECT_ROOT="/localstorage/home/f20221218/PEL"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

mkdir -p "$PROJECT_ROOT/results/slurm"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PEL_NUM_WORKERS="${PEL_NUM_WORKERS:-16}"

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Python: $VENV_PYTHON"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$VENV_PYTHON" prepare_environment.py

echo "Launching full rerun matrix on full H100 with modality-optimized batch sizes"
"$VENV_PYTHON" scripts/run_experiment_matrix.py "$@"

echo "Job finished on $(date)"
