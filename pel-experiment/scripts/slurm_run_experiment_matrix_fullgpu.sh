#!/bin/bash
#SBATCH --job-name=pel_rerun
#SBATCH --output=results/slurm/pel_rerun_%j.out
#SBATCH --error=results/slurm/pel_rerun_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="$(pwd)"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

mkdir -p "$PROJECT_ROOT/results/slurm"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PEL_NUM_WORKERS="${PEL_NUM_WORKERS:-16}"

"$VENV_PYTHON" prepare_environment.py
"$VENV_PYTHON" scripts/run_experiment_matrix.py "$@"