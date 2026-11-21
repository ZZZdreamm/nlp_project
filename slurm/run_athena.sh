#!/bin/bash
# Użycie:  sbatch slurm/run_athena.sh            (domyślnie 1 GPU)
# Dostosuj -A (grant PLGrid) i partycję. Athena: partycja plgrid-gpu-a100, Lem: plgrid-gpu-h100.
#SBATCH -J llmbench
#SBATCH -A plgXXXXXXX-gpu-a100        # <-- wpisz swój grant obliczeniowy
#SBATCH -p plgrid-gpu-a100            # Lem: plgrid-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export HF_HOME="${SCRATCH:-$PWD}/hf_cache"      # duże modele - nie w $HOME
python -m llmbench info
python -m llmbench run --backend "${BACKEND:-hf}"
