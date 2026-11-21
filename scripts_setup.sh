#!/bin/bash
# Tworzy venv i instaluje zależności. Użycie: ./scripts_setup.sh [hf|vllm]
set -euo pipefail
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# torch: na klastrze użyj modułu CUDA zgodnego z wersją (module load CUDA/...), lokalnie:
pip install torch transformers accelerate
[ "${1:-hf}" = "vllm" ] && pip install vllm
python -m llmbench info
