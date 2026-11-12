#!/bin/bash
# =====================================================
# Queue the 21 training jobs (9 flat + 12 hierarchical); each trains 3 seeds.
#   bash slurm/submit_train.sh                          # TaSc models
#   ECG_UNIFORM_ATTENTION=1 bash slurm/submit_train.sh  # TaSc uniform retrain (P3 only; optional)
#   ECG_TASC=none bash slurm/submit_train.sh            # original models
# Paste the tuned cells (python collect_tuned.py) into TUNED_TASC first if you
# tuned; otherwise TaSc reuses the original models' tuned settings.
# =====================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p logs
EXP="ALL,ECG_TASC=${ECG_TASC:-feat},ECG_UNIFORM_ATTENTION=${ECG_UNIFORM_ATTENTION:-0}"
f=$(sbatch --parsable --export="${EXP}" slurm/train_flat_seq.slurm)
h=$(sbatch --parsable --export="${EXP}" slurm/train_hierarchical_seq.slurm)
echo "flat array job ${f} (9 tasks), hierarchical array job ${h} (12 tasks)  [${EXP#ALL,}]"
