#!/bin/bash
#SBATCH --job-name=ekg_train_hier
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/ekg_train_hier_%A_%a.out
#SBATCH --error=logs/ekg_train_hier_%A_%a.err
# =====================================================
# HIERARCHICAL training: 4 variants x 3 schemes = 12 array tasks; each task trains its
# 3 seeds (42,0,1) one after another -> 12 jobs, 36 seed-runs.
# Submit from the CODE folder:   bash slurm/submit_train.sh
#   task 0-3 diagnostic, 4-7 rhythm, 8-11 form; within each: noenc, conv, bilstm_cnn, bilstm_full
# ECG_TASC=feat (default) -> checkpoints/seq_tasc/ ; ECG_TASC=none -> original
# ECG_UNIFORM_ATTENTION=1 -> the uniform-attention retrain (only needed for P3's
#   "retrained with uniform" number) -> checkpoints/seq_tasc_uniform/
# NOTE: unlike the old slurm_train_* scripts this does NOT set 5 seeds.
# =====================================================
SCHEMES=(diagnostic diagnostic diagnostic diagnostic \
         rhythm     rhythm     rhythm     rhythm     \
         form       form       form       form)
VARIANTS=(noenc conv bilstm_cnn bilstm_full \
          noenc conv bilstm_cnn bilstm_full \
          noenc conv bilstm_cnn bilstm_full)
idx=${SLURM_ARRAY_TASK_ID:-0}
export ECG_LABEL_SCHEME=${SCHEMES[$idx]}
export ECG_MODEL_VARIANT=${VARIANTS[$idx]}
export ECG_TASC=${ECG_TASC:-feat}
export ECG_UNIFORM_ATTENTION=${ECG_UNIFORM_ATTENTION:-0}
export ECG_SEEDS=${ECG_SEEDS:-42,0,1}
export ECG_EPOCHS=${ECG_EPOCHS:-50}

source ~/.bashrc
conda activate ${CONDA_ENV:-att_ekgs}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ECG_DATA_ROOT=${ECG_DATA_ROOT:-/home/yv33n/data}
export ECG_USE_FILTERED=1
export ECG_NUM_WORKERS=${ECG_NUM_WORKERS:-4}

echo "=========================================="
echo "TRAIN hier  task ${idx}: ${ECG_LABEL_SCHEME} / ${ECG_MODEL_VARIANT}  tasc=${ECG_TASC}"
echo "  uniform=${ECG_UNIFORM_ATTENTION}  seeds=${ECG_SEEDS}  epochs=${ECG_EPOCHS}"
echo "  node=${SLURM_NODELIST}  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
ulimit -n 65535
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1
mkdir -p logs
python -u train_hierarchical_seq.py
