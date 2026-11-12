#!/bin/bash
#SBATCH --job-name=ekg_tune_hier
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/ekg_tune_hier_%A_%a.out
#SBATCH --error=logs/ekg_tune_hier_%A_%a.err
# =====================================================
# HIERARCHICAL tuning, one Optuna study per (scheme, variant): 3 x 4 = 12 array tasks.
# Submit from the CODE folder (logs/ must exist):  bash slurm/submit_tunes.sh
# (that script also gives bilstm_full a longer --time than the default above).
#
#   task  scheme       variant        task  scheme   variant        task  scheme  variant
#    0    diagnostic   noenc           4    rhythm   noenc           8    form    noenc
#    1    diagnostic   conv            5    rhythm   conv            9    form    conv
#    2    diagnostic   bilstm_cnn      6    rhythm   bilstm_cnn     10    form    bilstm_cnn
#    3    diagnostic   bilstm_full     7    rhythm   bilstm_full    11    form    bilstm_full
#
# SAVED + RESUMABLE: each study is tune_results/<tag>.db (ECG_TUNE_STORAGE=auto) and
# tune_results/<tag>.json is rewritten after every finished trial. If a job runs
# out of time, submit the same command again: finished studies only re-report (a
# few seconds), unfinished ones continue where they stopped.
# ECG_TASC=feat (default) tunes the TaSc models; ECG_TASC=none the original ones.
#
# CHECK FIRST (see slurm/check_tunes.sh):
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=4 bash slurm/tune_hierarchical_seq.slurm   # login node: --check only
#   ECG_TUNE_SMOKE=1 sbatch --array=4 --time=00:20:00 slurm/tune_hierarchical_seq.slurm
#        # real 2-trial x 2-epoch GPU run; results in tune_results/_smoke/, saved studies untouched
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

source ~/.bashrc
conda activate ${CONDA_ENV:-att_ekgs}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ECG_DATA_ROOT=${ECG_DATA_ROOT:-/home/yv33n/data}
export ECG_USE_FILTERED=1
export ECG_NUM_WORKERS=${ECG_NUM_WORKERS:-4}

# --- search budget (FAST preset; every value can be overridden at submit time) ---
export ECG_TUNE_EPOCHS=${ECG_TUNE_EPOCHS:-40}            # max epochs per trial
export ECG_TUNE_PATIENCE=${ECG_TUNE_PATIENCE:-6}         # trial early stopping (epochs)
export ECG_TUNE_STUDY_PATIENCE=${ECG_TUNE_STUDY_PATIENCE:-10}  # study early stopping (trials)
export ECG_TUNE_MIN_TRIALS=${ECG_TUNE_MIN_TRIALS:-15}    # ... never before this many trials
export ECG_TUNE_MIN_DELTA=${ECG_TUNE_MIN_DELTA:-0.001}   # macro-AUC gain that counts
export ECG_TUNE_TRIALS=${ECG_TUNE_TRIALS:-30}            # cap on trials in the saved study
export ECG_TUNE_SEED=${ECG_TUNE_SEED:-42}
export ECG_TUNE_STORAGE=${ECG_TUNE_STORAGE:-auto}        # sqlite:///tune_results/<tag>.db

# --- time budget: stop cleanly a few minutes before SLURM's limit ---
TL=$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null || true)
LEFT_MIN=$(python -c "
import sys
s = sys.argv[1].strip()
try:
    d, _, hms = s.rpartition('-')
    p = [int(x) for x in hms.split(':')]
    while len(p) < 3: p.insert(0, 0)
    print(int((int(d or 0) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]) / 60))
except Exception:
    print(0)" "${TL}")
MARGIN_MIN=${TUNE_MARGIN_MIN:-6}
if [ "${LEFT_MIN}" -gt $((MARGIN_MIN + 5)) ]; then
    export ECG_TUNE_TIMEOUT_MIN=$((LEFT_MIN - MARGIN_MIN))
fi

echo "=========================================="
echo "TUNE hier  task ${idx}: ${ECG_LABEL_SCHEME} / ${ECG_MODEL_VARIANT}  tasc=${ECG_TASC}"
echo "  node=${SLURM_NODELIST}  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  time left=${TL:-?}  -> tuning budget=${ECG_TUNE_TIMEOUT_MIN:-none} min"
echo "  dry_run=${DRY_RUN:-0}  smoke=${ECG_TUNE_SMOKE:-0}"
echo "=========================================="
ulimit -n 65535
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1
mkdir -p logs tune_results
if [ "${DRY_RUN:-0}" = "1" ]; then
    python -u tune_hierarchical_seq.py --check          # no GPU, no data loading, no training
else
    python -u tune_hierarchical_seq.py
fi
