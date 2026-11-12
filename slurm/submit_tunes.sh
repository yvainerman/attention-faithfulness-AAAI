#!/bin/bash
# =====================================================
# Queue the 21 tuning studies (7 models x 3 schemes) as 4 short SLURM arrays.
# Run from anywhere inside the code folder:
#
#   bash slurm/submit_tunes.sh                 # TaSc models (ECG_TASC=feat)
#   ECG_TASC=none bash slurm/submit_tunes.sh   # the original models instead
#   bash slurm/submit_tunes.sh flat            # only the flat family (or: hier)
#
# Time limits are sized from your June A100 logs (seconds per epoch x ~30 trials,
# with pruning + early stopping). Short limits schedule faster (backfill). Nothing
# is lost if a job runs out of time: every study is saved (tune_results/<tag>.db)
# and its best cell is rewritten after each trial (tune_results/<tag>.json).
# Just run this script again: finished studies only re-report (seconds),
# unfinished ones resume. Collect the cells with:  python collect_tuned.py
#
# Override any budget at submit time, e.g.  ECG_TUNE_TRIALS=50 bash slurm/submit_tunes.sh
# =====================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p logs tune_results               # SLURM writes logs/ before the job starts

TASC=${ECG_TASC:-feat}
WHICH=${1:-all}
EXP="ALL,ECG_TASC=${TASC}"
tag=$([ "${TASC}" = "none" ] && echo orig || echo tasc)

submit() {   # name  array  time  script
    local id
    id=$(sbatch --parsable --export="${EXP}" --job-name="$1" --array="$2" --time="$3" "$4")
    printf "  %-18s array=%-16s time=%s  -> job %s\n" "$1" "$2" "$3" "${id}"
}

echo "submitting tuning arrays (ECG_TASC=${TASC}) from $(pwd)"
if [ "${WHICH}" = "all" ] || [ "${WHICH}" = "flat" ]; then
    submit "tune_${tag}_flat"     "0,1,3,4,6,7"    "01:15:00" slurm/tune_flat_seq.slurm   # noenc, conv
    submit "tune_${tag}_flat_gl"  "2,5,8"          "02:00:00" slurm/tune_flat_seq.slurm   # bilstm
fi
if [ "${WHICH}" = "all" ] || [ "${WHICH}" = "hier" ]; then
    submit "tune_${tag}_hier"     "0-2,4-6,8-10"   "01:00:00" slurm/tune_hierarchical_seq.slurm  # noenc, conv, bilstm_cnn
    submit "tune_${tag}_hier_gl"  "3,7,11"         "01:45:00" slurm/tune_hierarchical_seq.slurm  # bilstm_full
fi
echo "watch:   squeue -u \$USER"
echo "results: python collect_tuned.py   (paste the cells into TUNED_TASC / TUNED)"
