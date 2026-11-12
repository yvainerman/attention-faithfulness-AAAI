#!/bin/bash
# =====================================================
# LOGIN-NODE CHECK before queueing the tuning arrays. Safe on a login node:
# no GPU, no training, the multi-GB caches are not loaded, nothing is queued.
#   1. runs every tuning task's real SLURM script with DRY_RUN=1 (python --check):
#      packages, PTB-XL folder + cache, saved-study storage, the model's
#      forward/backward on a tiny CPU batch, and the Optuna early-stopping code;
#   2. asks SLURM to validate the four arrays exactly as submit_tunes.sh would
#      send them (sbatch --test-only: partition / GPU / time are checked and a
#      start-time estimate is printed, but nothing is submitted).
#
#   bash slurm/check_tunes.sh              # all 21 TaSc studies (~2-4 min)
#   bash slurm/check_tunes.sh quick        # one scheme (rhythm) per model: 7 checks
#   ECG_TASC=none bash slurm/check_tunes.sh
#
# Then, for a REAL end-to-end run on a GPU (~15 min, saved studies untouched):
#   ECG_TUNE_SMOKE=1 sbatch --array=4 --time=00:20:00 slurm/tune_flat_seq.slurm
#   ECG_TUNE_SMOKE=1 sbatch --array=7 --time=00:20:00 slurm/tune_hierarchical_seq.slurm
# =====================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p logs/check
export ECG_TASC=${ECG_TASC:-feat}
MODE=${1:-all}
if [ "${MODE}" = "quick" ]; then
    FLAT_TASKS="3 4 5"; HIER_TASKS="4 5 6 7"          # rhythm row of each array
else
    FLAT_TASKS=$(seq 0 8); HIER_TASKS=$(seq 0 11)
fi

n_ok=0; n_bad=0; bad=()
check() {   # script  task  label
    local log="logs/check/$(basename "$1" .slurm)_task$2.log"
    if DRY_RUN=1 SLURM_ARRAY_TASK_ID=$2 bash "$1" > "${log}" 2>&1; then
        n_ok=$((n_ok + 1)); printf "  PASS  %-28s task %-2s  %s\n" "$(basename "$1")" "$2" "$3"
    else
        n_bad=$((n_bad + 1)); bad+=("${log}")
        printf "  FAIL  %-28s task %-2s  %s   -> %s\n" "$(basename "$1")" "$2" "$3" "${log}"
        grep -E "\[FAIL\]|Error|error" "${log}" | head -5 | sed 's/^/          /'
    fi
}
FLAT=(diagnostic/noenc diagnostic/conv diagnostic/bilstm rhythm/noenc rhythm/conv rhythm/bilstm
      form/noenc form/conv form/bilstm)
HIER=(diagnostic/noenc diagnostic/conv diagnostic/bilstm_cnn diagnostic/bilstm_full
      rhythm/noenc rhythm/conv rhythm/bilstm_cnn rhythm/bilstm_full
      form/noenc form/conv form/bilstm_cnn form/bilstm_full)

echo "1) python --check for each tuning task (ECG_TASC=${ECG_TASC}); full output in logs/check/"
for i in ${FLAT_TASKS}; do check slurm/tune_flat_seq.slurm "$i" "${FLAT[$i]}"; done
for i in ${HIER_TASKS}; do check slurm/tune_hierarchical_seq.slurm "$i" "${HIER[$i]}"; done
echo "   ${n_ok} passed, ${n_bad} failed"

echo "2) sbatch --test-only (validates partition/GPU/time; queues nothing)"
if command -v sbatch > /dev/null; then
    EXP="ALL,ECG_TASC=${ECG_TASC}"
    for spec in "0,1,3,4,6,7|01:15:00|slurm/tune_flat_seq.slurm" \
                "2,5,8|02:00:00|slurm/tune_flat_seq.slurm" \
                "0-2,4-6,8-10|01:00:00|slurm/tune_hierarchical_seq.slurm" \
                "3,7,11|01:45:00|slurm/tune_hierarchical_seq.slurm"; do
        IFS="|" read -r arr tim scr <<< "${spec}"
        msg=$(sbatch --test-only --export="${EXP}" --array="${arr}" --time="${tim}" "${scr}" 2>&1)
        rc=$?
        if [ ${rc} -ne 0 ]; then n_bad=$((n_bad + 1)); fi
        printf "  %-4s %-31s array=%-13s time=%s\n         %s\n" "$([ ${rc} -eq 0 ] && echo OK || echo FAIL)" \
               "$(basename "${scr}")" "${arr}" "${tim}" "${msg}"
    done
else
    echo "  (sbatch not found -- run this on the cluster's login node)"
fi

if [ ${n_bad} -eq 0 ]; then
    echo "ALL CHECKS PASSED -> bash slurm/submit_tunes.sh"
else
    echo "${n_bad} CHECK(S) FAILED -- fix before submitting (logs: ${bad[*]:-see above})"
    exit 1
fi
