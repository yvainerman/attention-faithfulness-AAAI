# AI4TS extension: TaSc attention for EKG time series

This folder is a copy of `attention-faithfulness-for-EKGs` (the BIBM 2026 code)
with ONE attention intervention added: **Task-Scaling (TaSc)**, Chrysostomou &
Aletras, ACL 2021 (code: github.com/GChrysostomou/tasc), adapted to EKGs. It is
switched on with `ECG_TASC=feat`; with the variable unset every script is the
original code (the no-TaSc models are bit-identical to the original ones and
load the original checkpoints).

## The method in one equation

```
original:  c = sum_t  alpha_t        * h_t
TaSc:      c = sum_t  alpha_t * s_t  * h_t,     s_t = u . e_t
```

* `alpha_t` – the usual additive softmax attention over the CONTEXTUALISED
  reach-encoder outputs `h_t` (unchanged). This is still the map that P1–P5 test.
* `e_t` – position t's NON-contextualised representation, taken before the reach
  encoder mixes positions: the 100-ms PatchCNN token (flat; patches do not
  overlap) or the per-beat vector (hierarchical; each beat encoded alone).
* `s_t = u . e_t` – a task-specific score of the position's own content (one
  learned vector `u`, 256 parameters per model). TaSc indexes `u` by vocabulary
  word; EKG segments have no vocabulary, so one `u` is shared by all positions
  (the Feat-TaSc form). Details: `models/tasc.py`.

Only the FINAL attention is scaled (the one every test evaluates; hierarchical
models keep their within-beat attention unchanged).

## Compute: 21 jobs, not 63

There are 21 models (7 architectures x 3 label schemes) and each SLURM job
trains ONE model with its 3 seeds back to back, so a full training pass is
**21 jobs** (9 flat + 12 hierarchical), ~1–1.5 h wall-clock when they run in
parallel (estimate from the June A100 logs). Two passes are optional:

* **Uniform-attention retrain** (`ECG_UNIFORM_ATTENTION=1`) – only for P3's
  "retrained with uniform attention" number. Without it P3 still runs and
  reports "uniform at test time" (no retraining needed).
* **Tuning** – costs more than training. Without it, TaSc reuses the original
  models' tuned settings (the run record says so), which also keeps TaSc the
  only difference from the original models.

## What changed (vs. the original repo)

| file | change |
|---|---|
| `models/tasc.py` | NEW: TaSc score, `ECG_TASC` parsing, checkpoint-folder naming |
| `models/flat_seq.py`, `models/hierarchical_seq.py` | optional `tasc=` argument; `final_values()` = `h` or `s*h` |
| `train_flat_seq.py`, `train_hierarchical_seq.py` | `ECG_TASC`; **3 seeds by default (42,0,1)**; `TUNED_TASC` table (empty -> reuses the original model's tuned cell); TaSc checkpoints in `checkpoints/{flat_seq,seq}_tasc[_uniform]/`; run record stores `tasc` |
| `tune_common.py` | NEW: Optuna early stopping (per trial and per study), resumable saved studies, job time budget, best cell saved after every trial, `--check`, `ECG_TUNE_SMOKE` |
| `tune_flat_seq.py`, `tune_hierarchical_seq.py` | **40 epochs per trial**; use `tune_common.py`; `--report`, `--check`; `ECG_TASC=feat` tunes TaSc (study tagged `_tasc`, cell for `TUNED_TASC`) |
| `collect_tuned.py` | NEW: prints every saved tuned cell, grouped by the table to paste into |
| `slurm/` | NEW: tuning + training arrays for Hellbender, two submit helpers, and `check_tunes.sh` (login-node check) |
| `test_flat_seq.py`, `test_hierarchical_seq.py` | `ECG_TASC`; evaluate only the seeds in `ECG_SEEDS` (so old 5-seed folders are scored on the same 3 seeds) |
| `faithfulness_common.py` | tests receive `s*h` as the final attention's values, so every intervention still acts on `alpha` alone; TaSc-aware gradient / LOO / uniform; outputs to `attention_tests_tasc/` |
| six test scripts | docstrings only (`ECG_TASC` listed; 3-seed note) |

`CHANGES_vs_original.diff` (next to this file) has every line that changed.

## Tuning (optional): early stopping + saved studies

Speed-ups, all env-overridable (full list in `tune_common.py`):

| setting | default | what it does |
|---|---|---|
| `ECG_TUNE_EPOCHS` | 40 | max epochs per trial |
| `ECG_TUNE_PATIENCE` | 6 | trial early stopping: stop after 6 epochs without a macro-AUC gain >= 0.001; the trial still COMPLETES with its best AUC, so it stays a candidate |
| MedianPruner | (as before) | prunes trials below the median after 5 epochs |
| `ECG_TUNE_STUDY_PATIENCE` | 10 | study early stopping (`study.stop()`): stop after 10 trials in a row without the best AUC improving by >= 0.001 ... |
| `ECG_TUNE_MIN_TRIALS` | 15 | ... but not before 15 trials (TPE's first 10 are random) |
| `ECG_TUNE_TRIALS` | 50 (SLURM: 30) | cap on trials in the saved study, counting earlier jobs |
| `ECG_TUNE_STORAGE=auto` | (SLURM) | one saved study per model: `tune_results/<tag>.db` |
| `ECG_TUNE_TIMEOUT_MIN` | (SLURM sets it) | no epoch starts that would not finish before the job's limit; the cut trial is marked FAIL and the job exits after saving |

Every study is saved (as in the original SLURM scripts) and
`tune_results/<tag>.json` is rewritten after every trial, so a job that runs
out of time loses at most the trial in progress. **Resubmitting is always
safe**: finished studies only re-report (seconds, no data loading), unfinished
ones resume (the sampler is re-seeded so random start-up trials are not
repeated).

**Check before you submit.** Real tuning cannot run on a login node (no GPU,
multi-GB caches, shared machine), but everything else can be checked there:

```bash
cd /path/to/attention-faithfulness-EKGs-AI4TS
bash slurm/check_tunes.sh quick       # login node, ~1 min: 7 checks (one per model) + sbatch --test-only
bash slurm/check_tunes.sh             # all 21 studies (~3-4 min)
```

Each check runs the real SLURM script with `DRY_RUN=1` (python `--check`):
package versions, PTB-XL folder + this scheme's cache, sqlite storage on your
filesystem (a throwaway study is created and deleted; the real study is only
read), the model's forward/backward on a tiny CPU batch (TaSc included), and
the Optuna early-stopping code on a fake objective. `sbatch --test-only` then
validates partition / GPU / time for the four arrays without queueing anything.
For one REAL end-to-end run on a GPU (~15 min; in-memory study, results in
`tune_results/_smoke/`, saved studies untouched):

```bash
mkdir -p logs
ECG_TUNE_SMOKE=1 sbatch --array=4 --time=00:20:00 slurm/tune_flat_seq.slurm          # f-lc, rhythm
ECG_TUNE_SMOKE=1 sbatch --array=7 --time=00:20:00 slurm/tune_hierarchical_seq.slurm  # h-gl-gl, rhythm
```

Then queue the real thing:

```bash
bash slurm/submit_tunes.sh            # 21 TaSc studies as 4 short arrays (ECG_TASC=none: original models)
squeue -u $USER
bash slurm/submit_tunes.sh            # only if some studies said "NOT finished": resumes them
python collect_tuned.py               # paste the printed cells into TUNED_TASC in the two trainers
```

Time limits (from your June A100 logs, seconds/epoch x ~30 trials with pruning
and early stopping; estimates): flat noenc/conv 1h15, flat bilstm 2h,
hierarchical noenc/conv/bilstm_cnn 1h, bilstm_full 1h45. Short limits get
scheduled sooner; anything unfinished resumes on resubmission.

## Training and testing

```bash
bash slurm/submit_train.sh                           # 21 jobs x 3 seeds, TaSc models
ECG_UNIFORM_ATTENTION=1 bash slurm/submit_train.sh   # optional: P3's uniform retrain
```

NOTE: the old `slurm_train_*` scripts export `ECG_SEEDS=42,0,1,2,3`, which
overrides the new 3-seed default; use the ones in `slurm/`.

Then, from the code folder (same commands as before, plus `ECG_TASC=feat`):

```bash
ECG_TASC=feat python test_flat_seq.py              # performance on the test fold
ECG_TASC=feat python test_hierarchical_seq.py
ECG_TASC=feat python ss_erasure_test.py        # P1  decision alignment (erasure)
ECG_TASC=feat python jw_correlation_test.py    # P2  attention vs gradient vs LOO
ECG_TASC=feat python wp_necessity_test.py      # P3  uniform attention
ECG_TASC=feat python jw_permutation_test.py    # P4  permutation
ECG_TASC=feat python jw_adversarial_test.py    # P4  adversarial attention
ECG_TASC=feat python wp_seed_variance_test.py  # P4  seed-variance baseline (after the adversarial test)
# P5 (clinical coherence) = the same tests compared across schemes; every script
# runs diagnostic,rhythm,form by default (ECG_TEST_SCHEMES).
```

The `--no-summary` / `--summarize` flags for SLURM arrays work as before. Run the
same commands without `ECG_TASC` to get the original models' results in
`attention_tests/` (with 3 seeds, existing 5-seed checkpoints are reused for
seeds 42, 0, 1).

Outputs: `checkpoints/*_tasc*/`, `test_results/{flat,hier}_tasc[_uniform]_test.json`,
`tune_results/*_tasc.{db,json}`, `attention_tests_tasc/`.

## How the tests read a TaSc model

The explanation under test is `alpha` (the softmax distribution) – the same
object the TaSc paper evaluates as "attention" and the one our paper tests. `s_t`
is treated as part of what position t contributes:

* P1 erasure / P4 permutation / P4 adversarial: `alpha` is zeroed and renormalised,
  shuffled, or optimised; each `s_t` stays with its own position.
* P2: gradient with all attention frozen (gradient flows through the encoders and
  `s_t`); LOO deletes the position and re-runs the whole model.
* P3: `alpha := 1/N` (re-trained or at test time) with `s_t` kept, i.e. "is the
  learned attention distribution needed once each position has its own score?"

## Not implemented (decision pending)

TaSc's strongest results in the paper use **attention x gradient** (alpha times
d output / d alpha) rather than raw `alpha` as the explanation. That combines
attention with a gradient, so it is not used as the explanation in P2–P5 here.
P1 already ranks by Attention x Gradient as one of its four original erasure
orders, so that number is produced anyway.
