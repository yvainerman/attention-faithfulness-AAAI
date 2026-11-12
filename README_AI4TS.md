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

## What changed (vs. the original repo)

| file | change |
|---|---|
| `models/tasc.py` | NEW: TaSc score, `ECG_TASC` parsing, checkpoint-folder naming |
| `models/flat_seq.py`, `models/hierarchical_seq.py` | optional `tasc=` argument; `final_values()` = `h` or `s*h` |
| `train_flat_seq.py`, `train_hierarchical_seq.py` | `ECG_TASC`; **3 seeds by default (42,0,1)**; `TUNED_TASC` table (empty -> reuses the original model's tuned cell); TaSc checkpoints in `checkpoints/{flat_seq,seq}_tasc[_uniform]/`; run record stores `tasc` |
| `tune_flat_seq.py`, `tune_hierarchical_seq.py` | **50 epochs per trial by default** (was 15); `ECG_TASC=feat` tunes the TaSc model (study tagged `_tasc`, prints a `TUNED_TASC` cell) |
| `test_flat_seq.py`, `test_hierarchical_seq.py` | `ECG_TASC`; evaluate only the seeds in `ECG_SEEDS` (so old 5-seed folders are scored on the same 3 seeds) |
| `faithfulness_common.py` | tests receive `s*h` as the final attention's values, so every intervention still acts on `alpha` alone; TaSc-aware gradient / LOO / uniform; outputs to `attention_tests_tasc/` |
| six test scripts | docstrings only (`ECG_TASC` listed; 3-seed note) |

`CHANGES_vs_original.diff` (next to this file) has every line that changed.

## Running it (same scripts, same env vars as before, plus `ECG_TASC=feat`)

```bash
export ECG_DATA_ROOT=/home/yv33n/data          # as before
python prepare_data.py                          # unchanged; caches are shared

# (optional) tune TaSc, 50 epochs/trial; paste the printed cell into TUNED_TASC
ECG_TASC=feat ECG_MODEL_VARIANT=conv ECG_LABEL_SCHEME=rhythm python tune_flat_seq.py
#   skip tuning -> TaSc reuses the original model's TUNED cell (logged as such)

# train: every (variant, scheme), 3 seeds; then the uniform-attention version for P3
ECG_TASC=feat ECG_MODEL_VARIANT=conv ECG_LABEL_SCHEME=rhythm python train_flat_seq.py
ECG_TASC=feat ECG_UNIFORM_ATTENTION=1 ECG_MODEL_VARIANT=conv ECG_LABEL_SCHEME=rhythm python train_flat_seq.py
#   flat variants: noenc conv bilstm | hierarchical: noenc conv bilstm_cnn bilstm_full

# performance on the test fold
ECG_TASC=feat python test_flat_seq.py
ECG_TASC=feat python test_hierarchical_seq.py

# faithfulness tests -- each one runs on its own
ECG_TASC=feat python ss_erasure_test.py        # P1  decision alignment (erasure)
ECG_TASC=feat python jw_correlation_test.py    # P2  attention vs gradient vs LOO
ECG_TASC=feat python wp_necessity_test.py      # P3  uniform attention (needs the _uniform checkpoints)
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
`tune_results/*_tasc.json`, `attention_tests_tasc/`.

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

## Compute (3 seeds)

7 models x 3 schemes x 3 seeds = 63 TaSc training runs, plus 63 uniform-attention
TaSc runs for P3. Optional tuning: 21 studies x 50 trials x up to 50 epochs
(MedianPruner stops weak trials after epoch 5).
