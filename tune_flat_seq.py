"""
tune_flat_seq.py
----------------
Optuna hyperparameter search for ONE (variant, scheme) of the multi-label FLAT
pipeline. Mirrors tune_hierarchical_seq.py exactly, but imports the FLAT trainer
(train_flat_seq.py) so it reuses that pipeline's data loading (cleaned PTB-XL
signals), model construction inputs (ARCH), loss, training step and multi-label
`validate` (search optimises the SAME metric the paper reports -- threshold-free
macro-AUC on the validation fold).

Axes selected by env (same names as the trainer):
    ECG_MODEL_VARIANT  noenc | conv | bilstm
    ECG_LABEL_SCHEME   diagnostic | rhythm | form
    ECG_DIAG_LEVEL     superclass | subclass | statement
    ECG_TASC           none | feat  (feat: tune the TaSc model; study tagged *_tasc,
                       cell goes into TUNED_TASC in train_flat_seq.py)

WHAT IS TUNED (optimisation only): lr, dropout, weight_decay, pos_weight, batch_size
-- exactly the five keys of a TUNED cell in train_flat_seq.py.
Architecture WIDTH (ARCH) is held FIXED -- the study compares reach with capacity
constant, so tuning width would reintroduce the confound the design removes.
Only the normal (learned-attention) model is tuned; the uniform-attention
baseline reuses its settings.

SPEED / SAVING (see tune_common.py for every setting):
    40 epochs per trial (ECG_TUNE_EPOCHS), trial early stopping (ECG_TUNE_PATIENCE),
    MedianPruner, study early stopping (ECG_TUNE_STUDY_PATIENCE), a cap on trials in
    the saved study (ECG_TUNE_TRIALS), a job time budget (ECG_TUNE_TIMEOUT_MIN), and a
    resumable saved study (ECG_TUNE_STORAGE=auto -> tune_results/<tag>.db).

RUN   python tune_flat_seq.py            tune (resumes a saved study)
      python tune_flat_seq.py --report   print/save the best cell of the saved study only
      python tune_flat_seq.py --check    login-node check: no GPU, no data loading, no training
      ECG_TUNE_SMOKE=1 python tune_flat_seq.py   real 2-trial x 2-epoch run (GPU node),
                                                 never touches the saved studies
"""
import argparse
import os
import sys

import numpy as np
import torch

# IMPORTANT: env (ECG_MODEL_VARIANT / ECG_LABEL_SCHEME / ...) must be set BEFORE
# this import so the trainer's module-level globals match this run.
from train_flat_seq import (
    MODEL_VARIANT, LABEL_SCHEME, DIAG_LEVEL, USE_FILTERED, UNIFORM, TASC, ARCH, DEVICE,
    PREP_TAG, TOKEN_LEN, FS,
    resolve_data_path, load_and_split, build_loaders, make_pos_weight,
    train_epoch, validate, set_seed, check_setup,
)
from models.flat_seq import FlatSeqAttention
from models.tasc import method_tag
import tune_common as tc


def make_objective(dataset, num_classes, train_idx, val_idx, train_labels):
    def objective(trial):
        lr           = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        dropout      = trial.suggest_float("dropout", 0.2, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        pw_scheme    = trial.suggest_categorical("pos_weight", ["none", "inv", "sqrt"])
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

        set_seed(tc.TUNE_SEED)
        train_loader, val_loader = build_loaders(
            dataset, train_idx, val_idx, batch_size, tc.TUNE_SEED, tc.NUM_WORKERS)

        model = FlatSeqAttention(
            num_classes=num_classes, variant=MODEL_VARIANT,
            dropout=dropout, tasc=TASC, **ARCH[MODEL_VARIANT]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.TUNE_EPOCHS)
        pos_weight = make_pos_weight(pw_scheme, train_labels).to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return tc.run_trial_epochs(trial, model, train_loader, val_loader, criterion,
                                   optimizer, scheduler, train_epoch, validate)
    return objective


def data_checks():
    """PTB-XL folder + this scheme's cleaned-signal cache (existence only: the cache's
    settings are checked when it is loaded, which is too heavy for a login node)."""
    from data.ptbxl import check_data_path, _cache_path
    try:
        dp = check_data_path(resolve_data_path(), FS)
    except Exception as e:
        return [("PTB-XL folder", False, str(e))]
    out = [("PTB-XL folder", True, dp)]
    cp = _cache_path(dp, LABEL_SCHEME, DIAG_LEVEL, USE_FILTERED, FS)
    if os.path.exists(cp) and os.path.getsize(cp) > 0:
        out.append(("signal cache", True, f"{cp} ({os.path.getsize(cp) / 1e9:.1f} GB)"))
    else:
        out.append(("signal cache", False, f"missing {cp}: run `python prepare_data.py` first, "
                                           f"or every tuning job rebuilds it inside its time limit"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="only print/save the best cell of the saved study (no training)")
    ap.add_argument("--check", action="store_true",
                    help="login-node check: environment, data caches, storage, model, Optuna (no training)")
    args = ap.parse_args()

    if MODEL_VARIANT not in FlatSeqAttention.VARIANTS:
        raise ValueError(f"ECG_MODEL_VARIANT={MODEL_VARIANT!r} invalid "
                         f"(flat variants: {FlatSeqAttention.VARIANTS})")
    if LABEL_SCHEME not in ("diagnostic", "rhythm", "form"):
        raise ValueError(f"ECG_LABEL_SCHEME={LABEL_SCHEME!r} invalid")
    if UNIFORM:
        raise ValueError("ECG_UNIFORM_ATTENTION is set: only the normal "
                         "(learned-attention) model is tuned; unset it.")

    scheme_tag = f"diagnostic-{DIAG_LEVEL}" if LABEL_SCHEME == "diagnostic" else LABEL_SCHEME
    tag = (f"flat_{MODEL_VARIANT}_{scheme_tag}" + ("" if USE_FILTERED else "_all")
           + method_tag(TASC))
    meta = {"family": "flat", "variant": MODEL_VARIANT, "scheme": LABEL_SCHEME,
            "diagnostic_level": DIAG_LEVEL if LABEL_SCHEME == "diagnostic" else None,
            "preprocessing": PREP_TAG, "token_samples": TOKEN_LEN,
            "arch_fixed": ARCH[MODEL_VARIANT], "tasc": TASC,
            "paste_into": f"{'TUNED_TASC' if TASC else 'TUNED'} in train_flat_seq.py"}

    if args.check:
        ok = tc.run_check(
            tag, meta,
            build_model=lambda: FlatSeqAttention(num_classes=5, variant=MODEL_VARIANT, dropout=0.3,
                                                 tasc=TASC, **ARCH[MODEL_VARIANT]),
            dummy_input=lambda: torch.randn(2, 12, 5000),
            data_checks=data_checks)
        sys.exit(0 if ok else 1)

    study, storage = tc.make_study(tag)
    if args.report:
        tc.report(study, tag, meta)
        return
    if tc.nothing_to_do(study, tag, meta):    # finished study / no time: skip loading data
        return

    torch.backends.cudnn.benchmark = True
    data_path = resolve_data_path()
    print(f"{'='*64}")
    print(f"  tune_flat_seq.py  (MULTI-LABEL / macro-AUC)")
    print(f"  study={tag}  storage={storage or 'in-memory (not resumable)'}")
    print(f"  preprocessing={PREP_TAG}  tokens={TOKEN_LEN} samples")
    print(f"  arch(FIXED)={ARCH[MODEL_VARIANT]}  tasc={TASC or 'none'}")
    print(f"  trials<={tc.TUNE_TRIALS}  epochs/trial<={tc.TUNE_EPOCHS}  seed={tc.TUNE_SEED}")
    print(f"  early stopping: trial patience={tc.PATIENCE} epochs, study patience="
          f"{tc.STUDY_PATIENCE} trials (after {tc.MIN_TRIALS}), min_delta={tc.MIN_DELTA}")
    print(f"  time budget={f'{tc.TIMEOUT_MIN:.1f} min' if tc.DEADLINE else 'none'}")
    print(f"{'='*64}", flush=True)

    dataset, num_classes, _label2idx, train_idx, val_idx = load_and_split(data_path)
    T, n_tokens, params = check_setup(dataset, train_idx, num_classes)   # same check as the trainer
    print(f"[check] T={T} -> {n_tokens} tokens; params={params['total']:,}")
    train_labels = np.stack([np.asarray(dataset.data[i][1], dtype=np.float32)
                             for i in train_idx], axis=0)
    print(f"num_classes={num_classes}  train={len(train_idx)}  val={len(val_idx)}\n",
          flush=True)

    objective = make_objective(dataset, num_classes, train_idx, val_idx, train_labels)
    tc.optimize(study, objective, tag, meta)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
