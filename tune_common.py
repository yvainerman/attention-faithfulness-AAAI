"""
tune_common.py
--------------
Shared Optuna machinery for tune_flat_seq.py and tune_hierarchical_seq.py:
early stopping (inside a trial AND across the study), a resumable saved study,
a wall-clock deadline for short SLURM jobs, and the best cell saved after EVERY
trial. The search space, metric (validation macro-AUC) and training step are
unchanged; they live in the two tune scripts.

TIME SAVERS (all env-overridable)
  ECG_TUNE_EPOCHS        40     max epochs per trial (also the cosine schedule length)
  ECG_TUNE_PATIENCE       6     TRIAL early stopping: stop a trial after this many epochs
                                without a validation macro-AUC gain >= ECG_TUNE_MIN_DELTA.
                                The trial COMPLETES with its best AUC, so it stays a candidate
                                for the best configuration (a pruned trial would not).
  ECG_TUNE_MIN_DELTA     0.001  smallest macro-AUC gain that counts as an improvement
                                (used by both early-stopping levels)
  MedianPruner                  (unchanged) prunes a trial that is below the median of the
                                finished trials after 5 epochs
  ECG_TUNE_STUDY_PATIENCE 10    STUDY early stopping (Optuna callback -> study.stop()): stop
                                tuning after this many finished trials in a row without the
                                best macro-AUC improving by >= ECG_TUNE_MIN_DELTA ...
  ECG_TUNE_MIN_TRIALS    15     ... but never before this many finished trials (TPE's first
                                10 trials are random, so stopping earlier would be blind)
  ECG_TUNE_TRIALS        50     cap on finished trials IN THE STUDY. With a saved study this
                                counts the trials of earlier jobs, so a resubmitted job only
                                tops the study up instead of adding another full budget.
  ECG_TUNE_TIMEOUT_MIN   (none) wall-clock budget of THIS job, in minutes (the SLURM scripts
                                set it from the job's remaining time). No epoch is started that
                                would not finish in time; the cut trial is marked FAIL (never
                                used as a result) and the job exits cleanly after saving.
SAVING / RESUMING
  ECG_TUNE_STORAGE       (none) Optuna storage URL. "auto" = sqlite:///tune_results/<tag>.db,
                                one file per study (one writer per file, safe on a shared
                                filesystem). With storage, resubmitting the same job RESUMES
                                the study; a study that has already early-stopped or reached
                                its cap is only re-reported.
  tune_results/<tag>.json       rewritten after every finished trial: best params, the
                                paste-ready TUNED / TUNED_TASC cell, and progress.
  --report                      print + save the best cell from the saved study without
                                loading data or training (e.g. after a job hit its time limit).
"""
import json
import math
import os
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState


def _envint(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _envfloat(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


TUNE_TRIALS   = _envint("ECG_TUNE_TRIALS", 50)
TUNE_EPOCHS   = _envint("ECG_TUNE_EPOCHS", 40)
TUNE_SEED     = _envint("ECG_TUNE_SEED", 42)
NUM_WORKERS   = _envint("ECG_NUM_WORKERS", 4)
PATIENCE      = _envint("ECG_TUNE_PATIENCE", 6)
MIN_DELTA     = _envfloat("ECG_TUNE_MIN_DELTA", 0.001)
STUDY_PATIENCE = _envint("ECG_TUNE_STUDY_PATIENCE", 10)
MIN_TRIALS    = _envint("ECG_TUNE_MIN_TRIALS", 15)
TIMEOUT_MIN   = _envfloat("ECG_TUNE_TIMEOUT_MIN", 0.0)
STORAGE_RAW   = os.environ.get("ECG_TUNE_STORAGE") or ""

OUT_DIR = "tune_results"
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)

JOB_START = time.time()
DEADLINE = JOB_START + 60.0 * TIMEOUT_MIN if TIMEOUT_MIN > 0 else None


class DeadlineReached(Exception):
    """Raised inside a trial when the job's time budget would be exceeded."""


# ---------------------------------------------------------------- epoch clock
_max_epoch_s = 0.0      # slowest epoch seen in this job (estimate for the next one)


def _check_deadline():
    if DEADLINE is None:
        return
    if time.time() + 1.2 * _max_epoch_s > DEADLINE:
        raise DeadlineReached(
            f"time budget ({TIMEOUT_MIN:.1f} min) would be exceeded by the next epoch "
            f"(~{_max_epoch_s:.0f} s)")


def run_trial_epochs(trial, model, train_loader, val_loader, criterion, optimizer,
                     scheduler, train_epoch, validate):
    """The epoch loop of ONE trial: train, validate (macro-AUC), report to the
    pruner, stop early on a plateau, and respect the job deadline.
    Returns the best validation macro-AUC of the trial."""
    global _max_epoch_s
    best_auc, best_epoch = -1.0, 0
    ref_auc, wait = -math.inf, 0              # reference for patience (Keras-style)
    stop = "max_epochs"
    epochs_run = 0
    for epoch in range(TUNE_EPOCHS):
        _check_deadline()
        t0 = time.time()
        train_epoch(model, train_loader, criterion, optimizer)
        _vl, auc, *_ = validate(model, val_loader, criterion)   # macro-AUC
        scheduler.step()
        _max_epoch_s = max(_max_epoch_s, time.time() - t0)
        epochs_run = epoch + 1
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1
        if auc > ref_auc + MIN_DELTA:
            ref_auc, wait = auc, 0
        else:
            wait += 1
        trial.report(auc, epoch)
        if trial.should_prune():
            trial.set_user_attr("epochs_run", epochs_run)
            raise optuna.TrialPruned()
        if PATIENCE > 0 and wait >= PATIENCE:
            stop = "patience"
            break
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("epochs_run", epochs_run)
    trial.set_user_attr("stop", stop)
    return best_auc


# ---------------------------------------------------------------- study state
def _finished(study):
    return sorted(study.get_trials(deepcopy=False, states=FINISHED), key=lambda t: t.number)


# Study-level stopping settings are stored IN the saved study, so a later
# `--report` (or a resubmission) without the env vars judges the study by the
# settings it was tuned with. An env var that IS set always wins.
_STOP_SETTINGS = {"trial_cap":      ("ECG_TUNE_TRIALS", TUNE_TRIALS),
                  "study_patience": ("ECG_TUNE_STUDY_PATIENCE", STUDY_PATIENCE),
                  "min_trials":     ("ECG_TUNE_MIN_TRIALS", MIN_TRIALS),
                  "min_delta":      ("ECG_TUNE_MIN_DELTA", MIN_DELTA)}


def stop_settings(study):
    stored = study.user_attrs.get("stop_settings", {})
    return {k: (val if os.environ.get(env) not in (None, "") or k not in stored else stored[k])
            for k, (env, val) in _STOP_SETTINGS.items()}


def early_stop_status(study):
    """(stop?, trials since the last real improvement, best value) over the
    finished trials of the WHOLE saved study (so it survives resubmission)."""
    st = stop_settings(study)
    done = _finished(study)
    ref, last_improve, best = -math.inf, -1, None
    for i, t in enumerate(done):
        if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value):
            best = t.value if best is None else max(best, t.value)
            if t.value > ref + st["min_delta"]:
                ref, last_improve = t.value, i
    since = len(done) - 1 - last_improve
    stop = (st["study_patience"] > 0 and len(done) >= st["min_trials"] and best is not None
            and since >= st["study_patience"])
    return stop, since, best


def stop_reason(study):
    st = stop_settings(study)
    if len(_finished(study)) >= st["trial_cap"]:
        return "trial cap reached"
    stop, since, _ = early_stop_status(study)
    if stop:
        return f"early stopping: no gain >= {st['min_delta']} in the last {since} trials"
    return None


def resolve_storage(tag):
    if STORAGE_RAW.strip().lower() == "auto":
        os.makedirs(OUT_DIR, exist_ok=True)
        return f"sqlite:///{OUT_DIR}/{tag}.db"
    return STORAGE_RAW or None


def make_study(tag):
    """Create, or load and resume, the study. On resume the sampler is re-seeded
    with seed + #trials, so TPE's random start-up trials are not repeated."""
    storage = resolve_storage(tag)
    kw = dict(direction="maximize", sampler=TPESampler(seed=TUNE_SEED),
              pruner=MedianPruner(n_warmup_steps=5))
    if storage:
        study = optuna.create_study(study_name=tag, storage=storage, load_if_exists=True, **kw)
    else:
        study = optuna.create_study(study_name=tag, **kw)
    n = len(study.trials)
    if n:
        study.sampler = TPESampler(seed=TUNE_SEED + n)
    return study, storage


# ---------------------------------------------------------------- reporting
def _counts(study):
    states = [t.state for t in study.trials]
    return {"n_trials": len(states),
            "n_complete": states.count(TrialState.COMPLETE),
            "n_pruned": states.count(TrialState.PRUNED),
            "n_fail": states.count(TrialState.FAIL),
            "n_running_or_cut": states.count(TrialState.RUNNING)}


def best_record(study, tag, meta):
    """Everything worth saving about the study right now (None if nothing completed)."""
    c = _counts(study)
    if c["n_complete"] == 0:
        return None
    best = study.best_trial
    cell = ", ".join(f"{k}={v!r}" for k, v in best.params.items())
    paste = f'("{meta["scheme"]}", "{meta["variant"]}"): dict({cell}),'
    stop, since, _ = early_stop_status(study)
    return {
        **meta,
        "study": tag,
        "best_macro_auc": float(best.value),
        "best_trial": best.number,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "best_params": best.params,
        "tuned_cell": paste,
        "paste_into": meta["paste_into"],
        **c,
        "trials_since_improvement": since,
        "stopped": stop_reason(study),
        "tune_epochs": TUNE_EPOCHS, "seed": TUNE_SEED,
        "early_stopping": {"trial_patience": PATIENCE, **stop_settings(study)},
    }


def save_record(rec, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{tag}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)                        # never leaves a half-written file
    return path


def report(study, tag, meta):
    """Print the BEST block (with the paste-ready cell) and save the JSON."""
    rec = best_record(study, tag, meta)
    c = _counts(study)
    print(f"\n{'='*64}")
    if rec is None:
        print(f"  {tag}: no completed trial yet ({c['n_trials']} trials, {c['n_pruned']} pruned, "
              f"{c['n_fail']} failed) -- nothing to report")
        print(f"{'='*64}", flush=True)
        return None
    print(f"  BEST  {tag}")
    print(f"  macro-AUC = {rec['best_macro_auc']:.4f}  (trial {rec['best_trial']}, "
          f"epoch {rec['best_epoch']}/{TUNE_EPOCHS})")
    for k, v in rec["best_params"].items():
        print(f"    {k:14s}: {v}")
    print(f"  trials: {c['n_trials']} total, {c['n_complete']} complete, {c['n_pruned']} pruned, "
          f"{c['n_fail']} failed/cut")
    print(f"  status: {rec['stopped'] or 'NOT finished -- resubmit the same job to continue'}")
    print(f"\n  Paste into {meta['paste_into']}:\n    {rec['tuned_cell']}")
    path = save_record(rec, tag)
    print(f"  saved -> {path}")
    print(f"{'='*64}", flush=True)
    return path


def _progress_callback(tag, meta):
    def cb(study, trial):
        rec = best_record(study, tag, meta)
        if rec is not None:
            save_record(rec, tag)                # the best cell is on disk after every trial
        best = f"{rec['best_macro_auc']:.4f} (trial {rec['best_trial']})" if rec else "n/a"
        val = f"{trial.value:.4f}" if trial.value is not None else "-"
        print(f"  [trial {trial.number:3d}] {trial.state.name:8s} auc={val}  "
              f"epochs={trial.user_attrs.get('epochs_run', '?')} "
              f"stop={trial.user_attrs.get('stop', 'pruned' if trial.state == TrialState.PRUNED else '?')}  "
              f"| best {best} | finished {len(_finished(study))}/{stop_settings(study)['trial_cap']}",
              flush=True)
    return cb


def _stop_callback(study, trial):
    reason = stop_reason(study)
    if reason:
        print(f"  [study] stopping: {reason}", flush=True)
        study.stop()


def nothing_to_do(study, tag, meta):
    """Called BEFORE the data are loaded: if the saved study is already finished
    (cap reached or early-stopped) or this job has no time budget left, report
    and return True, so resubmitting a whole array costs seconds per done study."""
    reason = stop_reason(study)
    if reason:
        print(f"[tune] {tag}: study already finished ({reason}); reporting only.", flush=True)
        report(study, tag, meta)
        return True
    if DEADLINE is not None and DEADLINE - time.time() <= 60:
        print(f"[tune] {tag}: no time left in this job's budget; reporting only.", flush=True)
        report(study, tag, meta)
        return True
    return False


def optimize(study, objective, tag, meta):
    """Run the remaining trials, then report. Returns the JSON path."""
    if nothing_to_do(study, tag, meta):           # (re-checked: loading data takes minutes)
        return None
    study.set_user_attr("stop_settings", stop_settings(study))   # remembered for --report
    remaining = stop_settings(study)["trial_cap"] - len(_finished(study))
    timeout = None if DEADLINE is None else DEADLINE - time.time()
    print(f"[tune] {tag}: {len(study.trials)} saved trials, up to {remaining} more this job"
          + (f", {timeout/60:.1f} min of budget left" if timeout else ""), flush=True)
    try:
        study.optimize(objective, n_trials=remaining, timeout=timeout, gc_after_trial=True,
                       callbacks=[_progress_callback(tag, meta), _stop_callback])
    except DeadlineReached as e:
        print(f"[tune] {tag}: {e}; the cut trial is marked FAIL. Resubmit the same job "
              f"to continue the saved study.", flush=True)
    return report(study, tag, meta)
