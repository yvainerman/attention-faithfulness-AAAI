"""
collect_tuned.py
----------------
Print every tuned cell saved in tune_results/*.json, grouped by the table it
belongs in (TUNED / TUNED_TASC in train_flat_seq.py / train_hierarchical_seq.py),
ready to paste. The tune scripts rewrite these JSONs after every finished trial,
so this also works while studies are still running or after a job hit its time
limit (unfinished studies are flagged).

RUN   python collect_tuned.py                 # all studies
      python collect_tuned.py --tasc-only     # only the TaSc studies
"""
import argparse
import glob
import json
import os
from collections import defaultdict

FAMILY_ORDER = {"flat": ("noenc", "conv", "bilstm"),
                "hier": ("noenc", "conv", "bilstm_cnn", "bilstm_full")}
SCHEME_ORDER = ("diagnostic", "rhythm", "form")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tune_results")
    ap.add_argument("--tasc-only", action="store_true")
    args = ap.parse_args()

    recs = []
    for p in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        if "tuned_cell" not in r or "paste_into" not in r:
            continue                                   # not written by the current tuners
        if args.tasc_only and not r.get("tasc"):
            continue
        recs.append(r)
    if not recs:
        print(f"no tune results in {args.dir}/")
        return

    groups = defaultdict(list)
    for r in recs:
        groups[r["paste_into"]].append(r)
    for table in sorted(groups):
        rows = sorted(groups[table], key=lambda r: (
            SCHEME_ORDER.index(r["scheme"]) if r["scheme"] in SCHEME_ORDER else 9,
            FAMILY_ORDER.get(r["family"], ()).index(r["variant"])
            if r["variant"] in FAMILY_ORDER.get(r["family"], ()) else 9))
        print(f"\n# ---- paste into {table} ----")
        for r in rows:
            status = r.get("stopped") or "UNFINISHED (resubmit to continue)"
            print(f"    {r['tuned_cell']}   # AUC={r['best_macro_auc']:.4f}, "
                  f"{r.get('n_complete', '?')}/{r.get('n_trials', '?')} trials complete, {status}")
    for tasc in sorted({bool(r.get("tasc")) for r in recs}):
        missing = [f"{fam}/{v}/{s}"
                   for fam, variants in FAMILY_ORDER.items() for v in variants for s in SCHEME_ORDER
                   if not any(bool(r.get("tasc")) == tasc and r["family"] == fam
                              and r["variant"] == v and r["scheme"] == s for r in recs)]
        if missing:
            print(f"\n# {'TaSc' if tasc else 'original'} models with no saved result yet "
                  f"({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
