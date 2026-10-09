#!/usr/bin/env python3
# In-domain splits: train and test within a single disaster.
#
# Same column shape as selection_v1.csv, so train.py needs no changes - role_<fold> is
# train/val/test for that fold's own chips and "unused" for every other fold. Then
# --held-out fire --selection manifest/indomain_v1.csv trains and tests inside fire.
#
# The point of comparison: cross-disaster trains on ~82k buildings from five folds,
# in-domain on ~14k from one. The in-domain model gets ~6x less data, so if it still
# wins, the domain effect is larger than measured, not smaller.

import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "manifest"
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=761)
    p.add_argument("--version", default="v1")
    a = p.parse_args()

    sel = pd.read_csv(MAN / "selection_v1.csv")
    lodo = json.loads((MAN / "selection_v1.json").read_text())

    for k, fold in enumerate(FOLDS):
        role = pd.Series("unused", index=sel.index)
        own = sel[sel.fold == fold]
        rng = np.random.default_rng(a.seed + k)
        # stratify by damage bin so the test split keeps the fold's own damage profile
        for _, grp in own.groupby("bin", observed=True):
            idx = rng.permutation(grp.index.to_numpy())
            n = len(idx)
            ntr = max(1, int(round(a.train_frac * n)))
            nva = max(1, int(round(a.val_frac * n))) if n >= 3 else 0
            ntr = min(ntr, n - nva - 1) if n - nva - 1 >= 1 else ntr
            role[idx[:ntr]] = "train"
            role[idx[ntr:ntr + nva]] = "val"
            role[idx[ntr + nva:]] = "test"
        sel[f"role_{fold}"] = role

    cols = (["chip_id", "fold", "bin", "n_polygons", "n_labelled", "n_damaged", "weight"]
            + [f"role_{f}" for f in FOLDS])
    out = MAN / f"indomain_{a.version}.csv"
    sel[cols].sort_values(["fold", "chip_id"]).to_csv(out, index=False)

    # sigma per fold from that fold's own measured decay (unused by mlp/gate, but train.py
    # looks it up regardless, and it would be the right value if dist is run in-domain)
    per_fold = lodo.get("sigma_per_fold_px", {})
    med = float(np.median(list(per_fold.values()))) if per_fold else 400.0
    sigma = {f: round(per_fold.get(f, med), 1) for f in FOLDS}

    (MAN / f"indomain_{a.version}.json").write_text(json.dumps({
        "kind": "in-domain", "train_frac": a.train_frac, "val_frac": a.val_frac,
        "seed": a.seed, "stratified_by": "damage bin, within fold",
        "note": "role_<fold> covers only that fold's chips; all others are 'unused'. "
                "Training sets are ~6x smaller than the LODO runs - a confound that makes "
                "any in-domain advantage conservative.",
        "selection_csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "sigma_init_px": sigma, "sigma_source": "per-fold decay from selection_v1.json",
    }, indent=2) + "\n")

    print(f"{'fold':11} {'chips':>18} {'buildings (train)':>18} {'test bldg':>10} {'test dmg%':>10}")
    for f in FOLDS:
        g = sel[sel.fold == f]
        r = g[f"role_{f}"]
        tr, va, te = (r == "train"), (r == "val"), (r == "test")
        tb = int(g[te].n_labelled.sum())
        td = int(g[te].n_damaged.sum())
        print(f"{f:11} {f'{tr.sum()}/{va.sum()}/{te.sum()}':>18} "
              f"{int(g[tr].n_polygons.sum()):>18,} {tb:>10,} {100*td/max(tb,1):>9.1f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
