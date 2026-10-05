#!/usr/bin/env python3
# Layer 2: pick the chips and split them for LODO.
#
# manifest/chips.csv holds facts; this file holds decisions - budget, bin edges,
# allocation, seed - so revising them produces selection_v2 beside v1 rather than
# rewriting the census. Sampling is disproportionate (we oversample damaged chips),
# so every chip carries the weight needed to recover natural-distribution metrics.

import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "manifest"
EDGES = [0.0, 1e-12, 0.25, 0.50, 1.01]
BINS = ["0", "(0,.25]", "(.25,.5]", "(.5,1]"]
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]


def allocate(avail, budget):
    # equal share across bins that have data; a bin that cannot meet its share takes all
    # it has and spills the rest to the others, repeated until nothing more can move
    alloc = {b: 0.0 for b in avail}
    pool = {b for b in avail if avail[b] > 0}
    left = float(budget)
    while pool and left > 1e-9:
        share = left / len(pool)
        short = [b for b in pool if avail[b] - alloc[b] <= share]
        if not short:
            for b in pool:
                alloc[b] += share
            break
        for b in short:
            left -= avail[b] - alloc[b]
            alloc[b] = float(avail[b])
            pool.discard(b)
    return alloc


def pick(df, budget, rng):
    # whole chips only - the Delaunay graph is built per chip, so sampling buildings
    # would silently change the topology (design-notes §3)
    avail = {b: int(df[df.bin == b].n_polygons.sum()) for b in BINS}
    target = allocate(avail, budget)
    keep, stats = [], {}
    for b in BINS:
        sub = df[df.bin == b]
        if not len(sub) or target[b] <= 0:
            stats[b] = dict(n_chips=0, N_chips=len(sub), buildings=0, target=round(target[b]))
            continue
        sub = sub.sample(frac=1, random_state=rng.integers(2**31)).copy()
        take = sub[sub.n_polygons.cumsum() <= target[b]]
        if not len(take):
            take = sub.iloc[:1]
        keep.append(take)
        stats[b] = dict(n_chips=len(take), N_chips=len(sub),
                        buildings=int(take.n_polygons.sum()), target=round(target[b]))
    return pd.concat(keep), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=20000, help="buildings per fold")
    p.add_argument("--val-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=760)
    p.add_argument("--version", default="v1")
    a = p.parse_args()

    df = pd.read_csv(MAN / "chips.csv")
    df = df[df.usable].copy()
    df["bin"] = pd.cut(df.damage_ratio, bins=EDGES, labels=BINS, include_lowest=True)

    rng = np.random.default_rng(a.seed)
    sel, allstats = [], {}
    for fold in FOLDS:
        part, allstats[fold] = pick(df[df.fold == fold], a.budget, rng)
        sel.append(part)
    sel = pd.concat(sel).reset_index(drop=True)

    # weight = how many chips this one stands for in its (fold, bin) stratum, so a
    # weighted confusion matrix gives the natural-distribution estimate
    sel["weight"] = [allstats[f][b]["N_chips"] / allstats[f][b]["n_chips"]
                     for f, b in zip(sel.fold, sel.bin)]

    # six LODO runs: held-out fold is test, 12.5% of the rest is the inner validation
    # split used for early stopping. The test fold is never touched by either.
    for k, held in enumerate(FOLDS):
        role = pd.Series("train", index=sel.index)
        role[sel.fold == held] = "test"
        pool = sel[sel.fold != held]
        r = np.random.default_rng(a.seed + 1000 + k)
        for _, grp in pool.groupby(["fold", "bin"], observed=True):
            n = int(round(a.val_frac * len(grp)))
            if n:
                role[r.choice(grp.index, size=n, replace=False)] = "val"
        sel[f"role_{held}"] = role

    cols = (["chip_id", "fold", "bin", "n_polygons", "n_labelled", "n_damaged", "weight"]
            + [f"role_{f}" for f in FOLDS])
    out = MAN / f"selection_{a.version}.csv"
    sel[cols].sort_values(["fold", "chip_id"]).to_csv(out, index=False)

    summary = {}
    for fold in FOLDS:
        g = sel[sel.fold == fold]
        summary[fold] = dict(
            chips=len(g), buildings=int(g.n_polygons.sum()),
            labelled=int(g.n_labelled.sum()), damaged=int(g.n_damaged.sum()),
            damage_rate=round(float(g.n_damaged.sum() / g.n_labelled.sum()), 4),
            bins={b: allstats[fold][b] for b in BINS})
    (MAN / f"selection_{a.version}.json").write_text(json.dumps({
        "budget_per_fold": a.budget, "bin_edges": EDGES, "allocation": "equal share, shortfall redistributed",
        "val_frac": a.val_frac, "seed": a.seed,
        "chips_csv_sha256": hashlib.sha256((MAN / "chips.csv").read_bytes()).hexdigest(),
        "selection_csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "note": "sigma_init per run is derived separately from training-fold assortativity",
        "per_fold": summary}, indent=2) + "\n")

    print(f"{'fold':10} {'chips':>6} {'buildings':>10} {'damaged':>8} {'rate':>6}   bins (chips kept/available)")
    for fold in FOLDS:
        s = summary[fold]
        b = " ".join(f"{allstats[fold][x]['n_chips']}/{allstats[fold][x]['N_chips']}" for x in BINS)
        print(f"{fold:10} {s['chips']:>6,} {s['buildings']:>10,} {s['damaged']:>8,} "
              f"{100*s['damage_rate']:>5.1f}%   {b}")
    print(f"\n{'TOTAL':10} {len(sel):>6,} {sel.n_polygons.sum():>10,}")
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
