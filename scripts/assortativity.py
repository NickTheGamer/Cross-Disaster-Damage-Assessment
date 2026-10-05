#!/usr/bin/env python3
# Is damage spatially autocorrelated? Measures label agreement across Delaunay edges,
# per fold and as a function of edge length. This tests the premise the whole graph
# approach rests on, and needs no GPU and no model.
#
# Distances are in pixels; effective imagery scale is roughly 0.45 m/px.

import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_dataset"
DAMAGE = ("no-damage", "minor-damage", "major-damage", "destroyed")
LABEL = {c: i for i, c in enumerate(DAMAGE)}
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
DIST_BINS = [0, 25, 50, 100, 200, 400, 10**9]
FINE = list(range(0, 401, 20)) + [10**9]
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]


def chip_edges(feats):
    cen = np.empty((len(feats), 2))
    for i, ft in enumerate(feats):
        n = np.fromiter(map(float, NUM.findall(ft["wkt"])), float)
        xs, ys = n[0::2], n[1::2]
        cen[i] = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    uniq, inv = np.unique(cen, axis=0, return_inverse=True)
    if len(uniq) < 3:
        return cen, []
    try:
        simp = Delaunay(uniq).simplices
    except (QhullError, ValueError):
        return cen, []
    members = [np.flatnonzero(inv == u) for u in range(len(uniq))]
    pairs = {(min(a, b), max(a, b)) for s in simp
             for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2]))}
    return cen, [(i, j) for a, b in pairs for i in members[a] for j in members[b]]


def newman_r(mix):
    # categorical assortativity: 0 is chance, 1 is perfect agreement
    tot = mix.sum()
    if tot == 0:
        return float("nan")
    e = mix / tot
    a = e.sum(axis=1)
    denom = 1 - (a * a).sum()
    return float((np.trace(e) - (a * a).sum()) / denom) if denom > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", help="restrict to the chips in a selection csv")
    ap.add_argument("--write-sigma", help="write per-LODO-run sigma into this selection json")
    args = ap.parse_args()

    usable = {r["chip_id"]: (r["fold"], r["split"])
              for r in csv.DictReader(open(ROOT / "manifest" / "chips.csv"))
              if r["usable"] == "True"}
    if args.selection:
        keep = {r["chip_id"] for r in csv.DictReader(open(args.selection))}
        usable = {k: v for k, v in usable.items() if k in keep}
        print(f"restricted to {len(usable):,} selected chips", file=sys.stderr)

    mix4 = defaultdict(lambda: np.zeros((4, 4)))        # 4-class mixing matrix per fold
    mix2 = defaultdict(lambda: np.zeros((2, 2)))        # damaged / not
    bins = defaultdict(lambda: np.zeros((len(DIST_BINS) - 1, 2)))   # [same, total] per bin
    fine = defaultdict(lambda: np.zeros((len(FINE) - 1, 2)))
    node_agree = defaultdict(list)                      # per-node share of agreeing neighbours
    lens = defaultdict(list)

    for k, (cid, (fold, split)) in enumerate(usable.items(), 1):
        feats = json.loads((RAW / split / "labels" / f"{cid}_post_disaster.json").read_text())["features"]["xy"]
        cen, edges = chip_edges(feats)
        if not edges:
            continue
        y = np.array([LABEL.get(f["properties"]["subtype"], -1) for f in feats])
        nb = defaultdict(list)
        for i, j in edges:
            if y[i] < 0 or y[j] < 0:            # un-classified has no label to compare
                continue
            d = float(np.hypot(*(cen[i] - cen[j])))
            for m, n in ((i, j), (j, i)):       # both directions, symmetric matrix
                mix4[fold][y[m], y[n]] += 1
                mix2[fold][int(y[m] > 0), int(y[n] > 0)] += 1
            b = np.searchsorted(DIST_BINS, d, side="right") - 1
            bins[fold][b] += (y[i] == y[j], 1)
            fine[fold][np.searchsorted(FINE, d, side="right") - 1] += (y[i] == y[j], 1)
            lens[fold].append(d)
            nb[i].append(y[j]); nb[j].append(y[i])
        for i, ns in nb.items():
            node_agree[fold].append(np.mean([n == y[i] for n in ns]))
        if k % 1500 == 0:
            print(f"  {k:,}/{len(usable):,}", file=sys.stderr)

    folds = sorted(mix4)
    print(f"\n=== label agreement across Delaunay edges ({sum(int(m.sum()//2) for m in mix4.values()):,} edges) ===\n")
    print(f"{'fold':10} {'edges':>8} {'P(same)':>8} {'chance':>8} {'lift':>6} "
          f"{'r (4-cls)':>10} {'r (binary)':>11} {'med len':>8}")
    for f in folds:
        m = mix4[f]
        e = m / m.sum()
        obs, chance = np.trace(e), (e.sum(axis=1) ** 2).sum()
        print(f"{f:10} {int(m.sum()//2):>8,} {obs:>8.3f} {chance:>8.3f} {obs/chance:>5.2f}x "
              f"{newman_r(m):>10.3f} {newman_r(mix2[f]):>11.3f} {np.median(lens[f]):>7.0f}px")

    print(f"\n=== P(same class) by edge length — does agreement decay with distance? ===\n")
    hdr = [f"{DIST_BINS[i]}-{DIST_BINS[i+1] if DIST_BINS[i+1] < 10**8 else ''}" for i in range(len(DIST_BINS) - 1)]
    print(f"{'fold':10} " + " ".join(f"{h:>9}" for h in hdr))
    for f in folds:
        row = []
        for same, tot in bins[f]:
            row.append(f"{same/tot:>9.3f}" if tot >= 50 else f"{'-':>9}")
        print(f"{f:10} " + " ".join(row))

    print(f"\n=== excess agreement over chance, by distance (normalised to the 0-20px bin) ===\n")
    print(f"{'fold':10} " + " ".join(f"{FINE[i]:>5}" for i in range(0, 20, 2)) + "   half-at   sigma")
    for f in folds:
        e = mix4[f] / mix4[f].sum()
        chance = (e.sum(axis=1) ** 2).sum()
        obs = np.array([s_ / t if t >= 200 else np.nan for s_, t in fine[f]])
        exc = obs - chance
        base = exc[0]
        norm = exc / base if base > 0 else exc * np.nan
        row = " ".join(f"{norm[i]:>5.2f}" if np.isfinite(norm[i]) else f"{'-':>5}" for i in range(0, 20, 2))
        half = next((FINE[i] for i in range(len(norm)) if np.isfinite(norm[i]) and norm[i] < 0.5), None)
        sig = f"{half/np.log(2):>7.0f}px" if half else f"{'>400':>9}"
        print(f"{f:10} {row}   {str(half)+'px' if half else '>400px':>7} {sig}")

    print(f"\n=== what fraction of edges are long enough for weighting to matter? ===\n")
    print(f"{'fold':10} {'>50px':>7} {'>100px':>7} {'>200px':>7} {'med':>6}")
    for f in folds:
        a = np.array(lens[f])
        print(f"{f:10} {100*(a>50).mean():>6.1f}% {100*(a>100).mean():>6.1f}% "
              f"{100*(a>200).mean():>6.1f}% {np.median(a):>5.0f}px")

    if args.write_sigma:
        # sigma per fold from the measured decay; folds with no spatial structure carry no
        # information about a correlation length, so they are left out of the estimate
        sig = {}
        for f in folds:
            e = mix4[f] / mix4[f].sum()
            chance, obs = (e.sum(axis=1) ** 2).sum(), np.trace(e)
            if obs / chance < 1.05:
                continue
            exc = np.array([s_ / t - chance if t >= 200 else np.nan for s_, t in fine[f]])
            half = next((FINE[i] for i in range(len(exc))
                         if np.isfinite(exc[i]) and exc[i] < 0.5 * exc[0]), None)
            sig[f] = float((half or FINE[-2]) / np.log(2))      # censored -> last real bin edge
        doc = json.loads(Path(args.write_sigma).read_text())
        # each run's init uses only the folds it is allowed to see
        doc["sigma_init_px"] = {h: round(float(np.median([v for k, v in sig.items() if k != h])), 1)
                                for h in FOLDS}
        doc["sigma_per_fold_px"] = {k: round(v, 1) for k, v in sig.items()}
        doc["sigma_excluded"] = [f for f in folds if f not in sig]
        Path(args.write_sigma).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\n=== sigma written to {args.write_sigma} ===")
        print("  per fold :", {k: round(v) for k, v in sig.items()})
        print("  per run  :", doc["sigma_init_px"])
        print("  excluded :", doc["sigma_excluded"])

    print(f"\n=== per-node neighbour agreement — is there heterogeneity for a gate to exploit? ===\n")
    print(f"{'fold':10} {'nodes':>8} {'mean':>6} {'all agree':>10} {'none agree':>11} {'mixed':>7}")
    for f in folds:
        a = np.array(node_agree[f])
        print(f"{f:10} {len(a):>8,} {a.mean():>6.3f} {100*(a == 1).mean():>9.1f}% "
              f"{100*(a == 0).mean():>10.1f}% {100*((a > 0) & (a < 1)).mean():>6.1f}%")


if __name__ == "__main__":
    main()
