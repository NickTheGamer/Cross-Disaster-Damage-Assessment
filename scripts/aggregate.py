#!/usr/bin/env python3
# Turn the grid's per-run JSON into the tables the report needs.
#
# Fire is the dev fold - weights, epoch ceiling and patience were chosen on it, so it is
# reported but excluded from the clean mean. Macro-F1 is averaged over the five clean
# folds; per-fold n is always shown, because a fold with 44 major-damage buildings and
# one with 3,709 do not deserve equal reading.

import argparse, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]
ARMS = ["mlp", "sage", "gate", "dist"]
DEV = "fire"


def gm(r):
    return "-" if "gate_mean" not in r else f'{r["gate_mean"]:.3f}'


def load(out, weights, seed):
    R = {}
    for f in Path(out).glob(f"*_{weights}_s{seed}.json"):
        r = json.loads(f.read_text())
        R[(r["arm"], r["held_out"])] = r
    return R


def table(R, view, key="macro_f1"):
    rows = []
    for arm in ARMS:
        cells, clean = [], []
        for f in FOLDS:
            r = R.get((arm, f))
            if r is None:
                cells.append(None); continue
            v = r["test"][view][key]
            cells.append(v)
            if f != DEV:
                clean.append(v)
        rows.append((arm, cells, sum(clean) / len(clean) if clean else None))
    return rows


def show(R, view, title, key="macro_f1"):
    print(f"\n=== {title} ===")
    print(f"{'arm':6} " + "".join(f"{f[:9]:>10}" for f in FOLDS) + f"{'clean mean':>12}")
    for arm, cells, mean in table(R, view, key):
        line = f"{arm:6} " + "".join(f"{'-':>10}" if c is None else f"{c:>10.4f}" for c in cells)
        print(line + (f"{mean:>12.4f}" if mean is not None else f"{'-':>12}"))
    print(f"{'':6} " + "".join(f"{'(dev)':>10}" if f == DEV else f"{'':>10}" for f in FOLDS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--weights", default="sqrt")
    p.add_argument("--seed", type=int, default=760)
    a = p.parse_args()
    R = load(a.out, a.weights, a.seed)
    if not R:
        print("no results found"); return
    print(f"{len(R)} of {len(ARMS)*len(FOLDS)} runs present")

    for view, title in [("as_sampled", "4-class macro-F1 (headline)"),
                        ("collapsed_mid", "3-class, minor+major merged"),
                        ("collapsed_3", "3-class, major+destroyed merged (Ismail)"),
                        ("collapsed_2", "binary damaged / not"),
                        ("natural", "4-class, weighted to natural distribution")]:
        show(R, view, title)
    show(R, "as_sampled", "accuracy (diagnostic only - base rate dominates)", "accuracy")

    print("\n=== boundary vs interior (the oversmoothing test) ===")
    print(f"{'arm':6} " + "".join(f"{f[:9]:>10}" for f in FOLDS) + f"{'mean gap':>12}")
    for arm in ARMS:
        cells, gaps = [], []
        for f in FOLDS:
            r = R.get((arm, f))
            if r is None:
                cells.append(f"{'-':>10}"); continue
            g = r["test"]["interior"]["macro_f1"] - r["test"]["boundary"]["macro_f1"]
            cells.append(f"{g:>+10.4f}")
            if f != DEV:
                gaps.append(g)
        print(f"{arm:6} " + "".join(cells) + (f"{sum(gaps)/len(gaps):>+12.4f}" if gaps else ""))
    print("  positive = interior easier than boundary. A wider gap for graph arms is")
    print("  oversmoothing; gate narrowing it against sage is the gate working.")

    print("\n=== per-fold detail ===")
    for f in FOLDS:
        r = R.get(("mlp", f))
        if r is None:
            continue
        pc = r["test"]["as_sampled"]["per_class"]
        print(f"  {f:11} n={sum(v['n'] for v in pc.values()):>7,.0f}  "
              f"classes scored {r['test']['as_sampled']['n_classes_scored']}  "
              + " ".join(f"{k[:5]}:{int(v['n'])}" for k, v in pc.items()))

    print("\n=== run health ===")
    print(f"{'arm':6} {'fold':11} {'epochs':>7} {'best':>6} {'val mF1':>9} {'sigma':>9} {'gate':>7}")
    for arm in ARMS:
        for f in FOLDS:
            r = R.get((arm, f))
            if r is None:
                continue
            sig = r.get("sigma_final")
            print(f"{arm:6} {f:11} {len(r['history']):>7} {r['best_epoch']:>6} "
                  f"{r['best_val_macro_f1']:>9.4f} {'-' if sig is None else f'{sig:.0f}px':>9} "
                  f"{gm(r):>7}")


if __name__ == "__main__":
    main()
