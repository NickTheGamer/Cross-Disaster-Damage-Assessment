#!/usr/bin/env python3
# The LODO ablation grid: every arm against every held-out fold.
#
# Resumable - a completed run leaves a result file and is skipped on restart, so a crash
# at hour 14 does not start over. But a *stub* left by a short smoke run looks identical
# to a finished run, so --check refuses to start when anything would be skipped and
# makes you look at it.

import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]
ARMS = ["mlp", "sage", "gate", "dist"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--folds", nargs="+", default=FOLDS, choices=FOLDS)
    p.add_argument("--weights", default="sqrt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seed", type=int, default=760)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--force", action="store_true", help="rerun even if a result exists")
    a = p.parse_args()

    runs = [(arm, f) for f in a.folds for arm in a.arms]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = [(arm, f) for arm, f in runs
            if (out / f"{arm}_{f}_{a.weights}_s{a.seed}.json").exists()]

    if done and not a.force:
        print(f"{len(done)} of {len(runs)} already have result files and will be SKIPPED:")
        for arm, f in done:
            r = json.loads((out / f"{arm}_{f}_{a.weights}_s{a.seed}.json").read_text())
            print(f"  {arm:5} {f:11} {len(r['history']):>3} epochs, best {r['best_epoch']}")
        print("\nIf any of those are smoke-test stubs, delete them first. Re-run with "
              "--force to redo everything.")
        if input("continue, skipping them? [y/N] ").strip().lower() != "y":
            return 1

    todo = runs if a.force else [r for r in runs if r not in done]
    print(f"\n{len(todo)} runs | weights={a.weights} epochs={a.epochs} "
          f"patience={a.patience} budget={a.budget} seed={a.seed}\n")

    t0, failed = time.perf_counter(), []
    for i, (arm, fold) in enumerate(todo, 1):
        el = time.perf_counter() - t0
        eta = f", eta {(el/(i-1))*(len(todo)-i+1)/3600:.1f}h" if i > 1 else ""
        print(f"[{i}/{len(todo)}] {arm} / hold out {fold}   (elapsed {el/3600:.1f}h{eta})",
              flush=True)
        cmd = [sys.executable, str(ROOT / "model" / "train.py"), "--arm", arm,
               "--held-out", fold, "--weights", a.weights, "--epochs", str(a.epochs),
               "--patience", str(a.patience), "--budget", str(a.budget),
               "--seed", str(a.seed), "--out", a.out]
        if subprocess.run(cmd, cwd=ROOT / "model").returncode != 0:
            print(f"  FAILED: {arm}/{fold}")
            failed.append((arm, fold))

    print(f"\nfinished in {(time.perf_counter()-t0)/3600:.1f}h"
          + (f"; {len(failed)} failed: {failed}" if failed else "; all runs completed"))

    # a best epoch near the ceiling means the run was still climbing when we cut it off,
    # which would bias the ablation against whichever arms converge slowest
    near = []
    for arm, fold in runs:
        f = out / f"{arm}_{fold}_{a.weights}_s{a.seed}.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r["best_epoch"] > a.epochs - 10:
                near.append((arm, fold, r["best_epoch"]))
    if near:
        print(f"\nWARNING: {len(near)} runs peaked within 10 epochs of the {a.epochs} ceiling "
              f"- they may have been truncated:")
        for arm, fold, ep in near:
            print(f"  {arm:5} {fold:11} best epoch {ep}")
    else:
        print(f"\nno run peaked near the {a.epochs}-epoch ceiling; budget was sufficient")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
