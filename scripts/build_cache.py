#!/usr/bin/env python3
# Layer 3: materialise the selected chips as tensors.
#
# Loading a chip from PNG costs ~2x a training step and happens every epoch, so caching
# is roughly a 4x saving over the whole grid. Crops are stored uint8 and normalised on
# the GPU. Graphs are per chip with local node indices - a batcher combining several
# chips must offset edge_index by the running node count, or chip 2's edges will point
# at chip 1's nodes.

import argparse, csv, json, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from data import load_chip                                        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", default=str(ROOT / "manifest" / "selection_v1.csv"))
    p.add_argument("--out", default=str(ROOT / "processed"))
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--force", action="store_true", help="rebuild chips already cached")
    a = p.parse_args()

    split = {r["chip_id"]: r["split"] for r in csv.DictReader(open(ROOT / "manifest" / "chips.csv"))}
    sel = list(csv.DictReader(open(a.selection)))
    out = Path(a.out)

    t0, done, skipped, nodes, edges = time.perf_counter(), 0, 0, 0, 0
    for i, r in enumerate(sel, 1):
        cid, fold = r["chip_id"], r["fold"]
        dst = out / fold / f"{cid}.pt"
        if dst.exists() and not a.force:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        c = load_chip(cid, root=ROOT / "raw_dataset" / split[cid], crop=a.crop)
        torch.save({"chip_id": c.chip_id, "pre": c.pre, "post": c.post, "y": c.y,
                    "pos": c.pos, "edge_index": c.edge_index, "edge_len": c.edge_len,
                    "agree": c.agree}, dst)
        done += 1
        nodes += c.pre.shape[0]
        edges += c.edge_index.shape[1] // 2
        if i % 200 == 0:
            print(f"  {i:,}/{len(sel):,}  {time.perf_counter()-t0:.0f}s", flush=True)

    size = sum(f.stat().st_size for f in out.rglob("*.pt"))
    print(f"\ncached {done:,} chips ({skipped:,} already present) in {time.perf_counter()-t0:.0f}s")
    print(f"{nodes:,} buildings, {edges:,} edges, {size/2**30:.2f} GB on disk")
    for d in sorted(out.iterdir()):
        if d.is_dir():
            n = list(d.glob("*.pt"))
            print(f"  {d.name:11} {len(n):>5,} chips  {sum(f.stat().st_size for f in n)/2**30:>6.2f} GB")


if __name__ == "__main__":
    main()
