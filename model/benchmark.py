# End to end on one chip: load from disk, train a few steps, report timings.
# Wall and CPU time are both reported - the data path is CPU-bound and the GPU step is not,
# so the ratio tells us whether the compute budget is set by the 3090 or by PNG decoding.
#
# Usage:  python model/benchmark.py                  # every chip in testing/
#         python model/benchmark.py --nodes 1843     # synthetic worst case, no files needed
#         python model/benchmark.py --ckpt           # gradient checkpointing

import argparse, time

import torch

from data import chip_ids, load_chip
from loss import damage_loss
from net import DamageNet


class Timer:
    # wall clock and process CPU time together; sync=True waits for the GPU first
    def __init__(self, sync=False):
        self.sync = sync

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.w, self.c = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *_):
        if self.sync:
            torch.cuda.synchronize()
        self.wall = time.perf_counter() - self.w
        self.cpu = time.process_time() - self.c


def macro_f1(pred, y, n_classes=4):
    # mean of per-class F1 over labelled nodes; classes absent from this chip are skipped
    m = y >= 0
    pred, y = pred[m], y[m]
    scores = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(pred, y):
    m = y >= 0
    return float((pred[m] == y[m]).float().mean()) if m.any() else 0.0


def train_steps(net, pre, post, y, amp, steps):
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    def one():
        # bfloat16 rather than float16: same exponent range as fp32, so gradients cannot
        # underflow and no GradScaler is needed. fp16 here was skipping optimizer steps
        # and occasionally blowing the loss up.
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = net(pre, post)
            loss = damage_loss(out, y)
        loss.backward()
        opt.step()
        return out.detach(), float(loss)

    out0, loss0 = one()                     # warmup: cudnn autotune and allocator settle here
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with Timer(sync=True) as t:
        for _ in range(steps):
            out, loss = one()
    return {
        "wall": t.wall / steps, "cpu": t.cpu / steps,
        "gb": torch.cuda.max_memory_allocated() / 2**30,
        "loss0": loss0, "loss1": loss,
        "acc": accuracy(out.argmax(1), y), "f1": macro_f1(out.argmax(1), y),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, help="synthetic building count instead of real chips")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--combine", default="subtract", choices=["subtract", "concat"])
    p.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    p.add_argument("--fp32", action="store_true", help="also time in full fp32")
    a = p.parse_args()

    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | crop {a.crop} "
          f"chunk {a.chunk} combine {a.combine} ckpt {a.ckpt} steps {a.steps}\n")

    items = [(f"synthetic-{a.nodes}", a.nodes)] if a.nodes else [(c, None) for c in chip_ids()]
    modes = ([("fp32", False)] if a.fp32 else []) + [("bf16", True)]

    hdr = (f"{'chip':30} {'bldg':>5} {'load_s':>7} {'load_cpu':>8} {'mode':>5} "
           f"{'step_s':>7} {'cpu_s':>7} {'GB':>6} {'crops/s':>8} {'loss':>13} {'acc':>5} {'mF1':>5}")
    print(hdr); print("-" * len(hdr))

    tot_wall = tot_cpu = 0.0
    for name, n in items:
        with Timer() as tl:
            if n:
                pre = torch.randn(n, 3, a.crop, a.crop)
                post = torch.randn(n, 3, a.crop, a.crop)
                y = torch.randint(0, 4, (n,))
            else:
                c = load_chip(name, crop=a.crop)
                pre, post, y, n = c.pre, c.post, c.y, c.pre.shape[0]
        pre, post, y = pre.to(dev), post.to(dev), y.to(dev)
        tot_wall += tl.wall; tot_cpu += tl.cpu

        for label, amp in modes:
            net = DamageNet(combine=a.combine, chunk=a.chunk, use_ckpt=a.ckpt).to(dev).train()
            try:
                r = train_steps(net, pre, post, y, amp, a.steps)
                print(f"{name:30} {n:>5} {tl.wall:>7.2f} {tl.cpu:>8.2f} {label:>5} "
                      f"{r['wall']:>7.3f} {r['cpu']:>7.3f} {r['gb']:>6.2f} {2*n/r['wall']:>8.0f} "
                      f"{r['loss0']:>6.3f}->{r['loss1']:<6.3f} {r['acc']:>5.2f} {r['f1']:>5.2f}")
                tot_wall += r["wall"] * a.steps; tot_cpu += r["cpu"] * a.steps
            except torch.cuda.OutOfMemoryError:
                print(f"{name:30} {n:>5} {tl.wall:>7.2f} {tl.cpu:>8.2f} {label:>5} {'OOM':>7}")
            del net
            torch.cuda.empty_cache()
        del pre, post, y
        torch.cuda.empty_cache()

    print(f"\ntotal wall {tot_wall:.1f}s | total cpu {tot_cpu:.1f}s "
          f"(cpu > wall just means torch used several threads)")
    print("what matters: load_s vs step_s. Loading happens every epoch unless we cache crops.")


if __name__ == "__main__":
    main()
