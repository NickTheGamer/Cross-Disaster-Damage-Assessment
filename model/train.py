# One LODO run: train an arm with the held-out fold as test.
#
# Chips are forwarded one at a time - never packed together - so no edge_index offsetting
# is needed and a chip's graph can never leak into its neighbour's. Gradients accumulate
# over ~budget nodes before one optimizer step, which keeps the effective batch large
# enough that the rare classes still get signal on every step.
#
# Early stopping selects on validation macro-F1, not loss - loss is dominated by the
# 74% no-damage majority, so a collapsed model can post a good loss while scoring 0.25
# macro-F1. The test fold is touched exactly once, at the end.

import argparse, csv, json, time
from pathlib import Path

import torch

from data import load_cached, normalise
from loss import damage_loss, inverse_frequency_weights
from metrics import evaluate
from net import DamageNet

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]


def split_chips(selection, held, role, processed):
    rows = [r for r in csv.DictReader(open(selection)) if r[f"role_{held}"] == role]
    return [(Path(processed, r["fold"], r["chip_id"] + ".pt"), float(r["weight"])) for r in rows]


def batches(items, budget, shuffle=None):
    # pack chips until the node budget; one chip per step would leave the GPU idle,
    # and with BatchNorm frozen the packing cannot change what the model computes
    idx = torch.randperm(len(items), generator=shuffle).tolist() if shuffle is not None else range(len(items))
    cur, n = [], 0
    for i in idx:
        c = load_cached(items[i][0])
        if cur and n + c.pre.shape[0] > budget:
            yield cur
            cur, n = [], 0
        cur.append((c, items[i][1]))
        n += c.pre.shape[0]
    if cur:
        yield cur


def to_device(c, dev):
    return (normalise(c.pre.to(dev)), normalise(c.post.to(dev)), c.y.to(dev),
            c.edge_index.to(dev), c.edge_len.to(dev))


def weight_mass(y, cw):
    # how much of the group's loss this chip should carry. cross_entropy normalises by
    # sum(w) over whatever nodes it is handed, so a chip with one rare building would
    # otherwise see that building carry ~24% of its loss while the next chip gives it 0%.
    # Scaling by the chip's share of the group's weight mass restores the weighting the
    # class weights were meant to deliver.
    m = y >= 0
    if not m.any():
        return 0.0
    return float(m.sum()) if cw is None else float(cw[y[m]].sum())


def class_weights(items, scheme):
    # computed from the training folds only - counting over all data would leak the test
    # fold into the loss
    counts = torch.zeros(4)
    for p, _ in items:
        y = load_cached(p).y
        counts += torch.bincount(y[y >= 0], minlength=4)
    if scheme == "none":
        return None, counts
    w = inverse_frequency_weights(counts if scheme == "inverse" else counts.sqrt())
    return w, counts


@torch.no_grad()
def run_eval(net, items, dev, amp):
    net.eval()
    P, Y, W, A = [], [], [], []
    for path, weight in items:
        c = load_cached(path)
        pre, post, y, ei, el = to_device(c, dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
            out = net(pre, post, ei, el)
        P.append(out.float().argmax(1))
        Y.append(y)
        W.append(torch.full((y.shape[0],), weight, device=dev))
        A.append(c.agree.to(dev))
    net.train()
    return torch.cat(P), torch.cat(Y), torch.cat(W), torch.cat(A)


def run(arm, held, a):
    dev = torch.device("cuda")
    sel = a.selection
    meta = json.loads(Path(sel).with_suffix(".json").read_text())
    tr = split_chips(sel, held, "train", a.processed)
    va = split_chips(sel, held, "val", a.processed)
    te = split_chips(sel, held, "test", a.processed)

    cw, counts = class_weights(tr, a.weights)
    torch.manual_seed(a.seed)
    net = DamageNet(arm=arm, combine=a.combine, chunk=a.chunk, use_ckpt=a.ckpt,
                    sigma_init=meta["sigma_init_px"][held]).to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    cwd = cw.to(dev) if cw is not None else None
    gen = torch.Generator().manual_seed(a.seed)

    best, best_ep, best_state, hist = -1.0, -1, None, []
    for ep in range(1, a.epochs + 1):
        t0, tot, nb = time.perf_counter(), 0.0, 0
        for g in batches(tr, a.budget, shuffle=gen):
            mass = [weight_mass(c.y, cw) for c, _ in g]
            total = sum(mass)
            if total == 0:
                continue
            opt.zero_grad(set_to_none=True)
            step = 0.0
            for (c, _), m in zip(g, mass):
                if m == 0:
                    continue
                pre, post, y, ei, el = to_device(c, dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                    l = damage_loss(net(pre, post, ei, el), y, cwd) * (m / total)
                l.backward()          # gradients add up; memory only ever holds one chip
                step += float(l)
            opt.step()
            tot += step; nb += 1
        p, y, w, ag = run_eval(net, va, dev, bool(a.amp))
        vm = evaluate(p, y)["as_sampled"]["macro_f1"]
        sigma = float(net.neck.log_sigma.exp()) if getattr(net.neck, "dist", False) else None
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_macro_f1=vm,
                         sigma=sigma, secs=round(time.perf_counter() - t0, 1)))
        print(f"  ep{ep:>3} loss {tot/max(nb,1):.4f} val_mF1 {vm:.4f}"
              f"{'' if sigma is None else f' sigma {sigma:.0f}px'} ({hist[-1]['secs']:.0f}s)"
              + ("  *" if vm > best else ""), flush=True)
        if vm > best:
            best, best_ep = vm, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= a.patience:
            print(f"  early stop: no val gain for {a.patience} epochs")
            break

    net.load_state_dict(best_state)                 # the test fold is seen once, from here
    p, y, w, ag = run_eval(net, te, dev, bool(a.amp))
    res = evaluate(p, y, weight=w, agree=ag)
    out = dict(arm=arm, held_out=held, seed=a.seed, best_epoch=best_ep, best_val_macro_f1=best,
               class_weights=None if cw is None else cw.tolist(),
               train_class_counts=counts.tolist(), weights_scheme=a.weights,
               sigma_init=meta["sigma_init_px"][held],
               sigma_final=float(net.neck.log_sigma.exp()) if getattr(net.neck, "dist", False) else None,
               n_train=len(tr), n_val=len(va), n_test=len(te), history=hist, test=res)
    if getattr(net.neck, "last_gate", None) is not None:
        out["gate_mean"] = float(net.neck.last_gate.mean())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"{arm}_{held}_{a.weights}_s{a.seed}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="mlp", choices=["mlp", "sage", "gate", "dist"])
    p.add_argument("--held-out", default="fire", choices=FOLDS)
    p.add_argument("--weights", default="inverse", choices=["inverse", "sqrt", "none"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--budget", type=int, default=1000,
                   help="nodes accumulated before an optimizer step")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--combine", default="subtract", choices=["subtract", "concat"])
    p.add_argument("--seed", type=int, default=760)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--selection", default=str(ROOT / "manifest" / "selection_v1.csv"))
    p.add_argument("--processed", default=str(ROOT / "processed"))
    p.add_argument("--out", default=str(ROOT / "results"))
    a = p.parse_args()
    print(f"arm={a.arm} held_out={a.held_out} weights={a.weights} seed={a.seed}")
    r = run(a.arm, a.held_out, a)
    t = r["test"]["as_sampled"]
    print(f"\ntest macro_f1 {t['macro_f1']:.4f} over {t['n_classes_scored']} classes | "
          f"acc {t['accuracy']:.4f} | best epoch {r['best_epoch']}")


if __name__ == "__main__":
    main()
