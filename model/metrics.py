# Metrics for node-level damage classification.
#
# Every score is reported with the class counts behind it: tsunami has no minor-damage
# buildings at all and earthquake has two destroyed, so a macro average can silently be
# over three classes in one fold and four in another.

import torch

CLASSES = ("no-damage", "minor", "major", "destroyed")
# collapses reuse from_confusion, so names are looked up by matrix size
NAMES = {4: CLASSES, 3: ("no-damage", "minor", "major+destroyed"), 2: ("no-damage", "damaged")}


def confusion(pred, y, weight=None, n_classes=4):
    # weight is per node; passing the sampling weights gives the natural-distribution
    # estimate from the same predictions, without retraining
    m = y >= 0
    pred, y = pred[m], y[m]
    w = torch.ones(m.sum(), device=y.device) if weight is None else weight[m]
    cm = torch.zeros(n_classes, n_classes, device=y.device)
    cm.index_put_((y, pred), w, accumulate=True)
    return cm


def from_confusion(cm):
    names = NAMES.get(cm.shape[0], tuple(f"c{i}" for i in range(cm.shape[0])))
    tp = cm.diag()
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = tp / (tp + fp).clamp(min=1e-9)
    rec = tp / (tp + fn).clamp(min=1e-9)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    present = cm.sum(1) > 0            # classes absent from the truth cannot be scored
    return dict(
        accuracy=float(tp.sum() / cm.sum().clamp(min=1e-9)),
        macro_f1=float(f1[present].mean()) if present.any() else 0.0,
        n_classes_scored=int(present.sum()),
        per_class={names[i]: dict(f1=float(f1[i]), precision=float(prec[i]),
                                  recall=float(rec[i]), n=float(cm[i].sum()))
                   for i in range(len(names))},
        confusion=cm.tolist(),
    )


def collapse(cm, how):
    # 4-class predictions re-scored as 3-class or binary without retraining:
    # "3" merges major+destroyed (Ismail's choice), "2" is any damage vs none
    groups = {"3": [[0], [1], [2, 3]], "2": [[0], [1, 2, 3]]}[how]
    k = len(groups)
    out = torch.zeros(k, k, device=cm.device)
    for a, ga in enumerate(groups):
        for b, gb in enumerate(groups):
            out[a, b] = cm[ga][:, gb].sum()
    return out


def evaluate(pred, y, weight=None, agree=None):
    # as-sampled and natural-distribution views are kept strictly separate, collapses
    # included - mixing them silently reports a weighted number under an unweighted name
    raw = confusion(pred, y, None)
    out = {"as_sampled": from_confusion(raw)}
    for how in ("3", "2"):
        out[f"collapsed_{how}"] = from_confusion(collapse(raw, how))
    if weight is not None:
        wcm = confusion(pred, y, weight)
        out["natural"] = from_confusion(wcm)
        for how in ("3", "2"):
            out[f"natural_collapsed_{how}"] = from_confusion(collapse(wcm, how))
    if agree is not None:
        # boundary nodes are where uniform smoothing destroys information, so they are
        # where the gate has to earn its place. Interior nodes are easy either way.
        for name, m in (("boundary", agree < 1.0), ("interior", agree >= 1.0)):
            m = m & (y >= 0) & ~agree.isnan()
            out[name] = from_confusion(confusion(pred[m], y[m])) | {"n_nodes": int(m.sum())}
    return out
