#!/usr/bin/env python3
# Regenerates local/findings.md with the result tables computed from the run JSONs, so
# the prose and the numbers cannot drift apart.

import json, glob, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["earthquake", "fire", "flooding", "tsunami", "volcano", "wind"]
CLEAN = [f for f in FOLDS if f != "fire"]
ARMS = ["mlp", "sage", "gate", "dist"]

L, I = {}, {}
for f in glob.glob(str(ROOT / "results" / "*_sqrt_s760.json")):
    r = json.loads(Path(f).read_text()); L[(r["arm"], r["held_out"])] = r
for f in glob.glob(str(ROOT / "results_indomain" / "*_sqrt_s761.json")):
    r = json.loads(Path(f).read_text()); I[(r["arm"], r["held_out"])] = r

def val(D, a, f, view="as_sampled", key="macro_f1"):
    return D[(a, f)]["test"][view][key] if (a, f) in D else None

def grid(view, key="macro_f1", D=None, arms=None, getter=None):
    D = D or L; arms = arms or ARMS
    out = ["| arm | " + " | ".join(f.capitalize() for f in FOLDS) + " | clean mean |",
           "|---|" + "---|" * (len(FOLDS) + 1)]
    for a in arms:
        vs = [getter(D, a, f) if getter else val(D, a, f, view, key) for f in FOLDS]
        cl = [x for f, x in zip(FOLDS, vs) if f != "fire" and x is not None]
        out.append(f"| `{a}` | " + " | ".join("—" if x is None else f"{x:.4f}" for x in vs)
                   + (f" | **{sum(cl)/len(cl):.4f}** |" if cl else " | — |"))
    return "\n".join(out)

def ttest(getter_a, getter_b, folds=CLEAN):
    d = [getter_a(f) - getter_b(f) for f in folds]
    m = sum(d) / len(d); v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    se = math.sqrt(v / len(d))
    return m, se, (m / se if se else 0), sum(1 for x in d if x > 0), len(d)

gap = lambda D, a, f: (D[(a, f)]["test"]["interior"]["macro_f1"]
                       - D[(a, f)]["test"]["boundary"]["macro_f1"]) if (a, f) in D else None

arm_t = {}
for a in ("sage", "gate", "dist"):
    arm_t[a] = ttest(lambda f, a=a: val(L, a, f), lambda f: val(L, "mlp", f))
gap_t = {a: ttest(lambda f, a=a: gap(L, a, f), lambda f: gap(L, "mlp", f))
         for a in ("sage", "gate", "dist")}
dom = ttest(lambda f: val(I, "mlp", f), lambda f: val(L, "mlp", f), FOLDS)
gv = ttest(lambda f: I[("gate", f)]["gate_mean"], lambda f: L[("gate", f)]["gate_mean"], FOLDS)

health = ["| arm | fold | epochs | best | val mF1 | test mF1 | gate | sigma | test acc |",
          "|---|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    for f in FOLDS:
        if (a, f) not in L: continue
        r = L[(a, f)]; s = r.get("sigma_final"); gm = r.get("gate_mean")
        health.append(f"| `{a}` | {f} | {len(r['history'])} | {r['best_epoch']}"
                      f"{' ⚠' if r['best_epoch'] > 70 else ''} | {r['best_val_macro_f1']:.4f} | "
                      f"{val(L,a,f):.4f} | {'—' if gm is None else f'{gm:.3f}'} | "
                      f"{'—' if s is None else f'{s:.0f}px'} | {val(L,a,f,'as_sampled','accuracy'):.4f} |")

counts = ["| fold | test buildings | classes scored | no-damage | minor | major | destroyed |",
          "|---|---|---|---|---|---|---|"]
for f in FOLDS:
    pc = L[("mlp", f)]["test"]["as_sampled"]["per_class"]
    counts.append(f"| {f} | {sum(v['n'] for v in pc.values()):,.0f} | "
                  f"{L[('mlp',f)]['test']['as_sampled']['n_classes_scored']} | "
                  + " | ".join(f"{int(v['n']):,}" for v in pc.values()) + " |")

ind = ["| fold | cross-disaster | in-domain | delta |", "|---|---|---|---|"]
for f in sorted(FOLDS, key=lambda f: -(val(I,"mlp",f)-val(L,"mlp",f))):
    ind.append(f"| {f} | {val(L,'mlp',f):.4f} | {val(I,'mlp',f):.4f} | "
               f"**{val(I,'mlp',f)-val(L,'mlp',f):+.3f}** |")

print(Path(ROOT / "scripts" / "findings_template.md").read_text().format(
    grid_4=grid("as_sampled"), grid_bin=grid("collapsed_2"), grid_mid=grid("collapsed_mid"),
    grid_ismail=grid("collapsed_3"), grid_nat=grid("natural"),
    grid_acc=grid("as_sampled", "accuracy"), grid_gap=grid(None, getter=gap),
    health="\n".join(health), counts="\n".join(counts), indomain="\n".join(ind),
    t_sage=f"{arm_t['sage'][0]:+.4f}", t_sage_t=f"{arm_t['sage'][2]:+.2f}",
    t_gate=f"{arm_t['gate'][0]:+.4f}", t_gate_t=f"{arm_t['gate'][2]:+.2f}",
    t_dist=f"{arm_t['dist'][0]:+.4f}", t_dist_t=f"{arm_t['dist'][2]:+.2f}",
    g_sage=f"{gap_t['sage'][0]:+.4f}", g_sage_t=f"{gap_t['sage'][2]:+.2f}", g_sage_n=gap_t['sage'][3],
    g_gate=f"{gap_t['gate'][0]:+.4f}", g_gate_t=f"{gap_t['gate'][2]:+.2f}", g_gate_n=gap_t['gate'][3],
    g_dist=f"{gap_t['dist'][0]:+.4f}", g_dist_t=f"{gap_t['dist'][2]:+.2f}", g_dist_n=gap_t['dist'][3],
    dom_m=f"{dom[0]:+.3f}", dom_t=f"{dom[2]:.2f}", dom_n=dom[3],
    gv_m=f"{gv[0]:+.3f}", gv_t=f"{gv[2]:.2f}", gv_n=gv[3],
    gv_lodo=f"{sum(L[('gate',f)]['gate_mean'] for f in FOLDS)/6:.3f}",
    gv_ind=f"{sum(I[('gate',f)]['gate_mean'] for f in FOLDS)/6:.3f}",
))
