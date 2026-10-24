# how close xView2's official test/hold sets sit to its training data, and what location alone predicts
import json, re
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

CLS = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
chips = pd.read_csv("manifest/chips.csv")
gt = json.load(open("raw_dataset/xview_geotransforms.json"))
# competition setup: train + tier3 is the training pool
pool = lambda split: "train" if split in ("train", "tier3") else split

def metres(lng, lat, lng0, lat0):
    # local equirectangular per event - fine at city scale
    return np.c_[(lng - lng0) * 111320 * np.cos(np.radians(lat0)), (lat - lat0) * 110540]

def macro_f1(y, p):
    f = []
    for c in np.unique(y):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

# one row per labelled building, at its envelope centre
rows = []
for r in chips.itertuples():
    lab = json.load(open(f"raw_dataset/{r.split}/labels/{r.chip_id}_post_disaster.json"))
    for ft in lab["features"]["lng_lat"]:
        y = CLS.get(ft["properties"].get("subtype"))
        if y is None:
            continue
        pts = np.array(re.findall(r"(-?[\d.]+(?:e-?\d+)?) (-?[\d.]+(?:e-?\d+)?)", ft["wkt"]), float)
        rows.append((r.event, r.fold, pool(r.split), *(pts.min(0) + pts.max(0)) / 2, y))
B = pd.DataFrame(rows, columns=["event", "fold", "pool", "lng", "lat", "y"])

# chip footprints from the post-image geotransform (top-left origin, pixel size in degrees)
fp = []
for r in chips.itertuples():
    (x0, dx, _, y0, _, dy), _ = gt[f"{r.chip_id}_post_disaster.png"]
    fp.append((r.event, pool(r.split), x0, x0 + r.width * dx, y0 + r.height * dy, y0))
F = pd.DataFrame(fp, columns=["event", "pool", "x0", "x1", "y0", "y1"])

# chip level: gap to the nearest training chip of the same event, and how much of the chip it covers
C = []
for ev, f in F.groupby("event"):
    tr, kx, ky = f[f.pool == "train"], 111320 * np.cos(np.radians(f.y0.mean())), 110540
    for a in f[f.pool != "train"].itertuples():
        gx = np.maximum(0, np.maximum(tr.x0 - a.x1, a.x0 - tr.x1)) * kx
        gy = np.maximum(0, np.maximum(tr.y0 - a.y1, a.y0 - tr.y1)) * ky
        ox = np.clip(np.minimum(a.x1, tr.x1) - np.maximum(a.x0, tr.x0), 0, None)
        oy = np.clip(np.minimum(a.y1, tr.y1) - np.maximum(a.y0, tr.y0), 0, None)
        C.append((a.pool, np.hypot(gx, gy).min(), (ox * oy).max() / ((a.x1 - a.x0) * (a.y1 - a.y0))))
C = pd.DataFrame(C, columns=["split", "gap", "cover"])

# building level: label-copy from the nearest training building of the same event, no imagery
P = []
for ev, b in B.groupby("event"):
    tr = b[b.pool == "train"]
    tree = cKDTree(metres(tr.lng.values, tr.lat.values, b.lng.mean(), b.lat.mean()))
    for sp in ("test", "hold"):
        q = b[b.pool == sp]
        if len(q):
            d, i = tree.query(metres(q.lng.values, q.lat.values, b.lng.mean(), b.lat.mean()))
            P.append(pd.DataFrame({"fold": q.fold.values, "split": sp, "y": q.y.values, "p": tr.y.values[i],
                                   "d": d, "maj": np.bincount(tr.y, minlength=4).argmax()}))
P = pd.concat(P)

for sp in ("test", "hold"):
    c, p = C[C.split == sp], P[P.split == sp]
    yb, pb = (p.y > 0).values, (p.p > 0).values
    print(f"{sp}: {len(c)} chips | share ground with a training chip {(c.cover > 0).mean():.1%} "
          f"(median cover of those {c.cover[c.cover > 0].median():.0%}) | within 100 m {(c.gap <= 100).mean():.1%}")
    print(f"  {len(p):,} buildings | median distance to nearest training building {p.d.median():.0f} m")
    print(f"  label copy: macro-F1 {macro_f1(p.y.values, p.p.values):.3f}, binary F1 "
          f"{2 * np.sum(yb & pb) / (yb.sum() + pb.sum()):.3f} | majority class macro-F1 {macro_f1(p.y.values, p.maj.values):.3f}")
print("label copy by disaster type (test + hold):",
      {fo: round(macro_f1(p.y.values, p.p.values), 3) for fo, p in P.groupby("fold")})
