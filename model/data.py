# Load one xBD chip into tensors: per-building crops, labels, centroids.

import json, re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import Delaunay, QhullError

CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
LABEL = {c: i for i, c in enumerate(CLASSES)}

# backbone is ImageNet-pretrained so crops get ImageNet normalisation
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class Chip:
    chip_id: str
    pre: torch.Tensor     # [N,3,crop,crop] uint8 - normalise() on the GPU
    post: torch.Tensor    # [N,3,crop,crop] uint8
    y: torch.Tensor       # [N]  0-3, and -1 for un-classified
    pos: torch.Tensor     # [N,2] envelope centroids
    edge_index: torch.Tensor  # [2,2E] Delaunay neighbours, both directions
    edge_len: torch.Tensor    # [2E] edge length in pixels, for distance weighting
    agree: torch.Tensor = None   # [N] share of labelled neighbours with the same class


def _envelopes(feats):
    out = np.empty((len(feats), 4))
    for i, ft in enumerate(feats):
        n = np.fromiter(map(float, _NUM.findall(ft["wkt"])), float)
        xs, ys = n[0::2], n[1::2]
        out[i] = xs.min(), ys.min(), xs.max(), ys.max()
    return out


def _delaunay_edges(pos):
    # Delaunay needs distinct points, so triangulate the unique centroids and map back.
    # Only one chip in 6,286 has a co-located pair, but a silent index mismatch would be nasty.
    uniq, inv = np.unique(pos, axis=0, return_inverse=True)
    if len(uniq) < 3:
        return torch.zeros(2, 0, dtype=torch.long)
    try:
        simplices = Delaunay(uniq).simplices
    except (QhullError, ValueError):       # every point on one line
        return torch.zeros(2, 0, dtype=torch.long)

    members = [np.flatnonzero(inv == u) for u in range(len(uniq))]
    pairs = {(min(a, b), max(a, b)) for s in simplices
             for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2]))}

    e = [(i, j) for a, b in pairs for i in members[a] for j in members[b]]
    e += [(i, j) for g in members for i in g for j in g if i != j]   # co-located buildings
    e += [(j, i) for i, j in e]                                      # undirected
    return torch.tensor(sorted(set(e)), dtype=torch.long).T if e else torch.zeros(2, 0, dtype=torch.long)


def _to_u8(im):
    return torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1)


def normalise(x):
    # crops are stored as uint8 and normalised wherever they land - 4x less disk and 4x
    # less host-to-GPU traffic than keeping floats, and exact rather than quantised
    return (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)


def neighbour_agreement(y, edge_index):
    # share of a node's labelled neighbours that share its class. Nodes below 1.0 are the
    # boundary cases the gate exists for; nan where a node has no labelled neighbour.
    out = torch.full((len(y),), float("nan"))
    if edge_index.numel() == 0:
        return out
    src, dst = edge_index
    keep = (y[src] >= 0) & (y[dst] >= 0)
    src, dst = src[keep], dst[keep]
    same = (y[src] == y[dst]).float()
    num = torch.zeros(len(y)).index_add_(0, dst, same)
    den = torch.zeros(len(y)).index_add_(0, dst, torch.ones_like(same))
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


def load_chip(chip_id, root="testing", crop=128):
    root = Path(root)
    feats = json.loads((root / "labels" / f"{chip_id}_post_disaster.json").read_text())["features"]["xy"]
    env = _envelopes(feats)

    imgs = {t: Image.open(root / "images" / f"{chip_id}_{t}_disaster.png").convert("RGB")
            for t in ("pre", "post")}
    W, H = imgs["post"].size

    # post polygon crops both images - one box means pre and post cover the same ground
    out = {"pre": [], "post": []}
    for x0, y0, x1, y1 in env:
        box = (max(0, int(x0)), max(0, int(y0)),
               min(W, max(int(x1), int(x0) + 1)), min(H, max(int(y1), int(y0) + 1)))
        for t, im in imgs.items():
            out[t].append(_to_u8(im.crop(box).resize((crop, crop), Image.BILINEAR)))

    # un-classified stays a node but gets -1 so the loss skips it
    y = torch.tensor([LABEL.get(f["properties"]["subtype"], -1) for f in feats])
    cen = (env[:, :2] + env[:, 2:]) / 2
    pos = torch.tensor(cen, dtype=torch.float32)
    ei = _delaunay_edges(cen)
    el = (pos[ei[0]] - pos[ei[1]]).norm(dim=1) if ei.numel() else torch.zeros(0)
    return Chip(chip_id, torch.stack(out["pre"]), torch.stack(out["post"]), y, pos, ei, el,
                neighbour_agreement(y, ei))


def load_cached(path):
    return Chip(**torch.load(path, map_location="cpu", weights_only=True))


def chip_ids(root="testing"):
    return sorted(p.name.removesuffix("_post_disaster.json")
                  for p in Path(root, "labels").glob("*_post_disaster.json"))
