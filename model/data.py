# Load one xBD chip into tensors: per-building crops, labels, centroids.

import json, re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
LABEL = {c: i for i, c in enumerate(CLASSES)}

# backbone is ImageNet-pretrained so crops get ImageNet normalisation
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class Chip:
    chip_id: str
    pre: torch.Tensor     # [N,3,crop,crop]
    post: torch.Tensor    # [N,3,crop,crop]
    y: torch.Tensor       # [N]  0-3, and -1 for un-classified
    pos: torch.Tensor     # [N,2] envelope centroids, for the Delaunay graph later


def _envelopes(feats):
    out = np.empty((len(feats), 4))
    for i, ft in enumerate(feats):
        n = np.fromiter(map(float, _NUM.findall(ft["wkt"])), float)
        xs, ys = n[0::2], n[1::2]
        out[i] = xs.min(), ys.min(), xs.max(), ys.max()
    return out


def _to_tensor(im):
    x = torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


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
            out[t].append(_to_tensor(im.crop(box).resize((crop, crop), Image.BILINEAR)))

    # un-classified stays a node but gets -1 so the loss skips it
    y = torch.tensor([LABEL.get(f["properties"]["subtype"], -1) for f in feats])
    pos = torch.tensor((env[:, :2] + env[:, 2:]) / 2, dtype=torch.float32)
    return Chip(chip_id, torch.stack(out["pre"]), torch.stack(out["post"]), y, pos)


def chip_ids(root="testing"):
    return sorted(p.name.removesuffix("_post_disaster.json")
                  for p in Path(root, "labels").glob("*_post_disaster.json"))
