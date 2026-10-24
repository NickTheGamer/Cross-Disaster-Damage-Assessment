# Cross-Disaster Building Damage Assessment

COMPSCI 760 (University of Auckland), Group G6: Nicholas Coetzee, Balamurali Ragupathi,
Saketh Sudhirkumar Krishnapai, Kanishka Singh, Zixuan Huang.

Can a building-damage classifier trained on five xBD disaster types classify damage in a
sixth it has never seen, and does a graph over neighbouring buildings help? We train a
Siamese ResNet34 on per-building pre/post crops and compare four "necks" under strict
leave-one-disaster-out (LODO) evaluation:

| arm | neck |
|---|---|
| `mlp` | no message passing (dimension-matched control) |
| `sage` | GraphSAGE over a per-image Delaunay graph of buildings |
| `gate` | GraphSAGE + a learned per-building gate between local and neighbour evidence |
| `dist` | `gate` + a learned distance decay on edges |

Results and interpretation are in [`findings.md`](findings.md).

## Repository layout

```
scripts/fetch_xbd.sh        download + SHA1-verify the xBD challenge release
scripts/build_selection.py  layer 2: damage-ratio bins, 20k-building subset per fold, LODO roles
scripts/assortativity.py    label agreement across Delaunay edges; writes sigma init for `dist`
scripts/build_cache.py      layer 3: per-chip tensors (uint8 crops, labels, graph) in processed/
scripts/build_indomain.py   in-domain 70/15/15 splits for the supporting experiment
scripts/run_grid.py         runs the ablation grid (calls model/train.py per run)  <- replicates results
scripts/aggregate.py        prints every results table from the run JSONs
scripts/write_findings.py   regenerates findings.md from the run JSONs + findings_template.md
scripts/split_leakage.py    spatial overlap of xView2's official splits and the no-imagery label-copy baseline
model/data.py               chip loading, envelope crops, Delaunay edges
model/net.py                Siamese encoder (frozen BatchNorm) and the four-arm GraphNeck
model/loss.py               masked cross-entropy, class weights
model/metrics.py            macro-F1, collapsed schemes, natural-distribution and boundary views
model/train.py              one LODO run: training, early stopping, single test evaluation
model/benchmark.py          throughput / memory benchmark used for the compute budget
manifest/                   committed census and selections (chips.csv, selection_v1.*, indomain_v1.*)
```

Not in git (rebuildable or too large): `raw_dataset/`, `processed/`, `results/`,
`results_indomain/`, `.venv/`.

## Requirements

- One CUDA GPU with 24 GB (we used an RTX 3090). The densest image (1,843 buildings) peaks at
  about 20 GB in bf16; add `--ckpt` to `train.py` for gradient checkpointing on smaller GPUs.
- ~31 GB for the dataset, ~10 GB for the tensor cache.
- Python 3.12 with torch 2.5.1 + torchvision 0.20.1 (CUDA 12.1), numpy, scipy, pandas, pillow.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python numpy scipy pandas pillow
```

## Reproducing the results

Run from the repository root. Steps 2–4 are deterministic given the committed manifests, so
they can be skipped if `manifest/` is used as committed.

```bash
# 1. data: paste your signed URLs from https://xview2.org/download (see scripts/xbd_urls.txt
#    for the format) into local/xbd_urls.txt, then
XBD_URL_FILE=local/xbd_urls.txt ./scripts/fetch_xbd.sh

# 2. subset + LODO roles (seed 760, 20,000 buildings per fold)
.venv/bin/python scripts/build_selection.py

# 3. per-run sigma initialisation for the dist arm, from training folds only
.venv/bin/python scripts/assortativity.py --selection manifest/selection_v1.csv \
    --write-sigma manifest/selection_v1.json

# 4. in-domain splits
.venv/bin/python scripts/build_indomain.py

# 5. tensor cache (~2 min, 9.4 GB)
.venv/bin/python scripts/build_cache.py

# 6. main experiment: 4 arms x 6 held-out folds = 24 runs, ~20.5 h on one RTX 3090
.venv/bin/python scripts/run_grid.py --seed 760

# 7. supporting in-domain experiment: 12 runs, ~1 h
.venv/bin/python scripts/run_grid.py --arms mlp gate --seed 761 \
    --selection manifest/indomain_v1.csv --out results_indomain

# 8. tables and findings
.venv/bin/python scripts/aggregate.py --seed 760
.venv/bin/python scripts/write_findings.py > findings.md

# 9. spatial entanglement of xView2's official split (CPU, ~10 s)
.venv/bin/python scripts/split_leakage.py
```

`run_grid.py` uses the frozen configuration: sqrt class weights, 80-epoch ceiling, patience
10, gradient accumulation over ~1,000 buildings per step, bf16. It is resumable: finished runs
are skipped. A single run is `model/train.py --arm gate --held-out wind --weights sqrt
--epochs 80 --patience 10 --seed 760`.

Fire is the development fold (all settings were chosen with fire held out); it is reported
but excluded from the clean means.

**Census.** `manifest/chips.csv` (all 11,034 chips, with usable flags) is the committed census
that every later step reads. The script that generated it is not included in this
repository; its policy and source hashes are recorded in `manifest/manifest.json`.

**Gate values.** The stored `gate_mean` in the seed-760 and seed-761 results is the mean gate
over the final test chip of each run. `model/train.py` was corrected on 24 Oct 2026 to average
over every test building, so reruns with the current code record a fold-wide mean. A second
seed (42) is being run with the corrected code.

**Determinism.** Seeds are fixed, but some CUDA kernels (e.g. `index_add_`) are
non-deterministic, so reruns can differ slightly from the reported numbers.

## What is ours and what is not

**Ours:** all code in `model/` and `scripts/`: the data pipeline (census, damage-ratio
binning and allocation, sampling weights, tensor cache), Delaunay graph construction,
the four-arm `GraphNeck` (GraphSAGE layer implemented directly, gate, learned distance
decay), the training loop (masked loss, class-weight-preserving gradient accumulation, early
stopping), the metrics and the analysis scripts.

**From elsewhere:**
- Data: xBD (Gupta et al., 2019), downloaded from xview2.org under its licence.
- ResNet34 architecture and ImageNet weights from torchvision (He et al., 2016).
- GraphSAGE mean aggregation follows Hamilton et al. (2017).
- Neck hyperparameters (2 layers, 32 hidden, dropout 0.5, lr 3e-4) follow Ismail & Awad
  (2022); we did not tune them.
- Libraries: PyTorch, torchvision, NumPy, SciPy (`scipy.spatial.Delaunay`), pandas, Pillow.
- Generative AI (Claude, Anthropic) assisted with drafting and debugging code, as declared in
  our presentations. Design decisions, experiments and analysis are the group's own.
