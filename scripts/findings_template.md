# Findings — cross-disaster building damage assessment on xBD

**Status 2026-10-11.** Self-contained summary of the LODO ablation: what was run, every
result, and what may and may not be claimed. Decision history in `dev-diary.md`.

---

## The question

Can a building-damage classifier trained on some disaster types generalise to a type it
has never seen — and does a spatial graph over neighbouring buildings help?

Damage is spatially autocorrelated: floods are contiguous, fire perimeters connected,
tornado damage follows a track. A CNN seeing only a building's own crop cannot use that;
a graph over neighbouring buildings can. Ismail & Awad (2022) tried exactly this and
found no clear benefit — but tuned on their test set, so the result was not decisive.
**The contribution is testing it under a protocol that does not leak.**

## What was run

**Data.** Full xBD challenge release, SHA1-verified. 11,034 chips → 6,286 usable
(≥3 buildings, triangulable, ≥1 labelled) → **1,683 chips / 102,625 buildings**, 20,000
per fold allocated equally across four chip-level damage-ratio bins.

**Protocol.** Leave-one-disaster-out over six folds read from xBD's own
`metadata.disaster_type`. Early stopping on validation macro-F1 drawn only from training
folds; the test fold is read **once**, from the best-validation checkpoint. Fire is the
development fold — class weights, epoch ceiling and patience were set on it — and is
excluded from every clean mean.

**Model.** Siamese ResNet34 over per-building pre/post crops → per-chip Delaunay graph →
one of four necks. All four arms are one class with flags, so they cannot drift apart:

| arm | neck | neck params |
|---|---|---|
| `mlp` | no message passing (dimension-matched control) | 17,472 |
| `sage` | + GraphSAGE | 34,880 |
| `gate` | + learned per-node gate mixing local vs aggregated | 36,960 |
| `dist` | + learned distance kernel on edges | 36,961 |

**24 runs**, 4 arms × 6 folds, seed 760, `sqrt` class weights, 80-epoch ceiling,
patience 10.

---

# Full results

## 4-class macro-F1 — headline

{grid_4}

## Binary — damaged vs not

{grid_bin}

## 3-class — minor+major merged

{grid_mid}

## 3-class — major+destroyed merged (Ismail's grouping)

{grid_ismail}

## 4-class weighted to natural distribution

{grid_nat}

## Accuracy — diagnostic only

*Base rate dominates; compare against each fold's no-damage share below.*

{grid_acc}

## Interior-minus-boundary gap

*Positive = interior easier than boundary. A wider gap for a graph arm is oversmoothing.*

{grid_gap}

## Run health

*⚠ = best epoch within 10 of the 80-epoch ceiling, so possibly truncated.*

{health}

## Test-fold class counts

{counts}

---

# Interpretation

## Result 1 — No graph architecture beats the no-graph baseline

Paired over the five clean folds (df = 4, critical t = 2.776):

| comparison | mean | t | verdict |
|---|---|---|---|
| `sage` − `mlp` | {t_sage} | {t_sage_t} | not significant |
| `gate` − `mlp` | {t_gate} | {t_gate_t} | not significant |
| `dist` − `mlp` | {t_dist} | {t_dist_t} | not significant |

**None is significant.** The honest statement is **not** "the graph hurts" — it is that
**the graph shows no detectable benefit, with a slightly negative point estimate.** At
five folds, "neutral" and "mildly harmful" are indistinguishable.

This reproduces Ismail & Awad's negative finding under a protocol built to give the
graph a fair hearing: no test-set tuning, a dimension-matched control, frozen
BatchNorm so the baseline cannot see chip-level context, and a gate that can represent
either neighbouring arm exactly.

## Result 2 — The oversmoothing mechanism is real and measured

Boundary nodes are those whose Delaunay neighbours disagree on damage class — 9–46% of
nodes by fold. The prediction, made in advance, was that mean aggregation would smooth
them toward the local majority and destroy exactly the information that matters.

Change in the interior-minus-boundary gap relative to `mlp`:

| arm | mean | t | folds agreeing |
|---|---|---|---|
| **`sage`** | **{g_sage}** | **{g_sage_t}** | **{g_sage_n} / 5** |
| `gate` | {g_gate} | {g_gate_t} | {g_gate_n} / 5 |
| `dist` | {g_dist} | {g_dist_t} | {g_dist_n} / 5 |

**Message passing significantly widens the boundary penalty, same sign on every fold.**
The clearest confirmed mechanism in the study. The gate's repair of it is suggestive but
not established.

So: **we detect the pathology but not a performance consequence.**

## Result 3 — The gate measures neighbourhood trustworthiness

The learned gate value is interpretable — 1.0 means a node ignores its neighbours
entirely, 0.0 means it relies on them completely.

| setting | mean gate value |
|---|---|
| cross-disaster | **{gv_lodo}** |
| in-domain | **{gv_ind}** |

**Paired difference {gv_m} · t = {gv_t} · {gv_n}/6 folds · p < 0.01.**

With training and test in the same disaster the gate weighs neighbourhood and local
evidence about equally; under domain shift it moves decisively toward local. **The gate
correctly detects that neighbourhood information is less reliable across disasters** —
working as designed, without converting that into accuracy. Its contribution is
**diagnostic rather than predictive.**

## Why the graph fails despite strong spatial structure

Measured over 1.1M Delaunay edges **before any model was trained**: damage is strongly
spatially autocorrelated — Newman r = 0.64–0.87 on five of six folds, agreement lift
1.24–1.76× over chance, halving by roughly 140–200 px.

The structure is real; the method does not convert it into gain, and the reason is in
the same numbers. **Neighbours agree 74–99% of the time**, so message passing mostly
confirms what the CNN already concluded. At the minority of nodes where it would add
information — the boundaries — it actively misleads.

**High assortativity is simultaneously the justification for the graph and the reason it
is redundant.**

## Per-fold observations

**Flooding is the instructive failure.** Best class balance of any fold, predicted
0.35–0.45, scored 0.22. It also has the **most atypical class profile** — 3× the pooled
share of major damage, almost no destroyed. **Profile dissimilarity predicts transfer
failure better than class balance does.**

**Damage class profiles encode hazard physics.** Fire and tsunami destroy or spare
(28–29% destroyed, 0–2% minor); flooding inundates, damaging partially and rarely
flattening (19.9% major, 1.8% destroyed); wind is graduated (20.1% minor). A single model
must learn mutually inconsistent damage signatures.

**Tsunami is scored over 3 classes**, not 4 — the fold contains zero minor-damage
buildings — so its macro-F1 is not comparable to the others.

**`sage`'s earthquake "win" is majority collapse** — accuracy 0.9894 against a 0.9906
base rate. One class scoring ~1.0 and three scoring 0 macro-averages to 0.25.

**Severity grading fails; binary detection partly works.** Collapsing the same
predictions post-hoc: 4-class 0.30 → binary 0.54 clean mean, 0.73 on fire.

**Earthquake is degenerate, not a transfer failure.** 140 minor, 44 major and **2
destroyed** buildings — 0.5% damage, assortativity lift 1.01× (chance). xBD's only
earthquake event carries almost no damage signal, which plausibly explains why
xBD-pretrained models collapsed on the 2023 Turkey–Syria earthquake (MambaBDA reports
F1clf = 0.02 there).

---

# Supporting comparison — in-domain baseline

Added late, as a reference point rather than a main experiment: 12 runs training and
testing **within** a single disaster (70/15/15 by chip, seed 761), same protocol
otherwise.

{indomain}

**Mean {dom_m} · t = {dom_t} · {dom_n}/6 folds positive · p < 0.01.** Binary detection
moves further (flooding +0.259, tsunami +0.239).

**Conservative**: in-domain models trained on ~5× fewer buildings (~13,500 against
~82,000; volcano 25× fewer) and still won on every fold.

Two qualifications. In-domain test splits are not the LODO test folds, and chip lumpiness
moved their damage rates by 5–8 points — easier for fire, tsunami, wind and earthquake,
*harder* for flooding and volcano. The gain holds on both folds that got harder.
Separately, **earthquake's delta should be discarded**: its in-domain test split is 2.1%
damaged against the fold's 0.9%, so most of its gain is an easier test set.

The `gate` in-domain runs are **not usable** — patience 10 stopped two of six while still
improving (best epochs 2, 8, 4). The `mlp` runs are unaffected (best epochs 11–62) and
carry this comparison.

---

# What may be claimed

- **No graph architecture improves zero-shot transfer** over a dimension-matched
  baseline, under a protocol with no test-set tuning.
- **Message passing significantly worsens boundary nodes** — the oversmoothing mechanism,
  predicted in advance and confirmed on every fold.
- **Damage is strongly spatially autocorrelated** yet that structure does not convert into
  predictive gain, because neighbours mostly agree and the graph is therefore redundant
  where it is right and misleading where it is not.
- **The learned gate value tracks domain shift** — a diagnostic instrument, not a gain.
- **Binary damage detection transfers far better than severity grading.**
- **xBD's earthquake fold cannot support damage learning**, giving an evidence-based
  explanation for reported earthquake failures.
- **Supporting:** training within a disaster beats training across disasters by ~0.12
  macro-F1 despite 5× less data.

# What may not be claimed

- *"The graph hurts performance"* — not significant at n = 5 folds.
- *"The gate fixes oversmoothing"* — suggestive only.
- *"Published graph results are wrong"* — their test-set-tuning admissions are documented
  and our own validation/test gaps measured, but **their experiments were not
  reproduced**. The defensible claim is that in-domain evaluation overstates deployment
  performance.
- *"Graph complexity buys in-domain performance"* — LODO validation rises monotonically
  with complexity (`mlp` 0.5404 → `dist` 0.6015; val/test ratio 1.78× → 2.05×), but the
  in-domain experiment did not reproduce it and was underpowered. **Unresolved.**
- Direct comparison of these absolute values to MambaBDA's 0.75 — different task framing,
  metric and subset. Only internal comparisons are valid.

# Caveats

**Single seed.** One seed per experiment. `sage`'s deficit has SE 0.0131 at n = 5;
further seeds would roughly halve that and could separate "neutral" from "mildly
harmful". This is the largest weakness.

**Patience 10 was set on large validation sets.** Fine for the LODO grid (~10,000
validation buildings); too tight for the in-domain sets (478–3,537), where it truncated
two `gate` runs.

**Absolute scores are mildly understated** — small stopping margins are common, so values
sit a fraction of a point below converged. Affects arms roughly equally, so comparisons
are more robust than absolute numbers.

**One truncated run.** `gate`/fire peaked at epoch 80 of 80. Dev fold, so the clean mean
is unaffected.

# Next steps, by value per hour

1. **Additional seeds on the LODO grid** (~19 h each). Largest weakness; would settle
   whether the graph is neutral or mildly harmful.
2. **Gate-value versus boundary figure.** `last_gate` is stored per run; plotting it
   against each node's neighbour agreement shows directly whether the gate learned to
   distrust its neighbourhood where predicted. No GPU — strong figure.
3. **Re-run the six `gate` in-domain folds at patience 20–25** (~1 h). Would resolve the
   only open claim.
4. **Size-matched control** (~2 h). Subsample cross-disaster training to ~13,500
   buildings, separating domain effect from data volume.
5. **Per-class error analysis on flooding**, checking whether its confusions match the
   hazard-profile mismatch.
