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

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.2253 | 0.3559 | 0.2167 | 0.3910 | 0.3341 | 0.3485 | **0.3031** |
| `sage` | 0.2487 | 0.3547 | 0.2173 | 0.3434 | 0.2947 | 0.3233 | **0.2855** |
| `gate` | 0.2170 | 0.3696 | 0.2276 | 0.3735 | 0.3480 | 0.3511 | **0.3034** |
| `dist` | 0.2171 | 0.3586 | 0.2362 | 0.3205 | 0.3404 | 0.3520 | **0.2932** |

## Binary — damaged vs not

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.4468 | 0.7324 | 0.4578 | 0.5729 | 0.6079 | 0.6342 | **0.5439** |
| `sage` | 0.4973 | 0.7124 | 0.4588 | 0.5192 | 0.5336 | 0.6404 | **0.5299** |
| `gate` | 0.4334 | 0.7344 | 0.4411 | 0.5500 | 0.6493 | 0.6007 | **0.5349** |
| `dist` | 0.4330 | 0.6795 | 0.4641 | 0.4891 | 0.6277 | 0.6278 | **0.5283** |

## 3-class — minor+major merged

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.3000 | 0.4785 | 0.2864 | 0.3960 | 0.4556 | 0.4699 | **0.3816** |
| `sage` | 0.3316 | 0.4781 | 0.2922 | 0.3516 | 0.3883 | 0.4332 | **0.3594** |
| `gate` | 0.2891 | 0.4980 | 0.3026 | 0.3852 | 0.4759 | 0.4727 | **0.3851** |
| `dist` | 0.2891 | 0.4794 | 0.3169 | 0.3217 | 0.4583 | 0.4918 | **0.3756** |

## 3-class — major+destroyed merged (Ismail's grouping)

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.3008 | 0.4952 | 0.3044 | 0.5673 | 0.4068 | 0.4321 | **0.4023** |
| `sage` | 0.3316 | 0.4704 | 0.2965 | 0.5054 | 0.3733 | 0.4427 | **0.3899** |
| `gate` | 0.2890 | 0.4953 | 0.2943 | 0.5470 | 0.4365 | 0.4279 | **0.3990** |
| `dist` | 0.2894 | 0.4643 | 0.3108 | 0.4672 | 0.4227 | 0.4285 | **0.3837** |

## 4-class weighted to natural distribution

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.2230 | 0.3516 | 0.2237 | 0.4070 | 0.3341 | 0.3487 | **0.3073** |
| `sage` | 0.2491 | 0.3463 | 0.2289 | 0.3729 | 0.2947 | 0.3230 | **0.2937** |
| `gate` | 0.2175 | 0.3718 | 0.2414 | 0.3979 | 0.3480 | 0.3477 | **0.3105** |
| `dist` | 0.2163 | 0.3512 | 0.2495 | 0.3449 | 0.3404 | 0.3519 | **0.3006** |

## Accuracy — diagnostic only

*Base rate dominates; compare against each fold's no-damage share below.*

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.7659 | 0.6654 | 0.5570 | 0.6630 | 0.7043 | 0.6328 | **0.6646** |
| `sage` | 0.9894 | 0.6890 | 0.5938 | 0.6411 | 0.6319 | 0.6470 | **0.7006** |
| `gate` | 0.7308 | 0.6457 | 0.6310 | 0.6555 | 0.7208 | 0.6576 | **0.6792** |
| `dist` | 0.7244 | 0.5854 | 0.6326 | 0.5806 | 0.6831 | 0.6303 | **0.6502** |

## Interior-minus-boundary gap

*Positive = interior easier than boundary. A wider gap for a graph arm is oversmoothing.*

| arm | Earthquake | Fire | Flooding | Tsunami | Volcano | Wind | clean mean |
|---|---|---|---|---|---|---|---|
| `mlp` | 0.0547 | 0.0439 | 0.0535 | 0.0407 | -0.0527 | 0.0411 | **0.0275** |
| `sage` | 0.1156 | 0.0972 | 0.0688 | 0.0560 | 0.0297 | 0.1220 | **0.0784** |
| `gate` | 0.0311 | 0.1017 | 0.1017 | 0.0641 | -0.0407 | 0.1061 | **0.0524** |
| `dist` | 0.0289 | 0.0930 | 0.0911 | 0.0328 | -0.0378 | 0.0996 | **0.0429** |

## Run health

*⚠ = best epoch within 10 of the 80-epoch ceiling, so possibly truncated.*

| arm | fold | epochs | best | val mF1 | test mF1 | gate | sigma | test acc |
|---|---|---|---|---|---|---|---|---|
| `mlp` | earthquake | 35 | 25 | 0.5082 | 0.2253 | — | — | 0.7659 |
| `mlp` | fire | 61 | 51 | 0.5326 | 0.3559 | — | — | 0.6654 |
| `mlp` | flooding | 51 | 41 | 0.5345 | 0.2167 | — | — | 0.5570 |
| `mlp` | tsunami | 58 | 48 | 0.5636 | 0.3910 | — | — | 0.6630 |
| `mlp` | volcano | 51 | 41 | 0.5530 | 0.3341 | — | — | 0.7043 |
| `mlp` | wind | 46 | 36 | 0.5425 | 0.3485 | — | — | 0.6328 |
| `sage` | earthquake | 20 | 10 | 0.3695 | 0.2487 | — | — | 0.9894 |
| `sage` | fire | 32 | 22 | 0.4704 | 0.3547 | — | — | 0.6890 |
| `sage` | flooding | 62 | 52 | 0.5693 | 0.2173 | — | — | 0.5938 |
| `sage` | tsunami | 69 | 59 | 0.5746 | 0.3434 | — | — | 0.6411 |
| `sage` | volcano | 59 | 49 | 0.6488 | 0.2947 | — | — | 0.6319 |
| `sage` | wind | 46 | 36 | 0.5701 | 0.3233 | — | — | 0.6470 |
| `gate` | earthquake | 51 | 41 | 0.5875 | 0.2170 | 0.617 | — | 0.7308 |
| `gate` | fire | 80 | 80 ⚠ | 0.5859 | 0.3696 | 0.625 | — | 0.6457 |
| `gate` | flooding | 43 | 33 | 0.5490 | 0.2276 | 0.707 | — | 0.6310 |
| `gate` | tsunami | 71 | 61 | 0.6040 | 0.3735 | 0.602 | — | 0.6555 |
| `gate` | volcano | 70 | 60 | 0.6594 | 0.3480 | 0.691 | — | 0.7208 |
| `gate` | wind | 64 | 54 | 0.5952 | 0.3511 | 0.746 | — | 0.6576 |
| `dist` | earthquake | 68 | 58 | 0.5978 | 0.2171 | 0.641 | 749px | 0.7244 |
| `dist` | fire | 60 | 50 | 0.6076 | 0.3586 | 0.578 | 819px | 0.5854 |
| `dist` | flooding | 71 | 61 | 0.5961 | 0.2362 | 0.730 | 436px | 0.6326 |
| `dist` | tsunami | 56 | 46 | 0.6076 | 0.3205 | 0.566 | 572px | 0.5806 |
| `dist` | volcano | 72 | 62 | 0.6011 | 0.3404 | 0.680 | 675px | 0.6831 |
| `dist` | wind | 78 | 68 | 0.6048 | 0.3520 | 0.758 | 755px | 0.6303 |

## Test-fold class counts

| fold | test buildings | classes scored | no-damage | minor | major | destroyed |
|---|---|---|---|---|---|---|
| earthquake | 19,751 | 4 | 19,565 | 140 | 44 | 2 |
| fire | 19,217 | 4 | 13,109 | 333 | 380 | 5,395 |
| flooding | 18,662 | 4 | 12,325 | 2,287 | 3,709 | 341 |
| tsunami | 18,856 | 3 | 12,539 | 0 | 769 | 5,548 |
| volcano | 3,632 | 4 | 2,977 | 74 | 49 | 532 |
| wind | 19,331 | 4 | 12,828 | 3,888 | 1,414 | 1,201 |

---

# Interpretation

## Result 1 — No graph architecture beats the no-graph baseline

Paired over the five clean folds (df = 4, critical t = 2.776):

| comparison | mean | t | verdict |
|---|---|---|---|
| `sage` − `mlp` | -0.0176 | -1.34 | not significant |
| `gate` − `mlp` | +0.0003 | +0.05 | not significant |
| `dist` − `mlp` | -0.0099 | -0.63 | not significant |

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
| **`sage`** | **+0.0510** | **+3.39** | **5 / 5** |
| `gate` | +0.0250 | +1.63 | 4 / 5 |
| `dist` | +0.0154 | +1.02 | 3 / 5 |

**Message passing significantly widens the boundary penalty, same sign on every fold.**
The clearest confirmed mechanism in the study. The gate's repair of it is suggestive but
not established.

So: **we detect the pathology but not a performance consequence.**

## Result 3 — The gate measures neighbourhood trustworthiness

The learned gate value is interpretable — 1.0 means a node ignores its neighbours
entirely, 0.0 means it relies on them completely.

| setting | mean gate value |
|---|---|
| cross-disaster | **0.665** |
| in-domain | **0.544** |

**Paired difference -0.121 · t = -4.63 · 0/6 folds · p < 0.01.**

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

| fold | cross-disaster | in-domain | delta |
|---|---|---|---|
| tsunami | 0.3910 | 0.5891 | **+0.198** |
| flooding | 0.2167 | 0.3863 | **+0.170** |
| fire | 0.3559 | 0.4690 | **+0.113** |
| earthquake | 0.2253 | 0.3282 | **+0.103** |
| wind | 0.3485 | 0.4338 | **+0.085** |
| volcano | 0.3341 | 0.3782 | **+0.044** |

**Mean +0.119 · t = 5.17 · 6/6 folds positive · p < 0.01.** Binary detection
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

