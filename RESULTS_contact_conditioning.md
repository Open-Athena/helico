# Folding from contacts instead of MSAs — results so far

**Status:** exploratory. Results below are from a *warm-started* model.

**Real MarinFold predictions have now been tested, and the synthetic result does
not transfer** — see [Real predicted contacts](#real-predicted-contacts). At
matched precision and recall, real predictor errors cost a further 0.165 lDDT
beyond what our uniform noise model predicts. Treat the synthetic numbers as an
upper bound.

Weights: [timodonnell/helico](https://huggingface.co/timodonnell/helico)
(`contacts-msafree-01`, step 6000 — the checkpoint every number below describes).

Design doc and full research record:
[`.agents/project/20260806_contact_conditioned_folding.md`](.agents/project/20260806_contact_conditioned_folding.md).

---

## The question

Helico is an AlphaFold3 clone. AF3-family models lean heavily on multiple
sequence alignments: strip the MSA and accuracy collapses. MSAs are also the
expensive, slow, and least biologically satisfying part of the pipeline.

[MarinFold](https://github.com/Open-Athena/MarinFold) predicts residue–residue
side-chain contacts directly. So: **can a folding model take contacts as input
instead of an MSA, and reach the same accuracy?**

If yes, the alignment search comes out of the critical path and is replaced by a
contact predictor.

## What was built

- A three-state token×token contact matrix — `PRESENT` / `ABSENT` / `UNKNOWN` —
  computed by [pyconfind](https://github.com/Open-Athena/pyconfind) with the
  exact `contacts-v1` parameters MarinFold uses
  ([`src/helico/contacts.py`](src/helico/contacts.py)).
- The matrix is embedded and added into the pair representation `z_init`
  through a zero-initialised projection, so an untrained contact pathway is an
  exact no-op and warm starting from a Protenix checkpoint is lossless.
- Training samples the conditioning level per example — all-unknown, fully
  specified, and everything between — so one model serves any level of contact
  knowledge, including none.
- The MSA input is gated off (`use_msa=False`) for all runs here.
- Validation reports lDDT at 0%, 50%, 100%, and 100%-with-noise conditioning
  every validation step.

## Headline result (synthetic contacts — an upper bound)

**Given contacts degraded with our noise model to 60% precision and 60% recall,
an MSA-free model matches Protenix with MSAs.** With *real* MarinFold contacts at
a comparable operating point it does not — see
[Real predicted contacts](#real-predicted-contacts). The synthetic figure is the
ceiling this approach reaches if predictor errors were unstructured.

![MSA-free folding from contacts](.agents/project/figures/contact_conditioning_accuracy.png)

FoldBench, 27 protein targets scored by every arm (paired). Every Helico row is
genuinely MSA-free — no alignment, no conservation profile, at training or
inference.

| Arm | lDDT |
| --- | --- |
| Protenix v1, single sequence | 0.329 |
| Helico, contacts withheld | 0.316 |
| **Helico, contacts @ 60% precision / 60% recall** | **0.824** |
| Helico, oracle contacts (100%) | 0.836 |
| Protenix v1, with MSAs | 0.837 |

| Comparison | Δ lDDT | t | improved |
| --- | --- | --- | --- |
| MarinFold 60/60 vs contacts withheld | **+0.508 ± 0.027** | 18.6 | 27/27 |
| **MarinFold 60/60 vs Protenix + MSA** | **−0.013 ± 0.026** | −0.5 | 14/27 |
| oracle vs Protenix + MSA | −0.001 ± 0.025 | −0.0 | 16/27 |
| oracle vs MarinFold 60/60 | +0.012 ± 0.003 | 4.1 | 20/27 |

Two things matter here.

**Contact quality barely matters in this range.** Degrading a perfect contact
map to 60% precision and 60% recall costs **0.012 lDDT** — statistically real
(t=4.1) but tiny next to the +0.508 the contacts are worth. The contact map is
highly redundant: most of the fold is pinned by well under half of it, so losing
40% of contacts and adding 40% false ones is nearly free.

**So the predictor does not need to improve for this to work.** The 60/60 arm
lands within noise of Protenix+MSA (−0.013 ± 0.026). MarinFold's *current*
operating point is already enough to replace the alignment.

Both arms rise monotonically with training and plateau by ~step 4000; the
contacts-withheld arm is flat at ~0.31 throughout, which is the correct
signature for a model with no information to exploit.

### The load-bearing caveat

**The false positives are ours, not MarinFold's.** The 60/60 map is produced by
degrading the ground-truth map with our noise model, which draws false contacts
uniformly from the eligible region. Real predictor errors are spatially
correlated and concentrate near true contacts — geometrically plausible
near-misses are plausibly much harder to discount than uniform random ones,
because the model cannot reject them as inconsistent with everything else.

This established that *a* predictor at 60/60 would suffice **if its errors were
unstructured**. They are not: feeding real MarinFold output costs a further
0.165 lDDT at matched precision and recall
([Real predicted contacts](#real-predicted-contacts),
[helico#11](https://github.com/Open-Athena/helico/issues/11)).

Other controls:

- **Empirical null.** 11 nucleic-acid-only targets have no protein contacts, so
  the arms are identical by construction. Measured: +0.0004 (sd 0.026).
- **Zero-init no-op.** At step 0 the arms differ by +0.004 ± 0.005 (t=0.9).
- **Dead contact pathway.** A run whose contact projection never learned shows
  contacts off→on of +0.003 ± 0.003 (n.s.), so contacts leak nothing outside
  `linear_contact`.
- **No benchmark overlap.** 0 of the 49 FoldBench targets appear among the
  168,102 train-eligible structures.

## Real predicted contacts

The section above conditions on contacts derived from the answer, degraded with a
noise model whose false positives are drawn uniformly. This section feeds the
real thing.

![real vs synthetic contacts](.agents/project/figures/marinfold_real_contacts.png)

91 paired FoldBench monomer targets, all Helico arms MSA-free. Contacts come from
MarinFold `contacts-v1-exp199-1.5B` via
[MarinFold exp211](https://github.com/Open-Athena/MarinFold/issues/211)'s
rollouts. Each synthetic arm was generated at the precision/recall **measured for
its real counterpart**, so the real-vs-synthetic gap isolates error *structure*
from error *rate*.

Protenix v1 **and** v2 appear as baselines, each with and without MSAs. v2 runs
through the **official ByteDance implementation** (`protenix==2.0.0`, model
`protenix-v2`) rather than Helico's reimplementation, since v2 changes the
architecture. Both v2 arms use Protenix's own recommended inference settings
(5 samples / 10 cycles / 200 steps) — more compute than the Helico arms get at
3 samples / 6 cycles, which deliberately favours the baseline.

| arm | lDDT |
| --- | --- |
| no contacts | 0.368 |
| Protenix v1, single sequence | 0.386 |
| **Protenix v2, single sequence** | **0.409** |
| single rollout, top-L | 0.566 |
| **real MarinFold, top-L/5** | **0.575** |
| **real MarinFold, top-L/2** | **0.626** |
| **real MarinFold, top-L** | **0.638** |
| synthetic @ p=.795 r=.179 | 0.710 |
| synthetic @ p=.676 r=.379 | 0.790 |
| synthetic @ p=.505 r=.564 | 0.822 |
| Protenix v1 + MSA | 0.855 |
| oracle contacts | 0.862 |
| **Protenix v2 + MSA** | **0.865** |

n is 91 rather than 98 because FoldBench ships no pre-computed a3m for 7 targets,
so the Protenix MSA arm cannot run on them; every arm is restricted to the common
set so all comparisons stay paired.

### Against both Protenix generations

| comparison | Δ lDDT | t | better on |
| --- | --- | --- | --- |
| Protenix v2 vs v1, single sequence | +0.023 ± 0.011 | 2.2 | 51/91 |
| Protenix v2 vs v1, with MSA | +0.010 ± 0.004 | 2.5 | 60/91 |
| **real MarinFold vs v2 single sequence** | **+0.229 ± 0.022** | 10.4 | 76/91 |
| **v2 + MSA vs real MarinFold** | **+0.227 ± 0.019** | 11.8 | 87/91 |
| **v2 + MSA vs oracle contacts** | +0.004 ± 0.006 | 0.6 | 59/91 |

Three things follow.

**Real contacts beat the strongest single-sequence baseline.** v2 is genuinely
better than v1 without MSAs (+0.023), and real MarinFold contacts still clear it
by **+0.229 ± 0.022** on 76 of 91 targets.

**They still do not reach MSAs.** v2+MSA leads real contacts by
**+0.227 ± 0.019** on 87 of 91.

**Oracle contacts match the best MSA model.** v2+MSA vs oracle is
+0.004 ± 0.006 — indistinguishable. A perfect contact map is worth as much as an
alignment to the best available model; the entire shortfall is contact *quality*.

### Error structure dominates error rate

| budget | real − synthetic | t | worse on |
| --- | --- | --- | --- |
| top-L/5 | **−0.119 ± 0.018** | −6.5 | 77/98 |
| top-L/2 | **−0.147 ± 0.020** | −7.5 | 79/98 |
| top-L | **−0.165 ± 0.020** | −8.5 | 81/98 |

At top-L, degrading the oracle map to MarinFold's measured 50%/56% *rate* costs
0.037 lDDT; swapping uniform errors for real ones costs a further **0.165** —
about 4.5× more. Real predictor errors cluster near true contacts, where they are
geometrically plausible and cannot be rejected as inconsistent with the rest of
the map. The training noise model was the easy case.

### Against the honest single-sequence baseline

Stock Protenix v1 in single-sequence mode — original weights, depth-1 query-only
MSA, same 98 targets — scores **0.383**. Real MarinFold contacts beat it
decisively:

| vs stock Protenix single-sequence | Δ lDDT | t | better on |
| --- | --- | --- | --- |
| real MarinFold, top-L | **+0.239 ± 0.022** | +10.7 | 80/98 |
| real MarinFold, top-L/2 | +0.227 ± 0.021 | +10.8 | 81/98 |
| real MarinFold, top-L/5 | +0.180 ± 0.019 | +9.7 | 82/98 |
| single rollout, top-L | +0.168 ± 0.022 | +7.7 | 70/98 |

The control that makes this credible: **our own contacts-off arm scores 0.365,
slightly *below* Protenix's 0.383** (−0.018 ± 0.007, t=−2.63). The fine-tuned
model has no intrinsic advantage in the no-information condition — if anything it
gave up a little single-sequence capability during contact fine-tuning. All of
the +0.239 comes from the contacts.

### Where that leaves the project

The picture is two-sided. Real MarinFold contacts **clearly beat single-sequence
folding** — +0.257 ± 0.020 over no contacts (t=12.6, 87/98), +0.239 ± 0.022 over
stock Protenix single-sequence — and **clearly do not yet reach MSAs**: 0.622
against Protenix+MSA's 0.851. The contacts are doing real work; they are not yet
doing all of the MSA's work.

The 0.230 shortfall decomposes as roughly 0.028 (oracle vs MSA on this set) +
0.037 (error rate) + **0.165 (error structure)**. Structure dominates, which
makes the cheapest lever ours rather than MarinFold's: **sample training false
positives from near-miss pairs rather than uniformly**, so the model is trained
against the error distribution it actually faces.

Also measured:

- **Vote aggregation is worth +0.071 ± 0.008** (t=8.5) over a single rollout at
  the same budget, matching its ~0.10 precision advantage. Never mix the recipes.
- On monomers, Protenix+MSA edges out even oracle contacts by +0.028 ± 0.014
  (t=1.9, marginal) — unlike the assembly set, where they matched.

### The index map, which nearly broke this

MarinFold indexes residues into the published prompt (full SEQRES); the bench
derives its sequence from the *resolved* residues of the ground truth. Only 15 of
100 targets agree outright. 83 agree once prompt positions are mapped through
exp211's `resolved` list; 2 need real alignment and were dropped. A naive
identity map would have shifted contacts on 83% of targets and produced exactly
the same qualitative answer — "real contacts underperform" — for entirely the
wrong reason.

Verified by round trip: exp211's ground truth pushed through the map reproduces
helico's own `oracle_contact_state` at mean Jaccard 0.998 (min 0.984), pinned by
`tests/test_marinfold_export.py`.

## Two bugs that mattered

### The contact pathway could not learn

The contact projection is zero-initialised, and at the shared learning rate of
5e-5 **it never moved**. The first ~15k steps of training and the entire depth
sweep — about $1,100 of compute — measured a model that was ignoring its
contacts entirely, while reporting plausible-looking losses.

The fix is a per-parameter-group learning-rate multiplier
(`--contact-lr-multiplier=1000`); the contact weight norm then climbs to ~55 and
plateaus. `train/contact_weight_norm` is now logged every step so a dead
pathway is visible immediately rather than after a week of runs.

### `use_msa=False` did not disable MSAs

`use_msa` gated the MSA *module*. But `msa_profile` and `deletion_mean` — the
per-column conservation profile and insertion rate — live in `s_inputs`,
outside that module, and `build_s_inputs` read them unconditionally. Both
training ([`train.py`](src/helico/train.py) globbed the MSA tar indices
regardless of the flag) and benchmarking
([`modal/bench.py`](modal/bench.py) set `msa_server_url` unconditionally)
supplied them from real alignments.

So every "MSA-free" number before this fix was produced by a model receiving a
PSSM-style profile: for each residue, the frequency of all 32 residue classes
across up to 512 homologs, plus the mean insertion count. That encodes
conservation, tolerated substitutions, and gap structure — enough on its own for
secondary-structure and burial prediction. It does *not* contain co-evolution,
which is second-order and lives in the alignment's joint statistics.

Measured cost of the leak at step_8000:

| | with profile | MSA-free | Δ |
| --- | --- | --- | --- |
| contacts withheld | 0.622 | 0.311 | **+0.311 ± 0.021** (t=14.9) |
| contacts given | 0.853 | 0.841 | +0.012 ± 0.013 (t=0.9, n.s.) |

Two consequences. The apparent contact effect was understated — off→on is
+0.530 MSA-free, not +0.234 — because the profile was propping up the
contacts-withheld arm. And once the true contact map is supplied, the profile
adds nothing: contacts subsume the first-order conservation signal.

The fix gates `profile`/`deletion_mean` on `use_msa` in `build_s_inputs`, with
both `Helico.forward` call sites forwarding `config.use_msa`.
`TestNoMSALeak` poisons every MSA-derived batch key at once and asserts the
trunk output is unchanged, so a future feature that reintroduces alignment
information fails the suite.

### "Single sequence" meant three different things

Three distinct configurations were all being called "no MSA":

| | what it does | Protenix v1 lDDT |
| --- | --- | --- |
| `single_sequence_msa` | depth-1 MSA, row 0 = the query; module runs | **0.327** |
| `empty_msa` | depth-1 MSA of *gaps*; module runs | — |
| `use_msa=False` | MSA module never runs | 0.244 |

The first published single-sequence baseline here used the third: it removed a
module whose update the pairformer weights were trained to expect every
recycling iteration. That is a lesion, not a starved input, and it understated
Protenix by +0.082 ± 0.022 (t=3.80). The corrected baseline is 0.327.

This mattered because it made a fine-tuned model look like it beat a baseline it
was never fairly compared against. It does not affect the headline
contacts-off-vs-on result, which is same-checkpoint and paired.

## What did not hold up

The original proposal was to *also* shrink the trunk — the intuition being that
explicit contacts do the work the deep pairformer stack was doing implicitly.
That is wrong, at least under warm start: 48 blocks (0.815 val lDDT, +0.139
contact gain) beats 8 and 16 blocks (~0.66, ~+0.07) decisively. Explicit
contacts do not substitute for trunk depth.

This is confounded — every arm inherits Protenix's 48-block weights, which
favours the 48-block arm. A from-scratch sweep is still open.

## Caveats

These matter, and none of them are resolved yet.

0. **Synthetic contact errors — now measured.** The 60/60 arm draws false
   positives uniformly. Real MarinFold errors cost a further 0.165 lDDT at the
   same precision and recall, so every synthetic number here is an upper bound.
   See [Real predicted contacts](#real-predicted-contacts).
1. **Contacts come from the answer.** Even degraded, the contact map is derived
   from the ground-truth structure, so the Helico rows are conditioned on
   information a deployed system would have to predict. The Protenix rows are
   genuine end-to-end predictions; the Helico rows are not, and the two are not
   comparable as published numbers. What the comparison does establish is the
   *accuracy a contact predictor would have to reach* — and the answer is that
   60/60 is enough.
2. **Warm start.** All runs initialise from Protenix v1, which was trained with
   MSAs. The model is being adapted, not trained from scratch.
3. **Fine-tuned vs zero-shot.** The Helico rows are fine-tuned; the Protenix
   rows are zero-shot. Only the same-checkpoint contacts off-vs-on comparison
   isolates contacts.
4. **The noise model is not yet realistic.** See
   [Conditioning schedule](#conditioning-schedule-and-noise-model).
5. **The MSA module still exists.** `use_msa=False` removes the MSA *input*;
   the module is still constructed (~3M dead parameters). Deliberate for now, to
   keep warm starting simple.
6. **The 8000-step point is from a different run** than the 0-3000
   trajectory, so it is not directly comparable to it. See
   [Training progress](#training-progress).

## Training progress

Panel A is a single run (`contacts-lrmult1000`), so it is a real trajectory.
Step 0 is the warm start itself — Protenix v1 weights with `use_msa=False` and
the contact projection still at its zero initialisation.

| step | contacts given | contacts withheld |
| --- | --- | --- |
| 0 | 0.249 | 0.244 |
| 1000 | 0.820 | 0.557 |
| 2000 | 0.832 | 0.592 |
| 3000 | 0.837 | 0.560 |
| 8000 *(different run)* | 0.850 | 0.616 |

Almost all of the movement happens in the first 1000 steps; steps 1000 → 3000
add +0.017 with contacts given.

**Step 0 doubles as a control.** The contact projection is zero-initialised, so
conditioning should be an exact no-op there and the two arms should coincide.
Measured difference: **+0.0045 ± 0.0047 (t=0.96)** — consistent with zero. The
warm start is lossless and the pipeline is not leaking contact information
through some other path.

The in-training validation (panel C) covers steps 3000–9000 on a different
evaluation set and is flat over that range: averaged across the two runs with
full coverage, lDDT at 100% conditioning goes 0.759 (3000) → 0.795 (4000) →
0.801 (5000) → 0.805 (8000), while independent restarts at matched steps differ
by up to 0.029.

An earlier claim in this work — that a +0.013 FoldBench gain between two
checkpoints showed training was still helping — does not survive. Those
checkpoints came from different runs, and the effect is smaller than the
run-to-run spread.

## The validation set is sequence-contaminated

Use FoldBench numbers, not validation numbers, for anything absolute.

The train/val split is purely temporal (train < 2021-09-30, val
2022-05-01..2023-01-12, the AF3 convention). That removes almost no sequence
redundancy, because the PDB constantly re-deposits the same protein. Measured
against the 236k-entry manifest:

- **38.2%** of validation structures have a chain sequence appearing *verbatim*
  in training
- **18.4%** have every chain verbatim in training

Consequence, measured on one checkpoint:

| | val `@contacts0` | FoldBench, contacts off |
| --- | --- | --- |
| `contacts-msafree-01` step 500 | 0.680 | 0.289 |
| Protenix, single sequence | — | 0.329 |

The validation number is largely recall of memorised folds: given a sequence
seen in training, the model does not need conservation signal to place the
backbone. That is why removing the MSA profile cost +0.311 on FoldBench and
nothing at all on validation.

Between-level comparisons still hold — all conditioning levels score identical
structures under a fixed seed, so the conditioning *curve* is paired and
meaningful. It is the absolute values that are inflated.

## Conditioning schedule and noise model

What training samples per example ([`contacts.py`](src/helico/contacts.py)):

| mode | share | what it does |
| --- | --- | --- |
| `none` | 15% | everything unknown |
| `full` | 15% | every eligible pair specified |
| `pair-subset` | 35% | reveal a fraction of *pairs*, rest unknown |
| `contact-list` | 35% | MarinFold-shaped: a truncated top-k contact list |

MarinFold's operating point (2026-08) is **~60% precision, ~60% recall**, output
as a truncated top-k list. Three things were wrong for that and are now fixed:

1. **False positives landed where a predictor cannot produce them.** The FP
   candidate set was the whole upper triangle. Measured: **~40% of injected FPs
   were structurally impossible** — 8.7% at `|i-j| < 6` (filtered out of the
   true set) and 31% on non-protein tokens (pyconfind emits protein side-chain
   contacts only). That gave the model a free "this one is fake" cue that will
   not exist at deployment. FPs are now restricted to the eligible region;
   measured impossible FPs after the fix: **0**.
2. **A top-k list cannot assert absence.** `contact-list` previously flipped a
   coin and marked all unlisted eligible pairs ABSENT half the time. With
   truncation, an unlisted pair means "did not make the cut", which is
   uninformative — asserting millions of true negatives the predictor never
   claimed. Unlisted pairs now stay unknown.
3. **The noise range never reached the real operating point.** `eps_fp` counts
   false contacts relative to revealed ones, so precision `p` needs
   `eps_fp = (1-p)/p`. At p=0.6 that is **0.667**, well outside the old
   `U(0, 0.3)` — which corresponds to precision ≥ 0.77. The model had never
   been trained anywhere near where it has to work. `contact-list` now samples
   a precision in `[0.4, 1.0]` and derives `eps_fp`.

`conditioning_from_precision_recall(p, r)` converts an operating point into
`(reveal, eps_fp, eps_fn)`; recall loss folds into `reveal` because a top-k list
has no separate "reported but wrong" channel. A new validation level
`@contactsMarinFold` is pinned to the constants, so the tracked curve includes
the deployment condition rather than only bracketing it.

Remaining known gaps, not yet addressed:

- **`@contacts50` is not "half the information".** `pair-subset` at 0.5
  specifies half of *all pairs*, and since contacts are ~0.1% of pairs that
  asserts a very large number of true negatives. This is why the 50% level
  tracks close to 100% rather than sitting midway.
- **Revealed contacts are a uniform random subset.** A predictor finds
  high-confidence contacts first. pyconfind returns a *degree* per contact,
  currently thresholded and discarded — weighting reveal probability by degree
  would model this.
- **Errors are independent.** Real predictor errors are spatially correlated.
- **False positives are uniform within the eligible region.** Real ones
  concentrate near true contacts (near-miss pairs just beyond the distance
  threshold).

## Reproducing

```bash
uv run python .agents/project/figures/contact_conditioning_accuracy.py
```

Benchmark arms (oracle contacts on/off) are produced by `modal/bench.py`:

```bash
HELICO_BENCH_ORACLE_CONTACTS=1 modal run --detach modal/bench.py --checkpoint /ckpts/<run>/step_8000.pt --output-dir bench_on
```

Protenix single-sequence uses `HELICO_BENCH_SINGLE_SEQ=1` (depth-1 MSA whose one
row is the query). `HELICO_BENCH_NO_MSA=1` is a *different*, harsher ablation
that removes the MSA module outright — see
["Single sequence" meant three different things](#single-sequence-meant-three-different-things).

## Open questions

1. **Feed real MarinFold predictions** —
   [helico#11](https://github.com/Open-Athena/helico/issues/11). The remaining
   gap between this result and a working system: synthetic 60/60 suffices, real
   60/60 is untested. The design pairs every real-contact arm with a synthetic
   arm at the *same measured* precision/recall, so any gap isolates error
   *structure* from error *rate*.
2. **How accurate must contacts be, exactly?** 60/60 costs only 0.012 vs a
   perfect map. The floor is somewhere below 60% — worth locating, since it sets
   how much predictor headroom exists.
2. **Depth from scratch**, to remove the warm-start confound.
3. **Do partial contacts help proportionally?** 50% conditioning currently sits
   much closer to 100% than to 0% — worth understanding.
