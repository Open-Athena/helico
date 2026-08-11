# Folding from contacts instead of MSAs — results so far

**Status:** exploratory, results are from a *warm-started* model and use
**oracle contacts** (derived from the ground-truth structure). See
[Caveats](#caveats) before quoting any number.

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

## Headline result

**Given the true contact map, the model matches Protenix-with-MSAs.**

![Accuracy of contact-conditioned folding](.agents/project/figures/contact_conditioning_accuracy.png)

FoldBench, 28 protein targets scored by every arm (paired — same targets, same
scoring pipeline):

| Arm | lDDT | |
| --- | --- | --- |
| Protenix v1, single sequence | 0.327 | zero-shot |
| **Helico, contacts all-unknown** | **0.616** | fine-tuned |
| Protenix v1, with MSAs | 0.835 | zero-shot |
| **Helico, contacts given (100%)** | **0.850** | fine-tuned |

Paired differences:

| Comparison | Δ lDDT | t | improved |
| --- | --- | --- | --- |
| Helico: contacts off → on | **+0.234 ± 0.021** | 11.0 | 28/28 |
| Helico contacts-on vs Protenix + MSA | +0.016 ± 0.014 | 1.1 (n.s.) | 21/28 |
| Protenix: single sequence → +MSA | +0.508 ± 0.023 | 21.9 | 28/28 |

Read together: MSAs are worth +0.508 lDDT to Protenix on these targets, and
contact conditioning lands the model at the same place — the residual gap to
Protenix+MSA is not statistically distinguishable from zero. Starting from the
warm start at 0.244, the contacts-given arm reaches 0.820 within 1000 steps of
fine-tuning (see [Training progress](#training-progress)).

Controls that make the effect credible:

- **Empirical null.** 11 nucleic-acid-only targets have no protein contacts, so
  the two arms are identical by construction. Measured difference: +0.0004
  (sd 0.026) — the pipeline invents nothing.
- **Negative control.** An otherwise identical run whose contact pathway could
  not learn (see below) closed −1% of the gap through the same pipeline.
- **Zero-init no-op.** At step 0 the two arms differ by +0.0045 ± 0.0047
  (t=0.96) — no contact information reaches the model before training.

**Do not compare the two fine-tuned rows against the two zero-shot rows.**
Helico's contacts-withheld arm (0.616) beats zero-shot single-sequence Protenix
(0.327) by +0.289, but that is the expected payoff of fine-tuning a model for
the regime it is evaluated in, not evidence about contacts. The comparison that
isolates contacts is off-vs-on at the *same* checkpoint.

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

1. **Oracle contacts.** Contacts are computed from the ground-truth structure,
   so they leak the answer. This measures *structure realisation given a
   correct contact map*, not structure prediction. It is the right first
   experiment — it establishes the ceiling — but it is not comparable to
   published AF3/Protenix numbers, and the Protenix rows in the table above are
   genuine predictions while the Helico contacts-on row is not. The open
   question is how much accuracy survives *predicted* contacts, which is a
   sweep over false-positive/false-negative rates that has not been run.
2. **Warm start.** All runs initialise from Protenix v1, which was trained with
   MSAs. The model is being adapted, not trained from scratch.
3. **Fine-tuned vs zero-shot.** The Helico rows are fine-tuned for the no-MSA
   regime; the Protenix rows are zero-shot. Cross-comparisons between the two
   groups conflate "contacts help" with "fine-tuning helps". Only the
   same-checkpoint contacts off-vs-on comparison isolates contacts.
4. **The MSA module still exists.** `use_msa=False` removes the MSA *input*;
   the module is still constructed (~3M dead parameters). Deliberate for now, to
   keep warm starting simple.
5. **The 8000-step point is from a different run** than the 0-3000
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

1. **How accurate must contacts be?** The sweep over false-positive/false-negative
   rates that decides whether MarinFold-predicted contacts can drive this.
2. **Depth from scratch**, to remove the warm-start confound.
3. **Do partial contacts help proportionally?** 50% conditioning currently sits
   much closer to 100% than to 0% — worth understanding.
