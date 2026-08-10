# Contact-conditioned Helico: evaluation and implementation plan

**Date:** 2026-08-06
**Status:** phases 0–5 implemented; 6–8 (training runs) pending — see §10
**Scope:** replace the MSA-based trunk with a shallow trunk conditioned on a
three-state residue/residue contact matrix computed by
[pyconfind](https://github.com/timodonnell/pyconfind), using the same
parameters as [MarinFold](https://github.com/Open-Athena/MarinFold)
`contacts-v1`.

---

## 1. The proposal

Turn Helico from an MSA-based structure *predictor* into a contact-conditioned
structure *realizer*:

- **Drop MSAs entirely.**
- **Keep** all existing token- and atom-level features (restype, reference
  conformers, token bonds, relative position encoding).
- **Add** a token x token matrix with three states — `contact`, `no-contact`,
  `unknown`.
- **Shrink the trunk** from 48 Pairformer blocks to "a few".
- **Train across conditioning levels**, from an all-`unknown` matrix to a fully
  specified one.

Contacts come from `pyconfind.analyze(...)` with MarinFold's exact settings, so
a MarinFold contact-prediction model can drive Helico at inference time.

---

## 2. Verdict

**The idea is sound and the data supports it.** I validated the load-bearing
assumptions empirically (§3). Three things need to change or be added relative
to the proposal as stated:

1. **"A few layers" should be a measured quantity, not an assumption.** The
   contact matrix makes the *inference* problem much easier but leaves the
   *geometric* problem — turning pairwise constraints into coordinates — mostly
   intact. That problem is what triangle updates solve, and it needs depth
   proportional to the contact graph's diameter. Make trunk depth a config knob
   and sweep it (§7.1).

2. **Training must corrupt contacts, not just mask them.** The proposal
   describes masking (specify some fraction, leave the rest `unknown`). Masking
   alone teaches the model that every asserted contact is true. Real predicted
   contacts have false positives and false negatives, so a mask-only model will
   be brittle exactly where it matters. Add a corruption model (§6.3). This is
   the single highest-risk omission.

3. **Masking mode matters as much as masking rate.** A contact predictor emits
   a *contact list*, not a uniformly-sampled subset of matrix cells. Those are
   different conditioning distributions and a model trained on one transfers
   poorly to the other. Train on a mixture (§6.2).

One framing caveat that affects how results get reported: evaluating with
contacts derived from the ground-truth structure **leaks the answer**. Those
numbers measure structure realization, not structure prediction, and are not
comparable to AF3/Protenix/FoldBench baselines. See §8.3.

---

## 3. Evidence

All measurements below were run against this worktree with `pyconfind` 0.6.0
and real PDB entries. Scripts are in the session scratchpad; §5.1 promotes them
into the repo.

### 3.1 pyconfind integrates cleanly and cheaply

Stress-tested over 29 diverse entries (globular monomers, homo-oligomers,
antibody Fab/Fc, GPCRs, viral spike, RNA polymerase complex, photosystem II,
DNA-only):

| metric | result |
|---|---|
| structures processed | 28 OK, 0 failures (1 skipped upstream by `parse_mmcif`) |
| residue-identity round-trip | exact on every structure |
| throughput | **1.64 ms per protein residue**, linear in size |
| largest tested | 5XNL — 34,586 tokens / 9,364 protein residues in 15.3 s |
| DNA-only input (1BNA) | 0 protein residues, handled gracefully |

Extrapolating to the 236,326-structure processed set at ~400 residues each:
**~43 core-hours**, i.e. ~1.4 h on a 32-core Modal container. The existing
preprocess pass is ~85 min, so this roughly doubles preprocessing cost — not a
blocker.

### 3.2 Contact density is consistent and close to L

At `min_seq_separation >= 6` (MarinFold's definition), contacts per residue:

```
1UBQ 0.88   1MBN 0.83   3HTB 0.98   1AKE 0.96   4HHB 0.75   6VXX 1.09
1HTM 1.15   2SRC 1.08   1A2P 1.16   1GFL 1.23   7BV2 1.10   6M0J 1.11
1IGT 1.09   1CRN 0.61   1LYZ 0.96   1TIM 0.96   1BRS 1.16   1F88 0.91
```

Mean ~0.95, range 0.6–1.2. So a length-L protein gets ~L contacts —
comfortably above the ~L/2 long-range contacts classically needed to determine
a fold.

### 3.3 A contact is a sharp geometric constraint

Pooled over six structures, comparing pyconfind contacts against rep-atom
(CB, or CA for Gly) distances:

| | n | median | p5 | p95 | max |
|---|---|---|---|---|---|
| contacts | 4,368 | **6.2 Å** | 4.2 | 9.6 | 13.1 |
| non-contacts | 4,507,842 | 57.7 Å | 20.5 (p5) | — | — |

Separation is remarkably clean:

```
d < 10 Å:  97.0% of contacts,  0.42% of non-contacts
d < 12 Å:  99.7% of contacts,  0.94% of non-contacts
d < 14 Å: 100.0% of contacts,  1.60% of non-contacts
```

**A `contact` label asserts d < ~13 Å; a `no-contact` label asserts d > ~12 Å.**
The three-state matrix is effectively a high-quality binarized distance matrix
thresholded near 12 Å.

This has an important consequence that the proposal's framing understates: the
information in a *fully specified* matrix is far greater than "L contacts"
suggests, because each of the ~99.9% of pairs labeled `no-contact` is itself a
"> 12 Å" constraint. Full specification is close to a complete coarse distance
matrix — which is why a shallow trunk is plausible *at high conditioning*.

### 3.4 Signal is strong and monotone in conditioning level

Optimizer-free measurement: Spearman rho between graph-shortest-path distance
over the revealed contact graph and true CB–CB distance.

| pdb | nres | 100% | 50% | 25% | 10% | 5% |
|---|---|---|---|---|---|---|
| 1UBQ | 76 | 0.83 | 0.71 | 0.65 | 0.41 | 0.37 |
| 1MBN | 153 | 0.82 | 0.75 | 0.72 | 0.61 | 0.67 |
| 3HTB | 163 | 0.89 | 0.83 | 0.80 | 0.78 | 0.65 |
| 1AKE | 428 | 0.93 | 0.90 | 0.84 | 0.88 | 0.78 |
| 1SHG | 57 | 0.84 | 0.75 | 0.68 | 0.53 | 0.36 |
| 1PGB | 56 | 0.75 | 0.66 | 0.62 | 0.35 | 0.38 |
| 1TIM | 494 | 0.95 | 0.90 | 0.82 | 0.71 | 0.67 |
| 2SRC | 449 | 0.93 | 0.87 | 0.82 | 0.71 | 0.62 |

rho = 0.75–0.95 at full conditioning, degrading smoothly to ~0.4–0.8 at 5%.
**The conditioning axis the proposal asks for is real and well-behaved** —
there is a genuine continuum between "knows nothing" and "knows the fold".

### 3.5 …but naive decoders leave most of it on the table

Classical MDS on those same graph distances reconstructs poorly: TM 0.07–0.48
at 100% conditioning (best: 2SRC 0.48, 1TIM 0.42 — the large, well-connected
proteins). Gradient refinement with hinge losses did not improve on it.

This is a known limitation of MDS on contact graphs (graph paths overestimate
Euclidean chords, inflating the structure), **not** evidence that the
information is absent — §3.3 shows it is present. The reading is:

> The contact matrix determines the fold to a good approximation, but
> extracting it requires a decoder with a real structural prior. That is
> exactly what the diffusion module is. It also means **trunk capacity is
> load-bearing** — this is the empirical basis for recommending a depth sweep
> rather than committing to "a few layers" up front.

---

## 4. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Trunk too shallow to propagate transitive constraints | **high** | Keep triangle ops; sweep depth 2/4/8/16/48 (§7.1) |
| R2 | Train on clean contacts → brittle to predicted contacts | **high** | Corruption model, §6.3 |
| R3 | Uniform pair masking ≠ contact-list conditioning | **high** | Mode mixture, §6.2 |
| R4 | Oracle-contact eval leaks GT; not comparable to AF3 | **high** (reporting) | Frame as realization; report predicted-contact numbers separately (§8.3) |
| R5 | Contact maps are mirror-invariant | medium | Atom-level ref conformers + diffusion carry chirality; add an explicit mirror check (§8.2) |
| R6 | Modified residues tokenized per-atom by Helico, per-residue by pyconfind | medium | Group tokens by `(chain_idx, res_idx)`; attribute only to single standard protein tokens (§5.2) — **validated** |
| R7 | Dropping MSA discards Protenix warm-start value | medium | Keep `s_inputs` width, zero the MSA channels (§7.2) |
| R8 | New pair feature silently dropped at crop time | medium | `_subset_features` is a whitelist — must add explicitly (§5.4) |
| R9 | `--crop-size` is silently overridden by the stage schedule | low (pre-existing) | Fix at `train.py:613` (§9) |
| R10 | `modal/bench.py` hardcodes 2 config fields; new ones dropped | medium | Fix to pass through all fields (§9) |

### Note on R1

With full conditioning, the trunk's job is closer to distance-geometry
embedding than to coevolution inference — genuinely easier, so fewer blocks
should suffice. But constraint propagation over a contact graph needs depth on
the order of the graph diameter (~5–10 hops for a globular protein), and
triangle multiplicative update is precisely the operation that lets pair `(i,k)`
see `(i,j)` and `(j,k)`. My prior is **8–16 blocks**, not 2–4. The sweep is
cheap and settles it.

### Note on R4

This is a reporting hazard, not a technical one. A FoldBench run with contacts
derived from the answer will produce numbers that look spectacular and mean
something different from what readers will assume. Every such number needs to
carry the label "oracle contacts". The scientifically meaningful end-to-end
number requires MarinFold-predicted contacts.

---

## 5. Data pipeline

### 5.1 New module: `src/helico/contacts.py`

Promote the validated prototype. Public surface:

```python
PYCONFIND_KWARGS = dict(          # MarinFold contacts-v1, verified against
    native_only=True,             # experiments/exp74_.../pyconfind_contacts.py
    contact_distance=3.0,
    dcut=25.0,
    clash_distance=2.0,
    assembly=None,
)
MIN_CONTACT_DEGREE = 0.001
MIN_SEQ_SEPARATION = 6

def compute_contacts(tokens, rotamer_library) -> tuple[list[tuple[int,int]], list[int]]:
    """-> (contact edges as token-index pairs, eligible token indices)."""
```

Add `pyconfind>=0.6` to `pyproject.toml` dependencies (it pulls `gemmi` and
`pandas`).

### 5.2 Token → pyconfind mapping (validated)

The mapping is the one non-obvious part. pyconfind numbers positions by
input order, so building the `gemmi.Structure` ourselves makes
`Position.index` a direct index into our own ordering — no author/label chain-id
alignment needed. (Helico parses `label_asym_id`/`label_seq_id`; pyconfind reads
author numbering, so path-based alignment would otherwise be required.)

Rules, all verified over the 28-structure stress set:

1. Group tokens by `(chain_idx, res_idx)` to rebuild true residues. **Required**
   — Helico atom-tokenizes modified residues (MSE, SEP, TPO, …) because they
   miss the `THREE_TO_ONE` fast path in `tokenize_structure`, while pyconfind
   treats them as one residue. Grouping keeps them present as occluding geometry.
2. Skip groups whose residue name is not in `pyconfind.pdb.LEGAL_RESIDUE_NAMES`
   (31 names: standard 20 + HIS variants + MSE/SEC/CSO/SEP/TPO/PTR).
3. One `gemmi.Residue` per group, sequential `seqid` per chain.
4. Attribute contacts **only** to groups that are exactly one token with
   `token_type <= 20` (standard protein residue). Everything else — modified
   residues, ligands, nucleotides — is `unknown`.
5. Assert `len(analysis.positions) == len(slots)`; raise, don't silently
   misalign.

### 5.3 Storage: sparse, following the `token_bonds` precedent

Add two fields to `TokenizedStructure` ([data.py:768](src/helico/data.py:768)):

```python
contact_edges: list[tuple[int, int]] | None = None   # sparse; ~L entries
contact_eligible: np.ndarray | None = None           # (N_tok,) bool
```

Dense storage is not an option — the existing comment at
[data.py:776](src/helico/data.py:776) records that a dense `(N_tok, N_tok)`
`token_bonds` blew preprocess workers to 200+ GB RSS on ribosomes and capsids.
Contacts are ~1 per residue, so sparse is a natural fit.

`contact_eligible` is what makes the third state expressible: it records which
tokens pyconfind actually analyzed, so `no-contact` (both endpoints analyzed,
no contact found) is distinguishable from `unknown` (at least one endpoint not
analyzed).

Backward compatibility: old pickles lack both attributes. Use
`getattr(self, "contact_edges", None)`, mirroring the existing shims at
[data.py:2913](src/helico/data.py:2913) and [data.py:970](src/helico/data.py:970).

### 5.4 Materialization, cropping, collation

**`to_features()`** ([data.py:947](src/helico/data.py:947)) — densify to a
`(N_tok, N_tok)` `uint8` with `0=unknown, 1=no-contact, 2=contact`:

```
known(i,j) = eligible[i] & eligible[j] & (chain[i] != chain[j] or |res[i]-res[j]| >= 6)
state(i,j) = 2 if (i,j) in edges else (1 if known(i,j) else 0)
```

Note the seq-separation rule is applied **only within a chain** — MarinFold is
single-chain so its spec does not address this, but sequence separation is
meaningless across chains and every inter-chain pair should be knowable. Also
note same-chain pairs with `|i-j| < 6` become `unknown`, not `no-contact`: we
never determine them, and labeling them `no-contact` would be a lie (they are
frequently in contact).

**`_subset_features()`** ([data.py:2280](src/helico/data.py:2280)) — add
`result["contact_state"] = features["contact_state"][idx][:, idx]` next to
`token_bonds`. This function is a **whitelist**: any key not listed is silently
dropped, which would desync the feature after cropping with no error.

**`collate_fn()`** ([data.py:2399](src/helico/data.py:2399)) — pad exactly like
`token_bonds`, but pad with **0 (`unknown`)**, which `F.pad` gives for free.

**`make_synthetic_batch()`** ([data.py:2620](src/helico/data.py:2620)) — emit a
`contact_state`. The `--synthetic` training path goes through the real
tokenizer, so it picks the feature up automatically; this generator
hand-builds the batch dict and needs the key added. Note ~9 call sites across
tests and probes depend on it.

### 5.5 Preprocessing

`_process_single_structure` ([data.py:2699](src/helico/data.py:2699)) computes
contacts before pickling. The rotamer library must be loaded **once per worker**
in `_init_worker` ([data.py:2693](src/helico/data.py:2693)) — it is a ~3.4 s
parse, and per-structure loading would dominate runtime. `pyconfind` memoizes
in `DEFAULT_ROTAMER_LIBRARY`, but pass the `RotamerLibrary` explicitly to be sure.

This forces a **full re-preprocess** (~1.4 h on 32 cores) and a re-upload of the
processed dataset. Bump the schema marker in
[modal/upload_to_hf.py:127](modal/upload_to_hf.py:127) alongside
`token_bonds_format`.

---

## 6. Conditioning: masking and corruption

Applied per-example at data-loading time (not in the model), so it is visible to
tests and reproducible from a seed.

### 6.1 Where

In `collate_fn`, or a transform in `__getitem__`. Must be **per-example, not
per-batch** — a batch-level conditioning level halves the effective diversity per
step at `batch_size=1` and correlates the signal with the crop.

Symmetry is mandatory: build the upper triangle, then `m |= m.T`.

### 6.2 Modes

Sample a mode per example. This is the part the original proposal does not
cover, and it matters more than the rate:

| mode | p | behavior |
|---|---|---|
| `none` | 0.15 | everything `unknown` — the MSA-free ab initio baseline |
| `full` | 0.15 | every eligible pair specified |
| `pair-subset` | 0.35 | reveal fraction q ~ U(0,1) of eligible pairs; rest `unknown` |
| `contact-list` | 0.35 | reveal fraction q ~ U(0,1) of *contacts*; all other eligible pairs → `no-contact` (coin flip) or `unknown` |

`contact-list` is the mode that matches a real contact predictor: MarinFold
emits a list of asserted contacts, and everything it does not name is either
implicitly non-contact (if the list is meant to be complete) or unknown (if
truncated — and MarinFold *does* truncate to an 8192-token budget, dropping the
weakest contacts). Both readings occur in practice, hence the coin flip.

Including `none` and `full` as explicit atoms — rather than relying on
q ~ U(0,1) to reach the endpoints — matters because the endpoints are the two
headline operating points and uniform sampling almost never hits them exactly.

### 6.3 Corruption (do not skip this)

Applied in every mode except `none`, parameterized in **precision/recall terms**
so it maps onto how contact predictors are actually reported:

- **False negatives:** drop a fraction `eps_fn ~ U(0, 0.3)` of revealed contacts
  (to `no-contact` or `unknown`, matching the mode).
- **False positives:** add `n_fp = eps_fp * n_revealed_contacts` spurious
  contacts, `eps_fp ~ U(0, 0.3)`.

Expressing the FP count relative to the *number of true contacts* rather than
the number of pairs is the right parameterization — contacts are ~0.1% of pairs,
so a "1% of pairs" FP rate would swamp the true signal tenfold.

Refinement worth doing once the basic version trains: draw a share of the false
positives from pairs at 10–20 Å (near-misses) rather than uniformly. Real
predictors' errors are overwhelmingly near-misses, and uniform FPs are so
geometrically absurd that the model can learn to ignore them without learning
robustness. Coordinates are available at collate time, so this is cheap.

A held-out set of *fixed* (mode, rate, corruption) settings should be reserved
for validation so the robustness curve in §8.1 is measured, not sampled.

### 6.4 Curriculum

Sample uniformly from the start rather than annealing. This mirrors the existing
MSA row-subsampling pattern ([msa.py:319](src/helico/model/msa.py:319)), gets the
full conditioning curve from a single run, and avoids a schedule to tune. If
training proves unstable, fall back to annealing from high to low conditioning
via the existing `StageConfig` mechanism
([train.py:109](src/helico/train.py:109)).

---

## 7. Model changes

### 7.1 Contact embedding

Exactly parallel to `token_bonds`, which is the existing idiom for a
`(B, N, N)` pair feature:

```python
# Helico.__init__, next to linear_token_bond (model/helico.py:64)
self.linear_contact = linear_no_bias(3, config.d_pair, zeros_init=True)

# in BOTH forward (helico.py:120) and predict (helico.py:277)
contact_state = batch.get("contact_state")
if contact_state is not None:
    onehot = F.one_hot(contact_state.long(), 3).to(z_init.dtype)
    z_init = z_init + self.linear_contact(onehot)
```

Why this site: `z_init` is re-added at the top of every recycling iteration
([helico.py:130](src/helico/model/helico.py:130)), so the signal reaches
template, MSA, Pairformer, distogram, diffusion, and confidence for free.

`zeros_init=True` makes it an exact no-op at step 0, preserving warm-start
numerics — the pattern gh#9 established with `_init_distogram_proj_from_z`
([train.py:155](src/helico/train.py:155)), which was added after unwarmed
initialization produced grad norms of 30k–150k.

**One-hot into a Linear, not `nn.Embedding`** — matches `linear_token_bond`, and
extends cleanly if a fourth state or a continuous contact-degree channel is
added later.

> **Deliberately binary.** pyconfind returns a contact *degree* (0.001–~1.0),
> which is more informative than a binary label. It is excluded because
> MarinFold emits binary contacts; training on degree and inferring on binary
> would be a train/test mismatch. Revisit only if the inference-time source
> starts producing calibrated strengths.

### 7.2 Removing MSA

Model-side removal is small — `z` flows through `MSAModule` by plain assignment
([helico.py:132](src/helico/model/helico.py:132)), and nothing downstream reads
the MSA representation `m`. Delete the call in `forward` and `predict`, the
`build_msa_raw` calls, and the submodule.

**Keep `c_s_inputs` at 449.** 33 of the 65 non-`a_token` channels in `s_inputs`
are MSA-derived (`msa_profile` 32 + `deletion_mean` 1,
[features.py:118](src/helico/model/features.py:118)). Zero them rather than
removing them: shrinking to 416 would change `linear_sinit`,
`DiffusionConditioning.single_proj`, and `ConfidenceHead.linear_s1/s2`, breaking
Protenix warm-start for the diffusion and atom modules — which are unchanged by
this project, expensive to learn, and encode a lot of chemistry. The fallbacks at
[features.py:118-122](src/helico/model/features.py:118) already zero-fill when
the keys are absent, so this may require no code change at all beyond not
emitting the keys.

Gate on a config flag (`use_msa: bool = True`) rather than deleting code, so the
MSA baseline stays runnable for comparison.

### 7.3 Trunk depth

`n_pairformer_blocks` is already a config field with a CLI flag (`--n-blocks`).
No new plumbing needed — just sweep it. **Keep the triangle operations**; they
are the constraint-propagation mechanism (§4, R1). Depth is the thing to reduce,
not the block's internals.

### 7.4 Config plumbing checklist

A new `HelicoConfig` field must be added in **six** places or it silently
reverts to default:

1. `HelicoConfig` — [model/config.py:17](src/helico/model/config.py:17)
2. `TrainConfig` dataclass — [train.py:57](src/helico/train.py:57)
   (**not optional**: `asdict(config)` at [train.py:254](src/helico/train.py:254)
   is what lands in the checkpoint, and every loader reconstructs `HelicoConfig`
   from that dict by name)
3. argparse — [train.py:862](src/helico/train.py:862)
4. `TrainConfig(...)` construction — [train.py:912](src/helico/train.py:912)
5. `HelicoConfig(...)` construction — [train.py:936](src/helico/train.py:936)
6. `modal/train.py` — env var, `TRAIN_ARGS` ([modal/train.py:105](modal/train.py:105)),
   and `base_cli` ([modal/train.py:190](modal/train.py:190))

Plus `_TRAIN_ENV_MAPPING` in [experiment.py:504](src/helico/experiment.py:504) to
make it runnable through `ensure_training_run` — a gap gh#9 left open (it was
launched with raw `modal run` and never wired into the experiment library).

`infer_main` and `bench.py` pick new fields up for free via their
`hasattr(HelicoConfig, k)` filter. **`modal/bench.py:202` does not** — see §9.

### 7.5 Optional arm: contacts direct to diffusion

If the trunk ends up very shallow, the diffusion module may benefit from seeing
contacts directly rather than only through a thin `z`. gh#9 showed the diffusion
module works from a heavily bottlenecked pair input, so this is plausible.

Follow the gh#9 precedent exactly: add a **parallel** projection sized for the
extra channels ([diffusion.py:559](src/helico/model/diffusion.py:559)) rather
than widening `pair_proj`, so checkpoints round-trip in both directions. Treat
this as an ablation arm, not part of the main path.

---

## 8. Evaluation

### 8.1 The headline curve

**LDDT / TM vs conditioning fraction**, from 0% (ab initio, MSA-free) to 100%
(fully specified). This one plot is the result. Everything else supports it.

Second curve: **quality vs contact error rate** at fixed 100% conditioning,
sweeping `eps_fp` and `eps_fn` over a fixed grid. This measures the robustness
that §6.3 exists to create, and it is the number that predicts whether real
MarinFold contacts will work.

### 8.2 Ablations

- **Trunk depth** — 2 / 4 / 8 / 16 / 48 blocks at full conditioning. Settles R1.
  Run this early; it determines the shape of everything downstream.
- **Baselines** — MSA-free with all-`unknown` contacts (lower bound); current
  MSA model (reference). Following the gh#9 `none`-baseline pattern, the
  all-`unknown` arm should use the *same architecture with zeroed input* rather
  than reverting to the old architecture, so the comparison isolates the signal
  rather than the parameterization.
- **Mirror check** — fraction of predictions whose mirror image scores better
  than the original. Contact maps are reflection-invariant (R5); atom-level
  reference conformers should break the degeneracy, but it should be measured
  rather than assumed.
- **Corruption ablation** — train one arm without §6.3 corruption and compare
  both on corrupted-contact validation. Quantifies what the corruption model buys.

### 8.3 FoldBench and honest reporting

`predict_target` ([bench.py:370](src/helico/bench.py:370)) calls
`tokenized.to_features()` at [bench.py:406](src/helico/bench.py:406), so contacts
computed inside `to_features()` flow through automatically — but for a benchmark
target they would be computed from the *ground-truth* structure.

**Every such number must be labeled "oracle contacts".** It measures how well the
model realizes a structure given the answer's contact map, which is a legitimate
and interesting quantity — an upper bound on the MarinFold-driven pipeline — but
it is not structure prediction and is not comparable to AF3/Protenix results.
Reporting it unqualified would be actively misleading.

The end-to-end number requires MarinFold-predicted contacts. That needs a
contact source per target, which is new plumbing: `BenchTarget.extra` already
carries the target CSV row ([bench.py:153](src/helico/bench.py:153)) but is not
passed into `predict_target`, and `modal/bench.py:219` only forwards `pdb_id` and
`category`. Worth scoping as a follow-on once the oracle numbers justify it.

Note also that MarinFold `contacts-v1` is **single-chain only**, whereas
pyconfind handles complexes natively and ~10–15% of the contacts measured in
§3.2 on multi-chain entries were inter-chain. Helico can train on inter-chain
contacts; a MarinFold-driven pipeline initially cannot supply them. Train on
them anyway (they are free and correct) but expect the complex case to depend on
`unknown` handling until the contact source catches up.

---

## 9. Pre-existing bugs found along the way

Independent of this project, worth fixing before running experiments:

1. **`modal/bench.py:202` hardcodes two config fields** (`n_pairformer_blocks`,
   `n_diffusion_token_blocks`). Any other field is dropped, so a Modal bench
   silently runs a *different model* than the checkpoint specifies. This is
   already live for `diffusion_pair_source`, and would silently invalidate every
   contact-conditioned bench. **Blocking** — fix before any experiment.
2. **`--crop-size` cannot be swept.** [train.py:613](src/helico/train.py:613)
   overwrites `dataset.crop_size` from the stage schedule on every batch, so the
   CLI flag only sets a value that is immediately replaced. Unlike the adjacent
   `lr` case — which was fixed during gh#9's LR sweep — this one is *deliberate*
   (see the comment at [train.py:604](src/helico/train.py:604): "Crop-size
   staging below is unaffected"). Not a bug, but worth knowing before planning
   any experiment that varies crop size: it needs a `StageConfig` change, not a
   flag.
3. **A partially-resumed preprocess truncates the manifest.**
   `preprocess_structures` returns metadata only for structures processed *in
   that call*, and the `structures`/`all` branches then call `build_manifest`
   on it, overwriting `manifest.json`. When `skip_existing=True` skips
   *everything* the code correctly reloads the existing manifest, but when it
   skips only *some* files — the actual resume case, e.g. after a timeout —
   the manifest is rewritten containing just the newly-processed subset, and
   every previously-processed structure silently disappears from training.
   Not hit by a full `--no-skip-existing` run, but it makes resuming an
   interrupted preprocess unsafe. Worth fixing before anyone relies on resume.
4. `violation_loss` is imported at
   [helico.py:35](src/helico/model/helico.py:35) and never called.
5. `max_msa_len` at [data.py:2465](src/helico/data.py:2465) is computed and
   never used.

---

## 10. Phasing

| phase | work | status |
|---|---|---|
| **0** | `src/helico/contacts.py` + tests; `pyconfind` dep | **done** |
| **1** | `TokenizedStructure` fields, `to_features`, `_subset_features`, `collate_fn`, synthetic | **done** |
| **2** | Preprocess integration (rotamer library once per worker) | **done** — re-preprocess + re-upload still to run |
| **3** | `linear_contact` into `z_init` in `forward` + `predict`; config flags | **done** |
| **4** | Conditioning sampler + corruption | **done** |
| **5** | MSA gate; CLI/Modal/experiment plumbing; §9 item 1 | **done** |
| **6** | Depth sweep at full conditioning (2 / 4 / 8 / 16 / 48) | pending |
| **7** | Main training run | pending |
| **8** | Bench with oracle contacts; robustness curve | `oracle_contact_state` built; runs pending |

Phases 0–1 carried essentially all the integration risk, which §3 retired.
Phase 6 is the decision point for the architecture question.

### Dataset status: migrated (2026-08-07)

The `helico-train-data` Volume is **fully preprocessed with contacts**: all
236,326 structures, verified by sampling across the sorted subdir range
(`ak` … `zz`) at densities 0.86–1.11 contacts/residue. Phases 6–8 can run
against it as-is. Publishing to HF (`scripts/upload_to_hf.sh`) is still
outstanding.

To repeat or repair the migration — idempotent, so re-running after an
interruption processes only what is still missing:

```bash
HELICO_SKIP_DOWNLOAD=1 HELICO_STEP=structures HELICO_REQUIRE_CONTACTS=1 \
  modal run --detach modal/preprocess_on_modal.py
```

Training and benching are wired end to end:

```bash
HELICO_TRAIN_N_BLOCKS=8 HELICO_TRAIN_NO_MSA=1 \
  HELICO_TRAIN_RUN_NAME=contacts-depth8 modal run --detach modal/train.py
```

### How the migration actually went

Two full-corpus attempts were killed partway by their launching session exiting
— a plain `modal run` creates an *ephemeral* app tied to the client process.
**Use `--detach` for anything multi-hour.** Volume writes flush as they happen,
so the ~93% that had completed survived; only `build_manifest` never ran, and
since contacts don't change tokenization the existing manifest stayed valid.

Recovering the remaining ~6% exposed the deeper problem: `skip_existing` keys
off the pickle *existing*, so resuming skipped all 252K files and did nothing,
while `--no-skip-existing` meant redoing the whole corpus (~5 h) with the same
fragility. Hence `--require-contacts`, which skips a structure only when its
pickle already carries contacts. That reduced the repair to a 4-minute scan plus
~30 minutes of work, and made the operation idempotent — the property that
matters when runs keep getting interrupted.

Final numbers: scan found 220,065/236,326 already migrated; the run repaired
exactly the 16,261 missing, and the manifest merged to 236,326 (16,261 fresh +
220,065 carried). Without the §9 truncation fix that last step would have
written a 16,261-entry manifest and dropped 93% of the training set.

### Re-preprocessing gotchas

Three traps, all of which would have silently wasted a multi-hour run:

- **`skip_existing` defaults to `True`.** With `processed/structures` already
  populated from the previous run, the structures step skips every file and
  produces nothing — a silent no-op, not an error. The `structures` subcommand
  had `--no-skip-existing`; the `all` subcommand did not, and now does.
  `HELICO_NO_SKIP_EXISTING=1` is **required** after any schema change.
- **`maxtasksperchild=25` recycles workers.** Loading the rotamer library in
  `_init_worker` meant a ~3.4 s parse roughly 9,400 times over a full run (~9
  core-hours), plus a download race across concurrent workers on first use. It
  is now loaded once in the parent of `preprocess_structures` and inherited via
  fork COW, with a fallback in `_init_worker` for non-fork start methods so
  contacts can't be silently dropped.
- **`all` redoes work that doesn't need redoing.** It re-parses the 473 MB CCD
  and re-indexes 219 GB of MSA tars. `HELICO_STEP=structures` reuses both and
  still rebuilds the manifest.

The rotamer library is now baked into the preprocess image at build time, so a
long run cannot fail hours in on a transient GitHub download.

### Notes from the build

- **`make_synthetic_batch` must not draw from the global RNG.** The first
  implementation generated `contact_state` with `torch.rand`, which shifted the
  global stream that MSA row-sampling consumes and broke three numerical pins in
  `tests/test_snapshots.py`. It now uses a dedicated seeded generator.
- **Test the projection, not the forward pass.** The zero-init no-op property
  was originally tested by comparing two whole forward passes for exact
  equality; that is a proxy at the mercy of cuEquivariance kernel
  nondeterminism. The tests now hook `linear_contact` and assert on its output
  directly, which is exact.
- **Contacts closer than `MIN_SEQ_SEPARATION` must be `unknown`, not
  `contact`.** pyconfind reports contacts at every separation, but MarinFold
  filters short-range ones out before emitting. Marking them present would train
  on a signal no contact predictor will ever supply. Guarded by
  `test_present_respects_seq_separation`.
- **Oracle-contact index mapping needs no sequence alignment.** Because
  `structure_to_chains` derives each input sequence from the ground truth's
  resolved residues in order, the k-th protein residue of chain C corresponds
  positionally on both sides. Verified exact (contact-count ratio 1.000) across
  six structures including ones where crystallization aids shift the token
  count. `oracle_contact_state` still checks per-position residue identity and
  bails below 0.9, since a silent misalignment would scramble every contact.

Per [`experiments/AGENTS.md`](experiments/AGENTS.md) and the memory note that
baselines and characterizations default to `main`, phases 6–8 should each be
scaffolded as a numbered experiment against a GitHub issue.

---

## 10b. Results and infrastructure findings (2026-08-08/10)

### The depth sweep was inconclusive

Five arms (2/4/8/16/48 blocks, MSA-free, 5000 steps, ~$796). Mean
`val/lddt_hard` per arm, averaged over all validation points:

| arm | full | @none | full − none |
|---|---|---|---|
| d2 | 0.590 | 0.547 | +0.043 |
| d4 | 0.635 | 0.614 | +0.021 |
| d8 | 0.618 | 0.638 | −0.020 |
| d16 | 0.628 | 0.670 | −0.043 |
| d48 | **0.716** | 0.707 | +0.009 |

Mean effect **+0.002**, sign inconsistent across arms. Step-to-step SD was
~0.08 — roughly 2x any conditioning difference — and `@noisy` frequently beat
full conditioning, which a model genuinely using contacts would not do.

Three candidate explanations, unseparated: `@none` is already 0.55–0.71 so the
warm start leaves little headroom; 5000 steps at LR 5e-5 with grad norms of
1e4–1e5 against a clip of 1.0 means nearly every update was fully clipped, and
`linear_contact` starts at exactly zero; and n=16 validation batches cannot
resolve a ±0.04 effect. Depth is separately confounded — d48 inherits all 48
Protenix blocks while d2 keeps two, so the sweep partly measures *how much warm
start each arm retained*.

**The noise floor, measured directly.** Three consecutive validations of one
unchanged model gave full-vs-none gaps of −0.025, +0.179, −0.114 at n=8 per
level. Any effect below ~0.2 lddt is unmeasurable at that sample size.

### Infrastructure bugs found (all pre-existing unless noted)

- **DDP validation killed every run.** Validation is rank-0-only with no
  barrier, so other ranks waited inside a gradient all-reduce with NCCL's
  10-minute watchdog running; overrunning it aborts them all. Sweeping four
  conditioning levels made this far likelier (*my regression*). Fixed with a
  `val_max_seconds` cap, a 2h NCCL timeout, and a post-validation barrier.
  Caveat: d4/d8/d16b/d48 all died at *exactly* step 4675 despite different
  cadences, which suggests a data-dependent CUDA fault as well; unreproduced.
- **lDDT inflated by `n_diffusion_samples`.** `smooth_lddt_loss` broadcast
  `(B*N_d, N, 3)` predictions against `(B, N, 3)` truth, summing N_d samples
  over a one-row denominator. Reported lDDT of 2–7 for a [0,1] metric; exactly
  8.0x at N_d=8. Metric-only. Confirmed fixed in-run: `LDDT` and `LDDT_h`, from
  independent code paths, now agree (0.240 vs 0.241).
- **Oracle contacts silently off in the benchmark.** `modal/bench.py` read
  `HELICO_BENCH_ORACLE_CONTACTS` as a module global; Modal re-imports the
  module inside the container where the launching env does not exist, so it was
  always False. The bench ran without contacts and reported success — the
  contacts-on and contacts-off arms would have been byte-identical. Now baked
  into the image env, with a per-target `oracle_contacts=ON/OFF` log line.
- **`uv run modal` resolved to the anaconda binary**, which cannot import
  helico. Harmless for train/upload (their entrypoints don't import it), fatal
  for bench. Fixed by installing modal into the venv.

### Current experiment

`contacts-48b-v2`: 48 blocks, MSA-free, contact-conditioned, 15000 steps on
8xH100 (~20.5h, ~$650), warm-started from Protenix. Validation every 500 steps
at 200 samples reports `val/{lddt_hard,gdt_ts,rmsd}@contacts{0,50,100}` plus
`@contacts100noisy` and a `_gain` (100% − 0%) series.

Planned eval: two full FoldBench runs on the final checkpoint with oracle
contacts off and on, paired per target. Rank by the contacts-off score, take
the failures as the "MSA usually required" set, and report paired deltas with
counts — not means, given the noise floor above.

## 10c. The contact pathway could not learn (2026-08-10)

### What was wrong

Everything before this point — the depth sweep and the 15,000-step run — was
measuring a model that ignored contacts, because `linear_contact` never left
its zero initialisation.

`linear_contact` is deliberately zero-init so that enabling contacts is an
exact no-op on a warm-started checkpoint (§7.1). But at the shared LR of 5e-5
it then has to travel from *exactly* zero while the rest of the trunk is
already near a good solution. Adding `train/contact_weight_norm` made this
visible immediately: at 1x the projection's norm after 50 steps extrapolates
to ~0.0014, against ~1.0 for the sibling projections feeding `z_init`.

`--contact-lr-multiplier` puts the projection in its own AdamW param group.
At 1000x its norm reaches ~55 within 500 steps and then **plateaus** rather
than diverging — it finds a scale and settles.

### Result: the conditioning curve appears

`val/lddt_hard` on the paired validation subset, 48 blocks, MSA-free:

| step | 0% | 50% | 100% | gain |
|---|---|---|---|---|
| 500 | 0.648 | 0.736 | 0.831 | +0.184 |
| 1000 | 0.652 | 0.757 | 0.769 | +0.117 |
| 1500 | 0.675 | 0.768 | 0.820 | +0.145 |

Monotone in conditioning level at every point, and — unlike every earlier
apparent signal — it survives repeat validation. At 1x the same measurement
gave a mean of +0.016 (t=1.33) drifting toward zero.

### The MSA gap, and what fraction contacts close

Three-way paired FoldBench on 31 protein-containing targets (identical
targets, 3 samples, 6 cycles):

| arm | mean lddt |
|---|---|
| Helico, contacts off (single-sequence) | 0.644 |
| Helico, contacts on (1x checkpoint) | 0.642 |
| **Protenix v1 + MSA** | **0.840** |

**MSA gap = +0.195 +/- 0.017 (t=12.2)** — a well-resolved effect, which also
demonstrates the instrument can detect a real difference when one exists. On
the 1x checkpoint contacts closed -1% of it, as expected for a pathway pinned
at zero. The in-training +0.145 would correspond to ~74% closure if it holds
on FoldBench; that measurement is pending.

**Empirical null.** pyconfind is protein-only, so on `monomer_rna` /
`monomer_dna` targets `oracle_contact_state` returns None and both arms get
byte-identical input. Those 11 targets give a run-to-run noise floor of
sd 0.031, range [-0.008, +0.084] — three of the six largest apparent "gains"
in the first analysis were nucleic-acid targets, i.e. provably not caused by
contacts. Any future contact claim must clear this null.

### On the LR multiplier

100x initially looked like a failure regime (gain -0.069 then -0.029) and was
stopped early. Its final validation, which landed as it was being killed,
read **+0.109** with the weight norm at 30 and still climbing toward 1000x's
55. So 100x was converging to the same behaviour, just slower: the **weight
norm** appears to be what matters and the LR only sets how fast it is reached.
The "100x sits in a bad intermediate regime" story was wrong.

### Caveat that governs how these numbers may be reported

These are **oracle contacts**, computed from the ground-truth structure. If
contacts close most of the MSA gap, the supportable claim is "an accurate
contact predictor could substitute for alignments" — not that alignments have
been replaced. Protenix's MSAs are genuinely available at inference; ours are
not. Conversely the Helico model has ~3k steps against Protenix's full
training, so a shortfall would not be conclusive either.

## 11. Decisions taken

Answered by the user on 2026-08-06:

1. **Trunk shape** — keep the triangle operations, sweep depth. Try 8–16 blocks,
   and include **2 blocks** as an extreme point to test how far it degrades.
2. **Warm start** — warm-start from Protenix v1 rather than training from
   scratch. This is why `c_s_inputs` stays at 449 with the MSA channels zeroed
   (§7.2).
3. **Contact degree** — MarinFold will keep emitting binary contacts, so the
   3-state binary encoding stands and degree stays out (§7.1).
4. **Inter-chain contacts** — train on them. MarinFold will emit them later;
   pyconfind already handles complexes natively, and ~10–15% of contacts on
   multi-chain entries are inter-chain (§3.2).

---

## Appendix A — validated token→pyconfind mapping

This is the prototype behind every measurement in §3, reproduced here because it
is the load-bearing piece of phase 0 and encodes the non-obvious rules from §5.2.
Verified over 28 diverse PDB entries with zero failures and exact residue
round-trip.

```python
import gemmi
from pyconfind import analyze
from pyconfind.pdb import LEGAL_RESIDUE_NAMES

PYCONFIND_KWARGS = dict(
    native_only=True, contact_distance=3.0, dcut=25.0,
    clash_distance=2.0, assembly=None,
)
MIN_CONTACT_DEGREE = 0.001
MAX_PROTEIN_TOKEN_TYPE = 20   # helico token_types 0..20 == standard protein residue


def build_gemmi(tokens):
    """Group helico tokens into gemmi residues.

    Returns (structure, slot_to_token). Slot k is the k-th emitted residue,
    which is exactly pyconfind's Position index because positions_from_atoms
    preserves input order -- this is what removes any need to align
    label_asym_id against author numbering. slot_to_token[k] is the helico
    token index to attribute contacts to, or -1 for residues that are not a
    single standard protein token (modified residue / ligand -> unknown).
    """
    groups, key_prev, cur = [], None, None
    for ti, tok in enumerate(tokens):
        key = (tok.chain_idx, tok.res_idx)
        if key != key_prev:
            cur = []
            groups.append((key, cur))
            key_prev = key
        cur.append(ti)

    st, model, chains, slot_to_token = gemmi.Structure(), gemmi.Model("1"), {}, []
    for (chain_idx, _res_idx), tis in groups:
        resname = tokens[tis[0]].res_name
        if resname not in LEGAL_RESIDUE_NAMES:
            continue                      # nucleic / ligand: pyconfind drops it anyway
        ch = chains.setdefault(str(chain_idx), gemmi.Chain(str(chain_idx)))
        res = gemmi.Residue()
        res.name = resname
        res.seqid = gemmi.SeqId(len(ch) + 1, " ")
        n_atoms = 0
        for ti in tis:
            tok = tokens[ti]
            if tok.atom_coords is None:
                continue
            for name, elem, xyz in zip(tok.atom_names, tok.atom_elements, tok.atom_coords):
                a = gemmi.Atom()
                a.name, a.element = name, gemmi.Element(elem or "C")
                a.pos = gemmi.Position(*(float(v) for v in xyz))
                a.occ, a.altloc = 1.0, "\x00"
                res.add_atom(a)
                n_atoms += 1
        if n_atoms == 0:
            continue
        ch.add_residue(res)
        slot_to_token.append(
            tis[0] if len(tis) == 1 and tokens[tis[0]].token_type <= MAX_PROTEIN_TOKEN_TYPE
            else -1
        )

    for cname in chains:
        model.add_chain(chains[cname])
    st.add_model(model)
    return st, slot_to_token


def compute_contacts(tokens, rotamer_library, min_degree=MIN_CONTACT_DEGREE):
    """-> (edges as [(ti, tj, degree)], eligible token indices)."""
    st, slot_to_token = build_gemmi(tokens)
    if sum(1 for s in slot_to_token if s >= 0) < 2:
        return [], []
    a = analyze(st, rotamer_library=rotamer_library, **PYCONFIND_KWARGS)
    if len(a.positions) != len(slot_to_token):
        raise RuntimeError(
            f"position/slot mismatch: {len(a.positions)} != {len(slot_to_token)}")
    edges = []
    for c in a.report.contacts:          # Contact.pos_i / pos_j are ints
        if c.degree < min_degree:
            continue
        ti, tj = slot_to_token[c.pos_i], slot_to_token[c.pos_j]
        if ti < 0 or tj < 0:
            continue
        edges.append((min(ti, tj), max(ti, tj), float(c.degree)))
    return edges, [s for s in slot_to_token if s >= 0]
```

Load the rotamer library **once per worker** and pass it in — it is a ~3.4 s
parse:

```python
from pyconfind import cached_rotamer_library, load_library
lib = load_library(cached_rotamer_library())
```

The `min_seq_separation >= 6` rule is deliberately *not* applied here. It belongs
in `to_features()` where chain identity is available, so it can be applied
within a chain only (§5.4).
