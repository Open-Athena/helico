---
jupytext:
  text_representation:
    extension: .md
    format_name: markdown
    format_version: '1.3'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
helico_experiment:
  issue: 23
  title: Contact prediction, structural control and inference-time search
  branch: codex/clean-contact-pilot
---

# Contact prediction, structural control and inference-time search

Evaluate the exp22 step-20,000 checkpoint without additional training. The questions are whether its contact probabilities identify useful structural alternatives, whether conditioning controls generated structures, and whether inference-time search improves results at matched compute. The existing Protenix-v1 initialization supplies an inference baseline; no fine-tuning control arm is trained.

## Inputs and separation of development from test

The final training checkpoint has SHA256 `355788e40f610c8cc662e4c10f8225204668dfdcaf92ee1f04d744f8d5ea8b9a`, with regular and EMA weights. The dataset lock is unchanged (`2855c34acc97c8990eb49268f0f17a35f014f76fcd1fac9195dbdd9aaa699114`). Validation contains 384 full complexes, 18–1,020 tokens, disjoint from training and FoldBench. Their full upstream features are prepared with Protenix source `85767b811c40ed46e73a9b39519cf6bfca8701ba`, deterministic seeds, unchanged paired/unpaired protein and RNA MSA processing and no structural cropping. At most 512 MSA rows are retained, preserving the query and full-profile features. Every prepared input has a checksum in `data/validation_manifest.csv`.

A deterministic, nonoverlapping development panel selects eight protein–ligand, eight protein–protein and eight intraprotein targets from metadata before inference; selection uses a stable hash and the latter excludes the main scored interface categories. See `data/development_panel.csv`. All oracle contact diagnostics stay on validation. Native coordinates, contact targets, validity masks and observation masks are explicitly removed before every model call. Deployable search proposals depend only on input identities, predicted probabilities and generated coordinates. FoldBench settings are frozen after development; native contacts or pockets cannot be supplied during its predictions.

## Evaluation stages

1. Eight-target GPU preflight: regular weights, EMA and the Protenix initialization; 50 coordinate-diffusion steps. These measurements validate execution and are not the final benchmark.
2. Contact audit on all 384 validation entries: regular/EMA, MSA on/off, all contacts unknown. Report macro and pair-weighted BCE, positive prevalence, average precision, Brier score, calibration and top-k precision for interprotein, protein–ligand and intraprotein pairs. Empty/undefined groups remain explicit.
3. Generated structures using 200 diffusion steps: compare regular/EMA and the existing initialization. On the fixed panel, test 1/4/16 positive contacts, 16 negative contacts, full valid labels and no supplied labels, with MSA on/off and common seeds. Report requested satisfaction, unrequested contact recovery, observed-atom geometry, confidence and interchain clashes. Full-complex pair-weighted lDDT diagnostics use fixed atom/chain correspondence; they are not substituted for official symmetry-aware FoldBench metrics.
4. Compare discrepancy-guided beam search, progressive stochastic unmasking and ordinary batched diffusion. Tree branching uses high-probability absent contacts with generated distance at least 7 Å and confidence-based beam selection. The absorbing reverse schedule samples both contact and no-contact labels, preserving earlier reveals. Each candidate retains its coordinates, conditioning, parent, ranking, measured model time and independently computed quality. Ordinary sampling amortizes the trunk over batches; only complete batches count at a time budget. Report model-selected and oracle-best results separately, with paired bootstrap intervals.
5. Freeze raw/EMA and search settings using validation, then run full-input FoldBench with official scoring and the same baseline inference settings. Report every size/resource failure and category denominator. A separate alternative-state reference panel is required before claiming recovery of experimentally observed alternative states; candidate diversity alone is not that evidence.

## Resources and execution

The existing RNO2A H100 fleet is prepaid. The whole evaluation study, including preparation, failed attempts and all stages, has an initial ceiling of 512 reserved H100-hours and a conservative $96 incremental first-month storage/transfer allowance; no new nodes or fleet changes are requested. Validation preparation uses range reads from the already verified source-shard cache, retrieving 384 assemblies and 1,343 alignment files instead of downloading the entire terabyte. Per-member identity/size and content hashes are retained.

Each stage has immutable source/config/input identities, per-target durable results, a fixed timeout and a separate job name. Re-executing a notebook cell adopts its submission receipt; submission is not completion. Completion requires successful Iris exit, all expected rank/target artifacts and finished W&B status. Unknown failures are investigated explicitly; per-target error records prevent silent exclusions.

Select the stage using `HELICO_EVAL_STAGE`; the default is the eight-target preflight. The cost estimate below covers the complete planned study, and the dry run must pass the repository's $100 cost gate before a real dispatch.

```python
import os
from helico.experiment import set_experiment, ensure_bench_run
set_experiment('exp23_contact-evaluation')
stage = os.environ.get('HELICO_EVAL_STAGE', 'smoke')
stage_configs = {'smoke': ('contact-smoke-v2', 2), 'audit': ('contact-audit-v1', 2), 'development': ('contact-development-v1', 4), 'conditioning': ('contact-conditioning-v1', 4), 'search': ('contact-search-v1', 6), 'validation-structures': ('contact-validation-structures-v2', 6), 'alternative-states': ('contact-alternative-states-v1', 2), 'blind-preflight': ('contact-blind-preflight-v2', 3), 'contact-hypotheses': ('contact-contact-hypotheses-v1', 4)}
stage_configs.update({f'foldbench-{weight}-{shard}': (f'contact-foldbench-{weight}-{shard}-v1', 6)
    for weight in ['ema', 'protenix'] for shard in range(4)})
stage_configs.update({f'foldbench-{weight}-completion': (f'contact-foldbench-{weight}-completion-v1', 2)
    for weight in ['ema', 'protenix']})
# Corrected coordinate precision; original configurations remain immutable.
stage_configs.update({name: ('contact-' + name, hours) for name, hours in [
    ('development-v2', 4), ('conditioning-v2', 4), ('search-v2', 6),
    ('alternative-states-v2', 2), ('contact-hypotheses-v2', 4), ('validation-structures-v3', 6)]})
stage_configs.update({f'foldbench-v2-{weight}-{shard}': (f'contact-foldbench-{weight}-{shard}-v2', 6)
    for weight in ['ema', 'protenix'] for shard in range(4)})
stage_configs.update({f'foldbench-v2-{weight}-completion': (f'contact-foldbench-{weight}-completion-v2', 2)
    for weight in ['ema', 'protenix']})
stage_configs['conditioning-repeat-v1'] = ('contact-conditioning-repeat-v1', 2)
config_name, hours = stage_configs[stage]
# Whole-study allowance: $96 incremental storage/transfer, <=512 prepaid H100-hours.
gpus = 4 if stage.startswith('foldbench-') or stage.endswith(('-v2', '-v3')) or stage in {'alternative-states', 'contact-hypotheses', 'blind-preflight', 'validation-structures', 'conditioning-repeat-v1'} else 8
spec = dict(job_name=f'helico-exp23-{config_name}', mode='evaluate', gpus=gpus,
    cpu=12*gpus, memory=f'{125*gpus}g', disk='200g',
    image='pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca',
    timeout_seconds=hours * 3600, config=f'configs/eval/{config_name}.json',
    output_uri=f"s3://marin-us-east-02a/helico/evaluations/exp23/{config_name.removeprefix('contact-')}",
    priority_band='interactive', estimated_incremental_cost_usd=96,
    cost_accounting='Whole exp23 study: <=512 prepaid H100-hours and $96 first-month storage/transfer allowance')
run = ensure_bench_run(config_name, coreweave=spec, gpu='H100', workers=gpus,
    est_wall_hours=hours)
print(run.meta)
```

## Status

All corrected validation stages are complete with no target inference failures. The full-input FoldBench comparison is finishing; numerical benchmark results are pending final aggregation. The primary method decision is frozen in `data/foldbench_protocol_v2.json`: EMA versus the Protenix initialization, ordinary batched coordinate diffusion, four samples, 200 steps, one recycling cycle, at most 512 MSA rows, no templates or supplied contacts. These baseline settings are shared; the initialization arm is not the published Protenix default inference recipe.

### Model evaluated

Exp22 fine-tuned the full 368.5-million-parameter model initialized from Protenix `base_default_v1.0.0` for 20,000 updates (640,000 weighted 384-token crops). It retains the MSA and adds symmetric three-state contact conditioning and a binary contact head alongside the distogram. Contact means minimum heavy-atom separation below 5 Å; protein-involving pairs are supervised, excluding self pairs and nearby intraprotein sequence neighbors. Missing heavy atoms cannot establish a negative label. Coordinate, distogram and contact losses are trained together; confidence and dummy-template parameters are frozen. Training uses 20% MSA dropout and value-independent absorbing masks, with no positive-biased revelation. No Flock, curated nonbinding, pMHC or TCR additions were used.

The released regular/EMA weights are pinned at [timodonnell/helico-masked-contact-exp22@01a60254](https://huggingface.co/timodonnell/helico-masked-contact-exp22/tree/01a6025408d078aa8305f302d1d664a0291c45ee). Every released EMA tensor matches the full checkpoint exactly. Inference is numerically nondeterministic across identical-seed repeats; fixed seeds do not imply identical predictions.

### Inference precision correction and provenance

The original sampler performed a nominally rigid coordinate rotation under BF16 autocast, changing internal distances at each diffusion step. Commit `5806c0f` preserves input coordinate precision for this operation; CPU and CUDA distance-invariance tests pass. Across the 24 development targets, EMA's mean distance error on native-close intratoken heavy-atom pairs falls from 0.1024 to 0.0119 Å. This diagnostic uses reference separations of 1.1–1.8 Å, not a chemical bond assignment. Model weights are unchanged.

All structure-generation measurements below use the corrected sampler. The unaffected contact audit remains `audit-v1`; authoritative structure results are `development-v2`, `conditioning-v2`, `search-v2`, `contact-hypotheses-v2`, `alternative-states-v2`, `validation-structures-v3`, and `conditioning-repeat-v1`. Earlier structure measurements and FoldBench v1 predictions are superseded but retained in versioned archives. Historical FoldBench outputs helped diagnose the precision defect. Corrected blind numerical quality scores were not inspected when freezing the corrected validation decision at 2026-10-16 02:38:08 UTC. See `plots/corrected/rotation-geometry.png`.

### Contact prediction and structure quality

With MSA, EMA's macro contact average precision is 0.9066 within proteins, 0.4847 between proteins and 0.5345 for protein–ligand pairs. Without MSA these fall to 0.2563, 0.0539 and 0.1345. Positive prevalence is only 1.34%, 0.23% and 0.72%, respectively, so low BCE alone is insufficient evidence of useful predictions. Undefined target/group metrics remain explicit; the protein–ligand group includes ions and glycans.

The 24-target development comparison finds EMA and regular weights nearly tied: confidence-selected observed-atom lDDT 0.852545 versus 0.852411, paired difference +0.000134 (95% target-bootstrap interval −0.001187 to +0.001546). EMA remains the primary checkpoint.

On all 384 validation complexes, confidence-selected EMA observed-atom lDDT is 0.799787 versus 0.787484 for initialization: difference **+0.012303**, 95% assembly-bootstrap interval **+0.008583 to +0.016490**. Interface lDDT is 0.486186 versus 0.477638 on 331 evaluable complexes: difference +0.008547 (−0.004595 to +0.021932). This resolves a modest full-structure improvement, but not an interface improvement. These diagnostics use fixed atom/chain correspondence and are not official FoldBench metrics. See `data/validation-structures-v3/` and `plots/corrected/full-validation.png`.

For the highest-ranked missing-contact proposal per group/sample with generated distance at least 7 Å, mean predicted probabilities are 0.834/0.565/0.603 (intraprotein/protein–protein/protein–ligand); observed reference-contact fractions are 0.387/0.241/0.254. Unknown reference labels are excluded after proposal selection. This identifies overconfidence on selected disagreement proposals relative to the observed reference. One reference cannot exclude valid alternative contacts; these are not unconditional calibration statistics for the whole head.

### Sparse conditioning and inference repeat control

All 768 conditioning cases completed: 24 targets, two seeds, MSA on/off and eight modes. With MSA, supplying up to four native contacts missed by the original prediction recovers only **3.125% intraprotein, 3.125% protein–protein and 6.25% protein–ligand** requested contacts. Full native contact matrices help more without MSA (mean observed-lDDT gains about 0.045–0.049), but sparse steering remains weak.

A separate 96-case same-seed unconditioned inference repeat scores exactly the original requested pairs after prediction. Conditioned minus repeated-unconditioned satisfaction is −4.6875 percentage points within proteins (95% interval −9.375 to −1.5625), +3.125 between proteins (0 to +9.375), and +1.5625 for protein–ligand (0 to +4.6875). Thus none of the three groups shows a resolved positive structural-compliance gain over numerical repeatability. Each group has eight targets; these exploratory intervals are not multiple-comparison adjusted. See `data/conditioning-repeat-v1/` and `plots/corrected/contact-conditioning-repeat.png`.

The contact head does respond weakly to supplied labels: versus the original unconditioned prediction, missed-contact probabilities change by +0.00621 within proteins, +0.00724 between proteins and −0.00008 for protein–ligand. Only the intraprotein head change remains resolved against the repeated-unconditioned arm. The conditioning projection is enabled and nonzero (present-minus-unknown norm 5.03; absent-minus-unknown 2.26), and both new contact modules already used ten times the backbone learning rate. This is not explained by disabled conditioning. Continuous distances, negative corrections and MSA-off results are retained in the CSVs.

### Matched-time sampling and method decision

Tree search and progressive unmasking use predicted contacts and generated geometry only. At the common unmasking time horizon, confidence-selected observed-lDDT differences from ordinary batched sampling are +0.004659 for tree search (95% interval −0.000416 to +0.010735) and +0.002564 for unmasking (+0.000021 to +0.005495). The latter is a small nominal full-structure gain; neither method improves oracle-best quality, and unmasking's interface difference is −0.011924 (−0.027070 to +0.000037). Mean CA cluster counts at a 2 Å threshold are 5.33 ordinary, 3.73 tree and 3.94 unmasking. Candidate counts and fixed chain correspondence affect this diversity measure. Shorter horizons explicitly retain cases with no completed ordinary batch; paired analyses match target and seed before averaging seeds.

Contact-only diffusion creates four independent hypotheses with 7 or 32 trunk/contact-head transitions, generating coordinates only at the end. All 54 target/seed cases completed; primary analysis uses the prespecified 24-target panel, with three additional entries used only for alternative-reference diagnostics. At each method's own time budget, selected full-structure lDDT changes are +0.000164 for 7 steps (−0.006692 to +0.005994) and −0.004700 for 32 steps (−0.015713 to +0.003333). The 32-step interface difference is −0.039530 (−0.088501 to −0.004414); its oracle-best difference is −0.006902 (−0.012259 to −0.002847). Ordinary batching generates 8.83 and 19.33 mean candidates in the respective budgets, versus four contact hypotheses. See `plots/corrected/contact-search.png` and `contact-hypotheses.png`.

**Ordinary batched coordinate diffusion is retained for FoldBench.** The small selected unmasking gain without interface/oracle/diversity gains, weak constraint compliance, and adverse contact-only results do not justify another blind search arm. The decision, timestamp and validation-file hashes are recorded before reviewing corrected numerical blind scores.

### Alternative-reference diagnostic

Two identical-sequence apo/bound pairs were selected from validation reference geometry before inspecting predictions: EgtU (CA separation 4.397 Å) and BmCBP (2.173 Å). Recovery requires RMSD below min(2 Å, half the reference separation). At the progressive-unmasking time budget, ordinary sampling from apo EgtU reaches both states in both seeds; tree and unmasking reach only apo. From bound EgtU all three methods reach both states in one of two seeds. At the separate contact-only budgets, ordinary sampling reaches both EgtU states from both inputs in both seeds, while contact-only hypotheses remain in the input-associated state. No method reaches the strict BmCBP neighborhoods. These two ligand-dependent examples do not establish general ensemble recovery.

### FoldBench inputs, scoring and coverage

The initial immutable input release contains 1,482 ready full complexes, 27 explicit failures above the 2,048-token resource cap, and 13 assemblies awaiting one existing external alignment batch. Normal-length proteins with query-only placeholders cannot enter ready features. Inputs retain full sequences/chemical identities and independently prepared paired/unpaired MSAs; native coordinates, pockets, contacts and observation masks are unavailable to inference. No target is cropped. All 1,522 assemblies and 1,823 official category targets remain in final denominators, including unresolved inputs or scoring failures.

Primary interface success uses official DockQ ≥0.23, or ligand RMSD <2 Å and lDDT-PLI >0.8; monomers use OpenStructure lDDT with stereochemical checks. Confidence-selected performance and oracle best-of-four are separate. Paired intervals resample whole PDB assemblies while preserving target weighting. Reference-self checks cover all nine categories. An optional rigid-metric failure can be retried using the same primary metrics; the ligand CLI serialization defect is handled through a wrapper around public OpenStructure APIs, with numerical parity checked on all 16 preflight ligand predictions. No imported library is monkey-patched. Unassigned/unsupported interfaces remain failures; original errors and recovery receipts are retained.

The large-input preflight exposed excessive dummy-template attention allocation. Above 1,024 tokens, evaluation chunks independent anchor rows while retaining all attention keys and the learned dummy-template computation; CPU output/gradient and GPU BF16 parity checks pass. Both models completed the eight-target shape preflight through 2,043 tokens. A later CPU export process lost its Protenix environment; it was restarted with the correct environment and durable inference outputs, without GPU reruns. See `data/postprocessing_environment_recovery.json`.

### Next training recipe and candidate interaction data

The mask-support audit identifies a mismatch with the intended few-contact use case. For a fully observed 384-residue protein crop with the configured sequence exclusion, there are 71,631 eligible unordered pairs. At the sparsest nonterminal training time (999/1000), the expected revealed count is still 71.6; the probability of at most four reveals is about 8.75e-26. Actual crops vary, but the existing sparse bucket does not explicitly train 1/4/16-label inputs. See `data/mask_support_audit.json`.

A subsequent run should retain value-independent Bernoulli absorption while sampling noise levels according to the number of eligible pairs, explicitly covering expected revealed counts 1, 4, 16 and 64 alongside broad noise and endpoint cases. Use finer or continuous diffusion time. Establish contact-head response and geometric satisfaction of supplied constraints on held-out targets before spending more compute on search; do not force every marginal head prediction into one structure. The present study launches no further training.

A read-only audit of [Flock revision c2c993c3](https://huggingface.co/datasets/ternarytx/flock/tree/c2c993c31bf33184f12446bb33a3dc3c86e0f42b) finds 26,934 positive and 377,643 negative pairs: 368,937 negatives are PDB-derived and 8,706 literature-derived. Respect `published_as_negative` in the evidence table. Its 2023-06-01 cutoff does not preserve this project's 2021-09-30 split; its [source pipeline](https://github.com/TernaryTx/flock) uses Cβ separation above 8 Å, distinct from our heavy-atom definition. Recompute geometric labels and sequence/interface exclusions before release. Keep structural noncontact and assay noninteraction as distinct supervision types. Nonstructural positive binding evidence cannot justify invented dense contact maps or coordinates. These data may help specificity, but do not by themselves fix weak structural control.


## Reproduce one released-model prediction

With the repository's CUDA dependencies installed and source at `5806c0f` or later, download the pinned release and one prepared validation input:

```text
hf download timodonnell/helico-masked-contact-exp22 ema.pt --revision 01a6025408d078aa8305f302d1d664a0291c45ee --local-dir /tmp/helico-model
hf download timodonnell/helico-contact-eval-v1 7qeh.pt.gz --type dataset --revision ec4de019de89f79f03da31c5e026ddf7d858e77e --local-dir /tmp/helico-input
python experiments/exp23_contact-evaluation/predict_released.py --checkpoint /tmp/helico-model/ema.pt --features /tmp/helico-input/7qeh.pt.gz --output /tmp/helico-prediction.pt
```

The tensor output contains generated coordinates, contact probabilities and confidence. Its companion JSON records input/weight/output checksums and inference settings. All native coordinates, contact labels and observation masks are removed before the model call. This example uses full prepared inputs with MSA and unknown contacts; arbitrary sequences and ligands require the full input-preparation pipeline. Every released EMA tensor matches the full training checkpoint exactly, and the released-weight smoke produces finite predictions; see `data/released_model_inference_check.json`.

The completed same-seed unconditioned repeat is analyzed in `data/conditioning-repeat-v1/`; native-selected constraints are used only for subsequent scoring of this arm.
