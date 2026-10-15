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
config_name, hours = stage_configs[stage]
# Whole-study allowance: $96 incremental storage/transfer, <=512 prepaid H100-hours.
gpus = 4 if stage.startswith('foldbench-') or stage in {'alternative-states', 'contact-hypotheses', 'blind-preflight', 'validation-structures'} else 8
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

All 384 full validation inputs were prepared without failures. The GPU preflight, full 384-target contact audit and 24-target structural development evaluation completed without failures. Conditioning and matched-time search are complete; efficient contact-hypothesis sampling and the 384-target structure comparison are pending completion. FoldBench inputs and the official scoring harness are being prepared; blind benchmark results are pending.

### Validation weight selection

Across the fixed 24-target development panel, confidence-selected EMA predictions have mean observed-atom lDDT 0.84442, versus 0.84374 for regular weights (paired difference +0.00068; 95% target-bootstrap interval −0.00125 to +0.00245). Interface lDDT slightly favors regular weights (0.58495 vs 0.58180); neither difference is resolved by this panel. EMA is selected for the subsequent conditioning/search and blind test based on the full-complex mean, with the near-tie recorded explicitly. No FoldBench predictions or scores were used. These geometry diagnostics use fixed correspondence, not official benchmark scoring.

### Alternative-reference diagnostic

The two identical-sequence monomer pairs in `data/alternative_state_panel.json` were selected from validation reference geometry before inspecting predictions. The apo/bound CA separations are 4.397 Å (EgtU) and 2.173 Å (BmCBP). Matching uses common observed CA positions; state recovery requires RMSD below min(2 Å, half the reference separation). At the unmasking method's model-compute budget, ordinary sampling from apo EgtU reaches both reference neighborhoods in both seeds; tree and unmasking reach only the apo neighborhood. No method reaches the strict BmCBP neighborhoods. These are two ligand-dependent validation examples, not evidence of broad ensemble recovery. See `plots/alternative-states.png` and per-candidate CSVs.

### Conditioning and additional execution checks

All 768 conditioning arms completed (24 targets, two seeds, MSA on/off, eight modes). With MSA, supplying up to four native contacts missed by the unconditioned structure recovers only 10.9% of requested intraprotein contacts, 9.4% of protein–ligand contacts and 3.1% of protein–protein contacts. Sparse structural steering is weak despite good contact classification. These are oracle validation diagnostics; no native constraints enter blind inference.

A further contact-hypotheses stage uses 7 or 32 trunk/contact-head transitions and generates coordinates only at the end of each of four independently sampled absorbing trajectories. Its primary comparison uses the original 24-target panel; three extra entries complete the separately reported alternative-reference panel. Ordinary batched sampling gets the same measured model-compute budget. This stage separates contact generation cost from repeatedly generating coordinates at every revelation step.

The first blind shape preflight exposed a cubic dummy-template attention allocation at 1,672 and 2,043 tokens. Inference above 1,024 tokens now chunks independent anchor rows, retaining all attention keys and the learned dummy-template computation. CPU output/gradient parity and GPU BF16 parity are checked; smaller evaluation inputs and training retain the existing path. The failed attempt is preserved and a versioned preflight reruns all eight molecule/size examples. Resource failures remain explicit.

The complete 24-target matched-time comparison does not resolve an accuracy advantage: tree minus ordinary confidence-selected observed-atom lDDT is +0.00369 (95% target-bootstrap interval −0.00172 to +0.00990), and progressive unmasking is +0.00046 (−0.00339 to +0.00424). Oracle-best quality also does not improve. These numbers use the full unmasking time horizon, where all 24 targets have ordinary candidates; the shorter tree horizon has one target without a complete ordinary batch and reports that missing value explicitly. See `plots/contact-search.png`.

The 8-GPU validation-structures-v1 allocation was preempted before producing any predictions and then remained queued. It was canceled and replaced by validation-structures-v2, using four GPUs and deterministic token-count balancing. Input identities, seeds, model settings and the target set are unchanged. Full FoldBench is planned as four 4-GPU shards per model, for up to 32 GPUs without requiring eight free GPUs on a single node.

At the complete unmasking time horizon, ordinary sampling yields 5.40 mean CA-coordinate clusters (2 Å greedy threshold), versus 3.77 for tree search and 4.04 for progressive unmasking. Cluster-center geometry quality changes are small. This fixed-chain-correspondence diversity measure is affected by candidate count and homomer chain permutations; it is not evidence that these are stable or experimentally observed states. The matched-time alternative-reference diagnostic is reported separately.

### Blind inference and scoring readiness

The corrected eight-target blind shape preflight completed for both EMA and the initialization baseline, with four coordinate samples each: 64 structures, zero input/runtime/resource failures. Full input sizes include 1,672 and 2,043 tokens; RNA/DNA monomers and protein–nucleic-acid complexes are also covered. All outputs export to full mmCIF through the pinned Protenix writer. These preflight predictions were not scored against native coordinates to choose the protocol.

Reference-versus-reference checks now pass all nine official benchmark categories using OpenStructure 2.8 and FoldBench's modified DockQv2 for nucleic-acid interfaces. They exposed and fixed the wrapper's handling of polymer `interface_chain_id_*` columns, which differ from ligand `native_chain_id_*` columns. The corrected checks give DockQ/lDDT of 1 where applicable and ligand RMSD below 0.00001 Å. See `data/scorer_reference_selfcheck.csv`; these are harness checks, not model performance.
