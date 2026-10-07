---
jupytext:
  text_representation:
    extension: .md
    format_name: markdown
    format_version: '1.3'
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
helico_experiment:
  issue: 20
  title: Clean heavy-atom contact diffusion pilot
  branch: codex/clean-contact-pilot
---

# Clean heavy-atom contact diffusion pilot

Does clean contact conditioning control a pretrained structure predictor while retaining real MSAs? Compare two matched fine-tuning arms from Protenix v1: `masked` receives clean partially masked labels, and `unknown` always receives the unknown state.

Contacts mean minimum heavy-atom distance **<5 Å**, using protein residue tokens and ligand atom tokens. Missing atoms cannot establish an absence; within-chain protein pairs separated by fewer than six resolved residues and ligand–ligand pairs are excluded from supervision. No positive-biased revelation or false contact labels are used in training.

The schedule is an absorbing process with T=1000 and retention probability 1−t/T for each unordered pair. Time sampling is 20% fully masked, 10% fully visible, 25% sparse t∈[995,999], and 45% t∈[1,999]. These time-sampling weights define a denoising surrogate, not an unbiased estimate of the diffusion likelihood bound. The clean-target predictor does not require an explicit time embedding for this value-independent masking process.

Both arms train a symmetric binary contact head alongside the distogram, with loss `diffusion + 0.1 distogram + 0.1 masked-contact BCE + 0.01 observed-contact BCE`. The contact head and input projection learn at 1e-3; pretrained weights learn at 1e-5 after 20 warmup steps. Freeze the unsupervised confidence and template branches, preserve the MSA in 90% of steps, and disable both the MSA module and its profile/deletion features in the remaining 10%. Both arms center and randomly rotate/translate training coordinates, preserving contact labels, and pass ligand reference-space IDs through coordinate denoising. Training uses one recycle and one coordinate denoising sample, batch size one per GPU, eight GPUs, at most 256 steps or 3300 seconds of training.

Data are selected from a pinned byte prefix of the published Helico snapshot, with CCD reference coordinates rebuilt by atom name. Require exact MSA-query/token correspondence, at least two MSA rows per protein chain, 48–256 tokens, and no nucleotides. Use block-diagonal unpaired MSAs with up to 32 non-query rows per chain and 64 rows sampled by the model. Training predates 2021-09-30; validation starts 2022-05-01. Deduplicate exact complexes and remove training sequences whose global alignment score/minimum length against a held-out chain exceeds 0.4; this heuristic is not a homolog-clustered benchmark. The manifest records the realized sample sizes and categories. Asymmetric-unit contacts and ligand identity are not manually curated, so this pilot cannot establish biological binding specificity.

## Execution and budget

Run this notebook on the preallocated machine with `HELICO_PILOT_ASSETS` pointing to prepared assets and `HELICO_PILOT_ARM` set to one arm. A dry run always includes both arms; GPU smoke checks fit within an additional $2 reserve, making the total conservative reference budget $62 and requiring no new GPU instances.

```python
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from helico.experiment import set_experiment, ensure_training_run, is_dry_run, experiment_dir
set_experiment("exp20_clean-contact-pilot")
root = experiment_dir()
assets = Path(os.environ.get("HELICO_PILOT_ASSETS", str(Path.home() / "helico-pilot-assets")))
selected = os.environ.get("HELICO_PILOT_ARM", "both")
analyze_only = os.environ.get("HELICO_PILOT_ANALYZE_ONLY") == "1" or (
    selected == "both" and (root / "data" / "evaluation.csv").exists())
runs = {}
# Total expected reference cost: 2 * 8 A100-80GB * 1.5 hours * $2.50 = $60.
# Reserve another $2 for GPU smoke checks; complete experiment remains <= $62.
for arm in ("unknown", "masked"):
    if analyze_only and not is_dry_run():
        continue
    if not is_dry_run() and selected not in ("both", arm):
        continue
    runs[arm] = ensure_training_run(
        arm + "-v1", gpu="A100-80GB:8", max_steps=256, crop_size=256,
        batch_size=1, lr=1e-5, warmup_steps=20, n_diffusion_samples=1,
        est_wall_hours=1.5, command_timeout_seconds=5400,
        command=["env", "CUEQ_TRIMUL_FALLBACK_THRESHOLD=256", "CUEQ_TRIATTN_FALLBACK_THRESHOLD=256", sys.executable, "-m", "torch.distributed.run", "--standalone",
                 "--nproc_per_node=8", "-m", "helico.contact_pilot",
                 "--assets", str(assets), "--arm", arm, "--steps", "256"],
    )
```

The realized dataset has 53 training structures (17 monomers, 22 protein–ligand, 14 protein–protein) and 8 held-out structures (3, 4, and 1 respectively). Protein–protein validation is therefore anecdotal. GPU preflight uncovered fully masked attention rows producing NaNs in template attention and atom-attention backward; both are corrected with finite empty-row softmax. The cuEquivariance 0.8 fused gated-GEMM backward also produced NaNs on the A100 pilot, so this run explicitly selects its supplied PyTorch triangle-operation fallbacks through `CUEQ_TRIMUL_FALLBACK_THRESHOLD=256` and `CUEQ_TRIATTN_FALLBACK_THRESHOLD=256`, before library import. No runtime patching is used.

## Paired evaluation

Each validation example gets two fixed coordinate-noise seeds, one recycle, and 50 coordinate-diffusion steps. Compare no conditioning, 0.5% uniformly revealed true labels, full oracle labels, one search branch with four highest-probability contacts absent from the unconditioned structure, and an oracle branch restricted to true missing contacts; prefer interchain hypotheses when available. Oracle modes are diagnostics and leak structural labels deliberately. Search hypotheses use model probabilities only. Evaluate masked-pair BCE/AP/Brier separately from revealed labels, structural contact precision/recall, requested-contact satisfaction, and atom LDDT. These metrics do not use chain permutation or ligand atom symmetry correction, and the short sampler is a pilot setting.

Once the result CSV is present, the notebook defaults to analysis without dispatching training; `HELICO_PILOT_ANALYZE_ONLY=1` also selects this explicitly. Collect completed arm outputs into a portable CSV and summarize per target/seed before comparing arms. Save the plot and its source table so reviewing results never requires retraining.

```python
if not is_dry_run():
    paths = [root / ".cache" / "trainings" / (arm + "-v1") / "evaluation.csv"
             for arm in ("unknown", "masked") if selected in ("both", arm)]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames and (root / "data" / "evaluation.csv").exists():
        frames = [pd.read_csv(root / "data" / "evaluation.csv")]
    if frames:
        results = pd.concat(frames, ignore_index=True)
        (root / "data").mkdir(exist_ok=True)
        results.to_csv(root / "data" / "evaluation.csv", index=False)
        summary = results.groupby(["arm", "phase", "mode"])[
            ["lddt", "all_recall", "all_precision", "all_bce", "all_ap",
             "pp_recall", "pl_recall", "request_satisfaction"]].mean()
        summary.to_csv(root / "data" / "summary.csv")
        display(summary)
        final = summary.reset_index().query("phase == 'final'")
        if len(final):
            fig, ax = plt.subplots(figsize=(8, 4))
            for arm, group in final.groupby("arm"):
                ax.plot(group["mode"], group["all_recall"], marker="o", label=arm)
            ax.set(ylabel="True contact recovery", xlabel="Conditioning", ylim=(0, 1))
            ax.legend(); fig.tight_layout()
            (root / "plots").mkdir(exist_ok=True)
            fig.savefig(root / "plots" / "contact_recovery.png", dpi=160)
            plt.show()
```

## Paired differences and training diagnostics

Average the two seeds within each target before computing paired differences, so repeated samples are not treated as independent proteins. Compare head metrics only between identical masked-pair sets, because revealing labels changes the scoring subset. Bootstrap targets for descriptive 95% intervals, leaving single-target categories without an interval; eight targets are insufficient for a broad generalization claim.

```python
if not is_dry_run() and frames and set(results.arm) == {"masked", "unknown"}:
    from helico.contact_pilot_analysis import paired_comparisons
    paired = paired_comparisons(results)
    paired.to_csv(root / "data" / "paired_comparisons.csv", index=False)
    display(paired[paired.metric.isin(["lddt", "all_recall", "pl_recall"])])
    by_category = results.groupby(["arm", "phase", "mode", "kind"])[
        ["lddt", "all_recall", "all_precision", "all_bce", "all_ap",
         "pp_recall", "pl_recall", "request_precision", "request_satisfaction"]].mean()
    by_category.to_csv(root / "data" / "by_category.csv")
```

The contact losses during training use different visible subsets in the two arms and are diagnostic rather than a fair held-out comparison. Preserve the raw traces and plot their rolling means to expose instability or failed optimization.

```python
if not is_dry_run():
    traces = []
    for arm in ("unknown", "masked"):
        path = root / ".cache" / "trainings" / (arm + "-v1") / "training.csv"
        if path.exists():
            trace = pd.read_csv(path); trace["arm"] = arm; traces.append(trace)
    if traces:
        training = pd.concat(traces, ignore_index=True)
        training.to_csv(root / "data" / "training.csv", index=False)
    elif (root / "data" / "training.csv").exists():
        training = pd.read_csv(root / "data" / "training.csv")
    else:
        training = pd.DataFrame()
    if len(training):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for arm, trace in training.groupby("arm"):
            axes[0].plot(trace.step, trace.loss.rolling(20, min_periods=1).mean(), label=arm)
            axes[1].plot(trace.step, trace.contact_loss.rolling(20, min_periods=1).mean(), label=arm)
        axes[0].set(xlabel="Training step", ylabel="Total loss (20-step mean)")
        axes[1].set(xlabel="Training step", ylabel="Masked BCE (20-step mean)")
        axes[0].legend(); fig.tight_layout()
        fig.savefig(root / "plots" / "training.png", dpi=160)
        plt.show()
```

## Checkpoint reuse

Checkpoints carry `model_state_dict` and `model_config`, including the opt-in contact head, the retained MSA path, and the pilot sampler settings. The normal inference and benchmark loaders accept this configuration field; contact conditioning must use the new 5 Å heavy-atom labels, not the legacy pyconfind oracle option. The command in the execution cell records the kernel-fallback environment needed to reproduce the A100 run.

## Conclusion

Both runs completed all 256 updates on 2026-10-07. They consumed **3.69 GPU-hours** for training plus evaluation, equivalent to **$9.23** at the repository’s reference rate, plus the small preflight checks; the nodes were already allocated. All GPU workers exited after evaluation.

**The head learns contact probabilities, but this pilot does not establish reliable sparse structural control.** On the same all-masked held-out pairs, the masked arm achieved AP 0.587 and BCE 0.0830; the all-unknown control achieved AP 0.582 and BCE 0.0824. These are mean per-target metrics; the constant initial head had AP 0.040, the mean contact prevalence. Most of the improvement comes from training the new head, not from masked conditioning.

| Masked-arm condition | Contact recall | Contact precision | Atom LDDT |
|---|---:|---:|---:|
| No contacts | 51.52% | 49.54% | 0.6154 |
| 0.5% uniform revelation | 52.19% | 49.51% | 0.6112 |
| Full oracle map | 53.44% | 50.94% | 0.6207 |
| Four model-selected hypotheses | 52.91% | 50.40% | 0.6131 |
| Four true missing contacts | 52.81% | 49.99% | 0.6082 |

Full conditioning increased contact recovery by 1.92 percentage points (descriptive target-bootstrap 95% interval approximately +0.58 to +3.67 points), but its LDDT change was only +0.0054 and its interval included zero. The pretrained baseline already recovered 53.94% of contacts with LDDT 0.6180; the full-contact result therefore does not establish an improvement over the original model. The matched all-unknown fine-tune achieved 52.51% recall and LDDT 0.6175 with no contacts.

The model-selected four-contact branches had 45.3% true hypotheses, satisfied 20.3% of requested contacts, and changed contact recovery by +1.38 points with an interval including zero. The all-unknown control satisfied 15.6% of its own hypotheses despite never learning visible contact inputs, so branch response alone is insufficient evidence of learned control. More decisively, only **9.4% of the four true missing contacts** were satisfied in the masked arm (14.1% in the control). The sparse steering mechanism needs further work.

Protein–ligand recovery in the masked arm increased from 18.6% to 25.7% under model-selected hypotheses, but this averages only five structures with protein–ligand contacts, including the protein–protein category’s mixed complex. Much of the gain comes from 9CIV, whose LDDT decreased. With one held-out protein–protein example and no chain/ligand symmetry correction, these are case studies rather than evidence of improved complex prediction. Shared seeds do not make the GPU runs bitwise identical: their initial mean LDDTs both round to 0.618, but per-sample LDDT differs by 0.0127 on average.

The next experiment should first establish strong oracle control on a tiny overfit panel and an MSA-free ablation, then vary contact-projection learning rate and supervision strength while keeping independent no-contact controls. A geometry-based restraint loss on revealed contacts is a candidate if the coordinate decoder keeps ignoring the input, but should be an explicit ablation. Expand to more independent complexes and known alternative states before testing tree-search efficiency or adding Flock/weak interaction labels. This pilot does not test a full reverse discrete sampler, biological nonbinding labels, or recovery of multiple conformational states.

Run records: [masked](https://wandb.ai/timodonnell/helico/runs/y63wamer), [all-unknown](https://wandb.ai/timodonnell/helico/runs/yd1yog91), and [issue #20](https://github.com/Open-Athena/helico/issues/20). Checkpoints, predictions, and run metadata: [masked arm](https://huggingface.co/buckets/timodonnell/helico-experiments/tree/exp20-clean-contact-pilot/masked-v1), [all-unknown control](https://huggingface.co/buckets/timodonnell/helico-experiments/tree/exp20-clean-contact-pilot/unknown-v1). The masked-arm `data/` directory also contains the prepared dataset and provenance. [Rendered report and CSVs](https://huggingface.co/buckets/timodonnell/helico-experiments/tree/exp20-clean-contact-pilot/report).

GitHub branch publication returned repeated server errors during closeout, and the draft PR could not be created. The complete source change is also preserved as `source.patch` in the [report artifacts](https://huggingface.co/buckets/timodonnell/helico-experiments/tree/exp20-clean-contact-pilot/report); apply it with `git am` on base commit `b10385d`. The local branch retains the commits for a later push.
