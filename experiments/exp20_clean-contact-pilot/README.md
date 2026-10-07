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
runs = {}
# Total expected reference cost: 2 * 8 A100-80GB * 1.5 hours * $2.50 = $60.
# Reserve another $2 for GPU smoke checks; complete experiment remains <= $62.
for arm in ("unknown", "masked"):
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

Collect completed arm outputs into a portable CSV and summarize per target/seed before comparing arms. Save the plot and its source table so reviewing results never requires retraining.

```python
if not is_dry_run():
    frames = [pd.read_csv(run.cache_dir / "evaluation.csv") for run in runs.values()
              if (run.cache_dir / "evaluation.csv").exists()]
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

## Conclusion

Pending execution. This experiment tests whether the contact input and head learn useful control; it does not yet test a full reverse discrete sampler, tree-search efficiency, biological nonbinding labels, or recovery of multiple conformational states.
