"""Probe peak GPU memory at different ``n_diffusion_samples`` values (gh#9).

With the trunk frozen + diffusion_pair_source="distogram_logits", we should
be able to crank ``n_diffusion_samples`` well past 8 (the gh#6 default).
This probe loads a real Helico from the protenix-v1 seed, runs one
forward+backward at a representative crop_size for each candidate
``n_diffusion_samples`` value, and reports peak GPU memory.

Run:
    modal run modal/probe_diffusion_samples.py

Reports a small table: n_d → peak GB / status. The largest value that
stays under ~70 GB on H100 (80 GB) becomes the production knob for the
full gh#9 fine-tune.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# Mirror the train image so the cuDNN / torch / cuequivariance versions
# match the eventual training run.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "git")
    .pip_install(
        "torch>=2.10,<2.11",  # cuDNN 9.x — torch 2.11's cuDNN 13 broke val (gh#3)
        "cuequivariance-torch>=0.8,<0.9",
        "cuequivariance-ops-torch-cu12>=0.8,<0.9",
        "biopython>=1.80",
        "numpy",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "tqdm",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-probe-diffusion-samples", image=image)
ckpt_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=True)


@app.function(gpu="H100:1", timeout=1800, volumes={"/ckpts": ckpt_volume})
def probe(crop_size: int = 384) -> list:
    import os, subprocess, gc, sys
    subprocess.run(
        "cd /root/helico && uv venv --python 3.11 && uv pip install -e .",
        check=True, shell=True,
    )
    sys.path.insert(0, "/root/helico/.venv/lib/python3.11/site-packages")
    sys.path.insert(0, "/root/helico/src")

    import torch
    from helico.model import Helico, HelicoConfig
    from helico.data import make_synthetic_batch

    # Match the production fine-tune knobs — full-size model with the
    # gh#9 swap + trunk frozen.
    cfg = HelicoConfig(diffusion_pair_source="distogram_logits", n_diffusion_samples=8)
    model = Helico(cfg).cuda()
    # Freeze trunk via the same helper used by training.
    from helico.train import _freeze_trunk
    _freeze_trunk(model)

    results = []
    for n_d in (8, 16, 24, 32):
        # Override n_d on the model config — read by Helico.forward.
        cfg.n_diffusion_samples = n_d
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            # Real proteins have ~12 heavy atoms/token (vs default 5 in
            # make_synthetic_batch). Probe with the realistic value so
            # we don't undersize the activation footprint.
            batch = make_synthetic_batch(
                n_tokens=crop_size, n_atoms_per_token=12, device="cuda",
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch, compute_confidence=False)
            loss = out["diffusion_loss"]
            loss.backward()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            status = f"OK loss={loss.item():.3g}"
        except torch.cuda.OutOfMemoryError as e:
            peak_gb = float("nan")
            status = f"OOM {str(e)[:80]}"
        except Exception as e:
            peak_gb = float("nan")
            status = f"FAIL {type(e).__name__}: {str(e)[:80]}"
        # Drop grads + cache so the next iteration starts clean.
        for p in model.parameters():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        results.append((n_d, peak_gb, status))
        print(f"n_d={n_d:3d}: peak={peak_gb:6.2f} GB  {status}", flush=True)
    return results


@app.local_entrypoint()
def main(crop_size: int = 384):
    res = probe.remote(crop_size=crop_size)
    print("\n=== summary ===")
    print(f"{'n_d':>5} {'peak_GB':>8}  status")
    for n_d, peak, status in res:
        peak_str = f"{peak:.2f}" if peak == peak else "  —"
        print(f"{n_d:>5} {peak_str:>8}  {status}")
