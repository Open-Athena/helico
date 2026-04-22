"""Probe which token-sequence shapes crash cuDNN flash-attn on the Modal training image.

This is a one-shot diagnostic for gh#2. Local H100 doesn't reproduce the bug —
all shapes pass. The crash only happens on Modal's training container, which
suggests a cuDNN/driver/NVRTC mismatch specific to that image.
"""

import modal

ROOT = __import__("pathlib").Path(__file__).parent.parent

# Reuse the train image so we exercise the same cuDNN that crashes in val.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "git")
    .pip_install(
        "torch>=2.7",
        "cuequivariance-torch>=0.8",
        "cuequivariance-ops-torch-cu12>=0.8",
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

app = modal.App("helico-probe-shapes", image=image)


@app.function(gpu="H100:1", timeout=600)
def probe() -> list:
    import warnings
    warnings.filterwarnings("ignore")
    import subprocess
    subprocess.run(
        "cd /root/helico && uv venv --python 3.11 && uv pip install -e . wandb",
        check=True, shell=True,
    )
    # Re-import via the venv path
    import sys
    sys.path.insert(0, "/root/helico/.venv/lib/python3.11/site-packages")
    sys.path.insert(0, "/root/helico/src")

    import torch
    from helico.model import Helico, HelicoConfig
    from helico.data import make_synthetic_batch

    cfg = HelicoConfig(n_pairformer_blocks=2, n_diffusion_token_blocks=2)
    model = Helico(cfg).cuda().eval()

    results = []
    sizes = [7, 10, 11, 13, 17, 23, 29, 37, 41, 47, 50, 53, 64, 100, 128, 200, 256]
    for n in sizes:
        batch = make_synthetic_batch(n_tokens=n, device="cuda")
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            nan = any(
                isinstance(v, torch.Tensor) and torch.isnan(v).any().item()
                for v in out.values()
            )
            status = "OK" if not nan else "NAN"
        except Exception as e:
            status = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
        results.append((n, status))
        torch.cuda.empty_cache()
    for n, s in results:
        print(f"n_tokens={n:4d}: {s}", flush=True)
    return results


@app.local_entrypoint()
def main():
    res = probe.remote()
    print("\n=== summary ===")
    for n, s in res:
        print(f"n_tokens={n:4d}: {s}")
