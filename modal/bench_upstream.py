"""Run upstream (Bytedance) Protenix v1.0.9 on FoldBench targets — A/B vs Helico.

Answers exp8's question: does upstream Protenix succeed on the specific
targets where Helico fails (8q3j, 8v52)?

**Versioning decision**: we use Protenix **code v1.0.9** with the **v1.0.0
model checkpoint** (`protenix_base_default_v1.0.0`). This is the same
checkpoint Helico loads in exp4, so the A/B is apples-to-apples —
differences are in featurization / inference pipeline, not weights.

(Protenix v0.3.2 bundled in FoldBench uses a different model v0.2.0
checkpoint whose architecture predates v1.0.0's extra linear-layer
biases. Using the bundled version would have introduced a weights
mismatch.)

Protocol: 5 seeds × 5 samples × 200 diffusion steps × 10 cycles —
matches both the FoldBench published protocol and Helico's exp8.

Usage:
    modal run modal/bench_upstream.py \\
        --targets 8t59-assembly1,8q3j-assembly1,8v52-assembly1
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


ROOT = Path(__file__).parent.parent

# We auto-download the v1.0.0 checkpoint via the Protenix CLI at first use.
# The checkpoint lives at a known URL for cache-warming if needed.
PROTENIX_MODEL_NAME = "protenix_base_default_v1.0.0"


upstream_image = (
    # Protenix 1.0.x JIT-compiles a fused `fast_layer_norm_cuda_v2`
    # extension at import time via torch.utils.cpp_extension.load, which
    # needs ninja + a CUDA toolkit with nvcc. The default debian_slim
    # image has neither. Start from nvidia/cuda devel (has nvcc); pip
    # installs torch 2.7.1 + cuequivariance on top.
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11",
    )
    .apt_install("wget", "curl", "git", "build-essential", "ninja-build")
    .pip_install("ninja")
    # Let `pip install protenix==1.0.9` pull its full pinned dep set.
    # Key deps: torch==2.7.1, numpy==2.4.1, biotite==1.4.0 (no
    # PDBX_COVALENT_TYPES fight), cuequivariance-torch==0.8.0,
    # biopython==1.85. Skip deepspeed==0.17.5 to save image-build time
    # (~5-10 min on CUDA extension compile) — Protenix uses triton
    # kernels otherwise and can run without ds4sci.
    .pip_install("protenix==1.0.9", extra_options="--no-deps")
    # Then install Protenix's deps except deepspeed.
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "cuequivariance-ops-torch-cu12==0.8.0",
        "cuequivariance-torch==0.8.0",
        "scipy>=1.9.0",
        "ml_collections==1.1.0",
        "tqdm==4.67.1",
        "pandas==2.3.1",
        "PyYAML==6.0.2",
        "matplotlib==3.10.5",
        "ipywidgets==8.1.7",
        "py3Dmol==2.5.2",
        "rdkit==2025.9.3",
        "biopython==1.85",
        "biotite==1.4.0",
        "modelcif==1.4",
        "gemmi==0.6.7",
        "pdbeccdutils==1.0.0",
        "fair-esm==2.0.0",
        "scikit-learn==1.7.1",
        "scikit-learn-extra==0.3.0",
        "pydantic>=2.0.0",
        "triton==3.3.1",
        "optree==0.17.0",
        "protobuf==6.31.1",
        "icecream==2.1.7",
        "ipdb==0.13.13",
        "numpy==2.4.1",
        "click",
        "huggingface_hub>=0.20",
    )
    # Pre-warm the Protenix model cache so the first inference doesn't
    # spend time downloading. Protenix CLI caches to ~/.cache/protenix.
    .run_commands(
        "mkdir -p /root/.cache/protenix && "
        "python -c 'from protenix.web_service.dependency_url import URL; print(URL)' || true"
    )
)


app = modal.App("helico-upstream-protenix", image=upstream_image)

# Shared data volume (same one Helico's bench uses). Upstream Protenix
# writes output CIFs under /cache/helico-data/upstream_protenix/<pdb_id>/.
data_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=True)
DATA_CACHE = "/cache/helico-data"


@app.cls(image=upstream_image, gpu="H100", timeout=3600,
         max_containers=4,
         volumes={DATA_CACHE: data_volume})
class UpstreamPredictor:
    @modal.method()
    def predict(
        self,
        pdb_id: str,
        input_json_relpath: str,   # relative to DATA_CACHE
        dump_relpath: str,         # relative to DATA_CACHE
        seeds_csv: str = "42,66,101,2024,8888",
        model_name: str = PROTENIX_MODEL_NAME,
    ) -> dict:
        """Run Protenix 1.0.9 inference. Uses the `protenix pred` CLI.

        With `--use_default_params=true` (the default), Protenix picks its
        recommended N_cycle / N_sample / N_step for the chosen model, which
        for protenix_base_default_v1.0.0 matches the published protocol
        (5 samples, 200 steps, 10 cycles).
        """
        import logging
        import subprocess

        logger = logging.getLogger(__name__)

        input_path = Path(DATA_CACHE) / input_json_relpath
        dump_dir = Path(DATA_CACHE) / dump_relpath
        dump_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            return {"pdb_id": pdb_id, "status": "error",
                    "error": f"missing input json at {input_path}"}

        # Per Protenix docs, the CLI auto-downloads the named model into
        # ~/.cache/protenix on first use.
        cmd = [
            "protenix", "pred",
            "-i", str(input_path),
            "-o", str(dump_dir),
            "-n", model_name,
            "--seeds", seeds_csv,
            "--use_msa", "true",
            "--use_template", "false",
            "--use_rna_msa", "false",
        ]
        logger.info(f"[{pdb_id}] running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            data_volume.commit()
            produced = sorted(str(p.relative_to(dump_dir)) for p in dump_dir.rglob("*.cif"))
            return {
                "pdb_id": pdb_id,
                "status": "ok",
                "dump_relpath": dump_relpath,
                "n_cifs": len(produced),
                "cif_paths": produced,
            }
        except subprocess.CalledProcessError as e:
            return {"pdb_id": pdb_id, "status": "error",
                    "error": f"returncode={e.returncode}"}
        except Exception as e:
            return {"pdb_id": pdb_id, "status": "error", "error": repr(e)}


@app.local_entrypoint()
def run_triage(
    targets: str = "8t59-assembly1,8q3j-assembly1,8v52-assembly1",
    staging_dir: str = "/tmp/upstream-protenix-staging",
    out_dir: str = str(ROOT / "experiments/exp8_ab_ag_triage/data/upstream"),
):
    """Stage inputs + MSAs locally, upload to volume, dispatch Protenix,
    pull outputs back. Does NOT score here — run
    scripts/pm/score_upstream.py afterward.
    """
    import shutil
    import subprocess
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from helico.upstream_protenix import build_protenix_input

    foldbench_local = Path.home() / ".cache/helico/data/benchmarks/FoldBench"
    gt_dir_local = foldbench_local / "examples/ground_truths"
    msa_local = foldbench_local / "foldbench-msas"

    staging_root = Path(staging_dir)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    target_list = [t.strip() for t in targets.split(",") if t.strip()]

    # 1. Stage locally — build inputs.json + per-sequence a3m files
    staged: list[dict] = []
    for pdb_id in target_list:
        print(f"\n=== staging {pdb_id} ===")
        stage = staging_root / pdb_id
        stage.mkdir()
        # remote_base is where this stage dir will live on the volume
        remote_base = f"/upstream_protenix/{pdb_id}"
        info = build_protenix_input(
            pdb_id=pdb_id,
            gt_cif_path=gt_dir_local / f"{pdb_id}.cif.gz",
            foldbench_msa_dir=msa_local,
            out_dir=stage,
            remote_msa_prefix=f"{DATA_CACHE}{remote_base}/msa",
        )
        staged.append({
            "pdb_id": pdb_id,
            "stage_local": stage,
            "remote_base": remote_base,
            **info,
        })

    # 2. Upload staged dirs to the shared volume
    for s in staged:
        print(f"\n=== uploading {s['pdb_id']} ===")
        subprocess.run(
            ["modal", "volume", "put", "--force", "helico-bench-data",
             str(s["stage_local"]), s["remote_base"]],
            check=True,
        )

    # 3. Dispatch predictions in parallel
    print(f"\n=== dispatching {len(staged)} Protenix runs ===")
    predictor = UpstreamPredictor()
    results = list(predictor.predict.map(
        [s["pdb_id"] for s in staged],
        [f"{s['remote_base'].lstrip('/')}/inputs.json" for s in staged],
        [f"{s['remote_base'].lstrip('/')}/predictions" for s in staged],
    ))
    for r in results:
        print(f"  {r}")

    # 4. Pull outputs back
    for s, r in zip(staged, results):
        if r.get("status") != "ok":
            print(f"[skip pull] {s['pdb_id']}: status={r.get('status')}")
            continue
        local_dump = out_root / s["pdb_id"]
        if local_dump.exists():
            shutil.rmtree(local_dump)
        print(f"\n=== pulling {s['pdb_id']} outputs ===")
        subprocess.run(
            ["modal", "volume", "get", "--force", "helico-bench-data",
             f"{s['remote_base']}/predictions", str(local_dump)],
            check=True,
        )

    print(f"\nDone. Outputs under {out_root}")
    print("Score with: uv run python scripts/pm/score_upstream.py")


