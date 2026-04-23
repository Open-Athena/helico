"""Run upstream (Bytedance) Protenix v1 on FoldBench targets for A/B vs Helico.

Direct answer to exp8's question: does upstream Protenix succeed on the
specific targets where Helico fails (8q3j, 8v52)?

Protocol: 5 seeds × 5 samples × 200 diffusion steps × 10 cycles — exactly
the published FoldBench methodology (matches make_predictions.sh in
FoldBench/algorithms/Protenix/).

Protenix source is vendored from the FoldBench-bundled clone
(~/.cache/helico/data/benchmarks/FoldBench/algorithms/Protenix/Protenix)
which is version 0.3.2. The PyPI `protenix` package is 2.x with different
conventions; we don't use it.

Dependencies are isolated in a dedicated Modal image — Protenix pins
torch==2.3.1 and numpy==1.26.3, incompatible with Helico's torch>=2.10
and numpy>=2. Modal images are independent, so no conflict at runtime.

Usage:
    modal run modal/bench_upstream.py \\
        --target-pdb-ids 8t59-assembly1,8q3j-assembly1,8v52-assembly1
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


ROOT = Path(__file__).parent.parent

# Vendored Protenix source lives in the user's FoldBench cache.
# add_local_dir uploads this to the image at build time.
PROTENIX_SRC_LOCAL = Path(os.path.expanduser(
    "~/.cache/helico/data/benchmarks/FoldBench/algorithms/Protenix/Protenix"
))

PROTENIX_CKPT_URL = os.environ.get(
    "HELICO_PROTENIX_URL",
    "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt",
)
PROTENIX_CKPT_REMOTE = "/root/ckpts/" + PROTENIX_CKPT_URL.rsplit("/", 1)[-1]


upstream_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "git", "build-essential")
    # Protenix's pinned runtime deps (from its requirements.txt)
    .pip_install(
        "torch==2.3.1",
        "numpy==1.26.3",
        "biopython==1.83",
        "biotite==1.1.0",
        "modelcif==0.7",
        "protobuf==3.20.2",
        "PyYaml",
        "scipy",
        "ml_collections",
        "tqdm",
        "pandas",
        "dm-tree",
        "rdkit",
        "scikit-learn",
        "scikit-learn-extra",
        "matplotlib==3.9.2",
        "click",
        "huggingface_hub>=0.20",
        # Skipping deepspeed for now — compiling its CUDA extensions on
        # image build takes ~5-10min. We pass
        # --use_deepspeed_evo_attention=False to Protenix's inference.
    )
    # Protenix checkpoint baked into image (same URL as modal/bench.py).
    .run_commands(
        f"mkdir -p /root/ckpts && "
        f"curl -fL --retry 5 --retry-delay 5 --retry-connrefused "
        f"--connect-timeout 30 --max-time 900 "
        f"-o {PROTENIX_CKPT_REMOTE} {PROTENIX_CKPT_URL} && "
        f"ls -lh {PROTENIX_CKPT_REMOTE}"
    )
    # Vendor Protenix source — 1.8MB, copy=True bakes into image layer
    # so `pip install -e` below runs against a stable path.
    .add_local_dir(str(PROTENIX_SRC_LOCAL), remote_path="/algo/Protenix", copy=True)
    .run_commands("cd /algo/Protenix && pip install -e . --no-deps")
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
        dump_relpath: str,         # relative to DATA_CACHE, under which seed_*/predictions/ appear
        seeds_csv: str = "42,66,101,2024,8888",
        n_samples_per_seed: int = 5,
        n_steps: int = 200,
        n_cycles: int = 10,
    ) -> dict:
        """Run Protenix inference for a single target. Reads input JSON
        from the shared volume; writes CIFs + confidence JSONs under
        `dump_relpath` on the volume.
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

        # Protenix's inference.py imports relative to its own runner/, so we
        # must run with cwd=/algo/Protenix and set PYTHONPATH accordingly.
        env = os.environ.copy()
        env["PYTHONPATH"] = "/algo/Protenix"
        env["CUTLASS_PATH"] = "/dev/null"  # never used since ds attention disabled

        cmd = [
            "python", "runner/inference.py",
            "--seeds", seeds_csv,
            "--dump_dir", str(dump_dir),
            "--input_json_path", str(input_path),
            "--load_checkpoint_path", PROTENIX_CKPT_REMOTE,
            f"--model.N_cycle={n_cycles}",
            f"--sample_diffusion.N_sample={n_samples_per_seed}",
            f"--sample_diffusion.N_step={n_steps}",
            "--use_deepspeed_evo_attention=False",
        ]
        logger.info(f"[{pdb_id}] running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True, cwd="/algo/Protenix", env=env,
                capture_output=False,
            )
            data_volume.commit()
            # Enumerate what we produced
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
    pull outputs back to `out_dir`. Does NOT score here — run
    scripts/pm/score_upstream.py afterward (keeps scoring tooling outside
    this image).
    """
    import json
    import shutil
    import subprocess
    import sys
    import tempfile

    sys.path.insert(0, str(ROOT / "src"))
    from helico.upstream_protenix import build_protenix_input

    # Local paths
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

    # 1. Stage locally — build inputs.json + MSA dir per target
    staged: list[dict] = []
    for pdb_id in target_list:
        print(f"\n=== staging {pdb_id} ===")
        stage = staging_root / pdb_id
        stage.mkdir()
        info = build_protenix_input(
            pdb_id=pdb_id,
            gt_cif_path=gt_dir_local / f"{pdb_id}.cif.gz",
            foldbench_msa_dir=msa_local,
            out_dir=stage,
        )
        # Rewrite paths in inputs.json local→remote
        remote_base = f"/upstream_protenix/{pdb_id}"
        data = json.loads((stage / "inputs.json").read_text())
        for entry in data:
            for seq in entry.get("sequences", []):
                chain = seq.get("proteinChain")
                if not chain:
                    continue
                msa = chain.get("msa") or {}
                p = msa.get("precomputed_msa_dir")
                if p and p.startswith(str(stage)):
                    rel = Path(p).relative_to(stage)
                    msa["precomputed_msa_dir"] = f"{DATA_CACHE}{remote_base}/{rel}"
        (stage / "inputs.json").write_text(json.dumps(data, indent=2))
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
