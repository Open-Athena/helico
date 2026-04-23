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
    def predict_and_dump(
        self,
        pdb_id: str,
        input_json_relpath: str,
        out_relpath: str,
        dump_relpath: str,
        seeds_csv: str = "42",
        model_name: str = PROTENIX_MODEL_NAME,
    ) -> dict:
        """Run Protenix via the in-process Python API with a dump hook.

        Writes:
          - CIF predictions under DATA_CACHE/<out_relpath>/
          - input_feature_dict + sample_meta under DATA_CACHE/<dump_relpath>/

        Unlike `predict` (which shells out to `protenix pred`), this
        path imports Protenix as a library so we can wrap
        `runner.predict` and capture the batch the model actually sees.
        """
        import logging
        import subprocess

        logger = logging.getLogger(__name__)

        input_path = Path(DATA_CACHE) / input_json_relpath
        out_dir = Path(DATA_CACHE) / out_relpath
        dump_dir = Path(DATA_CACHE) / dump_relpath
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            return {"pdb_id": pdb_id, "status": "error",
                    "error": f"missing input json at {input_path}"}

        import json as _json
        import traceback
        import numpy as np
        import torch
        from runner.batch_inference import get_default_runner, preprocess_input
        from runner.inference import infer_predict

        seeds = [int(s) for s in seeds_csv.split(",") if s.strip()]

        def _to_np(v):
            return v.detach().to(dtype=torch.float32, device="cpu").numpy()

        def _dump_batch(feature_dict: dict) -> None:
            arrs: dict = {}
            for k, v in feature_dict.items():
                if isinstance(v, torch.Tensor):
                    arrs[k] = _to_np(v)
                elif isinstance(v, (int, float, bool)):
                    arrs[k] = np.array(v)
                elif isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor):
                            arrs[f"{k}.{sk}"] = _to_np(sv)
            np.savez_compressed(dump_dir / "00_batch.npz", **arrs)

        try:
            # If MSAs are already baked into the JSON (we do this via
            # helico.upstream_protenix.build_protenix_input), this is a
            # no-op; otherwise it fetches MSAs via the Protenix server.
            preproc_out = str(out_dir / "preproc")
            os.makedirs(preproc_out, exist_ok=True)
            preprocessed_json = preprocess_input(
                input_json=str(input_path),
                out_dir=preproc_out,
                use_msa=True,
                use_template=False,
                use_rna_msa=False,
            )

            runner = get_default_runner(
                seeds=seeds,
                model_name=model_name,
                use_msa=True,
                use_template=False,
                use_rna_msa=False,
            )
            configs = runner.configs
            configs.input_json_path = preprocessed_json
            configs.dump_dir = str(out_dir)
            configs.seeds = seeds

            orig_predict = runner.predict
            state = {"dumped": False, "pre_captured": False, "pairformer_dumped": False}

            def _dump_stage(fname: str, arrs: dict) -> None:
                np.savez_compressed(dump_dir / fname, **{k: _to_np(v) if isinstance(v, torch.Tensor) else v for k, v in arrs.items()})

            # Hook 1: forward-hook on input_embedder → capture s_inputs
            # Hook 2: forward-hook on linear_no_bias_sinit → capture s_init
            # Hook 3: wrap relative_position_encoding → capture its output
            pre_cache: dict = {}

            def _hook_s_inputs(module, inp, out):
                pre_cache["s_inputs"] = out.detach()

            def _hook_s_init(module, inp, out):
                pre_cache["s_init"] = out.detach()

            def _hook_relpe(module, inp, out):
                pre_cache["relpe_pair"] = out.detach()

            h1 = runner.model.input_embedder.register_forward_hook(_hook_s_inputs)
            h2 = runner.model.linear_no_bias_sinit.register_forward_hook(_hook_s_init)
            h3 = runner.model.relative_position_encoding.register_forward_hook(_hook_relpe)

            # Per-cycle trace hooks on msa_module and pairformer_stack.
            cycle_trace = {"cycles": []}  # list of dicts, one per cycle

            def _to_stats(t: torch.Tensor) -> dict:
                t = t.detach()
                return {
                    "shape": tuple(t.shape),
                    "mean": float(t.float().mean().item()),
                    "std": float(t.float().std().item()),
                    "min": float(t.float().min().item()),
                    "max": float(t.float().max().item()),
                }

            def _hook_msa(module, inp, out):
                # inp[1] is z (second positional arg based on msa_module signature)
                z_in = inp[1] if len(inp) > 1 else None
                rec = cycle_trace["cycles"]
                if not rec or "msa_out" in rec[-1]:
                    rec.append({})
                rec[-1]["msa_in"] = _to_stats(z_in) if z_in is not None else None
                rec[-1]["msa_out"] = _to_stats(out)

            def _hook_pairformer(module, inp, out):
                # Pairformer returns (s, z)
                rec = cycle_trace["cycles"]
                if not rec:
                    rec.append({})
                s_out, z_out = out if isinstance(out, tuple) else (None, out)
                rec[-1]["pf_s_in"] = _to_stats(inp[0]) if len(inp) > 0 else None
                rec[-1]["pf_z_in"] = _to_stats(inp[1]) if len(inp) > 1 else None
                rec[-1]["pf_s_out"] = _to_stats(s_out) if s_out is not None else None
                rec[-1]["pf_z_out"] = _to_stats(z_out)

            h4 = runner.model.msa_module.register_forward_hook(_hook_msa)
            h5 = runner.model.pairformer_stack.register_forward_hook(_hook_pairformer)

            # Wrap get_pairformer_output to capture the initial z_init
            # (reconstruct from captured s_init + relpe + token_bonds) and
            # the post-recycle (s, z).
            orig_gpo = runner.model.get_pairformer_output

            def hooked_gpo(*args, **kwargs):
                # Call original — pre_cache is populated by forward hooks
                s_inputs, s, z = orig_gpo(*args, **kwargs)
                if not state["pre_captured"]:
                    # Reconstruct z_init the same way Protenix does
                    ife = kwargs.get("input_feature_dict") or args[0]
                    s_init = pre_cache.get("s_init")
                    relpe_pair = pre_cache.get("relpe_pair")
                    tb = ife.get("token_bonds")
                    # z_init = linear_zinit1(s_init)[:, None, :] + linear_zinit2(s_init)[None, :, :] + relpe + linear_token_bond(tb)
                    z1 = runner.model.linear_no_bias_zinit1(s_init)
                    z2 = runner.model.linear_no_bias_zinit2(s_init)
                    z_init = z1[..., None, :] + z2[..., None, :, :]
                    if relpe_pair is not None:
                        z_init = z_init + relpe_pair
                    if tb is not None:
                        z_init = z_init + runner.model.linear_no_bias_token_bond(tb.unsqueeze(-1))
                    _dump_stage("01_pre_recycle.npz", {
                        "s_inputs": pre_cache.get("s_inputs"),
                        "s_init": s_init,
                        "z_init": z_init,
                        "relpe_pair": relpe_pair,
                    })
                    state["pre_captured"] = True
                if not state["pairformer_dumped"]:
                    _dump_stage("02_post_recycle.npz", {"s_inputs": s_inputs, "s": s, "z": z})
                    # Write cycle trace as JSON
                    with open(dump_dir / "cycle_trace.json", "w") as f:
                        _json.dump(cycle_trace, f, indent=2)
                    state["pairformer_dumped"] = True
                return s_inputs, s, z

            runner.model.get_pairformer_output = hooked_gpo

            def hooked_predict(data):
                if not state["dumped"]:
                    _dump_batch(data["input_feature_dict"])
                    meta = {
                        "sample_name": str(data.get("sample_name")),
                        "N_asym": int(data["N_asym"].item()) if "N_asym" in data else None,
                        "N_token": int(data["N_token"].item()) if "N_token" in data else None,
                        "N_atom": int(data["N_atom"].item()) if "N_atom" in data else None,
                        "N_msa": int(data["N_msa"].item()) if "N_msa" in data else None,
                        "entity_poly_type": {
                            str(k): str(v)
                            for k, v in (data.get("entity_poly_type") or {}).items()
                        },
                        "feature_keys": sorted(data["input_feature_dict"].keys()),
                        "top_level_keys": sorted(data.keys()),
                    }
                    with open(dump_dir / "sample_meta.json", "w") as f:
                        _json.dump(meta, f, indent=2, default=str)
                    state["dumped"] = True
                prediction = orig_predict(data)
                # Dump post-diffusion outputs (first seed only).
                # Key names in prediction: "coordinate", "pae", "plddt", "pde", ...
                if "post_diff_dumped" not in state:
                    arrs = {}
                    for k, v in prediction.items():
                        if isinstance(v, torch.Tensor):
                            arrs[k] = _to_np(v)
                    np.savez_compressed(dump_dir / "03_post_diffusion.npz", **arrs)
                    state["post_diff_dumped"] = True
                return prediction

            runner.predict = hooked_predict
            infer_predict(runner, configs)
            data_volume.commit()
            cifs = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.cif"))
            return {
                "pdb_id": pdb_id,
                "status": "ok",
                "out_relpath": out_relpath,
                "dump_relpath": dump_relpath,
                "dumped_batch": state["dumped"],
                "n_cifs": len(cifs),
            }
        except Exception as e:
            return {
                "pdb_id": pdb_id,
                "status": "error",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

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


@app.local_entrypoint()
def run_dump(
    target: str = "8t59-assembly1",
    seeds_csv: str = "42",
    staging_dir: str = "/tmp/upstream-protenix-staging-dump",
    out_dir: str = str(ROOT / "experiments/exp8_ab_ag_triage/data/upstream_dump"),
):
    """Dump Protenix's input_feature_dict for one target (Phase 2 diff).

    Stages + uploads the same inputs as run_triage, but calls
    predict_and_dump (single seed, single sample) and pulls back both
    CIFs and the feature-batch npz. Use one target at a time.
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

    pdb_id = target.strip()
    stage = staging_root / pdb_id
    stage.mkdir()
    remote_base = f"/upstream_protenix_dump/{pdb_id}"
    print(f"=== staging {pdb_id} ===")
    build_protenix_input(
        pdb_id=pdb_id,
        gt_cif_path=gt_dir_local / f"{pdb_id}.cif.gz",
        foldbench_msa_dir=msa_local,
        out_dir=stage,
        remote_msa_prefix=f"{DATA_CACHE}{remote_base}/msa",
    )

    print(f"=== uploading {pdb_id} ===")
    subprocess.run(
        ["modal", "volume", "put", "--force", "helico-bench-data",
         str(stage), remote_base],
        check=True,
    )

    print(f"=== dispatching predict_and_dump for {pdb_id} (seeds={seeds_csv}) ===")
    predictor = UpstreamPredictor()
    result = predictor.predict_and_dump.remote(
        pdb_id=pdb_id,
        input_json_relpath=f"{remote_base.lstrip('/')}/inputs.json",
        out_relpath=f"{remote_base.lstrip('/')}/predictions",
        dump_relpath=f"{remote_base.lstrip('/')}/dump",
        seeds_csv=seeds_csv,
    )
    print(f"  {result}")

    if result.get("status") != "ok":
        print(f"[error] aborting pull; result={result}")
        return

    local_dump = out_root / pdb_id
    if local_dump.exists():
        shutil.rmtree(local_dump)
    local_dump.mkdir(parents=True)
    print(f"=== pulling {pdb_id} dump ===")
    subprocess.run(
        ["modal", "volume", "get", "--force", "helico-bench-data",
         f"{remote_base}/dump", str(local_dump / "dump")],
        check=True,
    )
    subprocess.run(
        ["modal", "volume", "get", "--force", "helico-bench-data",
         f"{remote_base}/predictions", str(local_dump / "predictions")],
        check=True,
    )
    print(f"\nDone. Feature dump: {local_dump}/dump/00_batch.npz")
