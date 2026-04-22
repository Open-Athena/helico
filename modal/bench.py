"""Parallel FoldBench benchmark on Modal — fans out predictions across GPU workers.

Configure via environment variables before running:
    HELICO_BENCH_WORKERS=8 HELICO_BENCH_GPU=H100 modal run modal/bench.py
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
PROTENIX_URL_DEFAULT = "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt"
PROTENIX_URL = os.environ.get("HELICO_PROTENIX_URL", PROTENIX_URL_DEFAULT)
PROTENIX_CKPT_PATH = "/root/helico/checkpoints/" + PROTENIX_URL.rsplit("/", 1)[-1]

# Modal decorator params are static — configure via env vars before `modal run`
N_WORKERS = int(os.environ.get("HELICO_BENCH_WORKERS", "4"))
GPU_TYPE = os.environ.get("HELICO_BENCH_GPU", "H100")

bench_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl")
    .pip_install(
        "torch>=2.7",
        "cuequivariance-torch>=0.8",
        "cuequivariance-ops-torch-cu12>=0.8",
        "biopython>=1.80",
        "numpy<2.0",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "requests",
        "tmtools",
        "DockQ",
        "tqdm",
    )
    # Protenix checkpoint baked into image (1.4 GB, cached by Modal)
    # curl with retries + progress dots — wget -q hangs silently on stalls
    .run_commands(
        f"mkdir -p /root/helico/checkpoints && "
        f"curl -fL --retry 5 --retry-delay 5 --retry-connrefused "
        f"--connect-timeout 30 --max-time 900 "
        f"-o {PROTENIX_CKPT_PATH} {PROTENIX_URL} && "
        f"ls -lh {PROTENIX_CKPT_PATH}"
    )
    # Project code last (changes most frequently)
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-bench", image=bench_image)

# Shared Volume caches CCD + FoldBench data across workers and runs. First
# worker to start populates it via snapshot_download (~2 GB); subsequent
# workers see the files immediately. Persists across runs — no re-download.
data_volume = modal.Volume.from_name("helico-bench-data", create_if_missing=True)
DATA_CACHE = "/cache/helico-data"

# Read-only mount of the training checkpoints volume so we can bench
# Helico checkpoints at paths like /ckpts/<run>/final.pt.
ckpt_volume = modal.Volume.from_name("helico-checkpoints", create_if_missing=True)
CKPT_MOUNT = "/ckpts"


@app.cls(gpu=GPU_TYPE, timeout=600, max_containers=N_WORKERS,
         volumes={DATA_CACHE: data_volume, CKPT_MOUNT: ckpt_volume})
class Predictor:
    # "" / "protenix-v1" → the Protenix v1 checkpoint baked into the image.
    # Anything else is interpreted as a path on the helico-checkpoints
    # Volume (e.g. /ckpts/v1-finetune-01/final.pt) and loaded in the
    # Helico checkpoint format written by train.py.
    checkpoint_path: str = modal.parameter(default="")

    @modal.enter()
    def setup(self):
        import os
        import subprocess

        # Point HELICO cache at the shared Volume so CCD + FoldBench are persisted.
        os.environ["HELICO_DATA_DIR"] = DATA_CACHE
        os.makedirs(DATA_CACHE, exist_ok=True)

        subprocess.run(
            "cd /root/helico && uv venv --python 3.11 && uv pip install -e '.[bench]'",
            check=True, shell=True,
        )

        import sys
        sys.path.insert(0, "/root/helico/src")

        from collections import OrderedDict
        import torch
        from huggingface_hub import snapshot_download
        from helico.data import parse_ccd
        from helico.model import Helico
        from helico.load_protenix import infer_protenix_config, load_protenix_state_dict
        from helico.bench import download_foldbench

        # Populate shared volume on first worker. Chunks + retries keep this
        # robust against transient HF CDN stalls (see image-build history).
        chunks = [
            ["processed/ccd_cache.pkl"],
            ["benchmarks/FoldBench/targets/**", "benchmarks/FoldBench/examples/**"],
            ["benchmarks/FoldBench/foldbench-msas/**"],
        ]
        for i, patterns in enumerate(chunks):
            for attempt in range(5):
                try:
                    print(f"[data chunk {i}] {patterns} attempt {attempt+1}", flush=True)
                    snapshot_download(
                        "timodonnell/helico-data", repo_type="dataset",
                        local_dir=DATA_CACHE,
                        allow_patterns=patterns, max_workers=8,
                        etag_timeout=30,
                    )
                    break
                except Exception as e:
                    print(f"[data chunk {i}] failed: {e}", flush=True)
            else:
                raise RuntimeError(f"data chunk {i} failed after 5 attempts")
        data_volume.commit()

        # Load model. Two formats:
        #   Protenix checkpoint: {"model": state_dict} — infer config from
        #     shapes, then map Protenix keys onto Helico via
        #     load_protenix_state_dict.
        #   Helico checkpoint: {"step": N, "model_state_dict": sd,
        #     "config": TrainConfig-dict, ...} — load directly after
        #     constructing Helico with matching n_pairformer_blocks etc.
        ckpt_path = self.checkpoint_path or "protenix-v1"
        if ckpt_path in ("", "protenix-v1"):
            print(f"Loading Protenix v1 checkpoint: {PROTENIX_CKPT_PATH}")
            ckpt = torch.load(PROTENIX_CKPT_PATH, map_location="cpu", weights_only=False)
            ptx_sd = ckpt["model"]
            ptx_sd = OrderedDict(
                (k.removeprefix("module."), v) for k, v in ptx_sd.items()
            )
            config = infer_protenix_config(ptx_sd)
            print(f"Inferred Protenix config: d_pair={config.d_pair}, d_msa={config.d_msa}")
            self.model = Helico(config)
            load_protenix_state_dict(ptx_sd, self.model)
        else:
            from helico.model import HelicoConfig
            print(f"Loading Helico checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" not in state:
                raise RuntimeError(
                    f"checkpoint {ckpt_path} is not a Helico checkpoint "
                    f"(missing 'model_state_dict'; keys={list(state.keys())[:5]})"
                )
            # Pull the two shape-defining HelicoConfig fields from the
            # saved TrainConfig dict; fall back to defaults if absent.
            saved_cfg = state.get("config") or {}
            config = HelicoConfig(
                n_pairformer_blocks=saved_cfg.get("n_pairformer_blocks", 48),
                n_diffusion_token_blocks=saved_cfg.get("n_diffusion_token_blocks", 24),
            )
            print(f"Helico config: n_pairformer_blocks={config.n_pairformer_blocks}, "
                  f"n_diffusion_token_blocks={config.n_diffusion_token_blocks}, "
                  f"step={state.get('step', '?')}")
            self.model = Helico(config)
            self.model.load_state_dict(state["model_state_dict"])

        # Load CCD
        self.ccd = parse_ccd()

        # Ensure FoldBench data is available
        self.foldbench_dir = download_foldbench()

    @modal.method()
    def predict(
        self,
        pdb_id: str,
        category: str,
        n_samples: int = 5,
        max_tokens: int = 2048,
        n_cycles: int = 10,
    ) -> dict | None:
        """Run prediction for a single target. Returns serializable result dict or None."""
        import logging
        import numpy as np
        import torch
        from helico.data import parse_mmcif
        from helico.bench import (
            _find_gt_path,
            predict_target,
            structure_to_chains,
        )
        from helico.train import coords_to_pdb

        logger = logging.getLogger(__name__)

        gt_dir = self.foldbench_dir / "examples" / "ground_truths"
        msa_dir = self.foldbench_dir / "foldbench-msas"
        if not msa_dir.exists():
            msa_dir = None

        try:
            gt_path = _find_gt_path(gt_dir, pdb_id)
            gt_structure = parse_mmcif(gt_path, max_resolution=float("inf"))
            assert gt_structure is not None, f"Failed to parse ground truth: {gt_path}"
            chains = structure_to_chains(gt_structure)

            pred_result = predict_target(
                self.model,
                chains,
                self.ccd,
                target_name=pdb_id,
                n_samples=n_samples,
                max_tokens=max_tokens,
                msa_dir=msa_dir,
                n_cycles=n_cycles,
            )

            if pred_result is None:
                return {"pdb_id": pdb_id, "category": category, "status": "too_large"}

            tokenized, results = pred_result
            pred_coords_np = results["coords"][0].cpu().float().numpy()
            plddt_np = results["plddt"][0].cpu().float().numpy()
            pred_pdb_str = coords_to_pdb(
                results["coords"][0], results["plddt"][0], tokenized,
            )
            torch.cuda.empty_cache()

            return {
                "pdb_id": pdb_id,
                "category": category,
                "status": "ok",
                "pred_coords": pred_coords_np,
                "plddt": plddt_np,
                "pdb_str": pred_pdb_str,
                "tokenized": tokenized,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM on {pdb_id}")
                torch.cuda.empty_cache()
                return {"pdb_id": pdb_id, "category": category, "status": "oom"}
            logger.error(f"RuntimeError on {pdb_id}: {e}")
            return {"pdb_id": pdb_id, "category": category, "status": "error"}
        except Exception as e:
            logger.error(f"Error on {pdb_id}: {e}")
            return {"pdb_id": pdb_id, "category": category, "status": "error"}


@app.local_entrypoint()
def run_bench(
    n_samples: int = 5,
    categories: str = "",
    output_dir: str = "bench_results",
    resume: bool = False,
    max_tokens: int = 2048,
    n_cycles: int = 10,
    cutoff_date: str = "2024-01-01",
    max_targets: int = 0,
    checkpoint: str = "protenix-v1",
):
    import logging
    import pickle

    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    from helico.bench import (
        INTERFACE_CATEGORIES,
        _find_gt_path,
        _pdb_code,
        download_foldbench,
        fetch_release_dates,
        load_targets,
        match_atoms,
        print_summary,
        score_interface,
        score_ligand_interface,
        score_monomer,
        write_category_csv,
        write_summary_csv,
    )
    from helico.data import parse_mmcif

    logger.info(f"Using {N_WORKERS} {GPU_TYPE} workers (set HELICO_BENCH_WORKERS / HELICO_BENCH_GPU to change)")

    # Download FoldBench locally (just target CSVs + ground truths for scoring)
    foldbench_dir = download_foldbench()
    targets_dir = foldbench_dir / "targets"
    gt_dir = foldbench_dir / "examples" / "ground_truths"

    all_targets = load_targets(targets_dir)
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]
        all_targets = {k: v for k, v in all_targets.items() if k in cat_list}

    # Filter targets by release date
    if cutoff_date:
        logger.info(f"Filtering targets with release date > {cutoff_date}")
        all_pdb_codes = [_pdb_code(t.pdb_id) for ts in all_targets.values() for t in ts]
        release_dates = fetch_release_dates(all_pdb_codes)
        filtered_targets = {}
        for category, targets in all_targets.items():
            kept = [t for t in targets
                    if release_dates.get(_pdb_code(t.pdb_id), "") > cutoff_date]
            logger.info(f"  {category}: {len(kept)}/{len(targets)} targets after date filter")
            filtered_targets[category] = kept
        all_targets = filtered_targets

    # Cap total targets for shakedown runs (distributes across categories, round-robin)
    if max_targets > 0:
        total = sum(len(t) for t in all_targets.values())
        if total > max_targets:
            logger.info(f"Capping to max_targets={max_targets} (from {total})")
            remaining = max_targets
            capped = {c: [] for c in all_targets}
            cats = list(all_targets.keys())
            idx = 0
            pulled = True
            while remaining > 0 and pulled:
                pulled = False
                for c in cats:
                    if remaining <= 0:
                        break
                    if idx < len(all_targets[c]):
                        capped[c].append(all_targets[c][idx])
                        remaining -= 1
                        pulled = True
                idx += 1
            all_targets = capped
            for c, ts in all_targets.items():
                logger.info(f"  {c}: {len(ts)} targets after cap")

    output_path = Path(output_dir)
    predictions_dir = output_path / "predictions"
    results_dir = output_path / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build flat list of (pdb_id, category) to predict. Dedupe by pdb_id —
    # some targets appear in multiple category CSVs; predict once, score
    # against each category.
    to_predict = []
    cached_results = {}  # pdb_id -> cached prediction dict
    seen_pdb_ids = set()

    for category, targets in all_targets.items():
        for target in targets:
            pdb_id = target.pdb_id
            if pdb_id in seen_pdb_ids:
                continue
            seen_pdb_ids.add(pdb_id)
            pred_cache_path = predictions_dir / f"{pdb_id}.pkl"
            if resume and pred_cache_path.exists():
                try:
                    with open(pred_cache_path, "rb") as f:
                        cached = pickle.load(f)
                    cached["category"] = category
                    cached["status"] = "ok"
                    cached_results[pdb_id] = cached
                    logger.info(f"Cached: {pdb_id}")
                    continue
                except Exception:
                    logger.warning(f"Failed to load cache for {pdb_id}, re-predicting")
            to_predict.append((pdb_id, category))

    logger.info(
        f"Total targets: {sum(len(t) for t in all_targets.values())}, "
        f"cached: {len(cached_results)}, to predict: {len(to_predict)}"
    )

    # Fan out predictions across Modal workers
    prediction_results = {}  # pdb_id -> result dict
    prediction_results.update(cached_results)

    if to_predict:
        # "protenix-v1" and "" both signal the baked-in Protenix checkpoint.
        # Anything else is a path on the helico-checkpoints Volume.
        ckpt_param = "" if checkpoint == "protenix-v1" else checkpoint
        predictor = Predictor(checkpoint_path=ckpt_param)
        logger.info(f"Using checkpoint: {checkpoint}")
        results_iter = predictor.predict.map(
            [pdb_id for pdb_id, _ in to_predict],
            [category for _, category in to_predict],
            [n_samples] * len(to_predict),
            [max_tokens] * len(to_predict),
            [n_cycles] * len(to_predict),
        )

        for result in results_iter:
            if result is None:
                continue
            pdb_id = result["pdb_id"]
            logger.info(f"Received: {pdb_id} (status={result['status']})")

            # Cache successful predictions
            if result.get("status") == "ok":
                pred_cache_path = predictions_dir / f"{pdb_id}.pkl"
                with open(pred_cache_path, "wb") as f:
                    pickle.dump({
                        "tokenized": result["tokenized"],
                        "pred_coords": result["pred_coords"],
                        "plddt": result["plddt"],
                        "pdb_str": result["pdb_str"],
                    }, f)

            prediction_results[pdb_id] = result

    # Score all predictions locally
    logger.info("Scoring predictions...")
    category_summaries = []

    for category, targets in all_targets.items():
        logger.info(f"\nScoring {category} ({len(targets)} targets)")

        is_interface = category in INTERFACE_CATEGORIES
        is_ligand = category == "interface_protein_ligand"

        category_results = []
        n_predicted = 0
        n_success = 0

        for target in targets:
            pdb_id = target.pdb_id
            result_row = {"pdb_id": pdb_id, "status": "failed"}

            pred = prediction_results.get(pdb_id)
            if pred is None or pred.get("status") != "ok":
                result_row["status"] = pred["status"] if pred else "missing"
                category_results.append(result_row)
                continue

            try:
                tokenized = pred["tokenized"]
                pred_coords_np = pred["pred_coords"]
                pred_pdb_str = pred.get("pdb_str", "")

                gt_path = _find_gt_path(gt_dir, pdb_id)
                gt_structure = parse_mmcif(gt_path, max_resolution=float("inf"))
                assert gt_structure is not None

                matched = match_atoms(tokenized, pred_coords_np, gt_structure)
                assert len(matched.pred_coords) > 0, f"No atoms matched for {pdb_id}"

                n_predicted += 1
                if is_ligand:
                    scores = score_ligand_interface(matched)
                    success = (
                        not np.isnan(scores.get("lrmsd", float("nan")))
                        and scores["lrmsd"] < 2.0
                        and not np.isnan(scores.get("lddt_pli", float("nan")))
                        and scores["lddt_pli"] > 0.8
                    )
                elif is_interface:
                    scores = score_interface(pred_pdb_str, gt_path, matched)
                    success = scores.get("dockq", 0.0) >= 0.23
                else:
                    scores = score_monomer(matched)
                    success = False

                if success:
                    n_success += 1

                result_row["status"] = "ok"
                result_row["n_matched_atoms"] = len(matched.pred_coords)
                result_row.update(scores)
                category_results.append(result_row)

                logger.info(
                    f"  {pdb_id}: "
                    + " | ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in scores.items()
                    )
                )

            except Exception as e:
                logger.error(f"Scoring error on {pdb_id}: {e}")
                result_row["status"] = "error"
                category_results.append(result_row)

        write_category_csv(category_results, results_dir / f"{category}.csv")

        ok_results = [r for r in category_results if r["status"] == "ok"]
        mean_lddt = float(np.mean([r["lddt"] for r in ok_results])) if ok_results else 0.0
        mean_dockq = float("nan")
        if is_interface and not is_ligand and ok_results:
            dockq_vals = [r.get("dockq", float("nan")) for r in ok_results]
            dockq_vals = [v for v in dockq_vals if not np.isnan(v)]
            mean_dockq = float(np.mean(dockq_vals)) if dockq_vals else float("nan")

        success_pct = float("nan")
        if is_interface or is_ligand:
            success_pct = (
                100.0 * n_success / max(n_predicted, 1) if n_predicted > 0 else 0.0
            )

        category_summaries.append({
            "category": category,
            "n_total": len(targets),
            "n_predicted": n_predicted,
            "success_pct": success_pct,
            "mean_lddt": mean_lddt,
            "mean_dockq": mean_dockq,
        })

    print_summary(category_summaries)
    write_summary_csv(category_summaries, output_path / "summary.csv")
    logger.info(f"Results written to {output_path}")
