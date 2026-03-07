"""Parallel FoldBench benchmark on Modal — fans out predictions across GPU workers.

Configure via environment variables before running:
    HELICO_BENCH_WORKERS=8 HELICO_BENCH_GPU=H100 modal run modal/bench.py
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
PROTENIX_URL = "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt"

# Modal decorator params are static — configure via env vars before `modal run`
N_WORKERS = int(os.environ.get("HELICO_BENCH_WORKERS", "4"))
GPU_TYPE = os.environ.get("HELICO_BENCH_GPU", "H100")

bench_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .pip_install(
        "torch>=2.7",
        "cuequivariance-torch>=0.8",
        "cuequivariance-ops-torch-cu12>=0.8",
        "biopython>=1.80",
        "numpy",
        "scipy",
        "pyyaml>=6.0",
        "huggingface_hub>=0.20",
        "requests",
        "tmtools",
        "DockQ",
        "tqdm",
    )
    # Protenix checkpoint baked into image (1.4 GB, cached by Modal)
    .run_commands(
        f"mkdir -p /root/helico/checkpoints && "
        f"wget -q -O /root/helico/checkpoints/protenix_base_default_v1.0.0.pt {PROTENIX_URL}"
    )
    # CCD cache
    .run_commands(
        "mkdir -p /root/.cache/helico/data && "
        "hf download timodonnell/helico-data processed/ccd_cache.pkl "
        "--repo-type dataset --local-dir /root/.cache/helico/data"
    )
    # FoldBench data: targets, ground truths, AF3 inputs, MSAs
    .run_commands(
        "hf download timodonnell/helico-data 'benchmarks/FoldBench/**' "
        "--repo-type dataset --local-dir /root/.cache/helico/data"
    )
    # Project code last (changes most frequently)
    .add_local_dir(str(ROOT / "src"), remote_path="/root/helico/src")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/helico/pyproject.toml")
    .add_local_file(str(ROOT / "README.md"), remote_path="/root/helico/README.md")
)

app = modal.App("helico-bench", image=bench_image)


@app.cls(gpu=GPU_TYPE, timeout=600, max_containers=N_WORKERS)
class Predictor:
    @modal.enter()
    def setup(self):
        import subprocess
        subprocess.run(
            "cd /root/helico && uv venv --python 3.11 && uv pip install -e '.[bench]'",
            check=True, shell=True,
        )

        import sys
        sys.path.insert(0, "/root/helico/src")

        from collections import OrderedDict
        import torch
        from helico.data import parse_ccd
        from helico.model import Helico, HelicoConfig
        from helico.load_protenix import load_protenix_state_dict
        from helico.bench import download_foldbench

        # Load model
        config = HelicoConfig()
        self.model = Helico(config)
        ckpt = torch.load(
            "/root/helico/checkpoints/protenix_base_default_v1.0.0.pt",
            map_location="cpu", weights_only=False,
        )
        ptx_sd = ckpt["model"]
        ptx_sd = OrderedDict(
            (k.removeprefix("module."), v) for k, v in ptx_sd.items()
        )
        load_protenix_state_dict(ptx_sd, self.model)

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
):
    import logging
    import pickle

    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    from helico.data import parse_mmcif
    from helico.bench import (
        INTERFACE_CATEGORIES,
        _find_gt_path,
        download_foldbench,
        load_targets,
        match_atoms,
        print_summary,
        score_interface,
        score_ligand_interface,
        score_monomer,
        write_category_csv,
        write_summary_csv,
    )

    logger.info(f"Using {N_WORKERS} {GPU_TYPE} workers (set HELICO_BENCH_WORKERS / HELICO_BENCH_GPU to change)")

    # Download FoldBench locally (just target CSVs + ground truths for scoring)
    foldbench_dir = download_foldbench()
    targets_dir = foldbench_dir / "targets"
    gt_dir = foldbench_dir / "examples" / "ground_truths"

    all_targets = load_targets(targets_dir)
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]
        all_targets = {k: v for k, v in all_targets.items() if k in cat_list}

    output_path = Path(output_dir)
    predictions_dir = output_path / "predictions"
    results_dir = output_path / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build flat list of (pdb_id, category) to predict
    to_predict = []
    cached_results = {}  # pdb_id -> cached prediction dict

    for category, targets in all_targets.items():
        for target in targets:
            pdb_id = target.pdb_id
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
        predictor = Predictor()
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
