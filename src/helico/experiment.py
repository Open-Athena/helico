"""Helico experiment primitives.

Provides idempotent wrappers around Modal training and benchmarking so that
jupytext-markdown experiment notebooks can be re-run cheaply during
iteration. Cache key is the step name only: repeat calls with the same
name return instantly without re-dispatching Modal.

Typical use inside a notebook cell:

    from helico.experiment import ensure_bench_run
    bench = ensure_bench_run(
        "protenix-v1-baseline",
        checkpoint="protenix-v1",
        workers=8,
    )
    bench.summary  # pandas DataFrame indexed by category

Cache precedence on lookup:
    1. Local .cache/ under the experiment directory (fast re-read)
    2. Modal volume `helico-experiments` (authoritative; fetched on miss)
    3. Launch Modal; sync result to volume on success.

Dry-run mode: set HELICO_DRY_RUN=1. Every ensure_* records its cost
estimate and returns a sentinel without touching Modal. Used by the agent
to gate on total cost before a real run.

See .agents/project/20260422_experiment_system_design.md for the full
design. See experiments/AGENTS.md for rules on using this module.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration: repo paths, prices, config
# --------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent
# src/helico/experiment.py -> ../../ = repo root
REPO_ROOT = _MODULE_DIR.parent.parent
PRICES_PATH = REPO_ROOT / "scripts" / "pm" / "modal_prices.yaml"
CONFIG_PATH = REPO_ROOT / ".github" / "experiments.yaml"

EXPERIMENTS_VOLUME = "helico-experiments"
CHECKPOINTS_VOLUME = "helico-checkpoints"


def _load_prices() -> dict:
    with open(PRICES_PATH) as f:
        return yaml.safe_load(f)


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# Experiment context
# --------------------------------------------------------------------------

_EXPERIMENT_SLUG: Optional[str] = None


def set_experiment(slug_or_path: str | Path) -> None:
    """Declare which experiment the current notebook belongs to.

    Call once at the top of the notebook. Accepts either a slug
    (`"exp1_protenix_baseline"`) or a directory path.

    If not called, the library auto-detects from (in order):
        - `HELICO_EXPERIMENT` env var
        - walking up from cwd looking for `experiments/exp*/`
    """
    global _EXPERIMENT_SLUG
    p = Path(slug_or_path)
    if p.is_dir():
        _EXPERIMENT_SLUG = p.name
    else:
        _EXPERIMENT_SLUG = str(slug_or_path)


def _current_experiment() -> str:
    if _EXPERIMENT_SLUG:
        return _EXPERIMENT_SLUG
    if env := os.environ.get("HELICO_EXPERIMENT"):
        return env
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if parent.parent.name == "experiments" and parent.name.startswith("exp"):
            return parent.name
    raise RuntimeError(
        "Could not detect experiment. Call helico.experiment.set_experiment(...), "
        "set HELICO_EXPERIMENT, or run from an experiments/exp*/ directory."
    )


def _experiment_dir(slug: str) -> Path:
    return REPO_ROOT / "experiments" / slug


def _cache_dir(slug: str) -> Path:
    return _experiment_dir(slug) / ".cache"


def experiment_dir() -> Path:
    """Absolute path to the current experiment's directory.

    Use this in notebooks to build committed artifact paths (plots/, data/)
    so they resolve the same way regardless of cwd (kernel, jupytext, CI).
    """
    return _experiment_dir(_current_experiment())


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------


def estimate_cost(*, gpu: str, hours: float, count: int = 1) -> float:
    """Estimate Modal $ cost for `count` GPUs of `gpu` type running for `hours`."""
    prices = _load_prices()
    gpus = prices["gpus"]
    if gpu not in gpus:
        raise ValueError(
            f"Unknown GPU: {gpu!r}. Known: {sorted(gpus)}. "
            f"Edit scripts/pm/modal_prices.yaml to add it."
        )
    return round(float(gpus[gpu]) * hours * count, 2)


def _estimate_bench_cost(gpu: str, workers: int, wall_hours: float) -> float:
    prices = _load_prices()
    floor = float(prices.get("bench_flat_usd", 5.0))
    est = estimate_cost(gpu=gpu, hours=wall_hours, count=workers)
    return max(floor, est)


# --------------------------------------------------------------------------
# Dry-run accumulator
# --------------------------------------------------------------------------

_DRY_RUN_RECORDS: list[dict] = []


def is_dry_run() -> bool:
    return os.environ.get("HELICO_DRY_RUN", "") not in ("", "0", "false", "False")


def _record_dry_run(kind: str, name: str, usd: float) -> None:
    _DRY_RUN_RECORDS.append({"kind": kind, "name": name, "usd": usd})


def dry_run_records() -> list[dict]:
    """Return all recorded ensure_* estimates from this process's dry run."""
    return list(_DRY_RUN_RECORDS)


def dry_run_total_usd() -> float:
    return round(sum(r["usd"] for r in _DRY_RUN_RECORDS), 2)


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


@dataclass
class BenchRun:
    """Result of a FoldBench run.

    Artifacts live in `cache_dir` (local) and on Modal volume
    `helico-experiments` at `volume_path` (authoritative). `summary` lazy-
    loads summary.csv as a pandas DataFrame indexed by category.
    """

    name: str
    experiment: str
    cache_dir: Path
    volume_path: str
    cached: bool
    meta: dict = field(default_factory=dict)

    @property
    def summary_csv_path(self) -> Path:
        return self.cache_dir / "summary.csv"

    @property
    def summary(self):  # -> pd.DataFrame
        import pandas as pd

        return pd.read_csv(self.summary_csv_path).set_index("category")

    def per_category(self, category: str):
        import pandas as pd

        p = self.cache_dir / "results" / f"{category}.csv"
        return pd.read_csv(p)


@dataclass
class TrainingRun:
    name: str
    experiment: str
    cache_dir: Path
    volume_path: str  # /ckpts/<experiment>-<name>/ on helico-checkpoints
    cached: bool
    meta: dict = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> str:
        return f"{self.volume_path}/final.pt"


# --------------------------------------------------------------------------
# Modal volume helpers (via CLI subprocess)
# --------------------------------------------------------------------------


def _ensure_volume_exists(volume_name: str) -> None:
    """Create the Modal volume if it doesn't exist. Idempotent."""
    try:
        import modal

        modal.Volume.from_name(volume_name, create_if_missing=True)
    except Exception as e:
        logger.warning("Could not ensure Modal volume %s: %s", volume_name, e)


def _volume_path_exists(volume_name: str, remote_path: str) -> bool:
    """Check via `modal volume ls` whether a path exists on a volume."""
    try:
        result = subprocess.run(
            ["modal", "volume", "ls", volume_name, remote_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _volume_pull(volume_name: str, remote_path: str, local_dir: Path) -> None:
    """Download `remote_path` from volume into `local_dir`. Overwrites."""
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", volume_name, remote_path, str(local_dir)],
        check=True,
    )


def _volume_push(volume_name: str, local_dir: Path, remote_path: str) -> None:
    """Upload `local_dir` to volume at `remote_path`. Creates parents."""
    _ensure_volume_exists(volume_name)
    subprocess.run(
        ["modal", "volume", "put", "--force", volume_name, str(local_dir), remote_path],
        check=True,
    )


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _log_start(kind: str, name: str, status: str, est_cost: Optional[float] = None) -> None:
    cost = f" (~${est_cost:.2f})" if est_cost is not None else ""
    print(f"[helico.experiment] {kind}({name!r}) — {status}{cost}", flush=True)


# --------------------------------------------------------------------------
# ensure_bench_run
# --------------------------------------------------------------------------


def ensure_bench_run(
    name: str,
    *,
    checkpoint: str = "protenix-v1",
    workers: int = 8,
    gpu: str = "H100",
    n_samples: int = 5,
    max_tokens: int = 2048,
    n_cycles: int = 10,
    cutoff_date: str = "2024-01-01",
    categories: str = "",
    est_wall_hours: float = 0.5,
    force: bool = False,
    publish: bool = False,
) -> BenchRun:
    """Run FoldBench idempotently. Returns cached result if `name` has run before.

    Caching is keyed by `(experiment_slug, name)` only. To force a rerun,
    pass `force=True` or bump `name`.

    `checkpoint` is either `"protenix-v1"` (the baked-in Protenix checkpoint)
    or a path on the `helico-checkpoints` Volume, e.g.
    `"/ckpts/v1-finetune-01/final.pt"`. The path must exist on the Volume
    at the time the bench runs.

    On cache miss: invokes `modal run modal/bench.py` via subprocess, then
    syncs results to the `helico-experiments` volume at
    `/experiments/<slug>/<name>/`.
    """
    slug = _current_experiment()
    volume_path = f"/experiments/{slug}/{name}"

    est_cost = _estimate_bench_cost(gpu=gpu, workers=workers, wall_hours=est_wall_hours)

    if is_dry_run():
        _log_start("ensure_bench_run", name, "dry-run", est_cost)
        _record_dry_run("bench", name, est_cost)
        # Dry-run artifacts go to a separate scratch tree so they can never
        # be mistaken for a real cache hit by a subsequent non-dry run.
        scratch_dir = _cache_dir(slug) / "dry-run-scratch" / "benches" / name
        _write_placeholder_bench(scratch_dir, name=name, slug=slug, est_cost=est_cost)
        return _load_bench_run(name, slug, scratch_dir, volume_path, cached=False)

    cache_dir = _cache_dir(slug) / "benches" / name

    if not force:
        if _bench_complete_local(cache_dir):
            _log_start("ensure_bench_run", name, "cached (local)")
            return _load_bench_run(name, slug, cache_dir, volume_path, cached=True)
        if _volume_path_exists(EXPERIMENTS_VOLUME, f"{volume_path}/summary.csv"):
            _log_start("ensure_bench_run", name, "cached (volume); fetching")
            _volume_pull(EXPERIMENTS_VOLUME, volume_path, cache_dir.parent)
            return _load_bench_run(name, slug, cache_dir, volume_path, cached=True)

    _log_start("ensure_bench_run", name, "launching", est_cost)

    # If we have cached predictions from a prior run that didn't reach
    # the meta.json/summary.csv finish line (e.g. crashed in scoring),
    # preserve them and pass --resume so modal/bench.py picks them up
    # instead of re-predicting.
    have_predictions = (
        cache_dir.exists()
        and (cache_dir / "predictions").is_dir()
        and any((cache_dir / "predictions").iterdir())
    )
    if cache_dir.exists() and not have_predictions:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if have_predictions:
        n_cached_preds = sum(1 for _ in (cache_dir / "predictions").iterdir())
        print(
            f"[helico.experiment] ensure_bench_run({name!r}) — resuming from "
            f"{n_cached_preds} cached predictions",
            flush=True,
        )

    env = os.environ.copy()
    env["HELICO_BENCH_WORKERS"] = str(workers)
    env["HELICO_BENCH_GPU"] = gpu

    cmd = [
        "modal", "run", "modal/bench.py",
        "--n-samples", str(n_samples),
        "--max-tokens", str(max_tokens),
        "--n-cycles", str(n_cycles),
        "--cutoff-date", cutoff_date,
        "--output-dir", str(cache_dir.resolve()),
        "--checkpoint", checkpoint,
    ]
    if have_predictions:
        cmd.append("--resume")
    if categories:
        cmd += ["--categories", categories]

    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))

    meta = {
        "name": name,
        "experiment": slug,
        "checkpoint": checkpoint,
        "gpu": gpu,
        "workers": workers,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "n_cycles": n_cycles,
        "cutoff_date": cutoff_date,
        "categories": categories,
        "git_sha": _git_sha(),
        "est_cost_usd": est_cost,
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Sync to volume. Failures here don't invalidate the local run — log
    # a warning and leave the researcher with local results.
    try:
        _volume_push(EXPERIMENTS_VOLUME, cache_dir, volume_path)
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Failed to push bench run to Modal volume %s (%s). Local cache is intact.",
            EXPERIMENTS_VOLUME, e,
        )

    bench = _load_bench_run(name, slug, cache_dir, volume_path, cached=False)
    if publish:
        _maybe_publish_bench(bench)
    return bench


def _bench_complete_local(cache_dir: Path) -> bool:
    return (cache_dir / "meta.json").exists() and (cache_dir / "summary.csv").exists()


_BENCH_CATEGORIES = [
    "monomer_protein", "monomer_dna", "monomer_rna",
    "interface_protein_protein", "interface_antibody_antigen",
    "interface_protein_peptide", "interface_protein_ligand",
    "interface_protein_dna", "interface_protein_rna",
]


def _write_placeholder_bench(cache_dir: Path, *, name: str, slug: str, est_cost: float) -> None:
    """Write a minimal but non-empty bench output so dry-run notebooks can
    execute through their analysis cells (groupbys, plots) without crashing.
    Values are placeholders — the cost estimate is the only real output."""
    import csv

    results_dir = cache_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n_total", "n_predicted", "success_pct", "mean_lddt", "mean_dockq"])
        for cat in _BENCH_CATEGORIES:
            is_iface = cat.startswith("interface_")
            w.writerow([cat, 3, 3, 0.0 if is_iface else "nan", 0.5,
                        0.3 if is_iface and cat != "interface_protein_ligand" else "nan"])

    # Three placeholder rows per category with plausible-looking values so
    # downstream groupbys / violinplots / bar charts don't crash on all-NaN
    # data. pdb_id starts with "PLACEHOLDER" so it's obvious on inspection.
    for cat in _BENCH_CATEGORIES:
        is_iface = cat.startswith("interface_") and cat != "interface_protein_ligand"
        with open(results_dir / f"{cat}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pdb_id", "status", "n_matched_atoms", "lddt", "dockq"])
            for i, lddt in enumerate([0.4, 0.5, 0.6]):
                dockq = round(0.2 + 0.1 * i, 1) if is_iface else "nan"
                w.writerow([f"PLACEHOLDER-{i}", "ok", 100, lddt, dockq])

    with open(cache_dir / "meta.json", "w") as f:
        json.dump({
            "name": name, "experiment": slug, "dry_run": True,
            "est_cost_usd": est_cost,
        }, f, indent=2)


def _load_bench_run(
    name: str, slug: str, cache_dir: Path, volume_path: str, *, cached: bool,
) -> BenchRun:
    meta: dict = {}
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return BenchRun(
        name=name, experiment=slug, cache_dir=cache_dir,
        volume_path=volume_path, cached=cached, meta=meta,
    )


# --------------------------------------------------------------------------
# ensure_training_run
# --------------------------------------------------------------------------


# Mapping from ensure_training_run kwargs to HELICO_TRAIN_* env vars. All
# values are stringified before handing off to modal run (subprocess env
# must be str -> str).
_TRAIN_ENV_MAPPING = {
    "max_steps": "HELICO_TRAIN_MAX_STEPS",
    "crop_size": "HELICO_TRAIN_CROP",
    "batch_size": "HELICO_TRAIN_BATCH",
    "lr": "HELICO_TRAIN_LR",
    "warmup_steps": "HELICO_TRAIN_WARMUP",
    "save_every": "HELICO_TRAIN_SAVE_EVERY",
    "log_every": "HELICO_TRAIN_LOG_EVERY",
    "val_every": "HELICO_TRAIN_VAL_EVERY",
    "val_samples": "HELICO_TRAIN_VAL_SAMPLES",
    "n_diffusion_samples": "HELICO_TRAIN_N_DIFFUSION_SAMPLES",
    "resume": "HELICO_TRAIN_RESUME",
    "train_cutoff": "HELICO_TRAIN_CUTOFF",
    "val_cutoff_start": "HELICO_VAL_START",
    "val_cutoff_end": "HELICO_VAL_END",
    "wandb_project": "HELICO_TRAIN_WANDB_PROJECT",
}


def _estimate_train_cost(gpu: str, est_wall_hours: float) -> float:
    """gpu is `TYPE:COUNT`, e.g. `H100:8`."""
    parts = gpu.split(":")
    gpu_type = parts[0]
    count = int(parts[1]) if len(parts) > 1 else 1
    return estimate_cost(gpu=gpu_type, hours=est_wall_hours, count=count)


def ensure_training_run(
    name: str,
    *,
    gpu: str = "H100:8",
    max_steps: int = 10000,
    crop_size: int = 384,
    batch_size: int = 1,
    lr: float = 5e-5,
    warmup_steps: int = 200,
    save_every: int = 250,
    log_every: int = 10,
    val_every: int = 0,
    val_samples: int = 32,
    n_diffusion_samples: int = 8,
    resume: Optional[str] = None,
    protenix_init: bool = True,
    train_cutoff: str = "2021-09-30",
    val_cutoff_start: str = "2022-05-01",
    val_cutoff_end: str = "2023-01-12",
    wandb_project: str = "helico",
    est_wall_hours: float = 12.0,
    force: bool = False,
    publish: bool = False,
) -> TrainingRun:
    """Train a model idempotently.

    Cache is keyed by `(experiment_slug, name)`. Run name on Modal is
    `{experiment_slug}-{name}`; the final checkpoint lands at
    `/ckpts/<run_name>/final.pt` on the `helico-checkpoints` volume. A
    cache hit here (or in the matching local `.cache/trainings/<name>/`)
    short-circuits without launching Modal.

    On fresh run: subprocesses `modal run modal/train.py` with HELICO_TRAIN_*
    env vars mapped from the kwargs; blocks until completion (up to 24h
    timeout on the Modal side). Records a meta.json locally after success.

    Dry-run mode returns a sentinel without dispatching; cost estimate is
    accumulated for the gate.
    """
    slug = _current_experiment()
    run_name = f"{slug}-{name}"
    volume_path = f"/ckpts/{run_name}"
    local_cache = _cache_dir(slug) / "trainings" / name

    est_cost = _estimate_train_cost(gpu=gpu, est_wall_hours=est_wall_hours)

    if is_dry_run():
        _log_start("ensure_training_run", name, "dry-run", est_cost)
        _record_dry_run("training", name, est_cost)
        scratch = _cache_dir(slug) / "dry-run-scratch" / "trainings" / name
        scratch.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": name, "experiment": slug, "dry_run": True,
            "run_name": run_name, "volume_path": volume_path,
            "est_cost_usd": est_cost,
        }
        (scratch / "meta.json").write_text(json.dumps(meta, indent=2))
        return TrainingRun(
            name=name, experiment=slug, cache_dir=scratch,
            volume_path=volume_path, cached=False, meta=meta,
        )

    if not force:
        if _training_complete_local(local_cache):
            _log_start("ensure_training_run", name, "cached (local)")
            return _load_training_run(name, slug, local_cache, volume_path, cached=True)
        if _volume_path_exists(CHECKPOINTS_VOLUME, f"{volume_path}/final.pt"):
            _log_start("ensure_training_run", name, "cached (volume); recording meta")
            # Don't pull the checkpoint locally — it's multi-GB. Record a
            # meta.json pointing at the volume path so future cache lookups
            # succeed via the local fast path.
            local_cache.mkdir(parents=True, exist_ok=True)
            meta = {
                "name": name, "experiment": slug,
                "run_name": run_name, "volume_path": volume_path,
                "cached_from_volume": True,
            }
            (local_cache / "meta.json").write_text(json.dumps(meta, indent=2))
            return _load_training_run(name, slug, local_cache, volume_path, cached=True)

    _log_start("ensure_training_run", name, "launching", est_cost)

    local_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HELICO_TRAIN_GPU"] = gpu
    env["HELICO_TRAIN_RUN_NAME"] = run_name
    env["HELICO_TRAIN_PROTENIX_INIT"] = "1" if protenix_init else "0"
    # Map remaining kwargs
    kwargs = {
        "max_steps": max_steps, "crop_size": crop_size, "batch_size": batch_size,
        "lr": lr, "warmup_steps": warmup_steps, "save_every": save_every,
        "log_every": log_every, "val_every": val_every, "val_samples": val_samples,
        "n_diffusion_samples": n_diffusion_samples, "resume": resume or "",
        "train_cutoff": train_cutoff, "val_cutoff_start": val_cutoff_start,
        "val_cutoff_end": val_cutoff_end, "wandb_project": wandb_project,
    }
    for k, v in kwargs.items():
        env[_TRAIN_ENV_MAPPING[k]] = str(v)

    subprocess.run(
        ["modal", "run", "modal/train.py"],
        check=True, env=env, cwd=str(REPO_ROOT),
    )

    meta = {
        "name": name,
        "experiment": slug,
        "run_name": run_name,
        "volume_path": volume_path,
        "gpu": gpu,
        "max_steps": max_steps,
        "crop_size": crop_size,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "val_every": val_every,
        "protenix_init": protenix_init,
        "train_cutoff": train_cutoff,
        "val_cutoff_start": val_cutoff_start,
        "val_cutoff_end": val_cutoff_end,
        "wandb_project": wandb_project,
        "git_sha": _git_sha(),
        "est_cost_usd": est_cost,
    }
    (local_cache / "meta.json").write_text(json.dumps(meta, indent=2))

    run = _load_training_run(name, slug, local_cache, volume_path, cached=False)
    if publish:
        _maybe_publish_training(run)
    return run


def _training_complete_local(cache_dir: Path) -> bool:
    return (cache_dir / "meta.json").exists()


def _maybe_publish_bench(bench: BenchRun) -> None:
    """Best-effort HF upload after a fresh bench run. Failures are logged
    but don't raise — the bench artifacts are already on local disk and
    the Modal volume; the researcher can always `helico-publish` manually."""
    try:
        from helico.hf_publish import publish_bench_run
        url = publish_bench_run(
            experiment=bench.experiment, name=bench.name,
        )
        print(f"[helico.experiment] published to {url}", flush=True)
    except Exception as e:
        logger.warning("Auto-publish of bench run failed: %s", e)


def _maybe_publish_training(run: TrainingRun) -> None:
    try:
        from helico.hf_publish import publish_training_run
        url = publish_training_run(
            experiment=run.experiment, name=run.name,
        )
        print(f"[helico.experiment] published to {url}", flush=True)
    except Exception as e:
        logger.warning("Auto-publish of training run failed: %s", e)


def _load_training_run(
    name: str, slug: str, cache_dir: Path, volume_path: str, *, cached: bool,
) -> TrainingRun:
    meta: dict = {}
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return TrainingRun(
        name=name, experiment=slug, cache_dir=cache_dir,
        volume_path=volume_path, cached=cached, meta=meta,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

__all__ = [
    "BenchRun",
    "TrainingRun",
    "ensure_bench_run",
    "ensure_training_run",
    "estimate_cost",
    "experiment_dir",
    "set_experiment",
    "is_dry_run",
    "dry_run_records",
    "dry_run_total_usd",
]
