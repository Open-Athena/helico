"""Step 8 -- publish every intermediate result, so re-analysis is cheap.

The expensive things this experiment produces are the **predicted structures**
and the **per-target scores**. Neither is recoverable from a summary table, and
re-deriving either costs what the original run cost. So both are published,
together with enough metadata to say exactly how each number was produced:
model, checkpoint step, sampling settings, wall time and the hardware it ran on.

Published to ``hf://buckets/timodonnell/helico-experiments/exp14_foldbench_held_out_monomers/``:

``manifest.json``
    The index: one entry per method with its sampling spec, timing and
    hardware, plus a file inventory with sizes and digests.
``targets.csv``, ``index_map_report.csv``, ``arms/*.json``
    The inputs -- the 333 units, the prompt-to-token map, and the contact set
    each conditioned arm was given.
``scores/``
    ``per_target.csv`` (every arm x target: lDDT, TM-score, GDT-TS, RMSD,
    status, contacts given) and the derived tables. This is what a re-plot
    needs, and it is small enough to fetch in seconds.
``timings/``, ``runs/``
    Per-target wall time and the per-run manifests.
``structures/``
    One tarball per arm (Helico, gzipped PDB per target) and per mode
    (Protenix-v2, the full prediction tree: every diffusion sample and its
    confidence JSON, not just the one that was scored).

Re-running one method and re-plotting is then: run that arm, `--fetch` the
rest, `analyze.py`.

    uv run python publish_artifacts.py --dry-run   # stage and report sizes
    uv run python publish_artifacts.py             # ... and upload
    uv run python publish_artifacts.py --fetch     # pull scores back down
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

BUCKET = "timodonnell/helico-experiments"
PREFIX = "exp14_foldbench_held_out_monomers"
URI = f"hf://buckets/{BUCKET}/{PREFIX}"

STAGE = U.CACHE / "publish"
BYCLASS = U.CACHE / "byclass"
PROTENIX = U.CACHE / "protenix_v2"

#: Small enough that `--fetch` is instant; this is what re-plotting needs.
SCORE_TABLES = (
    "per_target.csv", "headline.csv", "paired_deltas.csv", "val_vs_test.csv",
    "strata.csv", "analysis_summary.json", "marinfold_arm_accuracy.csv",
    "v2_arm_accuracy.csv", "protenix_v2_baseline.csv", "index_map_report.csv",
    "eval_set_report.json", "exp245_inputs.json", "marinfold_inputs.json",
)


def tar_directory(source: Path, dest: Path, arcname: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(source, arcname=arcname)
    return dest.stat().st_size


def helico_runs() -> dict[str, dict]:
    """Per-arm metadata, read from what the runs actually recorded."""
    runs = {}
    for cache in sorted(BYCLASS.glob("*/")):
        name = cache.name
        if name.startswith("smoke-") or not (cache / "results.csv").exists():
            continue
        meta_path, manifest_path = cache / "meta.json", cache / "run_manifest.json"
        entry = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if manifest_path.exists():
            entry["run"] = json.loads(manifest_path.read_text())
        with (cache / "results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        entry["n_targets"] = len(rows)
        entry["n_ok"] = sum(1 for r in rows if r["status"] == "ok")
        entry["n_structures"] = len(list((cache / "predictions").glob("*.pdb.gz"))) \
            if (cache / "predictions").is_dir() else 0
        runs[name] = entry
    return runs


def protenix_runs() -> dict[str, dict]:
    runs = {}
    for mode in ("single_seq", "msa"):
        root = PROTENIX / mode
        if not root.exists():
            continue
        entry = {"predictor": "protenix-v2", "mode": mode}
        manifest = PROTENIX / f"{mode}.manifest.json"
        if manifest.exists():
            entry.update(json.loads(manifest.read_text()))
        else:
            # The first runs predate the instrumentation. Record what the output
            # tree still proves rather than leaving the fields blank or, worse,
            # guessing: the seeds and per-seed sample count are observable.
            seeds = sorted({p.name for p in root.rglob("seed_*") if p.is_dir()})
            samples = sorted({p.name.rsplit("_sample_", 1)[-1].removesuffix(".cif")
                              for p in root.rglob("*_sample_*.cif")})
            entry.update({
                "seeds": ",".join(s.removeprefix("seed_") for s in seeds),
                "n_samples_per_seed": len(samples),
                "sampling_params": "protenix-v2 built-in defaults (not overridden)",
                "timing_seconds": None,
                "hardware": None,
                "note": ("run before per-target timing was instrumented; the "
                         "sampling spec here is read off the output tree"),
            })
        entry["n_targets"] = len({p.parent.parent.parent.name
                                  for p in root.rglob("*_sample_*.cif")})
        runs[mode] = entry
    return runs


def stage(dry_run: bool) -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "scores").mkdir(parents=True)
    (STAGE / "arms").mkdir(parents=True)
    (STAGE / "timings").mkdir(parents=True)
    (STAGE / "runs").mkdir(parents=True)
    (STAGE / "structures/helico").mkdir(parents=True)
    (STAGE / "structures/protenix_v2").mkdir(parents=True)

    shutil.copyfile(U.DATA / "targets.csv", STAGE / "targets.csv")
    for name in SCORE_TABLES:
        source = U.DATA / name
        if source.exists():
            shutil.copyfile(source, STAGE / "scores" / name)
    for arm in sorted(U.ARMS.glob("*.json")):
        shutil.copyfile(arm, STAGE / "arms" / arm.name)

    inventory = {}
    for name, cache in ((c.name, c) for c in sorted(BYCLASS.glob("*/"))):
        if name.startswith("smoke-") or not (cache / "results.csv").exists():
            continue
        shutil.copyfile(cache / "results.csv", STAGE / "scores" / f"arm_{name}.csv")
        for source, dest in ((cache / "timings.csv", STAGE / "timings" / f"{name}.csv"),
                             (cache / "run_manifest.json",
                              STAGE / "runs" / f"{name}.json")):
            if source.exists():
                shutil.copyfile(source, dest)
        predictions = cache / "predictions"
        if predictions.is_dir() and any(predictions.iterdir()):
            size = tar_directory(predictions,
                                 STAGE / "structures/helico" / f"{name}.tar.gz", name)
            inventory[f"structures/helico/{name}.tar.gz"] = size

    for mode in ("single_seq", "msa"):
        root = PROTENIX / mode
        if not root.exists():
            continue
        # The a3m alignments are inputs, not results: FoldBench ships all but
        # 16 of them and those 16 are published under `msa/`. Excluding them
        # keeps the tarball to the structures and their confidence files.
        dest = STAGE / "structures/protenix_v2" / f"{mode}.tar.gz"
        with tarfile.open(dest, "w:gz") as tar:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix in (".cif", ".json"):
                    tar.add(path, arcname=str(path.relative_to(root)))
        inventory[f"structures/protenix_v2/{mode}.tar.gz"] = dest.stat().st_size
        timings = PROTENIX / f"{mode}.timings.csv"
        if timings.exists():
            shutil.copyfile(timings, STAGE / "timings" / f"protenix_{mode}.csv")

    manifest = {
        "experiment": PREFIX,
        "issue": "https://github.com/Open-Athena/helico/issues/14",
        "eval_sets": "MarinFold exp245 (eval-val 97 / eval-test 217 / eval-denovo 19)",
        "upstream": json.loads((U.DATA / "exp245_inputs.json").read_text()),
        "methods": {"helico": helico_runs(), "protenix_v2": protenix_runs()},
        "files": {},
    }
    for path in sorted(STAGE.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(STAGE))
            manifest["files"][relative] = {
                "bytes": path.stat().st_size,
                "sha256": U.sha256(path) if path.stat().st_size < 100 << 20 else None,
            }
    manifest["files"].update(
        {k: {"bytes": v, "sha256": None} for k, v in inventory.items()
         if k not in manifest["files"]})
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (STAGE / "README.md").write_text(bucket_readme(manifest))

    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {sum(1 for p in STAGE.rglob('*') if p.is_file())} files, "
          f"{total / 1e6:.0f} MB -> {STAGE}")
    if dry_run:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                print(f"  {path.stat().st_size / 1e6:8.2f} MB  "
                      f"{path.relative_to(STAGE)}")
    return STAGE


def bucket_readme(manifest: dict) -> str:
    helico = manifest["methods"]["helico"]
    lines = [
        f"# {PREFIX}",
        "",
        "Predicted structures, per-target scores and run metadata for Helico on",
        "MarinFold exp245's held-out FoldBench monomer sets. Produced by",
        "`experiments/exp14_foldbench_held_out_monomers/` in Open-Athena/helico",
        f"({manifest['issue']}).",
        "",
        "## What is here",
        "",
        "| path | what |",
        "|---|---|",
        "| `manifest.json` | run metadata per method: sampling spec, timing, hardware, file digests |",
        "| `targets.csv` | the 333 scored units with their eval set, viral/designed flags and homology stratum |",
        "| `arms/*.json` | the contact set each conditioned arm was given, in Helico token indices |",
        "| `scores/per_target.csv` | every arm x target: lDDT, TM-score, GDT-TS, RMSD |",
        "| `scores/arm_*.csv` | the raw per-arm result tables |",
        "| `timings/*.csv` | per-target wall time and the GPU it ran on |",
        "| `runs/*.json` | per-run manifests as recorded by the workers |",
        "| `structures/helico/<arm>.tar.gz` | one gzipped PDB per target |",
        "| `structures/protenix_v2/<mode>.tar.gz` | every diffusion sample and its confidence JSON |",
        "",
        "## Sampling settings",
        "",
        "| method | trunk recycles | trunk runs | diffusion samples | MSA |",
        "|---|---:|---:|---:|---|",
    ]
    for name, entry in sorted(helico.items()):
        lines.append(
            f"| Helico `{name}` | {entry.get('n_trunk_recycles', '?')} | "
            f"{entry.get('n_trunk_runs', '?')} | "
            f"{entry.get('n_diffusion_samples', '?')} | no |")
    for mode, entry in sorted(manifest["methods"]["protenix_v2"].items()):
        lines.append(
            f"| Protenix-v2 `{mode}` | built-in default | "
            f"{len(str(entry.get('seeds', '')).split(',')) if entry.get('seeds') else '?'} | "
            f"{entry.get('n_samples_per_seed', '?')} | "
            f"{'yes' if mode == 'msa' else 'no'} |")
    lines += [
        "",
        "## Re-analysing without re-running",
        "",
        "```bash",
        "uv run python publish_artifacts.py --fetch   # scores + metadata only",
        "uv run python analyze.py                     # rebuilds every table",
        "uv run python plot_results.py",
        "```",
        "",
        "Re-running one method and comparing against the rest is the same flow with",
        "that one arm re-run first.",
        "",
    ]
    return "\n".join(lines)


def fetch() -> int:
    """Pull the small tables back down, into the layout analyze.py reads."""
    BYCLASS.mkdir(parents=True, exist_ok=True)
    U.DATA.mkdir(parents=True, exist_ok=True)
    binary = U.hf_binary()
    with __import__("tempfile").TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for remote, local in ((f"{URI}/scores", scratch / "scores"),
                              (f"{URI}/manifest.json", scratch / "manifest.json")):
            subprocess.run([binary, "buckets", "cp", "-r", remote, str(local)],
                           check=True)
        for path in sorted((scratch / "scores").glob("arm_*.csv")):
            arm = path.name.removeprefix("arm_").removesuffix(".csv")
            (BYCLASS / arm).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, BYCLASS / arm / "results.csv")
        for path in sorted((scratch / "scores").glob("*")):
            if not path.name.startswith("arm_"):
                shutil.copyfile(path, U.DATA / path.name)
        shutil.copyfile(scratch / "manifest.json", U.DATA / "published_manifest.json")
    print(f"fetched scores and metadata from {URI}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="stage and list, but do not upload")
    parser.add_argument("--fetch", action="store_true",
                        help="download the scores and metadata instead")
    args = parser.parse_args()

    if args.fetch:
        return fetch()

    stage_dir = stage(args.dry_run)
    if args.dry_run:
        return 0
    subprocess.run([U.hf_binary(), "buckets", "sync", str(stage_dir), URI], check=True)
    print(f"published -> https://huggingface.co/buckets/{BUCKET}/{PREFIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
