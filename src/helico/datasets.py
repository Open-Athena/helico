"""Versioned Hugging Face datasets, mixture locks, and reproducible staging.

A dataset owns its split indices and preprocessing contract. Large source files
can live in another pinned Hub dataset. A run owns an immutable lock containing
all manifests, revisions, checksums and mixture weights; local paths never enter
its identity. This module deliberately does not import torch or unpickle data.
"""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import threading


SCHEMA_VERSION = 1
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    """Atomically publish JSON without leaving a truncated lock on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError(f"Unsafe or noncanonical relative path: {value!r}")
    return value


def _repo(value: str) -> str:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        raise ValueError(f"Invalid Hub repository ID: {value!r}")
    return value


def _files(files: list[dict]) -> None:
    seen = set()
    for entry in files:
        name = relative_path(entry["path"])
        if name in seen:
            raise ValueError(f"Duplicate file: {name}")
        seen.add(name)
        if not SHA256.fullmatch(entry["sha256"]):
            raise ValueError(f"Missing SHA256 for {name}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError(f"Invalid size for {name}")


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest schema")
    if not manifest.get("format") or not manifest.get("splits"):
        raise ValueError("Dataset must specify a format and splits")
    _files(manifest["files"])
    listed = {f["path"] for f in manifest["files"]}
    for split in manifest["splits"].values():
        if relative_path(split["index"]) not in listed:
            raise ValueError("Every split index must have a file checksum")
        if "pdb_ids" in split and relative_path(split["pdb_ids"]) not in listed:
            raise ValueError("Every split membership list must have a file checksum")
    excluded = manifest.get("split_policy", {}).get("excluded_pdb_ids")
    if excluded and relative_path(excluded) not in listed:
        raise ValueError("The exclusion list must have a file checksum")
    for source in manifest.get("sources", []):
        _repo(source["repo_id"])
        if not COMMIT.fullmatch(source["revision"]):
            raise ValueError("Source dependencies must use full immutable commit IDs")
        if source["format"] not in {"files", "tar"}:
            raise ValueError("Source format must be files or tar")
        _files(source["files"])
        for prefix in source.get("extract_prefixes", []):
            relative_path(prefix.rstrip("/"))
    # FoldBench is an evaluation-only split; never accept it as a train split.
    if "train" in manifest["splits"] and manifest["splits"]["train"].get("role") != "train":
        raise ValueError("The train split must declare role=train")


class Hub:
    """Small dependency-injection boundary for offline tests and private repos."""

    def __init__(self, cache_dir: Path | None = None):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.cache_dir = cache_dir

    def revision(self, repo_id, revision):
        return self.api.dataset_info(repo_id, revision=revision).sha

    def download(self, repo_id, revision, filename):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id, filename, repo_type="dataset",
                                    revision=revision, cache_dir=self.cache_dir))


def resolve_recipe(recipe: dict, hub=None) -> dict:
    """Resolve explicitly requested refs once; runs consume the resulting lock."""
    if recipe.get("schema_version") != SCHEMA_VERSION or not recipe.get("datasets"):
        raise ValueError("A recipe needs schema_version=1 and nonempty datasets")
    hub = hub or Hub()
    datasets = []
    seen = set()
    for item in recipe["datasets"]:
        repo = _repo(item["repo_id"])
        requested = item["revision"]  # No implicit main/latest.
        revision = hub.revision(repo, requested)
        if not COMMIT.fullmatch(revision):
            raise ValueError(f"Hub did not return an immutable revision for {repo}")
        filename = relative_path(item.get("manifest", "dataset.json"))
        manifest_path = hub.download(repo, revision, filename)
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest)
        train_split = item.get("train_split", "train")
        validation_split = item.get("validation_split", "validation")
        for split, role in [(train_split, "train"), (validation_split, "validation")]:
            if split is not None and manifest["splits"][split].get("role") != role:
                raise ValueError(f"{repo}/{split} cannot be used for {role}")
        weight = float(item.get("weight", 1))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Mixture weights must be finite and positive")
        key = (repo, revision, train_split)
        if key in seen:
            raise ValueError(f"Duplicate dataset selection: {key}")
        seen.add(key)
        datasets.append(dict(repo_id=repo, revision=revision, manifest_path=filename,
                             manifest_sha256=file_sha256(manifest_path),
                             manifest_size=manifest_path.stat().st_size, manifest=manifest,
                             weight=weight, train_split=train_split, validation_split=validation_split))
    payload = dict(schema_version=SCHEMA_VERSION, name=recipe.get("name", "unnamed"), datasets=datasets)
    return {**payload, "lock_sha256": digest(payload)}


def validate_lock(lock: dict) -> None:
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if lock.get("schema_version") != SCHEMA_VERSION or digest(payload) != lock.get("lock_sha256"):
        raise ValueError("Data lock checksum/schema mismatch")
    if not lock.get("datasets"):
        raise ValueError("Empty data lock")
    for item in lock["datasets"]:
        _repo(item["repo_id"])
        if not COMMIT.fullmatch(item["revision"]):
            raise ValueError("Locked dataset revision must be a full commit ID")
        if not math.isfinite(item["weight"]) or item["weight"] <= 0:
            raise ValueError("Invalid locked mixture weight")
        relative_path(item["manifest_path"])
        validate_manifest(item["manifest"])
        for key, role in [("train_split", "train"), ("validation_split", "validation")]:
            split = item[key]
            if split is not None and item["manifest"]["splits"][split].get("role") != role:
                raise ValueError(f"Split {split} has the wrong role")


def checked_download(hub, repo: str, revision: str, entry: dict) -> Path:
    path = hub.download(repo, revision, entry["path"])
    if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Checksum/size mismatch: {repo}@{revision}/{entry['path']}")
    return path


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def extract_archive(archive: Path, destination: Path, prefixes: list[str], *, claims=None) -> None:
    """Extract only regular files under selected prefixes, never links/devices."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            if member.isdir():
                continue
            relative_path(name)
            if not member.isfile():
                raise ValueError(f"Non-regular archive member: {member.name}")
            if prefixes and not any(name.startswith(p.rstrip("/") + "/") for p in prefixes):
                continue
            if claims is not None:
                seen, lock = claims
                with lock:
                    if name in seen:
                        raise ValueError(f"Duplicate archive path during parallel extraction: {name}")
                    seen.add(name)
            target = destination / name
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Archive path escapes destination: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".extract-")
            try:
                with os.fdopen(fd, "wb") as out, tar.extractfile(member) as src:
                    shutil.copyfileobj(src, out, 8 * 1024 * 1024)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)


def stage_source(source: dict, cache_dir: Path, hub, *, workers=1) -> dict:
    source_id = digest(source)
    dest = cache_dir / "sources" / source_id / "data"
    marker = dest.parent / "complete.json"
    with file_lock(dest.parent / "prepare.lock"):
        if marker.exists() and (not dest.is_dir() or
                json.loads(marker.read_text()).get("source_sha256") != source_id):
            raise ValueError(f"Invalid prepared-source cache: {dest}")
        if not marker.exists():
            claims = (set(), threading.Lock()) if workers > 1 else None
            def prepare(entry):
                path = checked_download(hub, source["repo_id"], source["revision"], entry)
                if source["format"] == "tar":
                    extract_archive(path, dest, source.get("extract_prefixes", []), claims=claims)
                else:
                    target = dest / entry["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(prepare, source["files"]))
            write_json(marker, {"source_sha256": source_id})
    return dict(repo_id=source["repo_id"], revision=source["revision"],
                root=str(dest), source_sha256=source_id)


def stage_lock(lock: dict, cache_dir: Path, *, metadata_only=False, hub=None, source_workers=1) -> dict:
    """Stage split metadata and (optionally) all pinned source assets.

    Metadata-only is for inspection/CI; its result cannot be used for a run.
    Sources are shared by revision, even across different dataset mixtures.
    """
    validate_lock(lock)
    cache_dir = Path(cache_dir).resolve()
    hub = hub or Hub(cache_dir / "hub")
    prepared = []
    for item in lock["datasets"]:
        repo, revision, manifest = item["repo_id"], item["revision"], item["manifest"]
        downloaded_manifest = checked_download(hub, repo, revision, dict(
            path=item["manifest_path"], sha256=item["manifest_sha256"],
            size=item["manifest_size"]))
        if json.loads(downloaded_manifest.read_text()) != manifest:
            raise ValueError("Locked manifest differs from the Hub manifest")
        files = {entry["path"]: str(checked_download(hub, repo, revision, entry))
                 for entry in manifest["files"]}
        prepared.append(dict(repo_id=repo, revision=revision, weight=item["weight"],
                             train_split=item["train_split"], validation_split=item["validation_split"],
                             format=manifest["format"], files=files, sources=[],
                             splits=manifest["splits"]))
    bundle = dict(schema_version=SCHEMA_VERSION, lock_sha256=lock["lock_sha256"],
                  metadata_only=metadata_only, datasets=prepared)
    validate_membership(lock, bundle)
    # Check all memberships before any multi-terabyte source staging.
    if not metadata_only:
        for item, staged in zip(lock["datasets"], prepared, strict=True):
            staged["sources"] = [stage_source(source, cache_dir, hub, workers=source_workers)
                                  for source in item["manifest"].get("sources", [])]
    return bundle


def validate_membership(lock: dict, bundle: dict) -> None:
    """Detect structural split leakage across a mixture, not just within a source."""
    train, held_out = set(), set()
    for item, staged in zip(lock["datasets"], bundle["datasets"], strict=True):
        manifest = item["manifest"]
        for name, split in manifest["splits"].items():
            membership = split.get("pdb_ids")
            if membership is None:
                continue
            ids = {pid.strip().lower() for pid in Path(staged["files"][membership]).read_text().splitlines() if pid.strip()}
            if name == item["train_split"]:
                train.update(ids)
            elif split["role"] in {"validation", "test"}:
                held_out.update(ids)
        excluded = manifest.get("split_policy", {}).get("excluded_pdb_ids")
        if excluded:
            held_out.update(Path(staged["files"][excluded]).read_text().lower().split())
    overlap = train & held_out
    if overlap:
        raise ValueError(f"Training/held-out PDB overlap across dataset mixture: {sorted(overlap)[:10]}")


def record_run_data(lock: dict, run_dir: Path) -> Path:
    """Persist provenance before model initialization; refuse changed-data resume."""
    validate_lock(lock)
    run_dir = Path(run_dir)
    path = run_dir / "data.lock.json"
    with file_lock(run_dir / ".data.lock"):
        if path.exists():
            old = json.loads(path.read_text())
            validate_lock(old)
            if old["lock_sha256"] != lock["lock_sha256"]:
                raise ValueError("Run already references different data; use a new run directory")
        else:
            write_json(path, lock)
    return path


def prepare_training_data(recipe: dict, run_dir: Path, cache_dir: Path, *, hub=None) -> dict:
    """One entry point for trainers: recipe -> frozen run provenance + inputs."""
    lock = resolve_recipe(recipe, hub=hub)
    record_run_data(lock, run_dir)
    bundle = stage_lock(lock, cache_dir, hub=hub)
    write_json(Path(run_dir) / "data.paths.json", bundle)
    return bundle


def protenix_config(lock: dict, bundle: dict, train_crop_size: int = 384) -> dict:
    """Translate a staged mixture into upstream Protenix data-config overrides.

    Uses the upstream featurizer, paired MSAs and cluster-weighted sampler.
    The returned dict is merged under the upstream config's `data` key. It is
    also the input contract for a Helico training adapter using that pipeline.
    """
    validate_lock(lock)
    if bundle["metadata_only"] or bundle["lock_sha256"] != lock["lock_sha256"]:
        raise ValueError("Training requires a fully staged, matching data bundle")
    configs = {"train_sets": [], "test_sets": [],
               "train_sampler": {"sampler_type": "weighted", "train_sample_weights": []}}
    roots = set()
    for number, (locked, staged) in enumerate(zip(lock["datasets"], bundle["datasets"], strict=True)):
        manifest = locked["manifest"]
        if staged["format"] != "protenix-pdb-v1" or len(staged["sources"]) != 1:
            raise ValueError(f"No Protenix adapter for dataset format {staged['format']}")
        root = Path(staged["sources"][0]["root"])
        roots.add(str(root))
        for key, role in [("train_split", "train"), ("validation_split", "validation")]:
            split_name = locked[key]
            if split_name is None:
                continue
            split = manifest["splits"][split_name]
            name = f"helico_{number}_{split_name}"
            training = role == "train"
            configs["train_sets" if training else "test_sets"].append(name)
            if training:
                configs["train_sampler"]["train_sample_weights"].append(locked["weight"])
            configs[name] = {
                "base_info": {"mmcif_dir": str(root / "mmcif"),
                              "bioassembly_dict_dir": str(root / split["bioassembly_dir"]),
                              "indices_fpath": staged["files"][split["index"]],
                              "pdb_list": staged["files"].get(split.get("pdb_ids", ""), ""),
                              "max_n_token": -1, "random_sample_if_failed": False,
                              "use_reference_chains_only": False,
                              "sort_by_n_token": False, "group_by_pdb_id": not training,
                              "find_eval_chain_interface": not training, "exclusion": {}},
                "sampler_configs": manifest["sampler"] if training else {"sampler_type": "uniform"},
                "cropping_configs": {"method_weights": [0.2, 0.4, 0.4] if training else [0., 0., 1.],
                                     "crop_size": train_crop_size if training else -1},
                "sample_weight": locked["weight"], "limits": -1,
                "lig_atom_rename": False, "shuffle_mols": training, "shuffle_sym_ids": training,
                "constraint": {"enable": False, "fix_seed": False},
            }
    # Shared MSA/template/CCD lookup paths are global in upstream Protenix.
    # Never silently point a heterogeneous mixture at the wrong source tree.
    if len(roots) != 1:
        raise ValueError("The Protenix adapter currently requires one shared asset source")
    root = Path(next(iter(roots)))
    configs["msa"] = {"enable_prot_msa": True, "enable_rna_msa": True,
                      "prot_seq_or_filename_to_msadir_jsons": [str(root / "common/seq_to_pdb_index.json")],
                      "prot_msadir_raw_paths": [str(root / "mmcif_msa_template")],
                      "rna_seq_or_filename_to_msadir_jsons": [str(root / "rna_msa/rna_sequence_to_pdb_chains.json")],
                      "rna_msadir_raw_paths": [str(root / "rna_msa/msas")]}
    configs["template"] = {
        "enable_prot_template": True,
        "prot_template_mmcif_dir": str(root / "mmcif"),
        "prot_template_raw_paths": [str(root / "mmcif_msa_template")],
        "prot_seq_or_filename_to_templatedir_jsons": [str(root / "common/seq_to_pdb_index.json")],
        "release_dates_path": str(root / "common/release_date_cache.json"),
        "obsolete_pdbs_path": str(root / "common/obsolete_to_successor.json"),
    }
    for key, filename in {
        "ccd_components_file": "components.cif",
        "ccd_components_rdkit_mol_file": "components.cif.rdkit_mol.pkl",
        "obsolete_release_data_csv": "obsolete_release_date.csv",
        "pdb_cluster_file": "clusters-by-entity-40.txt",
    }.items():
        configs[key] = str(root / "common" / filename)
    return {"PROTENIX_ROOT_DIR": str(root), "data": configs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    resolve = subs.add_parser("lock", help="Resolve a dataset recipe to immutable revisions")
    resolve.add_argument("recipe", type=Path)
    resolve.add_argument("--output", type=Path, required=True)
    prepare = subs.add_parser("prepare", help="Verify/stage a lock; large assets are shared in the cache")
    prepare.add_argument("lock", type=Path)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--metadata-only", action="store_true")
    init = subs.add_parser("init-run", help="Freeze a recipe and stage its inputs for a new training run")
    init.add_argument("recipe", type=Path)
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--cache-dir", type=Path, required=True)
    config = subs.add_parser("protenix-config", help="Generate upstream data configuration from staged inputs")
    config.add_argument("lock", type=Path)
    config.add_argument("bundle", type=Path)
    config.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        result = resolve_recipe(json.loads(args.recipe.read_text()))
    elif args.command == "prepare":
        result = stage_lock(json.loads(args.lock.read_text()), args.cache_dir, metadata_only=args.metadata_only)
    elif args.command == "init-run":
        prepare_training_data(json.loads(args.recipe.read_text()), args.run_dir, args.cache_dir)
        print(args.run_dir / "data.lock.json")
        return
    else:
        result = protenix_config(json.loads(args.lock.read_text()), json.loads(args.bundle.read_text()))
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
