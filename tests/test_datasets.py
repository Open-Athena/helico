"""Data integrity, split roles, archive safety, and changed-data resume tests."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

import pytest

from helico.datasets import (
    digest, extract_archive, file_sha256, protenix_config, record_run_data,
    resolve_recipe, stage_lock, validate_lock, validate_manifest, write_json,
)


class LocalHub:
    def __init__(self, root):
        self.root = root
        self.commit = "a" * 40
        self.calls = []

    def revision(self, repo, revision):
        return self.commit if revision == "main" else revision

    def download(self, repo, revision, filename):
        self.calls.append((repo, revision, filename))
        return self.root / repo / revision / filename


def entry(path, relative):
    return {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}


@pytest.fixture
def setup(tmp_path):
    hub = LocalHub(tmp_path / "hub")
    data = hub.root / "user/pdb" / hub.commit
    source = hub.root / "user/source" / ("b" * 40)
    data.mkdir(parents=True)
    source.mkdir(parents=True)
    (data / "train.csv").write_text("pdb_id\n1abc\n")
    (data / "validation.csv").write_text("pdb_id\n2abc\n")
    with tarfile.open(source / "part.tar", "w") as tar:
        for name, content in [("structures/1abc.txt", b"coordinates"), ("unused/file", b"skip")]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    manifest = {"schema_version": 1, "format": "protenix-pdb-v1",
                "files": [entry(data / p, p) for p in ["train.csv", "validation.csv"]],
                "splits": {"train": {"role": "train", "index": "train.csv", "bioassembly_dir": "structures"},
                           "validation": {"role": "validation", "index": "validation.csv", "bioassembly_dir": "structures"}},
                "sampler": {"sampler_type": "weighted"},
                "sources": [{"repo_id": "user/source", "revision": "b" * 40, "format": "tar",
                             "extract_prefixes": ["structures"], "files": [entry(source / "part.tar", "part.tar")]}]}
    write_json(data / "dataset.json", manifest)
    recipe = {"schema_version": 1, "name": "test", "datasets": [
        {"repo_id": "user/pdb", "revision": "main", "weight": 1.0}]}
    return hub, recipe, manifest


def test_moving_branch_does_not_change_locked_inputs(setup, tmp_path):
    hub, recipe, _ = setup
    lock = resolve_recipe(recipe, hub)
    hub.commit = "c" * 40
    bundle = stage_lock(lock, tmp_path / "cache", metadata_only=True, hub=hub)
    assert bundle["datasets"][0]["revision"] == "a" * 40
    assert lock["datasets"][0]["manifest"]["sources"][0]["revision"] == "b" * 40
    assert not any(call[0] == "user/source" for call in hub.calls)


def test_corrupt_file_rejected_even_when_cached(setup, tmp_path):
    hub, recipe, _ = setup
    lock = resolve_recipe(recipe, hub)
    (hub.root / "user/pdb" / hub.commit / "train.csv").write_text("pdb_id\n9bad\n")
    with pytest.raises(ValueError, match="Checksum/size"):
        stage_lock(lock, tmp_path / "cache", metadata_only=True, hub=hub)


@pytest.mark.parametrize("duplicate", [False, True])
def test_parallel_shards_require_disjoint_paths(setup, tmp_path, duplicate):
    hub, recipe, manifest = setup
    source = hub.root / "user/source" / ("b" * 40)
    with tarfile.open(source / "second.tar", "w") as tar:
        member = tarfile.TarInfo("structures/1abc.txt" if duplicate else "structures/2abc.txt")
        member.size = 5
        tar.addfile(member, io.BytesIO(b"other"))
    manifest["sources"][0]["files"].append(entry(source / "second.tar", "second.tar"))
    write_json(hub.root / "user/pdb" / hub.commit / "dataset.json", manifest)
    lock = resolve_recipe(recipe, hub)
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate archive path"):
            stage_lock(lock, tmp_path / "cache", hub=hub, source_workers=2)
        assert not list((tmp_path / "cache/sources").glob("*/complete.json"))
    else:
        bundle = stage_lock(lock, tmp_path / "cache", hub=hub, source_workers=2)
        root = Path(bundle["datasets"][0]["sources"][0]["root"])
        assert (root / "structures/1abc.txt").read_bytes() == b"coordinates"
        assert (root / "structures/2abc.txt").read_bytes() == b"other"


def test_lock_tampering_rejected(setup):
    hub, recipe, _ = setup
    lock = resolve_recipe(recipe, hub)
    lock["datasets"][0]["weight"] = 2
    with pytest.raises(ValueError, match="checksum"):
        validate_lock(lock)


def test_validation_cannot_be_selected_for_training(setup):
    hub, recipe, _ = setup
    recipe["datasets"][0]["train_split"] = "validation"
    with pytest.raises(ValueError, match="cannot be used for train"):
        resolve_recipe(recipe, hub)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_mixture_weights(setup, weight):
    hub, recipe, _ = setup
    recipe["datasets"][0]["weight"] = weight
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_recipe(recipe, hub)


def test_multiple_datasets_retain_independent_weights(setup, tmp_path):
    hub, recipe, _ = setup
    import shutil
    shutil.copytree(hub.root / "user/pdb", hub.root / "user/extra")
    recipe["datasets"].append({"repo_id": "user/extra", "revision": "a" * 40, "weight": 0.25})
    lock = resolve_recipe(recipe, hub)
    bundle = stage_lock(lock, tmp_path / "cache", hub=hub)
    config = protenix_config(lock, bundle)
    assert config["data"]["train_sampler"]["train_sample_weights"] == [1., .25]
    assert bundle["datasets"][0]["sources"] == bundle["datasets"][1]["sources"]
    source = Path(bundle["datasets"][0]["sources"][0]["root"])
    assert (source / "structures/1abc.txt").read_bytes() == b"coordinates"
    assert not (source / "unused/file").exists()
    assert config["data"]["helico_0_train"]["base_info"]["random_sample_if_failed"] is False


def test_metadata_only_cannot_be_used_for_training(setup, tmp_path):
    hub, recipe, _ = setup
    lock = resolve_recipe(recipe, hub)
    bundle = stage_lock(lock, tmp_path / "cache", metadata_only=True, hub=hub)
    with pytest.raises(ValueError, match="fully staged"):
        protenix_config(lock, bundle)


def test_changed_dataset_resume_refused(setup, tmp_path):
    hub, recipe, _ = setup
    lock = resolve_recipe(recipe, hub)
    run = tmp_path / "run"
    record_run_data(lock, run)
    record_run_data(lock, run)
    recipe["datasets"][0]["weight"] = 2
    changed = resolve_recipe(recipe, hub)
    with pytest.raises(ValueError, match="different data"):
        record_run_data(changed, run)
    assert json.loads((run / "data.lock.json").read_text()) == lock


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/escape", "file"), ("link", "symlink")])
def test_archive_escape_and_symlink_rejected(tmp_path, name, kind):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp"
        tar.addfile(member)
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out", [])
    assert not (tmp_path / "escape").exists()


def test_preexisting_destination_symlink_rejected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (out / "structures").symlink_to(elsewhere, target_is_directory=True)
    archive = tmp_path / "data.tar"
    with tarfile.open(archive, "w") as tar:
        tar.addfile(tarfile.TarInfo("structures/escape"))
    with pytest.raises(ValueError, match="escapes"):
        extract_archive(archive, out, [])
    assert not list(elsewhere.iterdir())


def test_manifest_rejects_unpinned_dependency_and_unlisted_split(setup):
    _, _, manifest = setup
    bad = deepcopy(manifest)
    bad["sources"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        validate_manifest(bad)
    bad = deepcopy(manifest)
    bad["splits"]["train"]["index"] = "not-tracked.csv"
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(bad)


def test_held_out_overlap_detected_across_datasets(setup, tmp_path):
    hub, recipe, manifest = setup
    import shutil
    a = hub.root / "user/pdb" / hub.commit
    (a / "train_ids.txt").write_text("1abc\n")
    manifest["files"].append(entry(a / "train_ids.txt", "train_ids.txt"))
    manifest["splits"]["train"]["pdb_ids"] = "train_ids.txt"
    write_json(a / "dataset.json", manifest)
    b = hub.root / "user/extra" / hub.commit
    shutil.copytree(a, b)
    other = deepcopy(manifest)
    (b / "held_out.txt").write_text("1abc\n")
    other["files"].append(entry(b / "held_out.txt", "held_out.txt"))
    other["split_policy"] = {"excluded_pdb_ids": "held_out.txt"}
    write_json(b / "dataset.json", other)
    recipe["datasets"].append({"repo_id": "user/extra", "revision": hub.commit})
    lock = resolve_recipe(recipe, hub)
    with pytest.raises(ValueError, match="overlap across dataset mixture"):
        stage_lock(lock, tmp_path / "cache", metadata_only=True, hub=hub)


def test_checkpoint_retains_lock_and_refuses_changed_resume(setup, tmp_path):
    import torch
    from helico.train import TrainConfig, load_checkpoint, save_checkpoint
    hub, recipe, _ = setup
    lock = resolve_recipe(recipe, hub)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    config = TrainConfig(data_lock=lock)
    save_checkpoint(model, optimizer, 5, config, path=str(path))
    saved = torch.load(path, weights_only=False)
    assert saved["config"]["data_lock"] == lock
    assert saved["data_lock_sha256"] == lock["lock_sha256"]
    assert load_checkpoint(path, model, expected_data_lock=lock)[0] == 5
    recipe["datasets"][0]["weight"] = 2
    changed = resolve_recipe(recipe, hub)
    with pytest.raises(ValueError, match="training data differ"):
        load_checkpoint(path, model, expected_data_lock=changed)
