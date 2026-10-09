"""CoreWeave staging and training supervisor with durable object storage."""
from __future__ import annotations

import argparse
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from helico.datasets import Hub, checked_download, validate_lock, stage_lock, write_json, file_sha256


def filesystem(*, durable=False):
    import fsspec
    if durable:
        # Write checkpoints directly to object storage. The node-local LOTA
        # endpoint is useful for reads but stalled two multipart cache uploads.
        options = json.loads(os.environ.get("FSSPEC_S3", "{}"))
        options["endpoint_url"] = "https://cwobject.com"
        options.setdefault("client_kwargs", {}).pop("endpoint_url", None)
        return fsspec.filesystem("s3", skip_instance_cache=True, **options)
    return fsspec.filesystem("s3")


def mirror(config):
    """Cache exact Hub shard bytes, keyed by SHA256, using a CPU-only job."""
    lock = json.loads(Path(config["data_lock"]).read_text())
    validate_lock(lock)
    fs = filesystem(durable=True)
    hub = Hub(Path("/tmp/helico/hub"))
    sources = {s["revision"]: s for d in lock["datasets"] for s in d["manifest"]["sources"]}
    def copy(item):
        source, entry = item
        dest = config["mirror_uri"] + "/" + entry["sha256"]
        marker = dest + ".json"
        if fs.exists(marker):
            with fs.open(marker) as f:
                if json.load(f) != entry:
                    raise ValueError("Mirror manifest mismatch")
            return
        started = time.monotonic()
        path = checked_download(hub, source["repo_id"], source["revision"], entry)
        fs.put_file(str(path), dest)
        if fs.size(dest) != entry["size"]:
            raise ValueError("Mirror upload size mismatch")
        with fs.open(marker, "w") as f:
            json.dump(entry, f)
        print(json.dumps({"mirrored": entry["path"], "bytes": entry["size"],
                          "seconds": time.monotonic() - started}), flush=True)
    items = [(s, e) for s in sources.values() for e in s["files"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy, items))
    with fs.open(config["mirror_uri"] + "/complete.json", "w") as f:
        json.dump({"lock_sha256": lock["lock_sha256"], "shards": len(items)}, f)
    print("MIRROR_COMPLETE", flush=True)


class MirroredHub(Hub):
    def __init__(self, lock, mirror_uri, cache_dir):
        super().__init__(cache_dir / "hub")
        self.mirror_uri, self.local = mirror_uri, cache_dir / "mirrored-shards"
        self.local.mkdir(parents=True, exist_ok=True)
        self.resolved = {}
        self.entries = {(s["repo_id"], s["revision"], e["path"]): e
                        for d in lock["datasets"] for s in d["manifest"]["sources"] for e in s["files"]}

    def download(self, repo_id, revision, filename):
        key = (repo_id, revision, filename)
        if key in self.resolved:
            return self.resolved[key]
        entry = self.entries.get(key)
        if entry is None:
            return super().download(repo_id, revision, filename)
        dest = self.local / entry["sha256"]
        if dest.exists():
            return dest
        fs = filesystem()
        uri = self.mirror_uri + "/" + entry["sha256"]
        if fs.exists(uri + ".json"):
            print(json.dumps({"staging": filename, "source": "CoreWeave SHA256 cache"}), flush=True)
            pending = dest.with_suffix(".pending")
            fs.get_file(uri, str(pending))
            pending.replace(dest)
            return dest  # stage_lock verifies its SHA256 before extraction
        print(json.dumps({"staging": filename, "source": "pinned Hugging Face revision"}), flush=True)
        path = super().download(repo_id, revision, filename)
        # A mirror may finish while prefetch is running. Reuse already verified
        # Hub-cache bytes rather than downloading the same shard a second time.
        self.resolved[key] = path
        return path


def fetch(url, path):
    if not path.exists():
        pending = path.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=180) as src, pending.open("wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        pending.replace(path)
    return path


def train(config, config_path):
    scratch = Path("/tmp/helico"); scratch.mkdir(exist_ok=True)
    fs = filesystem()
    lock = json.loads(Path(config["data_lock"]).read_text())
    validate_lock(lock)
    for local, remote in [(Path(config["data_lock"]), "data.lock.json"), (config_path, "training.json")]:
        fs.put_file(str(local), config["output_uri"] + "/" + remote)
    # Install source by immutable SHA without dependency mutation or monkey patches.
    sha = config["upstream_sha"]
    archive = fetch(f"https://codeload.github.com/bytedance/Protenix/tar.gz/{sha}", scratch / f"{sha}.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall(scratch, filter="data")
    upstream = scratch / f"Protenix-{sha}"
    env = {**os.environ, "PYTHONPATH": str(upstream) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    (scratch / "environment.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    fs.put_file(str(scratch / "environment.txt"), config["output_uri"] + "/environment.txt")
    checkpoint = fetch("https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt",
                       scratch / "protenix_base_default_v1.0.0.pt")
    provenance = {"initial_checkpoint_sha256": file_sha256(checkpoint), "upstream_sha": sha,
                  "code_sha": os.environ["HELICO_CODE_SHA"], "data_lock_sha256": lock["lock_sha256"]}
    with fs.open(config["output_uri"] + "/provenance.json", "w") as f:
        json.dump(provenance, f)
    data_cache = Path(config.get("data_cache_dir", str(scratch / "data")))
    hub = MirroredHub(lock, config["mirror_uri"], data_cache)
    # Warm the downloads concurrently, then use the existing verified extraction
    # path and its cross-split checks. Metadata is checked before bulk staging.
    stage_lock(lock, data_cache, metadata_only=True, hub=hub)
    entries = list(hub.entries)
    with ThreadPoolExecutor(max_workers=config.get("download_workers", 4)) as pool:
        list(pool.map(lambda key: hub.download(*key), entries))
    bundle = stage_lock(lock, data_cache, hub=hub, source_workers=config.get("extract_workers", 1))
    write_json(scratch / "data.paths.json", bundle)
    # A resumed job restores the last fully uploaded optimizer/EMA snapshot.
    resume = []
    latest = config["output_uri"] + "/latest.json"
    if fs.exists(latest):
        with fs.open(latest) as f:
            state = json.load(f)
        if state["data_lock_sha256"] != lock["lock_sha256"]:
            raise ValueError("Remote run data lock changed")
        local = scratch / "resume.pt"
        fs.get_file(state["checkpoint"], str(local)); resume = ["--resume", str(local)]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", f"--nproc-per-node={config['gpus']}", "-m", "helico.train_contacts",
        "--config", str(config_path), "--bundle", str(scratch / "data.paths.json"),
        "--checkpoint", str(checkpoint), "--output", str(scratch / "run"), *resume]
    print(json.dumps({"training_command": command, **provenance}), flush=True)
    subprocess.run(command, check=True, env=env)
    fs.put_file(str(scratch / "run/result.json"), config["output_uri"] + "/result.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["mirror", "train"])
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "mirror":
        mirror(config)
    else:
        train(config, args.config.resolve())


if __name__ == "__main__":
    main()
