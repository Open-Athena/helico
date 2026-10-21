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
import threading
import time
import urllib.request
import uuid

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
        fs.put_file(str(path), dest, chunksize=256 * 1024**2, max_concurrency=4)
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
            fs.get_file(uri, str(pending), chunksize=256 * 1024**2, max_concurrency=4)
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


def run_trainer(command, env, scratch, config, *, diagnostics_fs=None):
    """Tee native failures and preserve diagnostics independently of cluster logs."""
    attempt = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    diagnostics = Path(os.environ.get("IRIS_OUTPUT_DIR", str(scratch / "diagnostics"))) / attempt
    diagnostics.mkdir(parents=True, exist_ok=True)
    remote = config["output_uri"] + "/diagnostics/" + attempt
    env = {**env, "HELICO_DIAGNOSTICS_DIR": str(diagnostics), "PYTHONFAULTHANDLER": "1",
           "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000", "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
           "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(diagnostics / "nccl-trace-")}
    stopped = threading.Event()
    def upload():
        fs = diagnostics_fs if diagnostics_fs is not None else filesystem(durable=True)
        # A single upload thread owns the remote diagnostic files. Never include
        # environment dumps or credential-bearing files from the runtime.
        while True:
            final_pass = stopped.is_set()
            for path in diagnostics.rglob("*"):
                if not path.is_file() or path.suffix == ".pending":
                    continue
                try:
                    fs.put_file(str(path), remote + "/" + str(path.relative_to(diagnostics)))
                except Exception as error:
                    print(json.dumps({"diagnostic_upload_error": type(error).__name__}), flush=True)
            if final_pass:
                return
            stopped.wait(60)
    uploader = threading.Thread(target=upload, daemon=True)
    uploader.start()
    code = -1
    try:
        with (diagnostics / "trainer.log").open("w", buffering=1) as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, env=env) as process:
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                code = process.wait()
    finally:
        (diagnostics / "exit.json").write_text(json.dumps({"returncode": code, "timestamp": time.time()}))
        stopped.set()
        uploader.join(timeout=180)
    if code:
        raise subprocess.CalledProcessError(code, command)


def train(config, config_path):
    scratch = Path("/tmp/helico"); scratch.mkdir(exist_ok=True)
    fs = filesystem(durable=True)
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
    if config.get('overfit'):
        from helico.contact_overfit import stage_features
        bundle = dict(fixed_features=stage_features(config['overfit']['dataset'],
                      lock['lock_sha256'], scratch / 'overfit-features'))
    else:
        data_cache = Path(config.get("data_cache_dir", str(scratch / "data")))
        hub = MirroredHub(lock, config["mirror_uri"], data_cache)
        # Metadata is checked before bulk staging and verified extraction.
        stage_lock(lock, data_cache, metadata_only=True, hub=hub)
        entries = list(hub.entries)
        with ThreadPoolExecutor(max_workers=config.get("download_workers", 4)) as pool:
            list(pool.map(lambda key: hub.download(*key), entries))
        bundle = stage_lock(lock, data_cache, hub=hub, source_workers=config.get("extract_workers", 1))
    write_json(scratch / "data.paths.json", bundle)
    # A resumed job restores the last fully uploaded optimizer/EMA snapshot.
    resume = []
    latest = config["output_uri"] + "/latest.json"
    if not fs.exists(latest) and config.get("resume_uri"):
        latest = config["resume_uri"].rstrip("/") + "/latest.json"
    if fs.exists(latest):
        with fs.open(latest) as f:
            state = json.load(f)
        if state["data_lock_sha256"] != lock["lock_sha256"]:
            raise ValueError("Remote run data lock changed")
        local = scratch / "resume.pt"
        fs.get_file(state["checkpoint"], str(local), chunksize=256 * 1024**2, max_concurrency=4)
        if state.get("sha256") and file_sha256(local) != state["sha256"]:
            raise ValueError("Resume checkpoint checksum mismatch")
        resume = ["--resume", str(local)]
        print(json.dumps({"resume_checkpoint": state}), flush=True)
    elif config.get("continuation"):
        if config.get("resume_uri"):
            raise ValueError("Continuation stage requires its preceding stage checkpoint")
        parent = config["continuation"]
        local = scratch / "continuation-parent.pt"
        fs.get_file(parent["checkpoint_uri"], str(local), chunksize=256 * 1024**2, max_concurrency=4)
        if file_sha256(local) != parent["sha256"]:
            raise ValueError("Continuation parent checkpoint checksum mismatch")
        resume = ["--continue-from", str(local)]
        print(json.dumps({"continuation_parent": parent}), flush=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", f"--nproc-per-node={config['gpus']}", "-m", "helico.train_contacts",
        "--config", str(config_path), "--bundle", str(scratch / "data.paths.json"),
        "--checkpoint", str(checkpoint), "--output", str(scratch / "run"), *resume]
    print(json.dumps({"training_command": command, **provenance}), flush=True)
    run_trainer(command, env, scratch, config)
    fs.put_file(str(scratch / "run/result.json"), config["output_uri"] + "/result.json")


def evaluate(config, config_path):
    """Evaluate prepared inputs with one independent replica per GPU."""
    from helico.contact_evaluate import select_panel
    import wandb
    scratch = Path('/tmp/helico/evaluation'); scratch.mkdir(parents=True, exist_ok=True)
    fs = filesystem(durable=True)
    output_uri = config['output_uri']
    if config.get('checkpoint_pointer_uri'):
        pointer = json.loads(fs.cat_file(config['checkpoint_pointer_uri']))
        if pointer['step'] != config['checkpoint_step'] or pointer['data_lock_sha256'] != config['data_lock_sha256']:
            raise ValueError('Evaluation checkpoint pointer step/data mismatch')
        config = {**config, 'checkpoint_uri': pointer['checkpoint'], 'checkpoint_sha256': pointer['sha256']}
        config_path = scratch / 'resolved-evaluation.json'
        write_json(config_path, config)
    fs.put_file(str(config_path), output_uri + '/evaluation.json')
    checkpoint = scratch / 'final.pt'
    if not checkpoint.exists() or file_sha256(checkpoint) != config['checkpoint_sha256']:
        fs.get_file(config['checkpoint_uri'], str(checkpoint), chunksize=256 * 1024**2, max_concurrency=4)
    if file_sha256(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint checksum mismatch')
    features = scratch / 'features'; features.mkdir(exist_ok=True)
    manifest_bytes = fs.cat_file(config['features_uri'] + '/manifest.json')
    import hashlib
    if hashlib.sha256(manifest_bytes).hexdigest() != config['features_manifest_sha256']:
        raise ValueError('Evaluation manifest checksum mismatch')
    (features / 'manifest.json').write_bytes(manifest_bytes)
    records = json.loads(manifest_bytes)
    if config.get('panel_per_group'):
        records = select_panel(records, config['panel_per_group'], config.get('seed', 2301))
    if config.get('target_ids'):
        records = [r for r in records if r['pdb_id'] in config['target_ids']]
    def download(record):
        if record['status'] != 'ready': return
        path = features / (record['pdb_id'] + '.pt.gz')
        if not path.exists() or file_sha256(path) != record['sha256']:
            fs.get_file(config['features_uri'] + '/' + path.name, str(path))
        if file_sha256(path) != record['sha256']:
            raise ValueError('Evaluation feature checksum mismatch: ' + record['pdb_id'])
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, records))
    initial = []
    if 'protenix' in config['weights']:
        path = fetch('https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt',
                     scratch / 'protenix_base_default_v1.0.0.pt')
        if file_sha256(path) != '2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04':
            raise ValueError('Protenix initialization checksum mismatch')
        initial = ['--initial', str(path)]
    run = wandb.init(project='helico', entity='timodonnell', id=config['run_name'],
        name=config['run_name'], resume='allow', job_type='evaluation',
        config={**config, 'code_sha': os.environ['HELICO_CODE_SHA']})
    started = time.monotonic()
    try:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nnodes=1', f"--nproc-per-node={config['gpus']}", '-m', 'helico.contact_evaluate',
            '--config', str(config_path), '--features', str(features),
            '--checkpoint', str(checkpoint), '--output', str(scratch / 'results'), *initial]
        run_trainer(command, dict(os.environ), scratch, config)
        results = [json.loads(fs.cat_file(output_uri + f'/ranks/{r}.json')) for r in range(config['gpus'])]
        if any(r['status'] != 'complete' or r['failures'] for r in results):
            raise RuntimeError('An evaluation rank did not complete cleanly')
        result = dict(status='complete', targets=sum(r['targets'] for r in results),
            resource_failures=sum(r.get('resource_failures', 0) for r in results),
            ranks=len(results), wall_seconds=time.monotonic() - started,
            code_sha=os.environ['HELICO_CODE_SHA'])
        from helico.checkpoint_storage import publish_json
        publish_json(fs, output_uri + '/result.json', result)
        run.summary.update(result)
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["mirror", "train", "evaluate"])
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "mirror":
        mirror(config)
    elif args.mode == 'evaluate':
        evaluate(config, args.config.resolve())
    else:
        train(config, args.config.resolve())


if __name__ == "__main__":
    main()
