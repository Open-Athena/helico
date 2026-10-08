"""CoreWeave staging and training supervisor with durable object storage."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from helico.datasets import Hub, checked_download, validate_lock


def filesystem():
    import fsspec
    return fsspec.filesystem("s3")


def mirror(config):
    """Cache exact Hub shard bytes, keyed by SHA256, using a CPU-only job."""
    lock = json.loads(Path(config["data_lock"]).read_text())
    validate_lock(lock)
    fs = filesystem()
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["mirror", "train"])
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "mirror":
        mirror(config)
    else:
        raise NotImplementedError("Training supervisor is prepared in the next implementation step")


if __name__ == "__main__":
    main()
