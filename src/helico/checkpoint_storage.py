"""Bounded retries for durable checkpoint publication, never model execution."""
from __future__ import annotations

import json
from pathlib import Path
import time


def transient_storage_failure(text):
    text = text.lower()
    return any(message in text for message in (
        "incompletebody", "you did not provide the number of bytes specified by the content-length",
        "endpointconnectionerror", "readtimeouterror", "connecttimeouterror",
        "connection reset by peer", "temporarily unavailable",
        "(requesttimeout)", "(slowdown)", "(internalerror)", "(serviceunavailable)",
    ))


def publish_checkpoint(fs, path, output_uri, step, data_lock_sha256, *, attempts=3, sleep=time.sleep):
    """Upload immutable bytes before advancing latest.json.

    Retry the entire put_file call so each multipart attempt reopens the local
    file. A failed or size-mismatched upload cannot advance the resume pointer.
    Retry pointer publication separately to avoid re-uploading a good snapshot.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    path = Path(path)
    from helico.datasets import file_sha256
    sha256 = file_sha256(path)
    uri = output_uri + "/" + path.name

    def retry(operation, label):
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as error:
                if attempt == attempts or not transient_storage_failure(
                    type(error).__name__ + ": " + str(error)
                ):
                    raise
                print(json.dumps({"checkpoint_storage_retry": label, "step": step,
                                  "failed_attempt": attempt, "max_attempts": attempts,
                                  "error_type": type(error).__name__}), flush=True)
                sleep(2 ** attempt)

    def upload():
        fs.put_file(str(path), uri)
        if fs.size(uri) != path.stat().st_size:
            raise RuntimeError("Checkpoint upload size mismatch")

    retry(upload, "checkpoint")
    pointer = json.dumps({"step": step, "checkpoint": uri,
                          "data_lock_sha256": data_lock_sha256, "sha256": sha256,
                          "bytes": path.stat().st_size}).encode()
    retry(lambda: fs.pipe_file(output_uri + "/latest.json", pointer), "latest.json")
    return uri
