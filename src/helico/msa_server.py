"""MMseqs2 MSA server client for on-the-fly MSA generation.

Adapted from ColabFold (https://github.com/sokrypton/ColabFold) via
Boltz (https://github.com/jwohlwend/boltz).
"""

from __future__ import annotations

import logging
import os
import random
import tarfile
import time

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://api.colabfold.com"


def run_mmseqs2(
    sequences: str | list[str],
    result_dir: str,
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = DEFAULT_HOST,
) -> list[str]:
    """Query a ColabFold-compatible MMseqs2 server for MSA generation.

    Args:
        sequences: Single sequence string or list of sequences.
        result_dir: Directory to cache results (tar.gz + extracted a3m files).
        use_env: Include environmental databases (BFD, MGnify, etc.).
        use_filter: Apply sequence filtering.
        use_pairing: Use paired MSA mode (for multi-chain).
        pairing_strategy: "greedy" or "complete" pairing.
        host_url: MMseqs2 server URL.

    Returns:
        List of A3M-format strings, one per input sequence.
    """
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"
    headers = {"User-Agent": "helico/0.1"}

    def submit(seqs: list[str], mode: str, N: int = 101) -> dict:
        query = ""
        for i, seq in enumerate(seqs):
            query += f">{N + i}\n{seq}\n"
        for attempt in range(6):
            try:
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
                return res.json()
            except Exception as e:
                if attempt >= 5:
                    raise RuntimeError("Too many failed MSA server requests") from e
                logger.warning(f"MSA server error (attempt {attempt + 1}/6): {e}")
                time.sleep(5)

    def poll_status(job_id: str) -> dict:
        for attempt in range(6):
            try:
                res = requests.get(
                    f"{host_url}/ticket/{job_id}",
                    timeout=6.02,
                    headers=headers,
                )
                return res.json()
            except Exception as e:
                if attempt >= 5:
                    raise RuntimeError("Too many failed MSA server requests") from e
                logger.warning(f"MSA server error (attempt {attempt + 1}/6): {e}")
                time.sleep(5)

    def download(job_id: str, path: str) -> None:
        for attempt in range(6):
            try:
                res = requests.get(
                    f"{host_url}/result/download/{job_id}",
                    timeout=6.02,
                    headers=headers,
                )
                with open(path, "wb") as f:
                    f.write(res.content)
                return
            except Exception as e:
                if attempt >= 5:
                    raise RuntimeError("Too many failed MSA server requests") from e
                logger.warning(f"MSA server error (attempt {attempt + 1}/6): {e}")
                time.sleep(5)

    if requests is None:
        raise ImportError("requests is required for MSA server: pip install requests")

    seqs = [sequences] if isinstance(sequences, str) else list(sequences)

    # Setup mode
    if use_pairing:
        mode = "pairgreedy" if pairing_strategy == "greedy" else "paircomplete"
        if use_env:
            mode += "-env"
    elif use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    path = f"{result_dir}_{mode}"
    os.makedirs(path, exist_ok=True)

    # Deduplicate sequences while preserving order
    seqs_unique: list[str] = []
    for s in seqs:
        if s not in seqs_unique:
            seqs_unique.append(s)
    N = 101
    Ms = [N + seqs_unique.index(s) for s in seqs]

    # Submit and download if not cached
    tar_gz_file = f"{path}/out.tar.gz"
    if not os.path.isfile(tar_gz_file):
        logger.info(f"Submitting {len(seqs_unique)} sequences to MMseqs2 server at {host_url}")

        # Submit
        out = submit(seqs_unique, mode, N)
        while out.get("status") in ["UNKNOWN", "RATELIMIT"]:
            sleep_time = 5 + random.randint(0, 5)
            logger.info(f"MSA server: {out['status']}, retrying in {sleep_time}s")
            time.sleep(sleep_time)
            out = submit(seqs_unique, mode, N)

        if out.get("status") in ("ERROR", "MAINTENANCE"):
            raise RuntimeError(f"MSA server error: {out.get('status')}")

        # Poll for completion
        job_id = out["id"]
        logger.info(f"MSA job submitted (id={job_id}), waiting for completion...")
        while out.get("status") in ["UNKNOWN", "RUNNING", "PENDING"]:
            time.sleep(5 + random.randint(0, 5))
            out = poll_status(job_id)

        if out.get("status") == "ERROR":
            raise RuntimeError("MSA server returned ERROR")

        if out.get("status") == "COMPLETE":
            logger.info("MSA job complete, downloading results")
            download(job_id, tar_gz_file)
        else:
            raise RuntimeError(f"Unexpected MSA server status: {out.get('status')}")

    # Determine expected A3M files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract if needed
    if any(not os.path.isfile(f) for f in a3m_files):
        with tarfile.open(tar_gz_file) as tar:
            tar.extractall(path, filter="data")

    # Parse A3M lines per sequence
    a3m_lines: dict[int, list[str]] = {}
    for a3m_file in a3m_files:
        if not os.path.isfile(a3m_file):
            continue
        update_M, M = True, None
        with open(a3m_file) as fh:
            for line in fh:
                if not line:
                    continue
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                if M is not None:
                    a3m_lines[M].append(line)

    return ["".join(a3m_lines.get(n, [])) for n in Ms]
