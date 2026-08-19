"""Shared paths, upstream mirroring, and the exp245 ranking recipe.

Everything this experiment consumes from MarinFold is a public, immutable file
under one HF bucket prefix. Mirroring it by name and pinning size + sha256 is
how exp245 itself handles its inputs, and the same discipline applies here for
the same reason: a silently changed eval set would move every number in this
directory with no other visible symptom.

The two CoreWeave-only inputs -- the per-protein dense score matrices and the
rollout vote tables for `exp232-decontam-m2-p06` -- are pinned the same way but
mirrored from S3 by `export_marinfold_contacts.py`, because exp245 exported
only its ``results/`` prefix to the public bucket.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / ".cache"
UPSTREAM = CACHE / "upstream"
GT_DIR = DATA / "gt"
ARMS = DATA / "arms"

#: exp245's published inputs and results.
BUCKET = ("hf://buckets/open-athena/MarinFold/data/"
          "contacts-v1-foldbench-monomers-exp245")
RUN_ID = "fbmono-20260818-01"

#: The better of #232's two decontaminated checkpoints -- 0.538 R-precision on
#: eval-test against m1-p02's 0.493 (exp245 section 4). Its rollout outputs live
#: under this label in both the CoreWeave run tree and contact_precision_all.csv.
CHECKPOINT_DIR = "exp232_decontam_m2_p06_step145199"
CHECKPOINT_LABEL = "marinfold-exp232-decontam-m2-p06-step145199"

CW_ENDPOINT = "https://cwobject.com"
CW_BUCKET = "marin-us-east-02a"
CW_RUN_ROOT = (
    "marin/protein-structure/MarinFold/exp245_foldbench_held_out_monomers"
    f"/evals/rollout/{RUN_ID}"
)

#: Helico's training filter. Every eval unit must postdate this or its lDDT
#: measures recall, not folding. build_eval_sets.py asserts it.
TRAIN_CUTOFF = "2021-09-30"

FOLDBENCH = Path.home() / ".cache/helico/data/benchmarks/FoldBench"
FOLDBENCH_GT = FOLDBENCH / "examples/ground_truths"

#: exp89's contact-metric constants, taken by value rather than reimplemented
#: loosely: a true contact needs degree >= MIN_DEG and separation >= MIN_SEP,
#: and only pairs of *resolved* residues are candidates. rank_pairs below
#: reproduces exp245's published precision on 333/333 proteins at every cut, so
#: these are not approximations of the upstream rule -- they are the rule.
MIN_DEG, MIN_SEP = 0.001, 6

#: `hf` outside the venv: helico pins huggingface_hub below the version with the
#: `buckets` subcommand. Same resolution exp245 documents.
HF_BIN_CANDIDATES = ("/home/bizon/anaconda3/bin/hf", "hf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_binary() -> str:
    for candidate in HF_BIN_CANDIDATES:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise SystemExit(
        "no usable `hf` CLI found. The venv's huggingface_hub is pinned below "
        "the release with a `buckets` subcommand; use a system install."
    )


def fetch(name: str) -> Path:
    """Mirror one file from exp245's bucket into `.cache/upstream/`.

    Cached by name: these are immutable published artifacts, and the pin file
    written by build_eval_sets.py is what detects it if they ever stop being.
    """
    local = UPSTREAM / Path(name).name
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([hf_binary(), "buckets", "cp", f"{BUCKET}/{name}", str(local)],
                   check=True)
    return local


def cw_client():
    """S3 client for CoreWeave AI Object Storage.

    Virtual-hosted addressing is not optional -- path style is rejected outright
    (PathStyleRequestNotAllowed). Credentials come from the environment
    (CW_KEY_ID / CW_KEY_SECRET); the ambient AWS profile is a different account
    and fails with InvalidAccessKeyId, which is why they are read explicitly
    rather than left to boto3's default chain.
    """
    import os

    import boto3
    import botocore

    key, secret = os.environ.get("CW_KEY_ID"), os.environ.get("CW_KEY_SECRET")
    if not (key and secret):
        raise SystemExit(
            "CW_KEY_ID / CW_KEY_SECRET are required to mirror the MarinFold "
            "rollout outputs; see ~/.config/marin/cw-rno2a.env"
        )
    return boto3.client(
        "s3", endpoint_url=CW_ENDPOINT,
        aws_access_key_id=key, aws_secret_access_key=secret,
        config=botocore.config.Config(s3={"addressing_style": "virtual"}),
    )


def load_gt_universe() -> dict[str, dict]:
    """exp245's ground-truth records, keyed by stem.

    Each carries `L`, `resolved` (prompt indices of the resolved residues),
    `gt_chain`, and `contacts` as (i, j, degree) triples in prompt indices.
    """
    path = fetch("gt_universe_scored.jsonl")
    records = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        records[record["stem"]] = record
    return records


def true_matrix(length: int, contacts) -> np.ndarray:
    """exp89's truth definition, verbatim."""
    matrix = np.zeros((length, length), bool)
    for i, j, degree in contacts:
        i, j = int(i), int(j)
        if degree >= MIN_DEG and (j - i) >= MIN_SEP and i < j < length:
            matrix[i, j] = True
    return matrix


def candidate_pairs(resolved) -> tuple[np.ndarray, np.ndarray]:
    """The (i, j) prompt-index pairs exp245 ranks, in its own order.

    Candidates are upper-triangle pairs of *resolved* residues at separation
    >= MIN_SEP. The order matters: exp245 breaks score ties with a stable sort,
    so the tie order is this array's order, and reproducing it is the difference
    between matching the published precision and merely coming close to it.
    """
    resolved = np.asarray(resolved, dtype=np.int64)
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    keep = (j - i) >= MIN_SEP
    return i[keep], j[keep]


def rank_pairs(score: np.ndarray, resolved) -> list[tuple[int, int]]:
    """Predicted contacts in prompt indices, best first -- exp245's ranking.

    `score` is the dense L x L matrix the run wrote per protein. Verified
    against contact_precision_all.csv: this ordering reproduces the published
    precision at L, L/2 and L/5 on all 333 proteins to floating-point identity
    (check_ranking.py).
    """
    i, j = candidate_pairs(resolved)
    order = np.argsort(-score[i, j], kind="mergesort")
    return [(int(i[k]), int(j[k])) for k in order]
