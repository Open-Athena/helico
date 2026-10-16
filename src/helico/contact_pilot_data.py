"""Reproducible small-data preparation for the clean-contact mechanism pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import tarfile
import urllib.request

import numpy as np
import torch
from Bio.Align import PairwiseAligner

from helico.contact_diffusion import heavy_atom_contacts
from helico.data import _raw_msa_from_a3m, compute_msa_features

SNAPSHOT = "382ad3219097b764500c75475aaae3f82547f4fd"
MSA_URL = "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_raw_msa.tar"


def fetch_range(url, offset, size):
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset+size-1}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise ValueError("Range server did not return 206")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Unexpected HTTP range size")
    return data


def chain_key(sequence):
    return "rcsb_raw_msa/" + hashlib.sha256((sequence + "\n").encode()).hexdigest() + ".a3m.gz"


def make_features(ts, ccd, index, assets):
    """Rebuild CCD refs; require exact MSA/query/resolved-token correspondence."""
    complete = []
    for token, kind in zip(ts.tokens, ts.entity_types):
        comp = ccd.get(token.res_name)
        if comp is None or comp.ideal_coords is None:
            raise ValueError("Missing CCD reference")
        lookup = {name: i for i, name in enumerate(comp.atom_names)}
        ix = [lookup[name] for name in token.atom_names]
        token.ref_coords = np.asarray(comp.ideal_coords[ix], dtype=np.float32).copy()
        token.atom_charges = [comp.atom_charges[i] for i in ix] if comp.atom_charges else [0] * len(ix)
        expected = {name for name, keep in zip(comp.atom_names, comp.non_leaving_heavy_atom_mask) if keep}
        complete.append(kind == "ligand" or expected.issubset(token.atom_names))
        if any(e in {"H", "D"} for e in token.atom_elements):
            raise ValueError("Hydrogen/deuterium in token")
    features = ts.to_features()
    n = ts.n_tokens
    protein = torch.tensor([kind == "protein" for kind in ts.entity_types])
    same_chain = features["chain_same"].bool()
    separation = (features["res_indices"][:, None] - features["res_indices"][None, :]).abs()
    eligible = (protein[:, None] | protein[None, :]) & ~(same_chain & protein[:, None] & protein[None, :] & (separation < 6))
    eligible.fill_diagonal_(False)
    labels = heavy_atom_contacts(features["atom_coords"], features["atom_to_token"], n)
    complete = torch.tensor(complete)
    valid = eligible & (labels | (complete[:, None] & complete[None, :]))
    features.update(contact_target=labels, contact_valid=valid,
                    contact_state=torch.zeros(n, n, dtype=torch.uint8), protein_token=protein)

    # Use genuine unpaired chain MSAs block-diagonally, keeping 32 rows/chain.
    # Exclude missing-residue mappings in this pilot instead of guessing offsets.
    rows, dels, depths = [], [], []
    query = features["restype"].numpy().astype(np.int8)
    rows.append(query[None]); dels.append(np.zeros((1, n), dtype=np.int16))
    for cid in dict.fromkeys(ts.chain_ids):
        positions = [i for i, (c, k) in enumerate(zip(ts.chain_ids, ts.entity_types)) if c == cid and k == "protein"]
        if not positions:
            continue
        key = chain_key(ts.chain_sequences[cid])
        local = assets / "msa" / Path(key).name
        local.parent.mkdir(exist_ok=True)
        if not local.exists():
            offset, size = index.entries[key]
            local.write_bytes(fetch_range(MSA_URL, offset, size))
        raw = _raw_msa_from_a3m(gzip.decompress(local.read_bytes()).decode())
        if raw is None or raw.length != len(positions) or not np.array_equal(raw.msa[0], query[positions]):
            raise ValueError("MSA query does not match resolved protein tokens")
        if raw.n_seqs < 2:
            raise ValueError("No non-query MSA rows")
        depths.append(raw.n_seqs)
        block = np.full((min(32, raw.n_seqs - 1), n), 31, dtype=np.int8)
        deletion = np.zeros_like(block, dtype=np.int16)
        block[:, positions] = raw.msa[1:1 + len(block)]
        deletion[:, positions] = raw.deletion_matrix[1:1 + len(block)]
        rows.append(block); dels.append(deletion)
    msa = np.concatenate(rows); deletion = np.concatenate(dels)
    mf = compute_msa_features(msa, deletion, max_seqs=len(msa), n_clusters=len(msa))
    for name in ("profile", "cluster_msa", "cluster_profile", "deletion_mean", "cluster_deletion_mean"):
        features["msa_profile" if name == "profile" else name] = torch.from_numpy(getattr(mf, name))
    features["has_msa"] = torch.tensor(1)
    return features, min(depths), len(msa)


def entry_date(ts, cache):
    path = cache / f"{ts.pdb_id}.json"
    if not path.exists():
        with urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{ts.pdb_id}", timeout=30) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text())["rcsb_accession_info"]["initial_release_date"][:10]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--validation", type=int, default=12)
    args = p.parse_args()
    assets = args.assets
    torch.set_num_threads(1)
    np.random.seed(20); random.seed(20)
    with (assets / "ccd_cache.pkl").open("rb") as f: ccd = pickle.load(f)
    with (assets / "msa_index.pkl").open("rb") as f: index = pickle.load(f)
    candidates = []
    with tarfile.open(assets / "structures-prefix.tar") as archive:
        try:
            for member in archive:
                if not member.isfile(): continue
                ts = pickle.load(archive.extractfile(member))
                if not 48 <= ts.n_tokens <= 256 or "nucleotide" in ts.entity_types: continue
                chains = {c for c, k in zip(ts.chain_ids, ts.entity_types) if k == "protein"}
                if not chains or not all(chain_key(ts.chain_sequences[c]) in index.entries for c in chains): continue
                candidates.append(ts)
        except tarfile.ReadError:
            # Input is an explicitly declared byte prefix, not a full archive.
            pass
    random.shuffle(candidates)
    print(f"{len(candidates)} small candidates with indexed MSAs", flush=True)
    date_cache = assets / "entry_metadata"; date_cache.mkdir(exist_ok=True)
    def dated(ts):
        try: return ts, entry_date(ts, date_cache)
        except Exception as exc:
            print(f"metadata skip {ts.pdb_id}: {exc}", flush=True)
            return ts, ""
    with ThreadPoolExecutor(8) as pool: dated_candidates = list(pool.map(dated, candidates))
    # Reserve newer structures first, then remove similar training sequences.
    aligner = PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                             open_gap_score=-1, extend_gap_score=-0.1)
    held_sequences = []
    records = []; datasets = {"train": [], "validation": []}; seen = set()
    for split, cap in (("validation", args.validation), ("train", args.train)):
        for ts, date in dated_candidates:
            if len(datasets[split]) >= cap: break
            if not date or (split == "train" and date >= "2021-09-30") or (split == "validation" and date < "2022-05-01"): continue
            seqs = [seq for c, seq in ts.chain_sequences.items() if any(cid == c and k == "protein" for cid, k in zip(ts.chain_ids, ts.entity_types))]
            signature = tuple(sorted(seqs))
            if signature in seen: continue
            if split == "train" and any(aligner.score(a, b) / min(len(a), len(b)) > 0.4 for a in seqs for b in held_sequences): continue
            try:
                features, depth, rows = make_features(ts, ccd, index, assets)
            except Exception as exc:
                print(f"skip {ts.pdb_id}: {exc}", flush=True); continue
            target = features["contact_target"] & features["contact_valid"]
            prot = features["protein_token"]
            inter = ~features["chain_same"].bool()
            pp = int(torch.triu(target & inter & prot[:, None] & prot[None, :], 1).sum())
            pl = int(torch.triu(target & (prot[:, None] ^ prot[None, :]), 1).sum())
            kind = "protein_protein" if pp >= 5 else "protein_ligand" if pl >= 5 else "monomer"
            # Keep all three groups represented, avoiding a monomer-only pilot.
            count = sum(r["split"] == split and r["kind"] == kind for r in records)
            if count >= (cap + 2) // 3: continue
            datasets[split].append({"id": ts.pdb_id, "features": features, "kind": kind})
            seen.add(signature)
            if split == "validation": held_sequences.extend(seqs)
            record = dict(pdb_id=ts.pdb_id, split=split, kind=kind, release_date=date,
                          tokens=ts.n_tokens, atoms=ts.n_atoms, min_msa_depth=depth,
                          msa_rows=rows, pp_contacts=pp, pl_contacts=pl)
            records.append(record); print(record, flush=True)
    if len(datasets["train"]) < 12 or len(datasets["validation"]) < 3:
        raise RuntimeError("Insufficient pilot examples")
    torch.save(datasets, assets / "pilot.pt")
    with (assets / "manifest.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (assets / "provenance.json").write_text(json.dumps({"snapshot": SNAPSHOT,
        "prefix_bytes": (assets / "structures-prefix.tar").stat().st_size,
        "prefix_sha256": hashlib.sha256((assets / "structures-prefix.tar").read_bytes()).hexdigest(),
        "msa_archive": MSA_URL, "seed": 20,
        "split_filter": "temporal + global alignment score/min length <=0.4; not a clustered benchmark"}, indent=2))


if __name__ == "__main__":
    main()
