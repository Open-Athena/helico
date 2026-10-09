"""Build the versioned Protenix-v1 PDB dataset metadata (no GPU required).

The large payload already exists on the Hub. Preserve it by commit and SHA256,
and publish our split indices/contract independently. This build is deterministic
given the pinned inputs; it never uses current PDB release dates or a latest ref.

Run: python scripts/build_protenix_dataset.py --cache-dir /path/to/cache --output /path/to/release
Publish: hf upload timodonnell/helico-protenix-v1-pdb /path/to/release --type dataset
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request

from huggingface_hub import HfApi, hf_hub_download
from helico.datasets import file_sha256, validate_manifest, write_json

UPSTREAM = "85767b811c40ed46e73a9b39519cf6bfca8701ba"
FOLDBENCH = "4273f6877d82bd0b2fa476d1b2f34d121cbccc70"
MIRROR = "LiteFold/protenix-data"
REVISION = "47150f5244c15967e69315f3937d0ebcf863f317"
INDEX = "weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz"
INDEX_ARCHIVE_SHA = "f834b335bd037bdd9b3d5feea4776a7128c1be6bae2828f355d60868c6403b5e"
TARGET_FILES = ["monomer_protein", "monomer_rna", "monomer_dna", "interface_protein_protein",
                "interface_antibody_antigen", "interface_protein_ligand", "interface_protein_dna",
                "interface_protein_rna", "interface_protein_peptide"]


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(path)
    return path


def compressed_writer(path):
    # mtime=0 and filename='' remove build time and local paths from gzip headers.
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, io.TextIOWrapper(gz, newline="")


def csv_rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def build(cache: Path, output: Path):
    import tarfile
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset.json").exists():
        raise ValueError("Build into a new directory; do not mutate a published release")
    for name in ["upstream", "splits", "audit", "foldbench"]:
        (output / name).mkdir(exist_ok=True)
    archive = fetch("https://protenix.tos-cn-beijing.volces.com/indices.tar.gz", cache / "indices.tar.gz")
    if file_sha256(archive) != INDEX_ARCHIVE_SHA:
        raise ValueError("Official index archive changed; audit and explicitly update the pin")
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile():
                with tar.extractfile(member) as src, (output / "upstream" / Path(member.name).name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    # Use pinned GitHub target lists and retain them with the derived dataset.
    fold_ids = set()
    for name in TARGET_FILES:
        path = fetch(f"https://raw.githubusercontent.com/BEAM-Labs/FoldBench/{FOLDBENCH}/targets/{name}.csv",
                     cache / "foldbench" / f"{name}.csv")
        shutil.copyfile(path, output / "foldbench" / path.name)
        fold_ids.update(row["pdb_id"].lower().split("-")[0] for row in csv_rows(path))

    metadata_path = Path(hf_hub_download(MIRROR, "metadata.csv", repo_type="dataset", revision=REVISION))
    info = HfApi().dataset_info(MIRROR, revision=REVISION, files_metadata=True)
    by_name = {f.rfilename: f for f in info.siblings}
    if file_sha256(metadata_path) != by_name["metadata.csv"].lfs.sha256:
        raise ValueError("Mirror metadata checksum failed")
    available = set()
    source_shards = set()
    prefixes = ["common", "mmcif", "mmcif_bioassembly", "mmcif_msa_template", "recentPDB_bioassembly", "rna_msa"]
    index_locations = []
    offset = defaultdict(int)
    for row in csv_rows(metadata_path):
        # Used only for small-file verification below. Validate each tar header
        # against its expected member name and size before reading its payload.
        row["offset"] = offset[row["shard_path"]]
        offset[row["shard_path"]] += 1536 + ((int(row["size_bytes"]) + 511) // 512) * 512
        if row["top_level"] in prefixes:
            available.add(row["path"])
            source_shards.add(row["shard_path"])
        if row["top_level"] == "indices" or row["path"] == "common/obsolete_to_successor.json":
            index_locations.append(row)
    verified = []
    obsolete = None
    for row in index_locations:
        start = row["offset"]
        end = start + 1536 + int(row["size_bytes"]) - 1
        url = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/{row['shard_path']}?range={start}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise ValueError("Mirror did not serve the requested byte range")
            payload = response.read(end - start + 2)
        if len(payload) != end - start + 1:
            raise ValueError("Unexpected mirror range size")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as tar:
            member = tar.next()
            if member.name != row["path"] or member.size != int(row["size_bytes"]):
                raise ValueError("Mirror tar layout changed; rebuild the member index")
            data = tar.extractfile(member).read()
        if row["top_level"] == "indices":
            expected = output / "upstream" / row["filename"]
            if hashlib.sha256(data).hexdigest() != file_sha256(expected):
                raise ValueError(f"Mirror and official index differ: {row['path']}")
        else:
            obsolete = json.loads(data)
            (output / "upstream" / row["filename"]).write_bytes(data)
        verified.append(dict(path=row["path"], sha256=hashlib.sha256(data).hexdigest(),
                             size=len(data), shard=row["shard_path"], header_offset=start))
    if obsolete is None:
        raise ValueError("Missing PDB replacement mapping")

    def canonical(pid):
        seen = set()
        while pid in obsolete:
            if pid in seen:
                raise ValueError("Cycle in obsolete PDB mapping")
            seen.add(pid)
            pid = obsolete[pid]
        return pid

    held_canonical = {canonical(pid) for pid in fold_ids}
    forbidden = fold_ids | {pid for pid in obsolete if canonical(pid) in held_canonical} | held_canonical
    val_path = output / "upstream/recentPDB_low_homology_maxtoken1536.csv"
    val_rows = list(csv_rows(val_path))
    val_ids = {row["pdb_id"] for row in val_rows}
    selected_val = set((output / "upstream/recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt").read_text().split())
    if val_ids & forbidden:
        raise ValueError("FoldBench overlaps validation")

    counts = Counter()
    groups = Counter()
    group_pdbs = defaultdict(set)
    train_ids = set()
    excluded_overlap = set()
    raw, stream = compressed_writer(output / "splits/train.csv.gz")
    try:
        with gzip.open(output / "upstream" / INDEX, "rt", newline="") as src, stream:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                counts["upstream_rows"] += 1
                pid = row["pdb_id"].lower()
                if not row["release_date"] or row["release_date"] >= "2021-09-30":
                    counts["date_excluded_rows"] += 1
                    continue
                if row["mol_1_type"] == "ions" or row["mol_2_type"] == "ions":
                    counts["ion_excluded_rows"] += 1
                    continue
                if pid in forbidden:
                    counts["foldbench_excluded_rows"] += 1
                    excluded_overlap.add(pid)
                    continue
                if pid in val_ids:
                    raise ValueError(f"Train/validation overlap: {pid}")
                writer.writerow(row)
                train_ids.add(pid)
                counts["train_rows"] += 1
                key = (row["type"], row["eval_type"])
                groups[key] += 1
                group_pdbs[key].add(pid)
    finally:
        raw.close()
    for split, selected in [("validation", selected_val), ("validation_pool", val_ids)]:
        rows = [row for row in val_rows if row["pdb_id"] in selected]
        with (output / "splits" / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(val_rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "splits" / f"{split}_pdb_ids.txt").write_text("\n".join(sorted(selected)) + "\n")
        counts[f"{split}_rows"] = len(rows)
        counts[f"{split}_pdb_entries"] = len(selected)
    (output / "splits/train_pdb_ids.txt").write_text("\n".join(sorted(train_ids)) + "\n")
    (output / "audit/foldbench_excluded_pdb_ids.txt").write_text("\n".join(sorted(forbidden)) + "\n")
    counts["train_pdb_entries"] = len(train_ids)
    missing_train = sorted(pid for pid in train_ids if f"mmcif_bioassembly/{pid}.pkl.gz" not in available)
    missing_val = sorted(pid for pid in val_ids if f"recentPDB_bioassembly/{pid}.pkl.gz" not in available)
    if missing_train or missing_val:
        raise ValueError(f"Missing source structures: train={missing_train[:10]}, validation={missing_val[:10]}")
    write_json(output / "audit/summary.json", dict(counts))
    write_json(output / "audit/mirror_verification.json", {"verified_files": verified,
               "metadata_sha256": file_sha256(metadata_path), "missing_train_bioassemblies": missing_train,
               "missing_validation_bioassemblies": missing_val,
               "scope": "Full split membership and source-path presence; no exhaustive payload or MSA-quality audit"})
    write_json(output / "audit/split_audit.json", {
        "foldbench_pdb_entries": len(fold_ids), "foldbench_exclusion_ids_with_aliases": len(forbidden),
        "removed_training_pdb_ids": sorted(excluded_overlap), "train_foldbench_overlap": 0,
        "train_validation_overlap": 0, "validation_foldbench_overlap": 0,
        "inherited_pretraining_exposure": "8p7u replaces 6g23; exact checkpoint training manifest is unavailable",
        "scope": "PDB IDs and obsolete aliases; not a sequence/structural homology audit"})
    with (output / "audit/categories.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "category", "sampling_records", "pdb_entries"])
        for key, count in sorted(groups.items()):
            writer.writerow([*key, count, len(group_pdbs[key])])
    files = [{"path": str(path.relative_to(output)), "size": path.stat().st_size, "sha256": file_sha256(path)}
             for path in sorted(output.rglob("*")) if path.is_file()]
    shards = [{"path": name, "size": by_name[name].size, "sha256": by_name[name].lfs.sha256}
              for name in sorted(source_shards)]
    manifest = {
        "schema_version": 1, "name": "helico-protenix-v1-pdb", "release": "v1.0.0",
        "format": "protenix-pdb-v1", "files": files,
        "splits": {name: {"role": "train" if name == "train" else "validation",
                            "index": f"splits/{name}.csv" + (".gz" if name == "train" else ""),
                            "pdb_ids": f"splits/{name}_pdb_ids.txt",
                            "bioassembly_dir": "mmcif_bioassembly" if name == "train" else "recentPDB_bioassembly",
                            "sampling_records": counts[f"{name}_rows"], "pdb_entries": counts[f"{name}_pdb_entries"]}
                   for name in ["train", "validation", "validation_pool"]},
        "sources": [{"repo_id": MIRROR, "revision": REVISION, "format": "tar", "files": shards,
                     "extract_prefixes": prefixes}],
        "split_policy": {"training_release_date_before": "2021-09-30", "unknown_dates": "exclude",
                         "membership": "released default Protenix index; exclude ion-centered rows and FoldBench IDs/aliases",
                         "excluded_pdb_ids": "audit/foldbench_excluded_pdb_ids.txt",
                         "foldbench_git_revision": FOLDBENCH, "keep_foldbench_evaluation_unchanged": True},
        "sampler": {"sampler_type": "weighted", "beta_dict": {"chain": 0.5, "interface": 1},
                    "alpha_dict": {"prot": 3, "nuc": 3, "ligand": 1}, "force_recompute_weight": True},
        "contact_definition": {"version": "min-heavy-atom-5A-v1", "distance": "minimum between token heavy atoms",
                               "threshold_angstrom": 5.0, "comparison": "strictly_less_than",
                               "exclude_elements": ["H", "D"], "states": {"masked": 0, "no_contact": 1, "contact": 2},
                               "symmetric": True, "diagonal_supervised": False,
                               "unknown_geometry": "Observed close heavy atoms establish positives; absence requires complete heavy atoms for both tokens",
                               "storage": "Coordinates retained; contact labels computed during featurization, not precomputed in this release"},
        "provenance": {"upstream_git_revision": UPSTREAM, "upstream_index_archive_sha256": INDEX_ARCHIVE_SHA,
                       "upstream_data_version": "2024.05.22", "distillation_included": False,
                       "exact_checkpoint_training_manifest_verified": False,
                       "build_script": "scripts/build_protenix_dataset.py"},
    }
    validate_manifest(manifest)
    write_json(output / "dataset.json", manifest)
    (output / "README.md").write_text(f"""---
pretty_name: Helico Protenix v1 PDB
size_categories:
  - 100K<n<1M
task_categories:
  - other
tags:
  - biology
  - protein
  - structural-biology
  - contact-diffusion
configs:
  - config_name: sampling_records
    default: true
    data_files:
      - split: train
        path: splits/train.csv.gz
      - split: validation
        path: splits/validation.csv
---

# Helico Protenix v1 PDB — v1.0.0

A versioned experimental-structure dataset for contact-conditioned fine-tuning.
This repository owns the split membership and data contract. Large coordinates,
MSAs, templates and chemical components are already on Hugging Face and referenced
at an immutable revision, rather than copied into another unversioned directory.

| Split | PDB entries | Chain/interface sampling records |
| --- | ---: | ---: |
| Train | {counts['train_pdb_entries']:,} | {counts['train_rows']:,} |
| Validation (original default subset) | {counts['validation_pdb_entries']:,} | {counts['validation_rows']:,} |
| Full validation pool (optional) | {counts['validation_pool_pdb_entries']:,} | {counts['validation_pool_rows']:,} |

Sampling records are not independent structures. Entries occur in multiple chain
and interface categories. This dataset does not supply curated binding negatives,
binding affinities, or the Protenix monomer/disorder-distillation corpora.

## Immutable source assets

- Repository: [{MIRROR}](https://huggingface.co/datasets/{MIRROR}/tree/{REVISION})
- Revision: `{REVISION}`
- Required shards: 50, totaling {sum(s['size'] for s in shards):,} bytes (~1.06 TB).
- Extracted components: coordinates/mmCIF, biological assemblies, protein MSAs
  and template alignments, RNA MSAs, CCD/reference chemistry and lookup tables.
- Search-database shards are not needed because alignments are precomputed.

`dataset.json` pins each shard's SHA256 and size, all local metadata files, the
contact definition, sampler settings, upstream code revision and split policy.
The mirror's four indices were compared byte for byte with the official package.
Every selected training and validation structure has a biological-assembly path
in the mirror metadata. The complete 1 TB payload has not been exhaustively
downloaded/featurized as part of this release; MSA quality and feature generation
remain training-preflight checks.

## Split policy

Start from the official default Protenix training index, using its assigned
release dates before **2021-09-30**. Drop ion-centered sampling rows and all
FoldBench PDB IDs plus obsolete aliases. The latter removes 11 rows for **8P7U**,
which supersedes **6G23** and was assigned the predecessor's 2019 release date.
Missing dates are excluded. Retain the original 384-entry validation subset and
the larger 1,818-entry validation pool separately. FoldBench itself is unchanged.

Audit results: zero train/validation or train/FoldBench ID overlap after filtering.
This is an ID/replacement audit, not a new sequence or structural homology split.
The exact training manifest of the pretrained checkpoint has not been verified;
removing this example cannot undo possible exposure during pretraining.

The report describes approximately 150k experimental structures, while the public
index contains approximately 168k. This release reproduces the **public index**
with the documented exclusions; it does not claim to reproduce a hidden manifest.

## Contact labels

Labels are defined by minimum heavy-atom distance **strictly below 5 Å** between
tokens. Protein tokens are residues; ligand tokens are individual heavy atoms.
Exclude H/D, make the matrix symmetric, and do not supervise the diagonal.
Observed close atoms establish a positive contact. A negative label requires
complete relevant heavy-atom geometry for both tokens; missing geometry is unknown.
Coordinates are retained here and labels are computed during featurization,
using the pinned contact definition. Dense contact tensors are not stored.

Input states: masked=0, no-contact=1, contact=2. The absorbing-mask process reveals
unordered pairs independently of their contact value. The noise/time distribution
and MSA dropout are training settings, not changes to dataset membership.

## Training recipes

Use [Helico training recipes](https://huggingface.co/datasets/timodonnell/helico-training-recipes).
A recipe lists dataset repositories, full commit IDs, selected splits and mixture
weights. `helico-datasets lock` resolves the manifests; `prepare` verifies files and
stages source assets. The run records `data.lock.json`; its digest belongs in the
checkpoint and W&B configuration. Do not train from a mutable `main` revision.

Use the upstream paired-MSA/template featurizer with the generated data configuration.
These are source-data indices, not the legacy Helico `TokenizedStructure` pickle
format or the small pilot's `pilot.pt`. The full-scale Helico feature adapter is
a separate training integration step; no model training is launched by this release.

## Provenance and attribution

- [Protenix code/configuration](https://github.com/bytedance/Protenix/tree/{UPSTREAM})
- [Official index archive](https://protenix.tos-cn-beijing.volces.com/indices.tar.gz)
- Index archive SHA256: `{INDEX_ARCHIVE_SHA}`
- [FoldBench target definitions](https://github.com/BEAM-Labs/FoldBench/tree/{FOLDBENCH}/targets)
- [8P7U replacement record](https://www.rcsb.org/structure/8P7U)

Underlying data and software remain attributed to wwPDB, Protenix, the alignment
database providers and LiteFold's mirror. Their original terms apply; this release
does not relicense third-party assets. The new tooling is in the Apache-2.0 Helico
repository. See `audit/` for the build counts and verification evidence, and
`build/build_protenix_dataset.py` for the exact build script.
""")
    (output / "build").mkdir(exist_ok=True)
    shutil.copyfile(__file__, output / "build/build_protenix_dataset.py")
    print(json.dumps(dict(counts), indent=2), flush=True)
    print(f"Source: {len(shards)} pinned shards, {sum(s['size'] for s in shards):,} bytes already on Hugging Face")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.cache_dir, args.output)
