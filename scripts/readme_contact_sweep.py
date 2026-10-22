"""Generate the real predictions used by the README contact sweep.

Run from the repository root with a CUDA GPU and Helico installed:
    python scripts/readme_contact_sweep.py

The rendering script reads the committed CA coordinates; no GPU is needed to
rebuild the GIF. Each condition is an independent prediction, not a trajectory.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from helico.bench import compute_lddt, structure_to_chains
from helico.data import CONTACT_PRESENT, parse_ccd, parse_mmcif, tokenize_sequences
from helico.inference import contacts_from_pairs, contacts_from_structure, fold, load_model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/images/contact_sweep"
REVISION = "52e5a9de1dd1b2333c6a31c687fc29c40521cbf2"
FILENAME = "contacts-msafree-01-step6000.pt"


def main():
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_path = OUTPUT / "8b2g.cif"
    if not reference_path.exists():
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/8B2G.cif", reference_path
        )
    reference = parse_mmcif(str(reference_path))
    reference = replace(reference, chains=[c for c in reference.chains if c.chain_id == "A"])
    chains = structure_to_chains(reference)
    assert len(chains) == 1 and chains[0]["id"] == "A"
    sequence = chains[0]["sequence"]
    ccd = parse_ccd()
    tokenized = tokenize_sequences(chains, ccd)
    oracle = contacts_from_structure(tokenized, reference)
    pairs = torch.triu(oracle == CONTACT_PRESENT, diagonal=1).nonzero()
    pairs = pairs[torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0))]
    pairs = pairs.tolist()
    (OUTPUT / "sequence.txt").write_text(sequence + "\n")
    (OUTPUT / "contacts.txt").write_text("".join(f"{i} {j}\n" for i, j in pairs))
    counts = sorted(set(min(n, len(pairs)) for n in [0, 5, 10, 20, 30, len(pairs)]))

    truth = np.array([
        atom.coords for chain in reference.chains if chain.chain_id == "A"
        for residue in chain.residues for atom in residue.atoms if atom.name == "CA"
    ])
    atom_names = [name for token in tokenized.tokens for name in token.atom_names]
    ca_indices = [i for i, name in enumerate(atom_names) if name == "CA"]
    assert len(truth) == len(ca_indices) == len(sequence) == 74

    checkpoint = hf_hub_download("timodonnell/helico", FILENAME, revision=REVISION)
    model = load_model(checkpoint)
    assert model.config.use_contacts and not model.config.use_msa
    with open(checkpoint, "rb") as handle:
        checkpoint_hash = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    metadata = {
        "target": "8B2G", "chain": "A", "sequence": sequence,
        "reference_url": "https://files.rcsb.org/download/8B2G.cif",
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "checkpoint": f"timodonnell/helico/{FILENAME}",
        "checkpoint_revision": REVISION,
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(), "dtype": str(model._helico_dtype),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "pyconfind", "cuequivariance-torch")},
        "contact_order_seed": 0, "inference_seed": 0,
        "n_samples": 3, "n_cycles": 10,
        "n_diffusion_steps": model.config.n_diffusion_steps,
        "sample_selection": "model ranking_score (no reference used for selection)",
        "contact_source": "pyconfind contacts-v1, derived from the deposited structure",
        "unlisted_pairs": "unknown at every contact count, including the final condition",
        "display": "CA traces, aligned to the reference by a rigid Kabsch transform",
        "reference_ca": truth.tolist(), "ordered_contact_pairs_zero_indexed": pairs,
        "conditions": [],
    }
    for count in counts:
        print(f"Predicting {count}/{len(pairs)} contacts", flush=True)
        started = time.monotonic()
        contacts = contacts_from_pairs(pairs[:count], seq_len=len(sequence), strict=True)
        assert int((contacts == CONTACT_PRESENT).sum()) == 2 * count
        result = fold({"A": sequence}, contacts=contacts, model=model,
                      n_samples=3, ccd=ccd, seed=0)
        prediction_path = OUTPUT / f"contacts_{count:03d}.pdb"
        result.write_pdb(prediction_path)
        predicted = result.coords.numpy()[ca_indices]
        assert predicted.shape == truth.shape and np.isfinite(predicted).all()
        row = {
            "n_contacts": count, "pdb": prediction_path.name,
            "pdb_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "mean_plddt": result.mean_plddt, "ranking_score": result.ranking_score,
            "ca_lddt": compute_lddt(predicted, truth),
            "seconds": time.monotonic() - started,
            "predicted_ca": predicted.tolist(),
        }
        metadata["conditions"].append(row)
        (OUTPUT / "data.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print({k: v for k, v in row.items() if k != "predicted_ca"}, flush=True)


if __name__ == "__main__":
    main()
