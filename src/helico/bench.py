"""Helico FoldBench benchmark: predict structures and score against ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from helico.data import (
    Structure,
    TarIndex,
    TokenizedStructure,
    load_msa_for_chain,
    parse_ccd,
    parse_mmcif,
    tokenize_sequences,
)
from helico.model import Helico, HelicoConfig
from helico.train import coords_to_pdb, run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# FoldBench categories (matching actual CSV filenames in BEAM-Labs/FoldBench)
# ============================================================================

INTERFACE_CATEGORIES = [
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_peptide",
    "interface_protein_ligand",
    "interface_protein_dna",
    "interface_protein_rna",
]

MONOMER_CATEGORIES = [
    "monomer_protein",
    "monomer_rna",
    "monomer_dna",
]

ALL_CATEGORIES = INTERFACE_CATEGORIES + MONOMER_CATEGORIES


# ============================================================================
# Data loading
# ============================================================================

@dataclass
class BenchTarget:
    """A single FoldBench benchmark target."""
    pdb_id: str       # e.g. "8tuz-assembly1"
    category: str
    extra: dict = field(default_factory=dict)


def load_targets(targets_dir: Path) -> dict[str, list[BenchTarget]]:
    """Parse category CSVs from the FoldBench targets/ directory.

    Returns dict mapping category name -> list of BenchTarget.
    """
    results: dict[str, list[BenchTarget]] = {}
    for csv_path in sorted(targets_dir.glob("*.csv")):
        category = csv_path.stem
        targets = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdb_id = row.get("pdb_id", "")
                targets.append(BenchTarget(
                    pdb_id=pdb_id,
                    category=category,
                    extra=dict(row),
                ))
        results[category] = targets
        logger.info(f"Loaded {len(targets)} targets for {category}")
    return results


def load_af3_inputs(json_path: Path) -> dict[str, dict]:
    """Load alphafold3_inputs.json and index by target name.

    Returns dict mapping target name (pdb_id) -> AF3 input dict.
    """
    with open(json_path) as f:
        data = json.load(f)
    return {entry["name"]: entry for entry in data}


def af3_entry_to_chains(entry: dict) -> list[dict]:
    """Convert a single AF3 input entry to Helico chain dicts.

    Handles:
    - "id" as list for homomers: {"protein": {"id": ["A","B"], "sequence": "..."}}
    - protein/rna/dna/ligand types
    """
    chains: list[dict] = []
    for seq_entry in entry.get("sequences", []):
        for mol_type in ("protein", "rna", "dna", "ligand"):
            if mol_type not in seq_entry:
                continue
            info = seq_entry[mol_type]

            # Expand homomer ids
            ids = info.get("id", info.get("ids", []))
            if isinstance(ids, str):
                ids = [ids]
            elif isinstance(ids, list) and len(ids) == 0:
                ids = ["A"]

            for chain_id in ids:
                chain_id = str(chain_id)
                if mol_type in ("protein", "rna", "dna"):
                    seq = info.get("sequence", "")
                    chains.append({
                        "type": mol_type,
                        "id": chain_id,
                        "sequence": seq,
                    })
                elif mol_type == "ligand":
                    ccd_codes = info.get("ccdCodes", info.get("ccd_codes", []))
                    if isinstance(ccd_codes, str):
                        ccd_codes = [ccd_codes]
                    if ccd_codes:
                        for ccd_code in ccd_codes:
                            chains.append({
                                "type": "ligand",
                                "id": chain_id,
                                "ccd": ccd_code,
                            })
    return chains


def structure_to_chains(structure: Structure) -> list[dict]:
    """Extract chain dicts from a parsed Structure (ground truth CIF).

    Extracts sequences and ligand CCD codes so we can re-predict from sequence.
    """
    THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    RNA_CODES = {"A", "C", "G", "U"}
    DNA_CODES = {"DA", "DC", "DG", "DT"}

    chains: list[dict] = []
    for chain in structure.chains:
        if chain.entity_type == "polymer" and chain.polymer_type.startswith("polypeptide"):
            seq = "".join(THREE_TO_ONE.get(r.name, "X") for r in chain.residues)
            if seq:
                chains.append({"type": "protein", "id": chain.chain_id, "sequence": seq})
        elif chain.entity_type == "polymer" and "ribonucleotide" in chain.polymer_type:
            is_dna = "deoxy" in chain.polymer_type
            if is_dna:
                seq = "".join(r.name[-1] if r.name in DNA_CODES else "N" for r in chain.residues)
                if seq:
                    chains.append({"type": "dna", "id": chain.chain_id, "sequence": seq})
            else:
                seq = "".join(r.name if r.name in RNA_CODES else "N" for r in chain.residues)
                if seq:
                    chains.append({"type": "rna", "id": chain.chain_id, "sequence": seq})
        elif chain.entity_type == "non-polymer":
            for res in chain.residues:
                chains.append({
                    "type": "ligand",
                    "id": chain.chain_id,
                    "ccd": res.name,
                })
    return chains


# ============================================================================
# Prediction pipeline
# ============================================================================

def predict_target(
    model: Helico,
    chains: list[dict],
    ccd: dict,
    target_name: str = "",
    n_samples: int = 5,
    max_tokens: int = 2048,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    msa_tar_indices: list[TarIndex] | None = None,
    msa_dir: Path | None = None,
    msa_server_url: str | None = None,
    msa_cache_dir: Path | None = None,
    n_cycles: int | None = None,
) -> tuple[TokenizedStructure, dict[str, torch.Tensor]] | None:
    """Run Helico inference on a target defined by chain dicts.

    Mirrors infer_main() logic from train.py.
    Returns (tokenized, results_dict) or None if target exceeds max_tokens.
    """
    tokenized = tokenize_sequences(chains, ccd)

    if tokenized.n_tokens > max_tokens:
        logger.warning(f"Target has {tokenized.n_tokens} tokens > max_tokens={max_tokens}, skipping")
        return None

    if tokenized.n_tokens == 0:
        logger.warning("Target produced 0 tokens, skipping")
        return None

    features = tokenized.to_features()

    # Add batch dimension
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    for key in ["n_tokens", "n_atoms"]:
        if key in batch and not isinstance(batch[key], torch.Tensor):
            batch[key] = torch.tensor([batch[key]])

    # Add masks
    n_tok = features["n_tokens"]
    n_atoms = features["n_atoms"]
    batch["token_mask"] = torch.ones(1, n_tok, dtype=torch.bool)
    batch["atom_mask"] = torch.ones(1, n_atoms, dtype=torch.bool)

    # Load MSA for the first polymer chain (same as training pipeline)
    msa_feat = None
    if msa_tar_indices or msa_dir:
        seen_chains = set()
        for chain_id, etype in zip(tokenized.chain_ids, tokenized.entity_types):
            if chain_id in seen_chains or etype != "protein":
                continue
            seen_chains.add(chain_id)
            seq = tokenized.chain_sequences.get(chain_id, "")
            if not seq:
                continue
            for tar_idx in (msa_tar_indices or []):
                msa_feat = load_msa_for_chain(
                    tokenized.pdb_id, chain_id,
                    sequence=seq,
                    tar_index=tar_idx,
                )
                if msa_feat is not None:
                    break
            if msa_feat is None and msa_dir:
                msa_feat = load_msa_for_chain(
                    tokenized.pdb_id, chain_id,
                    sequence=seq,
                    msa_dir=msa_dir,
                )
            if msa_feat is not None:
                break

    # Fallback: query MSA server if no pre-computed MSA found
    if msa_feat is None and msa_server_url:
        from helico.msa_server import run_mmseqs2
        from helico.data import parse_a3m, a3m_to_msa_matrix, compute_msa_features

        seen_chains = set()
        for chain_id, etype in zip(tokenized.chain_ids, tokenized.entity_types):
            if chain_id in seen_chains or etype != "protein":
                continue
            seen_chains.add(chain_id)
            seq = tokenized.chain_sequences.get(chain_id, "")
            if not seq:
                continue
            cache_dir = msa_cache_dir or Path("msa_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = target_name or tokenized.pdb_id
            try:
                a3m_lines = run_mmseqs2(
                    seq,
                    result_dir=str(cache_dir / f"{cache_name}_{chain_id}"),
                    host_url=msa_server_url,
                )
                if a3m_lines and a3m_lines[0].strip():
                    seqs, _ = parse_a3m(a3m_lines[0])
                    if seqs:
                        msa, dels = a3m_to_msa_matrix(seqs)
                        msa_feat = compute_msa_features(msa, dels)
                        logger.debug(f"MSA from server: {msa_feat.n_seqs} seqs")
            except Exception as e:
                logger.warning(f"MSA server error for {tokenized.pdb_id} chain {chain_id}: {e}")
            break

    if msa_feat is not None:
        logger.debug(f"MSA found: {msa_feat.n_seqs} seqs, length {msa_feat.length}")
        # MSA profile is (L, 32) for one chain; pad to (n_tok, 32)
        profile = torch.tensor(msa_feat.profile, dtype=torch.float32)
        cluster_msa = torch.tensor(msa_feat.cluster_msa, dtype=torch.long)
        cluster_profile = torch.tensor(msa_feat.cluster_profile, dtype=torch.float32)
        deletion_mean = torch.tensor(msa_feat.deletion_mean, dtype=torch.float32)
        cluster_deletion_mean = torch.tensor(msa_feat.cluster_deletion_mean, dtype=torch.float32)
        pad_len = n_tok - profile.shape[0]
        if pad_len > 0:
            profile = torch.nn.functional.pad(profile, (0, 0, 0, pad_len))
            cluster_msa = torch.nn.functional.pad(cluster_msa, (0, pad_len))
            cluster_profile = torch.nn.functional.pad(cluster_profile, (0, 0, 0, pad_len))
            deletion_mean = torch.nn.functional.pad(deletion_mean, (0, pad_len))
            cluster_deletion_mean = torch.nn.functional.pad(cluster_deletion_mean, (0, pad_len))
        batch["msa_profile"] = profile[:n_tok].unsqueeze(0)
        batch["cluster_msa"] = cluster_msa[:, :n_tok].unsqueeze(0)
        batch["cluster_profile"] = cluster_profile[:, :n_tok].unsqueeze(0)
        batch["deletion_mean"] = deletion_mean[:n_tok].unsqueeze(0)
        batch["cluster_deletion_mean"] = cluster_deletion_mean[:, :n_tok].unsqueeze(0)
        batch["has_msa"] = torch.ones(1)
    else:
        from helico.data import PROTENIX_NUM_MSA_CLASSES
        batch["msa_profile"] = torch.zeros(1, n_tok, PROTENIX_NUM_MSA_CLASSES)
        batch["cluster_msa"] = torch.zeros(1, 1, n_tok, dtype=torch.long)
        batch["cluster_profile"] = torch.zeros(1, 1, n_tok, PROTENIX_NUM_MSA_CLASSES)
        batch["deletion_mean"] = torch.zeros(1, n_tok)
        batch["cluster_deletion_mean"] = torch.zeros(1, 1, n_tok)
        batch["has_msa"] = torch.zeros(1)

    results = run_inference(model, batch, n_samples=n_samples, device=device, dtype=dtype, n_cycles=n_cycles)
    return tokenized, results


# ============================================================================
# Atom matching
# ============================================================================

@dataclass
class MatchedAtoms:
    """Paired predicted and ground-truth atom coordinates."""
    pred_coords: np.ndarray   # (N_matched, 3)
    gt_coords: np.ndarray     # (N_matched, 3)
    chain_ids: list[str]      # chain_id per matched atom
    res_seq_ids: list[int]    # residue seq_id per matched atom
    atom_names: list[str]     # atom name per matched atom
    elements: list[str]       # element symbol per matched atom
    entity_types: list[str]   # "protein", "nucleotide", "ligand" per matched atom


def match_atoms(
    tokenized: TokenizedStructure,
    pred_coords: np.ndarray,
    gt_structure: Structure,
) -> MatchedAtoms:
    """Match predicted atoms to ground truth by (chain_id, residue_seq_id, atom_name).

    For sequence-based predictions, token res_idx is 0-based sequential per chain.
    GT residues have seq_id from the CIF. We align by position within each chain.
    """
    # Build per-chain residue lists from ground truth
    gt_chain_residues: dict[str, list] = {}
    for chain in gt_structure.chains:
        gt_chain_residues[chain.chain_id] = chain.residues

    # Build ground truth lookup: (chain_id, position_in_chain, atom_name) -> coords
    # position_in_chain is 0-based index within the chain's residues
    gt_lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for chain in gt_structure.chains:
        for res_pos, res in enumerate(chain.residues):
            for atom in res.atoms:
                if atom.element == "H":
                    continue
                key = (chain.chain_id, res_pos, atom.name)
                gt_lookup[key] = atom.coords

    # Track per-chain token position to align with GT residue position
    chain_token_counts: dict[str, int] = {}

    matched_pred = []
    matched_gt = []
    chain_ids = []
    res_seq_ids = []
    atom_names = []
    elements = []
    entity_types = []

    atom_offset = 0
    for tok_idx, token in enumerate(tokenized.tokens):
        chain_id = tokenized.chain_ids[tok_idx]
        etype = tokenized.entity_types[tok_idx]

        # Track position within this chain
        if chain_id not in chain_token_counts:
            chain_token_counts[chain_id] = 0
        pos_in_chain = chain_token_counts[chain_id]
        chain_token_counts[chain_id] += 1

        for ai, aname in enumerate(token.atom_names):
            global_ai = atom_offset + ai
            if global_ai >= len(pred_coords):
                break

            gt_key = (chain_id, pos_in_chain, aname)
            if gt_key in gt_lookup:
                matched_pred.append(pred_coords[global_ai])
                matched_gt.append(gt_lookup[gt_key])
                chain_ids.append(chain_id)
                res_seq_ids.append(pos_in_chain)
                atom_names.append(aname)
                elements.append(token.atom_elements[ai] if ai < len(token.atom_elements) else "")
                entity_types.append(etype)

        atom_offset += len(token.atom_names)

    if not matched_pred:
        return MatchedAtoms(
            pred_coords=np.zeros((0, 3), dtype=np.float32),
            gt_coords=np.zeros((0, 3), dtype=np.float32),
            chain_ids=[], res_seq_ids=[], atom_names=[], elements=[], entity_types=[],
        )

    return MatchedAtoms(
        pred_coords=np.array(matched_pred, dtype=np.float32),
        gt_coords=np.array(matched_gt, dtype=np.float32),
        chain_ids=chain_ids,
        res_seq_ids=res_seq_ids,
        atom_names=atom_names,
        elements=elements,
        entity_types=entity_types,
    )


def extract_backbone_coords(matched: MatchedAtoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CA (protein) or C3' (nucleic acid) backbone atoms for TM-score/GDT-TS.

    Returns (pred_bb, gt_bb, mask) where mask is boolean.
    """
    backbone_names = {"CA", "C3'"}
    mask = np.array([n in backbone_names for n in matched.atom_names], dtype=bool)
    return matched.pred_coords[mask], matched.gt_coords[mask], mask


# ============================================================================
# Metrics
# ============================================================================

def compute_lddt(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    cutoff: float = 15.0,
) -> float:
    """Compute local Distance Difference Test (lDDT).

    Uses hard thresholds at 0.5, 1.0, 2.0, 4.0 Angstroms with 15A inclusion cutoff.
    """
    if len(pred_coords) < 2:
        return 0.0

    pred_dists = np.linalg.norm(pred_coords[:, None] - pred_coords[None, :], axis=-1)
    gt_dists = np.linalg.norm(gt_coords[:, None] - gt_coords[None, :], axis=-1)

    mask = (gt_dists < cutoff) & (gt_dists > 0.01)

    if not mask.any():
        return 0.0

    diff = np.abs(pred_dists - gt_dists)

    thresholds = [0.5, 1.0, 2.0, 4.0]
    score = 0.0
    for t in thresholds:
        score += (diff[mask] < t).mean()
    score /= len(thresholds)

    return float(score)


def compute_tm_score(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute TM-score using tmtools package."""
    try:
        import tmtools
    except ImportError:
        logger.warning("tmtools not installed, skipping TM-score")
        return float("nan")

    if len(pred_coords) < 3:
        return 0.0

    # tmtools requires float64 arrays
    pred_f64 = pred_coords.astype(np.float64)
    gt_f64 = gt_coords.astype(np.float64)
    seq_dummy = "A" * len(pred_coords)
    result = tmtools.tm_align(pred_f64, gt_f64, seq_dummy, seq_dummy)
    return float(result.tm_norm_chain2)


def _kabsch_superpose(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """Superpose pred onto gt using Kabsch algorithm.

    Returns (superposed_pred, rmsd).
    """
    if len(pred) < 3:
        diff = pred - gt
        rmsd = float(np.sqrt((diff ** 2).sum() / max(len(pred), 1)))
        return pred, rmsd

    pred_center = pred.mean(axis=0)
    gt_center = gt.mean(axis=0)
    pred_centered = pred - pred_center
    gt_centered = gt - gt_center

    rot, rssd = Rotation.align_vectors(gt_centered, pred_centered)
    pred_rotated = rot.apply(pred_centered) + gt_center

    diff = pred_rotated - gt
    rmsd = float(np.sqrt((diff ** 2).mean()))
    return pred_rotated, rmsd


def compute_rmsd(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute RMSD after Kabsch superposition."""
    if len(pred_coords) == 0:
        return float("nan")
    _, rmsd = _kabsch_superpose(pred_coords, gt_coords)
    return rmsd


def compute_gdt_ts(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute GDT-TS: fraction of atoms within 1/2/4/8 A after superposition."""
    if len(pred_coords) < 3:
        return 0.0

    superposed, _ = _kabsch_superpose(pred_coords, gt_coords)
    dists = np.linalg.norm(superposed - gt_coords, axis=-1)

    thresholds = [1.0, 2.0, 4.0, 8.0]
    score = sum((dists < t).mean() for t in thresholds) / len(thresholds)
    return float(score)


def compute_dockq(
    pred_pdb_str: str,
    gt_cif_path: Path,
    chain_ids: list[str] | None = None,
) -> dict[str, float]:
    """Compute DockQ score using the DockQ package."""
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError:
        logger.warning("DockQ not installed, skipping interface scoring")
        return {"dockq": float("nan"), "irmsd": float("nan"), "lrmsd": float("nan"), "fnat": float("nan")}

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pred_pdb_str)
        pred_path = f.name

    try:
        model_struct = load_PDB(pred_path)
        native_struct = load_PDB(str(gt_cif_path))

        # Build chain map: identity mapping for chains present in both
        model_chain_ids = {c.id for c in model_struct.get_chains()}
        native_chain_ids = {c.id for c in native_struct.get_chains()}
        common_chains = model_chain_ids & native_chain_ids
        if not common_chains or len(common_chains) < 2:
            return {"dockq": 0.0, "irmsd": float("nan"), "lrmsd": float("nan"), "fnat": 0.0}

        chain_map = {c: c for c in sorted(common_chains)}

        result_mapping, total_dockq = run_on_all_native_interfaces(
            model_struct, native_struct, chain_map=chain_map
        )
        if not result_mapping:
            return {"dockq": 0.0, "irmsd": float("nan"), "lrmsd": float("nan"), "fnat": 0.0}

        dockqs = []
        irmsds = []
        lrmsds = []
        fnats = []
        for interface_id, interface_result in result_mapping.items():
            dockqs.append(interface_result.get("DockQ", 0.0))
            irmsds.append(interface_result.get("iRMS", float("nan")))
            lrmsds.append(interface_result.get("LRMS", float("nan")))
            fnats.append(interface_result.get("fnat", 0.0))

        return {
            "dockq": float(np.nanmean(dockqs)) if dockqs else 0.0,
            "irmsd": float(np.nanmean(irmsds)) if irmsds else float("nan"),
            "lrmsd": float(np.nanmean(lrmsds)) if lrmsds else float("nan"),
            "fnat": float(np.nanmean(fnats)) if fnats else 0.0,
        }
    except Exception as e:
        logger.warning(f"DockQ failed: {e}")
        return {"dockq": float("nan"), "irmsd": float("nan"), "lrmsd": float("nan"), "fnat": float("nan")}
    finally:
        os.unlink(pred_path)


def compute_lddt_pli(matched: MatchedAtoms, cutoff: float = 15.0) -> float:
    """Compute LDDT restricted to protein-ligand cross-boundary pairs."""
    protein_mask = np.array([e == "protein" for e in matched.entity_types], dtype=bool)
    ligand_mask = np.array([e == "ligand" for e in matched.entity_types], dtype=bool)

    if protein_mask.sum() == 0 or ligand_mask.sum() == 0:
        return float("nan")

    pred_dists = np.linalg.norm(
        matched.pred_coords[:, None] - matched.pred_coords[None, :], axis=-1
    )
    gt_dists = np.linalg.norm(
        matched.gt_coords[:, None] - matched.gt_coords[None, :], axis=-1
    )

    cross_mask = (protein_mask[:, None] & ligand_mask[None, :]) | (ligand_mask[:, None] & protein_mask[None, :])
    pair_mask = cross_mask & (gt_dists < cutoff) & (gt_dists > 0.01)

    if not pair_mask.any():
        return float("nan")

    diff = np.abs(pred_dists - gt_dists)
    thresholds = [0.5, 1.0, 2.0, 4.0]
    score = sum((diff[pair_mask] < t).mean() for t in thresholds) / len(thresholds)
    return float(score)


def compute_ligand_rmsd(matched: MatchedAtoms) -> float:
    """Compute RMSD of ligand atoms after superposing on receptor (protein) atoms."""
    protein_mask = np.array([e == "protein" for e in matched.entity_types], dtype=bool)
    ligand_mask = np.array([e == "ligand" for e in matched.entity_types], dtype=bool)

    if protein_mask.sum() < 3 or ligand_mask.sum() == 0:
        return float("nan")

    pred_protein = matched.pred_coords[protein_mask]
    gt_protein = matched.gt_coords[protein_mask]

    pred_center = pred_protein.mean(axis=0)
    gt_center = gt_protein.mean(axis=0)
    pred_c = pred_protein - pred_center
    gt_c = gt_protein - gt_center

    rot, _ = Rotation.align_vectors(gt_c, pred_c)

    pred_ligand = matched.pred_coords[ligand_mask]
    gt_ligand = matched.gt_coords[ligand_mask]

    pred_ligand_aligned = rot.apply(pred_ligand - pred_center) + gt_center
    diff = pred_ligand_aligned - gt_ligand
    return float(np.sqrt((diff ** 2).sum(axis=-1).mean()))


# ============================================================================
# Scoring functions
# ============================================================================

def score_monomer(matched: MatchedAtoms) -> dict[str, float]:
    """Score a monomer prediction."""
    lddt = compute_lddt(matched.pred_coords, matched.gt_coords)
    pred_bb, gt_bb, _ = extract_backbone_coords(matched)
    tm_score = compute_tm_score(pred_bb, gt_bb)
    gdt_ts = compute_gdt_ts(pred_bb, gt_bb)
    rmsd = compute_rmsd(pred_bb, gt_bb)
    return {"lddt": lddt, "tm_score": tm_score, "gdt_ts": gdt_ts, "rmsd": rmsd}


def score_interface(
    pred_pdb_str: str,
    gt_cif_path: Path,
    matched: MatchedAtoms,
) -> dict[str, float]:
    """Score an interface prediction."""
    lddt = compute_lddt(matched.pred_coords, matched.gt_coords)
    dockq_results = compute_dockq(pred_pdb_str, gt_cif_path)
    return {"lddt": lddt, **dockq_results}


def score_ligand_interface(matched: MatchedAtoms) -> dict[str, float]:
    """Score a protein-ligand interface prediction."""
    lddt = compute_lddt(matched.pred_coords, matched.gt_coords)
    lddt_pli = compute_lddt_pli(matched)
    lrmsd = compute_ligand_rmsd(matched)
    return {"lddt": lddt, "lddt_pli": lddt_pli, "lrmsd": lrmsd}


# ============================================================================
# Results output
# ============================================================================

def write_category_csv(results: list[dict], output_path: Path):
    """Write per-target results for a category to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(category_summaries: list[dict]):
    """Print a summary table to stdout."""
    header = f"{'Category':<35} | {'N':>4} | {'Predicted':>9} | {'Success%':>8} | {'Mean LDDT':>9} | {'Mean DockQ':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in category_summaries:
        dockq_str = f"{s['mean_dockq']:.2f}" if not np.isnan(s["mean_dockq"]) else "-"
        success_str = f"{s['success_pct']:.1f}%" if not np.isnan(s["success_pct"]) else "-"
        print(
            f"{s['category']:<35} | {s['n_total']:>4} | {s['n_predicted']:>9} | "
            f"{success_str:>8} | {s['mean_lddt']:>9.2f} | {dockq_str:>10}"
        )
    print("=" * len(header) + "\n")


def write_summary_csv(category_summaries: list[dict], output_path: Path):
    """Write summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not category_summaries:
        return
    fieldnames = list(category_summaries[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(category_summaries)


# ============================================================================
# Main benchmark loop
# ============================================================================

def run_benchmark(
    model: Helico,
    foldbench_dir: Path,
    output_dir: Path,
    ccd: dict,
    categories: list[str] | None = None,
    n_samples: int = 5,
    max_tokens: int = 2048,
    resume: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    msa_tar_indices: list[TarIndex] | None = None,
    msa_dir: Path | None = None,
    msa_server_url: str | None = None,
    n_cycles: int | None = None,
):
    """Run the full FoldBench benchmark.

    Directory layout expected:
        foldbench_dir/
            targets/           # CSV files per category
            ground_truths/     # CIF files per target
            alphafold3_inputs.json  # (optional) AF3-format inputs
    """
    from tqdm import tqdm

    targets_dir = foldbench_dir / "targets"
    gt_dir = foldbench_dir / "ground_truths"

    # Load AF3 inputs if available
    af3_inputs: dict[str, dict] = {}
    af3_json_path = foldbench_dir / "alphafold3_inputs.json"
    if af3_json_path.exists():
        af3_inputs = load_af3_inputs(af3_json_path)
        logger.info(f"Loaded {len(af3_inputs)} AF3 inputs from {af3_json_path}")

    all_targets = load_targets(targets_dir)

    if categories:
        all_targets = {k: v for k, v in all_targets.items() if k in categories}

    results_dir = output_dir / "results"
    predictions_dir = output_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    category_summaries = []

    for category, targets in all_targets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Category: {category} ({len(targets)} targets)")
        logger.info(f"{'='*60}")

        is_interface = category in INTERFACE_CATEGORIES
        is_ligand = category == "interface_protein_ligand"

        category_results = []
        n_predicted = 0
        n_success = 0

        for target in tqdm(targets, desc=category):
            pdb_id = target.pdb_id
            result_row = {"pdb_id": pdb_id, "status": "failed"}

            # Check for cached prediction
            pred_cache_path = predictions_dir / f"{pdb_id}.pkl"
            cached = None
            if resume and pred_cache_path.exists():
                try:
                    with open(pred_cache_path, "rb") as f:
                        cached = pickle.load(f)
                    tokenized = cached["tokenized"]
                    pred_coords_np = cached["pred_coords"]
                    plddt_np = cached["plddt"]
                    pred_pdb_str = cached.get("pdb_str", "")
                except Exception:
                    logger.warning(f"Failed to load cache for {pdb_id}, re-predicting")
                    cached = None

            try:
                if cached is None:
                    # Build chain dicts: prefer AF3 JSON, fall back to GT CIF
                    chains = None
                    if pdb_id in af3_inputs:
                        chains = af3_entry_to_chains(af3_inputs[pdb_id])

                    if not chains:
                        # Extract from ground truth CIF
                        gt_path = gt_dir / f"{pdb_id}.cif"
                        if not gt_path.exists():
                            logger.warning(f"No AF3 input and no GT CIF for {pdb_id}")
                            result_row["status"] = "no_input"
                            category_results.append(result_row)
                            continue
                        gt_for_chains = parse_mmcif(gt_path, max_resolution=float("inf"))
                        if gt_for_chains is None:
                            result_row["status"] = "gt_parse_failed"
                            category_results.append(result_row)
                            continue
                        chains = structure_to_chains(gt_for_chains)

                    if not chains:
                        logger.warning(f"No chains for {pdb_id}")
                        result_row["status"] = "no_chains"
                        category_results.append(result_row)
                        continue

                    # Predict
                    pred_result = predict_target(
                        model, chains, ccd,
                        target_name=pdb_id,
                        n_samples=n_samples,
                        max_tokens=max_tokens,
                        device=device,
                        dtype=dtype,
                        msa_tar_indices=msa_tar_indices,
                        msa_dir=msa_dir,
                        msa_server_url=msa_server_url,
                        msa_cache_dir=output_dir / "msa_cache",
                        n_cycles=n_cycles,
                    )

                    if pred_result is None:
                        result_row["status"] = "too_large"
                        category_results.append(result_row)
                        continue

                    tokenized, results = pred_result
                    pred_coords_np = results["coords"][0].cpu().float().numpy()
                    plddt_np = results["plddt"][0].cpu().float().numpy()

                    # Generate PDB string for DockQ
                    pred_pdb_str = coords_to_pdb(results["coords"][0], results["plddt"][0], tokenized)

                    # Cache prediction
                    with open(pred_cache_path, "wb") as f:
                        pickle.dump({
                            "tokenized": tokenized,
                            "pred_coords": pred_coords_np,
                            "plddt": plddt_np,
                            "pdb_str": pred_pdb_str,
                        }, f)

                    torch.cuda.empty_cache()

                # Load ground truth for scoring
                gt_path = gt_dir / f"{pdb_id}.cif"
                if not gt_path.exists():
                    gt_path = gt_dir / f"{pdb_id}.cif.gz"
                if not gt_path.exists():
                    logger.warning(f"Ground truth not found for {pdb_id}")
                    result_row["status"] = "no_gt"
                    category_results.append(result_row)
                    continue

                gt_structure = parse_mmcif(gt_path, max_resolution=float("inf"))
                if gt_structure is None:
                    logger.warning(f"Failed to parse ground truth: {gt_path}")
                    result_row["status"] = "gt_parse_failed"
                    category_results.append(result_row)
                    continue

                # Match atoms
                matched = match_atoms(tokenized, pred_coords_np, gt_structure)
                if len(matched.pred_coords) == 0:
                    logger.warning(f"No atoms matched for {pdb_id}")
                    result_row["status"] = "no_match"
                    category_results.append(result_row)
                    continue

                # Score
                n_predicted += 1
                if is_ligand:
                    scores = score_ligand_interface(matched)
                    success = (
                        not np.isnan(scores.get("lrmsd", float("nan")))
                        and scores["lrmsd"] < 2.0
                        and not np.isnan(scores.get("lddt_pli", float("nan")))
                        and scores["lddt_pli"] > 0.8
                    )
                elif is_interface:
                    scores = score_interface(pred_pdb_str, gt_path, matched)
                    success = scores.get("dockq", 0.0) >= 0.23
                else:
                    scores = score_monomer(matched)
                    success = False  # No success criterion for monomers

                if success:
                    n_success += 1

                result_row["status"] = "ok"
                result_row["n_matched_atoms"] = len(matched.pred_coords)
                result_row.update(scores)
                category_results.append(result_row)

                logger.info(
                    f"  {pdb_id}: "
                    + " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in scores.items())
                )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on {pdb_id}, skipping")
                    result_row["status"] = "oom"
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"RuntimeError on {pdb_id}: {e}")
                    result_row["status"] = "error"
                category_results.append(result_row)
            except Exception as e:
                logger.error(f"Error on {pdb_id}: {e}")
                result_row["status"] = "error"
                category_results.append(result_row)

        # Write category results
        write_category_csv(category_results, results_dir / f"{category}.csv")

        # Compute category summary
        ok_results = [r for r in category_results if r["status"] == "ok"]
        mean_lddt = float(np.mean([r["lddt"] for r in ok_results])) if ok_results else 0.0
        mean_dockq = float("nan")
        if is_interface and not is_ligand and ok_results:
            dockq_vals = [r.get("dockq", float("nan")) for r in ok_results]
            dockq_vals = [v for v in dockq_vals if not np.isnan(v)]
            mean_dockq = float(np.mean(dockq_vals)) if dockq_vals else float("nan")

        success_pct = float("nan")
        if is_interface or is_ligand:
            success_pct = 100.0 * n_success / max(n_predicted, 1) if n_predicted > 0 else 0.0

        summary = {
            "category": category,
            "n_total": len(targets),
            "n_predicted": n_predicted,
            "success_pct": success_pct,
            "mean_lddt": mean_lddt,
            "mean_dockq": mean_dockq,
        }
        category_summaries.append(summary)

    # Print and write summary
    print_summary(category_summaries)
    write_summary_csv(category_summaries, output_dir / "summary.csv")
    logger.info(f"Results written to {output_dir}")


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Helico FoldBench Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Helico checkpoint")
    parser.add_argument("--protenix", type=str, default=None, help="Path to Protenix checkpoint (.pt)")
    parser.add_argument("--foldbench-dir", type=str, required=True, help="Path to FoldBench directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category names (default: all)")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of diffusion samples")
    parser.add_argument("--ccd", type=str, default=None, help="Path to CCD cache pickle")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per target (skip larger)")
    parser.add_argument("--resume", action="store_true", help="Resume from cached predictions")
    parser.add_argument("--msa-tar", type=str, nargs="+", default=None,
                        help="Path(s) to MSA tar archives (with corresponding .pkl index files)")
    parser.add_argument("--msa-dir", type=str, default=None,
                        help="Path to extracted MSA directory (a3m.gz files)")
    parser.add_argument("--use-msa-server", action="store_true",
                        help="Generate MSA using the public ColabFold MMseqs2 server (fallback when tar/dir miss)")
    parser.add_argument("--msa-server-url", type=str, default="https://api.colabfold.com",
                        help="MMseqs2 server URL (default: https://api.colabfold.com)")
    parser.add_argument("--n-cycles", type=int, default=10,
                        help="Number of recycling cycles (default: 10, matching Protenix)")
    args = parser.parse_args()

    if args.checkpoint is None and args.protenix is None:
        parser.error("Must specify either --checkpoint or --protenix")

    # Load model
    if args.protenix is not None:
        from collections import OrderedDict
        from helico.load_protenix import load_protenix_state_dict
        config = HelicoConfig()
        model = Helico(config)
        ckpt = torch.load(args.protenix, map_location="cpu", weights_only=False)
        ptx_sd = ckpt["model"]
        ptx_sd = OrderedDict((k.removeprefix("module."), v) for k, v in ptx_sd.items())
        stats = load_protenix_state_dict(ptx_sd, model)
        logger.info(f"Loaded Protenix checkpoint: {stats['n_transferred']} params transferred")
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = HelicoConfig(**{k: v for k, v in state.get("config", {}).items() if hasattr(HelicoConfig, k)})
        model = Helico(config)
        model.load_state_dict(state["model_state_dict"])

    # Load CCD
    ccd_cache = Path(args.ccd) if args.ccd else None
    ccd = parse_ccd(cache_path=ccd_cache)
    logger.info(f"CCD loaded with {len(ccd)} components")

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Load MSA tar indices
    msa_tar_indices = []
    if args.msa_tar:
        for tar_path_str in args.msa_tar:
            tar_path = Path(tar_path_str).resolve()
            # Look for the pre-built index pickle
            # Convention: index is in processed dir as <stem>_index.pkl
            stem = tar_path.stem  # e.g. "rcsb_raw_msa" from "rcsb_raw_msa.tar"
            index_path = tar_path.with_name(stem + "_index.pkl")
            if not index_path.exists():
                processed_dir = Path(os.environ.get("HELICO_PROCESSED_DIR", ""))
                if processed_dir.exists():
                    index_path = processed_dir / (stem + "_index.pkl")
            if index_path.exists():
                with open(index_path, "rb") as f:
                    tar_index = pickle.load(f)
                # Override tar_path in case index was built on a different machine
                tar_index.tar_path = tar_path
                logger.info(f"Loaded MSA tar index: {index_path} ({len(tar_index.entries)} entries)")
                msa_tar_indices.append(tar_index)
            else:
                logger.warning(f"No tar index found for {tar_path}, skipping")

    msa_dir = Path(args.msa_dir) if args.msa_dir else None

    # Run benchmark
    run_benchmark(
        model=model,
        foldbench_dir=Path(args.foldbench_dir),
        output_dir=Path(args.output_dir),
        ccd=ccd,
        categories=categories,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        resume=args.resume,
        msa_tar_indices=msa_tar_indices if msa_tar_indices else None,
        msa_dir=msa_dir,
        msa_server_url=args.msa_server_url if args.use_msa_server else None,
        n_cycles=args.n_cycles,
    )


if __name__ == "__main__":
    main()
