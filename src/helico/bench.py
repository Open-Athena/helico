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
    TokenizedStructure,
    parse_ccd,
    parse_mmcif,
    tokenize_sequences,
)
from helico.model import Helico, HelicoConfig
from helico.train import coords_to_pdb, run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# FoldBench categories
# ============================================================================

INTERFACE_CATEGORIES = [
    "interface_protein_protein",
    "interface_protein_nucleic_acid",
    "interface_protein_ligand",
    "interface_protein_peptide",
    "interface_antibody_protein",
    "interface_protein_metal_ion",
]

MONOMER_CATEGORIES = [
    "monomer_protein",
    "monomer_rna",
    "monomer_peptide",
]

ALL_CATEGORIES = INTERFACE_CATEGORIES + MONOMER_CATEGORIES


# ============================================================================
# Data loading
# ============================================================================

@dataclass
class BenchTarget:
    """A single FoldBench benchmark target."""
    target_id: str
    category: str
    pdb_id: str
    chains: str  # chain info from CSV
    extra: dict = field(default_factory=dict)


def load_targets(targets_dir: Path) -> dict[str, list[BenchTarget]]:
    """Parse 9 category CSVs from the FoldBench targets/ directory.

    Returns dict mapping category name -> list of BenchTarget.
    """
    results: dict[str, list[BenchTarget]] = {}
    for csv_path in sorted(targets_dir.glob("*.csv")):
        category = csv_path.stem
        targets = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                target_id = row.get("id") or row.get("target_id") or row.get("name", "")
                pdb_id = row.get("pdb_id") or row.get("pdb", "")
                chains = row.get("chains") or row.get("chain_ids", "")
                targets.append(BenchTarget(
                    target_id=target_id,
                    category=category,
                    pdb_id=pdb_id,
                    chains=chains,
                    extra=dict(row),
                ))
        results[category] = targets
        logger.info(f"Loaded {len(targets)} targets for {category}")
    return results


def af3_json_to_chains(json_path: Path) -> list[dict]:
    """Convert an AF3-style input JSON to Helico chain dicts.

    Handles:
    - "id" as list for homomers: {"protein": {"id": ["A","B"], "sequence": "..."}}
    - protein: {"type": "protein", "id": ..., "sequence": ...}
    - rna/dna: {"type": "rna"/"dna", "id": ..., "sequence": ...}
    - ligand with CCD codes: {"type": "ligand", "id": ..., "ccd": "ATP"}
    """
    with open(json_path) as f:
        data = json.load(f)

    sequences = data if isinstance(data, list) else data.get("sequences", data.get("modelSeeds", data))
    if not isinstance(sequences, list):
        sequences = [data]

    chains: list[dict] = []
    for entry in sequences:
        for mol_type in ("protein", "rna", "dna", "ligand"):
            if mol_type not in entry:
                continue
            info = entry[mol_type]

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
                    smiles = info.get("smiles", "")
                    if isinstance(ccd_codes, str):
                        ccd_codes = [ccd_codes]
                    if ccd_codes:
                        for ccd_code in ccd_codes:
                            chains.append({
                                "type": "ligand",
                                "id": chain_id,
                                "ccd": ccd_code,
                            })
                    elif smiles:
                        logger.warning(f"SMILES ligands not supported, skipping chain {chain_id}")

    return chains


# ============================================================================
# Prediction pipeline
# ============================================================================

def predict_target(
    model: Helico,
    chains: list[dict],
    ccd: dict,
    n_samples: int = 5,
    max_tokens: int = 2048,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
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

    # Add empty MSA features
    if "msa_profile" not in batch:
        batch["msa_profile"] = torch.zeros(1, n_tok, 22)
        batch["cluster_msa"] = torch.zeros(1, 1, n_tok, dtype=torch.long)
        batch["cluster_profile"] = torch.zeros(1, 1, n_tok, 22)
        batch["has_msa"] = torch.zeros(1)

    results = run_inference(model, batch, n_samples=n_samples, device=device, dtype=dtype)
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
    gt_structure,
) -> MatchedAtoms:
    """Match predicted atoms to ground truth by (chain_id, residue_seq_id, atom_name).

    Args:
        tokenized: TokenizedStructure from prediction input
        pred_coords: (N_atoms, 3) predicted coordinates
        gt_structure: parsed Structure from ground truth CIF

    Returns:
        MatchedAtoms with parallel arrays of matched coordinates.
    """
    # Build ground truth lookup: (chain_id, seq_id, atom_name) -> coords
    gt_lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for chain in gt_structure.chains:
        for res in chain.residues:
            for atom in res.atoms:
                if atom.element == "H":
                    continue
                key = (chain.chain_id, res.seq_id, atom.name)
                gt_lookup[key] = atom.coords

    # Walk through tokenized structure and match
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
        res_idx = token.res_idx

        # For sequence-based prediction, res_idx is 0-based within chain.
        # We need to map to gt seq_id. Build chain residue list from gt.
        for ai, aname in enumerate(token.atom_names):
            global_ai = atom_offset + ai
            if global_ai >= len(pred_coords):
                break

            # Try matching by chain_id + res_idx (1-based in gt) + atom_name
            # FoldBench ground truths use label_asym_id as chain_id
            # res_idx in tokenized is 0-based sequential, gt seq_id is 1-based
            gt_key = (chain_id, res_idx + 1, aname)
            if gt_key in gt_lookup:
                matched_pred.append(pred_coords[global_ai])
                matched_gt.append(gt_lookup[gt_key])
                chain_ids.append(chain_id)
                res_seq_ids.append(res_idx + 1)
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
    Reference: smooth_lddt_loss in model.py:1723 but with hard step functions.
    """
    if len(pred_coords) < 2:
        return 0.0

    # Pairwise distances
    pred_dists = np.linalg.norm(pred_coords[:, None] - pred_coords[None, :], axis=-1)
    gt_dists = np.linalg.norm(gt_coords[:, None] - gt_coords[None, :], axis=-1)

    # Mask: pairs within cutoff in ground truth, excluding self
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


def compute_lddt_per_chain(
    matched: MatchedAtoms,
    chain_id: str,
    cutoff: float = 15.0,
) -> float:
    """Compute LDDT restricted to atoms in a specific chain."""
    mask = np.array([c == chain_id for c in matched.chain_ids], dtype=bool)
    if mask.sum() < 2:
        return 0.0
    return compute_lddt(matched.pred_coords[mask], matched.gt_coords[mask], cutoff)


def compute_tm_score(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute TM-score using tmtools package."""
    try:
        import tmtools
    except ImportError:
        logger.warning("tmtools not installed, skipping TM-score")
        return float("nan")

    if len(pred_coords) < 3:
        return 0.0

    result = tmtools.tm_align(pred_coords, gt_coords, pred_coords, gt_coords)
    return float(result.tm_norm_chain2)


def _kabsch_superpose(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """Superpose pred onto gt using Kabsch algorithm.

    Returns (superposed_pred, rmsd).
    """
    if len(pred) < 3:
        diff = pred - gt
        rmsd = float(np.sqrt((diff ** 2).sum() / max(len(pred), 1)))
        return pred, rmsd

    # Center
    pred_center = pred.mean(axis=0)
    gt_center = gt.mean(axis=0)
    pred_centered = pred - pred_center
    gt_centered = gt - gt_center

    # Use scipy for Kabsch
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
) -> dict[str, float]:
    """Compute DockQ score using the DockQ package.

    Writes pred to temp PDB, runs DockQ against ground truth.
    Returns dict with dockq, irmsd, lrmsd, fnat.
    """
    try:
        from DockQ.DockQ import run_on_all_native_interfaces
    except ImportError:
        logger.warning("DockQ not installed, skipping interface scoring")
        return {"dockq": float("nan"), "irmsd": float("nan"), "lrmsd": float("nan"), "fnat": float("nan")}

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pred_pdb_str)
        pred_path = f.name

    try:
        result = run_on_all_native_interfaces(pred_path, str(gt_cif_path))
        if not result:
            return {"dockq": 0.0, "irmsd": float("nan"), "lrmsd": float("nan"), "fnat": 0.0}

        # Average over all interfaces
        dockqs = []
        irmsds = []
        lrmsds = []
        fnats = []
        for interface_id, interface_result in result.items():
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
    finally:
        os.unlink(pred_path)


def compute_lddt_pli(matched: MatchedAtoms, cutoff: float = 15.0) -> float:
    """Compute LDDT restricted to protein-ligand cross-boundary pairs."""
    protein_mask = np.array([e == "protein" for e in matched.entity_types], dtype=bool)
    ligand_mask = np.array([e == "ligand" for e in matched.entity_types], dtype=bool)

    if protein_mask.sum() == 0 or ligand_mask.sum() == 0:
        return float("nan")

    n = len(matched.pred_coords)
    pred_dists = np.linalg.norm(
        matched.pred_coords[:, None] - matched.pred_coords[None, :], axis=-1
    )
    gt_dists = np.linalg.norm(
        matched.gt_coords[:, None] - matched.gt_coords[None, :], axis=-1
    )

    # Cross-boundary mask: one atom protein, other atom ligand
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

    # Superpose on protein
    pred_protein = matched.pred_coords[protein_mask]
    gt_protein = matched.gt_coords[protein_mask]

    pred_center = pred_protein.mean(axis=0)
    gt_center = gt_protein.mean(axis=0)
    pred_c = pred_protein - pred_center
    gt_c = gt_protein - gt_center

    rot, _ = Rotation.align_vectors(gt_c, pred_c)

    # Apply same transform to ligand
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

def write_category_csv(
    results: list[dict],
    output_path: Path,
):
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
    header = f"{'Category':<40} | {'N':>4} | {'Predicted':>9} | {'Success%':>8} | {'Mean LDDT':>9} | {'Mean DockQ':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in category_summaries:
        dockq_str = f"{s['mean_dockq']:.2f}" if not np.isnan(s["mean_dockq"]) else "-"
        success_str = f"{s['success_pct']:.1f}%" if not np.isnan(s["success_pct"]) else "-"
        print(
            f"{s['category']:<40} | {s['n_total']:>4} | {s['n_predicted']:>9} | "
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
):
    """Run the full FoldBench benchmark."""
    from tqdm import tqdm

    targets_dir = foldbench_dir / "targets"
    inputs_dir = foldbench_dir / "inputs"
    gt_dir = foldbench_dir / "ground_truths"

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
            target_id = target.target_id
            result_row = {"target_id": target_id, "pdb_id": target.pdb_id, "status": "failed"}

            # Check for cached prediction
            pred_cache = predictions_dir / f"{target_id}.pkl"
            if resume and pred_cache.exists():
                try:
                    with open(pred_cache, "rb") as f:
                        cached = pickle.load(f)
                    tokenized = cached["tokenized"]
                    pred_coords_np = cached["pred_coords"]
                    plddt_np = cached["plddt"]
                    pred_pdb_str = cached.get("pdb_str", "")
                    logger.debug(f"Loaded cached prediction for {target_id}")
                except Exception:
                    logger.warning(f"Failed to load cache for {target_id}, re-predicting")
                    pred_cache = None
            else:
                pred_cache = None

            try:
                if pred_cache is None:
                    # Load input
                    json_path = inputs_dir / f"{target_id}.json"
                    if not json_path.exists():
                        logger.warning(f"Input not found: {json_path}")
                        result_row["status"] = "no_input"
                        category_results.append(result_row)
                        continue

                    chains = af3_json_to_chains(json_path)
                    if not chains:
                        logger.warning(f"No chains parsed from {json_path}")
                        result_row["status"] = "no_chains"
                        category_results.append(result_row)
                        continue

                    # Predict
                    pred_result = predict_target(
                        model, chains, ccd,
                        n_samples=n_samples,
                        max_tokens=max_tokens,
                        device=device,
                        dtype=dtype,
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
                    with open(predictions_dir / f"{target_id}.pkl", "wb") as f:
                        pickle.dump({
                            "tokenized": tokenized,
                            "pred_coords": pred_coords_np,
                            "plddt": plddt_np,
                            "pdb_str": pred_pdb_str,
                        }, f)

                    torch.cuda.empty_cache()

                # Load ground truth
                gt_path = gt_dir / f"{target_id}.cif"
                if not gt_path.exists():
                    gt_path = gt_dir / f"{target_id}.cif.gz"
                if not gt_path.exists():
                    logger.warning(f"Ground truth not found: {gt_dir / target_id}.*")
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
                    logger.warning(f"No atoms matched for {target_id}")
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
                    f"  {target_id}: "
                    + " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in scores.items())
                )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on {target_id}, skipping")
                    result_row["status"] = "oom"
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"RuntimeError on {target_id}: {e}")
                    result_row["status"] = "error"
                category_results.append(result_row)
            except Exception as e:
                logger.error(f"Error on {target_id}: {e}")
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
    )


if __name__ == "__main__":
    main()
