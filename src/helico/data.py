"""Helico data pipeline: CCD parsing, mmCIF parsing, tokenization, MSA features, cropping."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _env_path(var: str) -> Path | None:
    v = os.environ.get(var)
    return Path(v) if v else None

RAW_DIR = _env_path("HELICO_RAW_DIR")
PROCESSED_DIR = _env_path("HELICO_PROCESSED_DIR")


def _require_raw_dir() -> Path:
    if RAW_DIR is None:
        raise EnvironmentError("HELICO_RAW_DIR is not set")
    return RAW_DIR


def _require_processed_dir() -> Path:
    if PROCESSED_DIR is None:
        raise EnvironmentError("HELICO_PROCESSED_DIR is not set")
    return PROCESSED_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
UNK_AA_IDX = len(STANDARD_AAS)  # 20

NUCLEOTIDES = "ACGTU"
NUC_TO_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}
UNK_NUC_IDX = len(NUCLEOTIDES)

# Element types for atom-level features
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Se", "B", "Si", "Fe", "Zn",
            "Mg", "Ca", "Mn", "Cu", "Co", "Ni", "Mo", "Na", "K"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
UNK_ELEM_IDX = len(ELEMENTS)

# Token type offsets
TOKEN_PROTEIN = 0        # 0-20 (20 AAs + UNK)
TOKEN_NUCLEOTIDE = 21    # 21-26 (5 nucleotides + UNK)
TOKEN_LIGAND_ATOM = 27   # ligand atoms start here

NUM_TOKEN_TYPES = 28 + UNK_ELEM_IDX + 1  # protein + nucleotide + per-element ligand tokens

# Amino acid 3-letter <-> 1-letter mappings
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
RNA_ONE_TO_CCD = {"A": "A", "C": "C", "G": "G", "U": "U"}
DNA_ONE_TO_CCD = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}

# ============================================================================
# CCD Parser
# ============================================================================

@dataclass
class CCDComponent:
    """Chemical Component Dictionary entry."""
    comp_id: str
    name: str
    comp_type: str  # "L-PEPTIDE LINKING", "RNA LINKING", "NON-POLYMER", etc.
    formula: str
    atom_names: list[str] = field(default_factory=list)
    atom_elements: list[str] = field(default_factory=list)
    atom_charges: list[int] = field(default_factory=list)
    atom_leaving: list[bool] = field(default_factory=list)
    ideal_coords: np.ndarray | None = None  # (N_atoms, 3) ideal coordinates
    bonds: list[tuple[str, str, str]] = field(default_factory=list)  # (atom1, atom2, order)
    smiles: str = ""

    @property
    def heavy_atom_mask(self) -> list[bool]:
        return [e != "H" for e in self.atom_elements]

    @property
    def heavy_atom_names(self) -> list[str]:
        return [n for n, e in zip(self.atom_names, self.atom_elements) if e != "H"]

    @property
    def n_heavy_atoms(self) -> int:
        return sum(1 for e in self.atom_elements if e != "H")


def parse_ccd(cif_path: Path | None = None, cache_path: Path | None = None) -> dict[str, CCDComponent]:
    """Parse the Chemical Component Dictionary into a lookup table.

    Returns dict mapping 3-letter code -> CCDComponent.
    Caches result as pickle for fast reload.
    """
    if cif_path is None:
        cif_path = _require_raw_dir() / "components.cif"
    if cache_path is None:
        cache_path = _require_processed_dir() / "ccd_cache.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    components: dict[str, CCDComponent] = {}
    current_id = None
    current_comp: CCDComponent | None = None

    # State machine for parsing loop_ sections
    in_atom_loop = False
    in_bond_loop = False
    in_descriptor_loop = False
    atom_fields: list[str] = []
    bond_fields: list[str] = []
    descriptor_fields: list[str] = []

    coords_x: list[float] = []
    coords_y: list[float] = []
    coords_z: list[float] = []

    def _flush():
        nonlocal current_comp, coords_x, coords_y, coords_z
        if current_comp is not None and coords_x:
            current_comp.ideal_coords = np.array(
                list(zip(coords_x, coords_y, coords_z)), dtype=np.float32
            )
        if current_comp is not None:
            components[current_comp.comp_id] = current_comp
        coords_x, coords_y, coords_z = [], [], []

    in_semicolon_block = False

    with open(cif_path, "r") as f:
        for line in f:
            line_stripped = line.strip()

            # Handle multi-line text blocks delimited by semicolons
            if line.startswith(";"):
                in_semicolon_block = not in_semicolon_block
                continue
            if in_semicolon_block:
                continue

            # New component block
            if line_stripped.startswith("data_"):
                _flush()
                comp_id = line_stripped[5:]
                current_comp = CCDComponent(comp_id=comp_id, name="", comp_type="", formula="")
                current_id = comp_id
                in_atom_loop = False
                in_bond_loop = False
                in_descriptor_loop = False
                continue

            if current_comp is None:
                continue

            # Single-value fields
            if line_stripped.startswith("_chem_comp.name"):
                val = line_stripped.split(None, 1)
                if len(val) > 1:
                    current_comp.name = val[1].strip().strip('"')
            elif line_stripped.startswith("_chem_comp.type"):
                val = line_stripped.split(None, 1)
                if len(val) > 1:
                    current_comp.comp_type = val[1].strip().strip('"')
            elif line_stripped.startswith("_chem_comp.formula "):
                val = line_stripped.split(None, 1)
                if len(val) > 1:
                    current_comp.formula = val[1].strip().strip('"')
            elif line_stripped.startswith("_chem_comp.pdbx_formal_charge"):
                pass  # component-level charge, not per-atom

            # Loop detection
            elif line_stripped == "loop_":
                in_atom_loop = False
                in_bond_loop = False
                in_descriptor_loop = False
                atom_fields = []
                bond_fields = []
                descriptor_fields = []
                continue

            # Atom loop fields
            elif line_stripped.startswith("_chem_comp_atom."):
                field_name = line_stripped.split(".")[1].strip()
                atom_fields.append(field_name)
                in_atom_loop = True
                continue

            # Bond loop fields
            elif line_stripped.startswith("_chem_comp_bond."):
                field_name = line_stripped.split(".")[1].strip()
                bond_fields.append(field_name)
                in_bond_loop = True
                continue

            # Descriptor loop fields
            elif line_stripped.startswith("_pdbx_chem_comp_descriptor."):
                field_name = line_stripped.split(".")[1].strip()
                descriptor_fields.append(field_name)
                in_descriptor_loop = True
                continue

            # End of loop (comment or new section)
            elif line_stripped.startswith("#") or line_stripped.startswith("_"):
                in_atom_loop = False
                in_bond_loop = False
                in_descriptor_loop = False
                continue

            elif line_stripped == "":
                continue

            # Parse atom data rows
            if in_atom_loop and atom_fields and not line_stripped.startswith("_"):
                parts = line_stripped.split()
                if len(parts) < len(atom_fields):
                    continue
                row = dict(zip(atom_fields, parts))
                if row.get("comp_id") != current_id:
                    continue
                current_comp.atom_names.append(row.get("atom_id", ""))
                current_comp.atom_elements.append(row.get("type_symbol", ""))
                charge_str = row.get("charge", "0")
                try:
                    charge = int(charge_str) if charge_str not in ("?", ".", "") else 0
                except (ValueError, TypeError):
                    charge = 0
                current_comp.atom_charges.append(charge)
                leaving = row.get("pdbx_leaving_atom_flag", "N") == "Y"
                current_comp.atom_leaving.append(leaving)
                # Ideal coordinates
                x = row.get("pdbx_model_Cartn_x_ideal", "?")
                y = row.get("pdbx_model_Cartn_y_ideal", "?")
                z = row.get("pdbx_model_Cartn_z_ideal", "?")
                try:
                    coords_x.append(float(x))
                    coords_y.append(float(y))
                    coords_z.append(float(z))
                except (ValueError, TypeError):
                    coords_x.append(0.0)
                    coords_y.append(0.0)
                    coords_z.append(0.0)

            # Parse bond data rows
            elif in_bond_loop and bond_fields and not line_stripped.startswith("_"):
                parts = line_stripped.split()
                if len(parts) < len(bond_fields):
                    continue
                row = dict(zip(bond_fields, parts))
                if row.get("comp_id") != current_id:
                    continue
                current_comp.bonds.append((
                    row.get("atom_id_1", ""),
                    row.get("atom_id_2", ""),
                    row.get("value_order", "SING"),
                ))

            # Parse descriptor data rows (for SMILES)
            elif in_descriptor_loop and descriptor_fields and not line_stripped.startswith("_"):
                parts = line_stripped.split(None, len(descriptor_fields) - 1)
                if len(parts) >= len(descriptor_fields):
                    row = dict(zip(descriptor_fields, parts))
                    if (row.get("comp_id") == current_id
                            and "SMILES" in row.get("type", "")
                            and "CANONICAL" in row.get("type", "")
                            and not current_comp.smiles):
                        current_comp.smiles = row.get("descriptor", "").strip('"')

    _flush()

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(components, f, protocol=pickle.HIGHEST_PROTOCOL)

    return components


# ============================================================================
# mmCIF Structure Parser
# ============================================================================

@dataclass
class Atom:
    name: str
    element: str
    coords: np.ndarray  # (3,)
    b_factor: float
    occupancy: float
    charge: int = 0


@dataclass
class Residue:
    name: str          # 3-letter code
    seq_id: int        # author sequence number
    atoms: list[Atom] = field(default_factory=list)
    is_standard: bool = True  # standard AA/nucleotide vs modified/ligand


@dataclass
class Chain:
    chain_id: str
    entity_type: str  # "polymer", "non-polymer", "water"
    polymer_type: str = ""  # "polypeptide(L)", "polyribonucleotide", "polydeoxyribonucleotide", ""
    residues: list[Residue] = field(default_factory=list)
    sequence: str = ""  # one-letter sequence for polymers


@dataclass
class Structure:
    pdb_id: str
    chains: list[Chain]
    resolution: float = float("inf")
    release_date: str = ""
    method: str = ""
    bioassembly_ops: list[np.ndarray] = field(default_factory=list)  # list of 4x4 transform matrices

    @property
    def all_atoms(self) -> list[tuple[str, str, int, Atom]]:
        """Returns list of (chain_id, res_name, res_seq, atom)."""
        result = []
        for chain in self.chains:
            for res in chain.residues:
                for atom in res.atoms:
                    result.append((chain.chain_id, res.name, res.seq_id, atom))
        return result

    @property
    def n_residues(self) -> int:
        return sum(len(c.residues) for c in self.chains)


def parse_mmcif(cif_path: str | Path, max_resolution: float = 9.0) -> Structure | None:
    """Parse an mmCIF file into our internal Structure representation.

    Uses BioPython's MMCIF2Dict for parsing, with custom post-processing.
    Filters: removes water/hydrogen, validates chains, checks resolution.
    Returns None if structure doesn't pass filters.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif_path = Path(cif_path)

    # Handle gzipped files: decompress to a temp file for MMCIF2Dict
    if cif_path.suffix == ".gz" or cif_path.name.endswith(".cif.gz"):
        with gzip.open(cif_path, "rt") as gz:
            content = gz.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mmcif_dict = MMCIF2Dict(tmp.name)
        finally:
            os.unlink(tmp.name)
    else:
        mmcif_dict = MMCIF2Dict(str(cif_path))

    # Extract PDB ID
    pdb_id = mmcif_dict.get("_entry.id", [""])[0].upper()

    # Resolution check
    resolution = float("inf")
    for key in ["_refine.ls_d_res_high", "_em_3d_reconstruction.resolution", "_reflns.d_resolution_high"]:
        vals = mmcif_dict.get(key, [])
        for v in vals:
            try:
                r = float(v)
                resolution = min(resolution, r)
            except (ValueError, TypeError):
                pass
    if resolution > max_resolution:
        return None

    # Release date
    release_date = ""
    dates = mmcif_dict.get("_pdbx_audit_revision_history.revision_date", [])
    if dates:
        release_date = dates[0]

    # Method
    method = ""
    methods = mmcif_dict.get("_exptl.method", [])
    if methods:
        method = methods[0]

    # Build entity type lookup
    entity_ids = mmcif_dict.get("_entity.id", [])
    entity_types = mmcif_dict.get("_entity.type", [])
    entity_type_map = dict(zip(entity_ids, entity_types))

    # Entity polymer type
    poly_entity_ids = mmcif_dict.get("_entity_poly.entity_id", [])
    poly_types = mmcif_dict.get("_entity_poly.type", [])
    poly_type_map = dict(zip(poly_entity_ids, poly_types))

    # Entity-to-chain mapping
    struct_entity_ids = mmcif_dict.get("_struct_asym.entity_id", [])
    struct_asym_ids = mmcif_dict.get("_struct_asym.id", [])
    asym_to_entity = dict(zip(struct_asym_ids, struct_entity_ids))

    # Parse atom_site records
    atom_ids = mmcif_dict.get("_atom_site.id", [])
    if not atom_ids:
        return None

    atom_names = mmcif_dict.get("_atom_site.label_atom_id", [])
    atom_elements = mmcif_dict.get("_atom_site.type_symbol", [])
    atom_x = mmcif_dict.get("_atom_site.Cartn_x", [])
    atom_y = mmcif_dict.get("_atom_site.Cartn_y", [])
    atom_z = mmcif_dict.get("_atom_site.Cartn_z", [])
    atom_bfactor = mmcif_dict.get("_atom_site.B_iso_or_equiv", [])
    atom_occ = mmcif_dict.get("_atom_site.occupancy", [])
    atom_comp_ids = mmcif_dict.get("_atom_site.label_comp_id", [])
    atom_asym_ids = mmcif_dict.get("_atom_site.label_asym_id", [])
    atom_seq_ids = mmcif_dict.get("_atom_site.label_seq_id", [])
    atom_alt_ids = mmcif_dict.get("_atom_site.label_alt_id", [])
    atom_model_nums = mmcif_dict.get("_atom_site.pdbx_PDB_model_num", [])

    # Build chains from atom records (first model only)
    chains_dict: dict[str, Chain] = {}
    residues_dict: dict[tuple[str, str, str], Residue] = {}  # (asym_id, comp_id, seq_id) -> Residue

    for i in range(len(atom_ids)):
        # Only first model
        if atom_model_nums and atom_model_nums[i] != "1":
            continue

        # Skip alt conformations except first
        alt = atom_alt_ids[i] if i < len(atom_alt_ids) else "."
        if alt not in (".", "A", "?"):
            continue

        element = atom_elements[i] if i < len(atom_elements) else ""
        # Filter hydrogen and water
        if element == "H":
            continue

        asym_id = atom_asym_ids[i] if i < len(atom_asym_ids) else ""
        comp_id = atom_comp_ids[i] if i < len(atom_comp_ids) else ""

        # Skip water
        if comp_id == "HOH":
            continue

        seq_id = atom_seq_ids[i] if i < len(atom_seq_ids) else "."
        entity_id = asym_to_entity.get(asym_id, "")
        etype = entity_type_map.get(entity_id, "polymer")

        if asym_id not in chains_dict:
            ptype = poly_type_map.get(entity_id, "")
            chains_dict[asym_id] = Chain(
                chain_id=asym_id,
                entity_type=etype,
                polymer_type=ptype,
            )

        res_key = (asym_id, comp_id, seq_id)
        if res_key not in residues_dict:
            try:
                seq_num = int(seq_id) if seq_id not in (".", "?") else 0
            except ValueError:
                seq_num = 0
            is_std = comp_id in AA_TO_IDX or len(comp_id) == 3 and comp_id in (
                "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
                "DA", "DC", "DG", "DT", "A", "C", "G", "U",
            )
            res = Residue(name=comp_id, seq_id=seq_num, is_standard=is_std)
            residues_dict[res_key] = res
            chains_dict[asym_id].residues.append(res)

        # Create atom
        try:
            coords = np.array([float(atom_x[i]), float(atom_y[i]), float(atom_z[i])], dtype=np.float32)
        except (ValueError, IndexError):
            continue

        bfactor = float(atom_bfactor[i]) if i < len(atom_bfactor) and atom_bfactor[i] not in (".", "?") else 0.0
        occ = float(atom_occ[i]) if i < len(atom_occ) and atom_occ[i] not in (".", "?") else 1.0

        atom = Atom(
            name=atom_names[i] if i < len(atom_names) else "",
            element=element,
            coords=coords,
            b_factor=bfactor,
            occupancy=occ,
        )
        residues_dict[res_key].atoms.append(atom)

    # Build one-letter sequences for polymer chains
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    chains = list(chains_dict.values())
    for chain in chains:
        if chain.entity_type == "polymer":
            chain.sequence = "".join(
                three_to_one.get(r.name, "X") for r in chain.residues
            )

    # Bioassembly transforms
    bioassembly_ops = []
    oper_ids = mmcif_dict.get("_pdbx_struct_oper_list.id", [])
    for idx, oid in enumerate(oper_ids):
        mat = np.eye(4, dtype=np.float32)
        for row in range(3):
            for col in range(3):
                key = f"_pdbx_struct_oper_list.matrix[{row+1}][{col+1}]"
                vals = mmcif_dict.get(key, [])
                if idx < len(vals):
                    try:
                        mat[row, col] = float(vals[idx])
                    except (ValueError, TypeError):
                        pass
            key = f"_pdbx_struct_oper_list.vector[{row+1}]"
            vals = mmcif_dict.get(key, [])
            if idx < len(vals):
                try:
                    mat[row, 3] = float(vals[idx])
                except (ValueError, TypeError):
                    pass
        bioassembly_ops.append(mat)

    # Filter: must have at least one polymer chain with residues
    polymer_chains = [c for c in chains if c.entity_type == "polymer" and len(c.residues) > 0]
    if not polymer_chains:
        return None

    return Structure(
        pdb_id=pdb_id,
        chains=chains,
        resolution=resolution,
        release_date=release_date,
        method=method,
        bioassembly_ops=bioassembly_ops,
    )


# ============================================================================
# Tokenizer
# ============================================================================

@dataclass
class Token:
    """A single token in the model's input."""
    token_type: int       # index into token vocabulary
    chain_idx: int        # which chain this token belongs to
    res_idx: int          # residue index within structure
    atom_names: list[str]     # atom names belonging to this token
    atom_elements: list[str]  # element symbols
    atom_coords: np.ndarray   # (N_atoms, 3) coordinates
    ref_coords: np.ndarray | None  # (N_atoms, 3) reference coordinates from CCD
    res_name: str = ""
    entity_id: int = 0        # entity identifier (chains with same sequence share entity_id)
    sym_id: int = 0           # symmetry copy index (for bioassemblies)
    token_index: int = 0      # token position within residue (0 for protein/nucleotide)


@dataclass
class TokenizedStructure:
    """Tokenized representation of a structure."""
    pdb_id: str
    tokens: list[Token]
    chain_ids: list[str]          # chain_id per token
    entity_types: list[str]       # entity type per token
    chain_sequences: dict[str, str] = field(default_factory=dict)  # chain_id -> sequence

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_atoms(self) -> int:
        return sum(len(t.atom_names) for t in self.tokens)

    def to_features(self) -> dict[str, torch.Tensor]:
        """Convert to model-ready feature tensors."""
        n_tok = self.n_tokens

        # Token-level features
        token_types = torch.tensor([t.token_type for t in self.tokens], dtype=torch.long)
        chain_indices = torch.tensor([t.chain_idx for t in self.tokens], dtype=torch.long)
        res_indices = torch.tensor([t.res_idx for t in self.tokens], dtype=torch.long)

        # Relative position encoding (within each chain)
        rel_pos = torch.zeros(n_tok, dtype=torch.long)
        for i, tok in enumerate(self.tokens):
            rel_pos[i] = tok.res_idx

        # Atom-level features: flatten all atoms across tokens
        all_atom_coords = []
        all_atom_elements = []
        all_ref_coords = []
        atom_to_token = []  # maps each atom to its token index
        atoms_per_token = []

        for ti, tok in enumerate(self.tokens):
            n_atoms = len(tok.atom_names)
            atoms_per_token.append(n_atoms)
            atom_to_token.extend([ti] * n_atoms)
            all_atom_coords.append(tok.atom_coords)
            all_atom_elements.extend(tok.atom_elements)
            if tok.ref_coords is not None:
                all_ref_coords.append(tok.ref_coords)
            else:
                all_ref_coords.append(np.zeros((n_atoms, 3), dtype=np.float32))

        total_atoms = sum(atoms_per_token)
        atom_coords = torch.tensor(np.concatenate(all_atom_coords, axis=0), dtype=torch.float32) if total_atoms > 0 else torch.zeros(0, 3)
        ref_coords = torch.tensor(np.concatenate(all_ref_coords, axis=0), dtype=torch.float32) if total_atoms > 0 else torch.zeros(0, 3)
        atom_to_token_idx = torch.tensor(atom_to_token, dtype=torch.long)
        atoms_per_token_t = torch.tensor(atoms_per_token, dtype=torch.long)

        # Element indices for atoms
        atom_element_idx = torch.tensor(
            [ELEM_TO_IDX.get(e, UNK_ELEM_IDX) for e in all_atom_elements],
            dtype=torch.long,
        )

        # Chain-pair mask (same chain = 1, different chain = 0)
        chain_same = (chain_indices.unsqueeze(0) == chain_indices.unsqueeze(1)).long()

        # Per-token relpe features
        token_index = torch.tensor([t.token_index for t in self.tokens], dtype=torch.long)
        entity_id = torch.tensor([t.entity_id for t in self.tokens], dtype=torch.long)
        sym_id = torch.tensor([t.sym_id for t in self.tokens], dtype=torch.long)

        return {
            "token_types": token_types,            # (N_tok,)
            "chain_indices": chain_indices,          # (N_tok,)
            "res_indices": res_indices,              # (N_tok,)
            "rel_pos": rel_pos,                      # (N_tok,)
            "token_index": token_index,              # (N_tok,)
            "entity_id": entity_id,                  # (N_tok,)
            "sym_id": sym_id,                        # (N_tok,)
            "atom_coords": atom_coords,              # (N_atoms, 3)
            "ref_coords": ref_coords,                # (N_atoms, 3)
            "atom_to_token": atom_to_token_idx,      # (N_atoms,)
            "atoms_per_token": atoms_per_token_t,    # (N_tok,)
            "atom_element_idx": atom_element_idx,    # (N_atoms,)
            "chain_same": chain_same,                # (N_tok, N_tok)
            "n_tokens": n_tok,
            "n_atoms": total_atoms,
        }


def tokenize_structure(
    structure: Structure,
    ccd: dict[str, CCDComponent] | None = None,
) -> TokenizedStructure:
    """Tokenize a parsed structure.

    Proteins: 1 token per residue (20 standard AAs + unknown).
    Nucleic acids: 1 token per nucleotide.
    Ligands/modified residues: 1 token per heavy atom (using CCD definitions).
    """
    tokens: list[Token] = []
    chain_ids: list[str] = []
    entity_types: list[str] = []
    res_counter = 0

    # Build entity_id mapping: chains with the same sequence share an entity_id
    seq_to_entity: dict[str, int] = {}
    chain_to_entity: dict[str, int] = {}
    next_entity_id = 0
    for chain in structure.chains:
        seq = chain.sequence if chain.sequence else f"__nonpoly_{chain.chain_id}"
        if seq not in seq_to_entity:
            seq_to_entity[seq] = next_entity_id
            next_entity_id += 1
        chain_to_entity[chain.chain_id] = seq_to_entity[seq]

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    nuc_codes = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}

    for chain_idx, chain in enumerate(structure.chains):
        for res in chain.residues:
            heavy_atoms = [a for a in res.atoms if a.element != "H"]
            if not heavy_atoms:
                continue

            is_protein = chain.polymer_type.startswith("polypeptide") and res.name in three_to_one
            is_nucleotide = ("ribonucleotide" in chain.polymer_type or "deoxyribonucleotide" in chain.polymer_type) and res.name in nuc_codes

            if is_protein:
                # One token per residue
                one_letter = three_to_one.get(res.name, "X")
                token_type = AA_TO_IDX.get(one_letter, UNK_AA_IDX)

                atom_names = [a.name for a in heavy_atoms]
                atom_elements = [a.element for a in heavy_atoms]
                atom_coords = np.stack([a.coords for a in heavy_atoms])

                ref_c = None
                if ccd and res.name in ccd:
                    comp = ccd[res.name]
                    ref_c = comp.ideal_coords
                    if ref_c is not None:
                        # Match by heavy atoms only
                        mask = comp.heavy_atom_mask
                        ref_c = ref_c[mask] if len(mask) == len(ref_c) else None
                        if ref_c is not None and len(ref_c) != len(heavy_atoms):
                            ref_c = None  # shape mismatch, skip

                tokens.append(Token(
                    token_type=token_type,
                    chain_idx=chain_idx,
                    res_idx=res_counter,
                    atom_names=atom_names,
                    atom_elements=atom_elements,
                    atom_coords=atom_coords,
                    ref_coords=ref_c,
                    res_name=res.name,
                    entity_id=chain_to_entity[chain.chain_id],
                    sym_id=0,
                    token_index=0,
                ))
                chain_ids.append(chain.chain_id)
                entity_types.append("protein")
                res_counter += 1

            elif is_nucleotide:
                # One token per nucleotide
                nuc_letter = res.name[-1] if len(res.name) <= 2 else res.name
                if nuc_letter == "T":
                    nuc_letter = "U"  # treat DNA T as U for tokenization
                token_type = TOKEN_NUCLEOTIDE + NUC_TO_IDX.get(nuc_letter, UNK_NUC_IDX)

                atom_names = [a.name for a in heavy_atoms]
                atom_elements = [a.element for a in heavy_atoms]
                atom_coords = np.stack([a.coords for a in heavy_atoms])

                ref_c = None
                if ccd and res.name in ccd:
                    comp = ccd[res.name]
                    ref_c = comp.ideal_coords
                    if ref_c is not None:
                        mask = comp.heavy_atom_mask
                        ref_c = ref_c[mask] if len(mask) == len(ref_c) else None
                        if ref_c is not None and len(ref_c) != len(heavy_atoms):
                            ref_c = None

                tokens.append(Token(
                    token_type=token_type,
                    chain_idx=chain_idx,
                    res_idx=res_counter,
                    atom_names=atom_names,
                    atom_elements=atom_elements,
                    atom_coords=atom_coords,
                    ref_coords=ref_c,
                    res_name=res.name,
                    entity_id=chain_to_entity[chain.chain_id],
                    sym_id=0,
                    token_index=0,
                ))
                chain_ids.append(chain.chain_id)
                entity_types.append("nucleotide")
                res_counter += 1

            else:
                # Ligand / modified residue: one token per heavy atom
                lig_res_idx = res_counter
                for atom_idx, atom in enumerate(heavy_atoms):
                    elem_idx = ELEM_TO_IDX.get(atom.element, UNK_ELEM_IDX)
                    token_type = TOKEN_LIGAND_ATOM + elem_idx

                    tokens.append(Token(
                        token_type=token_type,
                        chain_idx=chain_idx,
                        res_idx=res_counter,
                        atom_names=[atom.name],
                        atom_elements=[atom.element],
                        atom_coords=atom.coords.reshape(1, 3),
                        ref_coords=None,
                        res_name=res.name,
                        entity_id=chain_to_entity[chain.chain_id],
                        sym_id=0,
                        token_index=atom_idx,
                    ))
                    chain_ids.append(chain.chain_id)
                    entity_types.append("ligand")
                    res_counter += 1

    chain_sequences = {c.chain_id: c.sequence for c in structure.chains if c.sequence}

    return TokenizedStructure(
        pdb_id=structure.pdb_id,
        tokens=tokens,
        chain_ids=chain_ids,
        entity_types=entity_types,
        chain_sequences=chain_sequences,
    )


def tokenize_sequences(
    chains: list[dict],
    ccd: dict[str, CCDComponent],
) -> TokenizedStructure:
    """Tokenize from sequence descriptions (no existing 3D coordinates needed).

    Each chain dict has:
      - {"type": "protein", "id": "A", "sequence": "MKFLILF..."}
      - {"type": "rna", "id": "B", "sequence": "AUGCCU..."}
      - {"type": "dna", "id": "C", "sequence": "ATGCCT..."}
      - {"type": "ligand", "id": "D", "ccd": "ATP"}

    Uses CCD ideal coordinates for ref_coords and atom_coords.
    """
    tokens: list[Token] = []
    chain_ids: list[str] = []
    entity_types: list[str] = []
    chain_sequences: dict[str, str] = {}
    res_counter = 0

    # Build entity_id mapping: chains with identical (type, sequence/ccd) share entity_id
    entity_key_to_id: dict[tuple[str, str], int] = {}
    chain_to_entity: dict[str, int] = {}
    next_entity_id = 0
    for chain in chains:
        ctype = chain["type"]
        key_str = chain.get("sequence", chain.get("ccd", ""))
        entity_key = (ctype, key_str)
        if entity_key not in entity_key_to_id:
            entity_key_to_id[entity_key] = next_entity_id
            next_entity_id += 1
        chain_to_entity[chain["id"]] = entity_key_to_id[entity_key]

    for chain_idx, chain in enumerate(chains):
        chain_id = chain["id"]
        ctype = chain["type"]
        entity_id = chain_to_entity[chain_id]

        if ctype == "protein":
            sequence = chain["sequence"]
            chain_sequences[chain_id] = sequence
            for char in sequence:
                three_code = ONE_TO_THREE.get(char, "UNK")
                token_type = AA_TO_IDX.get(char, UNK_AA_IDX)

                # Look up CCD for atom info and ideal coords
                if three_code in ccd:
                    comp = ccd[three_code]
                    mask = comp.heavy_atom_mask
                    atom_names = comp.heavy_atom_names
                    atom_elements = [e for e, m in zip(comp.atom_elements, mask) if m]
                    if comp.ideal_coords is not None and len(mask) == len(comp.ideal_coords):
                        ref_c = comp.ideal_coords[mask]
                    else:
                        ref_c = np.zeros((len(atom_names), 3), dtype=np.float32)
                else:
                    # Fallback: minimal backbone
                    atom_names = ["N", "CA", "C", "O"]
                    atom_elements = ["N", "C", "C", "O"]
                    ref_c = np.zeros((4, 3), dtype=np.float32)

                tokens.append(Token(
                    token_type=token_type,
                    chain_idx=chain_idx,
                    res_idx=res_counter,
                    atom_names=atom_names,
                    atom_elements=atom_elements,
                    atom_coords=ref_c.copy(),
                    ref_coords=ref_c,
                    res_name=three_code,
                    entity_id=entity_id,
                    sym_id=0,
                    token_index=0,
                ))
                chain_ids.append(chain_id)
                entity_types.append("protein")
                res_counter += 1

        elif ctype in ("rna", "dna"):
            sequence = chain["sequence"]
            chain_sequences[chain_id] = sequence
            ccd_map = RNA_ONE_TO_CCD if ctype == "rna" else DNA_ONE_TO_CCD
            for char in sequence:
                ccd_code = ccd_map.get(char)
                if ccd_code is None:
                    continue
                nuc_letter = char
                if ctype == "dna" and char == "T":
                    nuc_letter = "U"
                token_type = TOKEN_NUCLEOTIDE + NUC_TO_IDX.get(nuc_letter, UNK_NUC_IDX)

                if ccd_code in ccd:
                    comp = ccd[ccd_code]
                    mask = comp.heavy_atom_mask
                    atom_names = comp.heavy_atom_names
                    atom_elements = [e for e, m in zip(comp.atom_elements, mask) if m]
                    if comp.ideal_coords is not None and len(mask) == len(comp.ideal_coords):
                        ref_c = comp.ideal_coords[mask]
                    else:
                        ref_c = np.zeros((len(atom_names), 3), dtype=np.float32)
                else:
                    atom_names = [char]
                    atom_elements = ["C"]
                    ref_c = np.zeros((1, 3), dtype=np.float32)

                tokens.append(Token(
                    token_type=token_type,
                    chain_idx=chain_idx,
                    res_idx=res_counter,
                    atom_names=atom_names,
                    atom_elements=atom_elements,
                    atom_coords=ref_c.copy(),
                    ref_coords=ref_c,
                    res_name=ccd_code,
                    entity_id=entity_id,
                    sym_id=0,
                    token_index=0,
                ))
                chain_ids.append(chain_id)
                entity_types.append("nucleotide")
                res_counter += 1

        elif ctype == "ligand":
            ccd_code = chain["ccd"]
            if ccd_code not in ccd:
                logger.warning(f"CCD code {ccd_code} not found, skipping ligand chain {chain_id}")
                continue
            comp = ccd[ccd_code]
            mask = comp.heavy_atom_mask
            heavy_names = comp.heavy_atom_names
            heavy_elements = [e for e, m in zip(comp.atom_elements, mask) if m]
            if comp.ideal_coords is not None and len(mask) == len(comp.ideal_coords):
                heavy_coords = comp.ideal_coords[mask]
            else:
                heavy_coords = np.zeros((len(heavy_names), 3), dtype=np.float32)

            for atom_idx, (aname, aelem) in enumerate(zip(heavy_names, heavy_elements)):
                elem_idx = ELEM_TO_IDX.get(aelem, UNK_ELEM_IDX)
                token_type = TOKEN_LIGAND_ATOM + elem_idx
                coord = heavy_coords[atom_idx].reshape(1, 3)

                tokens.append(Token(
                    token_type=token_type,
                    chain_idx=chain_idx,
                    res_idx=res_counter,
                    atom_names=[aname],
                    atom_elements=[aelem],
                    atom_coords=coord.copy(),
                    ref_coords=coord,
                    res_name=ccd_code,
                    entity_id=entity_id,
                    sym_id=0,
                    token_index=atom_idx,
                ))
                chain_ids.append(chain_id)
                entity_types.append("ligand")
                res_counter += 1

    return TokenizedStructure(
        pdb_id="PREDICT",
        tokens=tokens,
        chain_ids=chain_ids,
        entity_types=entity_types,
        chain_sequences=chain_sequences,
    )


def parse_sequences_arg(sequences_str: str) -> list[dict]:
    """Parse --sequences arg like "A:MKFLILF,B:ACDEF" into chain dicts.

    Assumes protein-only. For RNA/DNA/ligands, use YAML input.
    """
    chains = []
    for part in sequences_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid sequence format '{part}', expected 'CHAIN_ID:SEQUENCE'")
        chain_id, sequence = part.split(":", 1)
        chains.append({"type": "protein", "id": chain_id.strip(), "sequence": sequence.strip()})
    return chains


def parse_input_yaml(yaml_path: str | Path) -> list[dict]:
    """Parse a YAML input file (Boltz2-inspired format) into chain dicts.

    Expected format:
        sequences:
          - protein: {id: A, sequence: MKFLILF}
          - rna: {id: B, sequence: AUGCCU}
          - ligand: {id: C, ccd: ATP}
    """
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    chains = []
    for entry in data.get("sequences", []):
        if "protein" in entry:
            info = entry["protein"]
            chains.append({"type": "protein", "id": str(info["id"]), "sequence": info["sequence"]})
        elif "rna" in entry:
            info = entry["rna"]
            chains.append({"type": "rna", "id": str(info["id"]), "sequence": info["sequence"]})
        elif "dna" in entry:
            info = entry["dna"]
            chains.append({"type": "dna", "id": str(info["id"]), "sequence": info["sequence"]})
        elif "ligand" in entry:
            info = entry["ligand"]
            chains.append({"type": "ligand", "id": str(info["id"]), "ccd": info["ccd"]})
    return chains


# ============================================================================
# MSA Feature Extraction
# ============================================================================

@dataclass
class MSAFeatures:
    """MSA features for a single chain."""
    msa: np.ndarray             # (N_seqs, L) integer-encoded MSA
    deletion_matrix: np.ndarray  # (N_seqs, L) deletion counts
    profile: np.ndarray          # (L, 22) residue frequency profile (20 AA + gap + unknown)
    cluster_msa: np.ndarray      # (N_clusters, L) clustered MSA
    cluster_profile: np.ndarray  # (N_clusters, L, 22) cluster profiles
    n_seqs: int
    length: int


def parse_a3m(content: str) -> tuple[list[str], list[str]]:
    """Parse A3M format content into sequences and descriptions.

    A3M is like FASTA but lowercase letters indicate insertions (deletions in query).
    """
    seqs: list[str] = []
    descs: list[str] = []
    current_seq = []
    current_desc = ""

    for line in content.split("\n"):
        if line.startswith(">"):
            if current_seq:
                seqs.append("".join(current_seq))
                descs.append(current_desc)
            current_desc = line[1:].strip()
            current_seq = []
        elif line.strip():
            current_seq.append(line.strip())

    if current_seq:
        seqs.append("".join(current_seq))
        descs.append(current_desc)

    return seqs, descs


def a3m_to_msa_matrix(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convert A3M sequences to MSA matrix and deletion matrix.

    Lowercase chars are insertions (removed from alignment, counted as deletions).
    Returns:
        msa: (N_seqs, L) integer encoded (0-20=AA, 21=gap, 22=unknown)
        deletions: (N_seqs, L) deletion counts
    """
    if not seqs:
        return np.zeros((0, 0), dtype=np.int8), np.zeros((0, 0), dtype=np.int8)

    # Process query to get alignment length
    query = seqs[0]
    query_aligned = "".join(c for c in query if not c.islower())
    L = len(query_aligned)

    aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    gap_idx = 20
    unk_idx = 21

    n_seqs = len(seqs)
    msa = np.full((n_seqs, L), gap_idx, dtype=np.int8)
    deletions = np.zeros((n_seqs, L), dtype=np.int8)

    for si, seq in enumerate(seqs):
        pos = 0
        del_count = 0
        for c in seq:
            if c.islower():
                # Insertion in this sequence = deletion relative to query
                del_count += 1
            elif c == "-":
                msa[si, pos] = gap_idx
                deletions[si, pos] = min(del_count, 255)
                del_count = 0
                pos += 1
            else:
                aa = c.upper()
                msa[si, pos] = aa_map.get(aa, unk_idx)
                deletions[si, pos] = min(del_count, 255)
                del_count = 0
                pos += 1
            if pos >= L:
                break

    return msa, deletions


def compute_msa_features(
    msa: np.ndarray,
    deletions: np.ndarray,
    max_seqs: int = 512,
    n_clusters: int = 64,
) -> MSAFeatures:
    """Compute MSA features: profile, clustering.

    Args:
        msa: (N_seqs, L) integer encoded MSA
        deletions: (N_seqs, L) deletion counts
        max_seqs: maximum number of sequences to use
        n_clusters: number of clusters for summary

    Returns:
        MSAFeatures with profile, cluster summaries, etc.
    """
    n_seqs, L = msa.shape

    # Subsample if too many sequences
    if n_seqs > max_seqs:
        # Keep query, randomly sample rest
        indices = np.concatenate([[0], np.random.choice(range(1, n_seqs), max_seqs - 1, replace=False)])
        msa = msa[indices]
        deletions = deletions[indices]
        n_seqs = max_seqs

    # Compute profile (amino acid frequencies at each position)
    n_classes = 22  # 20 AA + gap + unknown
    profile = np.zeros((L, n_classes), dtype=np.float32)
    for pos in range(L):
        counts = np.bincount(msa[:, pos].astype(np.int32), minlength=n_classes)[:n_classes]
        total = counts.sum()
        if total > 0:
            profile[pos] = counts / total

    # Simple clustering by sequence identity to query
    query = msa[0]
    n_clust = min(n_clusters, n_seqs)

    if n_seqs <= n_clust:
        cluster_msa = msa
        cluster_profile = np.zeros((n_seqs, L, n_classes), dtype=np.float32)
        for ci in range(n_seqs):
            for pos in range(L):
                cluster_profile[ci, pos, msa[ci, pos]] = 1.0
    else:
        # Greedy clustering by hamming distance
        assigned = np.full(n_seqs, -1, dtype=np.int32)
        centers = [0]
        assigned[0] = 0

        for i in range(1, n_seqs):
            if len(centers) < n_clust:
                # Check identity to existing centers
                best_id = -1
                best_match = -1
                for ci, center_idx in enumerate(centers):
                    matches = np.sum(msa[i] == msa[center_idx])
                    if matches > best_match:
                        best_match = matches
                        best_id = ci
                # If < 80% identity to best center, make new center
                if best_match < 0.8 * L:
                    centers.append(i)
                    assigned[i] = len(centers) - 1
                else:
                    assigned[i] = best_id
            else:
                # Assign to closest center
                best_id = 0
                best_match = -1
                for ci, center_idx in enumerate(centers):
                    matches = np.sum(msa[i] == msa[center_idx])
                    if matches > best_match:
                        best_match = matches
                        best_id = ci
                assigned[i] = best_id

        cluster_msa = msa[centers]
        cluster_profile = np.zeros((len(centers), L, n_classes), dtype=np.float32)
        for ci in range(len(centers)):
            members = msa[assigned == ci]
            for pos in range(L):
                counts = np.bincount(members[:, pos].astype(np.int32), minlength=n_classes)[:n_classes]
                total = counts.sum()
                if total > 0:
                    cluster_profile[ci, pos] = counts / total

    return MSAFeatures(
        msa=msa,
        deletion_matrix=deletions,
        profile=profile,
        cluster_msa=cluster_msa,
        cluster_profile=cluster_profile,
        n_seqs=n_seqs,
        length=L,
    )


def _read_a3m_features(raw_gz: bytes) -> MSAFeatures | None:
    """Decompress gzipped a3m content and compute MSA features."""
    content = gzip.decompress(raw_gz).decode()
    seqs, _ = parse_a3m(content)
    if seqs:
        msa, dels = a3m_to_msa_matrix(seqs)
        return compute_msa_features(msa, dels)
    return None


def load_msa_for_chain(
    pdb_id: str,
    chain_id: str,
    sequence: str = "",
    msa_dir: Path | None = None,
    msa_tar_path: Path | None = None,
    tar_index: TarIndex | None = None,
) -> MSAFeatures | None:
    """Load and process MSA for a given PDB chain.

    Lookup strategy (tried in order):
    1. Extracted directory: tries sha256(sequence+"\\n") and pdb_chain hash filenames
    2. RCSB tar index: files are named sha256(sequence+"\\n").a3m.gz
    3. OpenFold tar index: files are named {uniprot_accession}.a3m.gz (not supported
       without a mapping — caller must provide extracted dir or appropriate tar_index)
    4. Fallback sequential tar scan

    Args:
        sequence: the chain's amino acid sequence, used to compute the RCSB hash
    """
    # Compute RCSB-style hash: sha256(sequence + "\n")
    seq_hash = hashlib.sha256((sequence + "\n").encode()).hexdigest() if sequence else ""

    # Try extracted directory first
    if msa_dir and msa_dir.exists():
        for hash_name in [seq_hash] if seq_hash else []:
            msa_path = msa_dir / f"{hash_name}.a3m.gz"
            if msa_path.exists():
                with gzip.open(msa_path, "rt") as f:
                    content = f.read()
                seqs, _ = parse_a3m(content)
                if seqs:
                    msa, dels = a3m_to_msa_matrix(seqs)
                    return compute_msa_features(msa, dels)

    # Try tar index for O(1) access
    if tar_index is not None and seq_hash:
        # RCSB naming convention
        target = f"rcsb_raw_msa/{seq_hash}.a3m.gz"
        if target in tar_index.entries:
            offset, size = tar_index.entries[target]
            raw = read_tar_member(tar_index.tar_path, offset, size)
            return _read_a3m_features(raw)

    # Fallback: sequential tar scan (slow)
    if msa_tar_path is not None and msa_tar_path.exists() and seq_hash:
        target = f"rcsb_raw_msa/{seq_hash}.a3m.gz"
        try:
            with tarfile.open(msa_tar_path, "r") as tar:
                member = tar.getmember(target)
                f = tar.extractfile(member)
                if f:
                    return _read_a3m_features(f.read())
        except (KeyError, tarfile.TarError):
            pass

    return None


# ============================================================================
# PDB Sequence Database
# ============================================================================

def load_pdb_seqres(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load pdb_seqres.txt.gz into dict: pdb_id -> {chain_id: sequence}.

    Format: >PDBID_CHAIN mol:TYPE length:N  DESCRIPTION
    """
    if path is None:
        path = _require_raw_dir() / "pdb_seqres.txt.gz"
    seqres: dict[str, dict[str, str]] = {}

    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as f:
        current_pdb = ""
        current_chain = ""
        current_seq = []

        for line in f:
            if line.startswith(">"):
                # Flush previous
                if current_pdb and current_seq:
                    if current_pdb not in seqres:
                        seqres[current_pdb] = {}
                    seqres[current_pdb][current_chain] = "".join(current_seq)

                # Parse header: >PDBID_CHAIN mol:TYPE ...
                parts = line[1:].split()
                if parts:
                    id_chain = parts[0].split("_")
                    current_pdb = id_chain[0].upper()
                    current_chain = id_chain[1] if len(id_chain) > 1 else "A"
                current_seq = []
            else:
                current_seq.append(line.strip())

        if current_pdb and current_seq:
            if current_pdb not in seqres:
                seqres[current_pdb] = {}
            seqres[current_pdb][current_chain] = "".join(current_seq)

    return seqres


# ============================================================================
# Tar Indexing (O(1) random access into tar archives)
# ============================================================================

@dataclass
class TarIndex:
    """Index for O(1) random access into a tar archive."""
    tar_path: Path
    entries: dict[str, tuple[int, int]]  # member_name -> (data_offset, data_size)


def build_tar_index(tar_path: Path) -> TarIndex:
    """Build an index by iterating through the tar once, recording data offsets."""
    entries: dict[str, tuple[int, int]] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if member.isfile():
                entries[member.name] = (member.offset_data, member.size)
    return TarIndex(tar_path=tar_path, entries=entries)


def save_tar_index(index: TarIndex, path: Path) -> None:
    """Save a TarIndex as a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tar_index(path: Path) -> TarIndex:
    """Load a TarIndex from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def read_tar_member(tar_path: Path, offset: int, size: int) -> bytes:
    """Read a single member from a tar archive by seeking directly to its data offset."""
    with open(tar_path, "rb") as f:
        f.seek(offset)
        return f.read(size)


# ============================================================================
# Cropping
# ============================================================================

def spatial_crop(
    features: dict[str, torch.Tensor],
    crop_size: int,
    interface_bias: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Spatial cropping: pick a center token and keep closest tokens.

    For multi-chain complexes, biased toward picking interface tokens.

    Args:
        features: dict from TokenizedStructure.to_features()
        crop_size: number of tokens to keep
        interface_bias: probability of picking an interface token as center
    """
    n_tok = features["n_tokens"]
    if n_tok <= crop_size:
        return features

    atom_coords = features["atom_coords"]        # (N_atoms, 3)
    atom_to_token = features["atom_to_token"]     # (N_atoms,)

    # Compute token center coordinates (mean of atom coords per token)
    token_centers = torch.zeros(n_tok, 3)
    token_counts = torch.zeros(n_tok)
    token_centers.scatter_add_(0, atom_to_token.unsqueeze(1).expand(-1, 3), atom_coords)
    token_counts.scatter_add_(0, atom_to_token, torch.ones(atom_to_token.shape[0]))
    token_counts = token_counts.clamp(min=1)
    token_centers = token_centers / token_counts.unsqueeze(1)

    # Identify interface tokens (tokens close to a different chain)
    chain_indices = features["chain_indices"]
    chain_same = features["chain_same"]  # (N_tok, N_tok)

    # Distance matrix between token centers
    dists = torch.cdist(token_centers.unsqueeze(0), token_centers.unsqueeze(0)).squeeze(0)

    # Interface tokens: close to (< 10Å) tokens from a different chain
    different_chain = 1 - chain_same
    close_different = (dists < 10.0) & (different_chain.bool())
    is_interface = close_different.any(dim=1)

    # Pick center token
    if is_interface.any() and torch.rand(1).item() < interface_bias:
        interface_indices = torch.where(is_interface)[0]
        center_idx = interface_indices[torch.randint(len(interface_indices), (1,))].item()
    else:
        center_idx = torch.randint(n_tok, (1,)).item()

    # Find closest tokens to center
    center_dists = dists[center_idx]
    _, sorted_indices = center_dists.sort()
    keep_indices = sorted_indices[:crop_size].sort().values

    return _subset_features(features, keep_indices)


def contiguous_crop(
    features: dict[str, torch.Tensor],
    crop_size: int,
) -> dict[str, torch.Tensor]:
    """Contiguous cropping: preserves sequence continuity within chains.

    Picks a random start point and takes a contiguous block of tokens.
    """
    n_tok = features["n_tokens"]
    if n_tok <= crop_size:
        return features

    start = torch.randint(0, n_tok - crop_size + 1, (1,)).item()
    keep_indices = torch.arange(start, start + crop_size)

    return _subset_features(features, keep_indices)


def _subset_features(
    features: dict[str, torch.Tensor],
    token_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Subset all feature tensors to keep only the given token indices."""
    n_new = len(token_indices)

    # Build atom mask: which atoms belong to kept tokens
    atom_to_token = features["atom_to_token"]
    kept_set = set(token_indices.tolist())
    atom_mask = torch.tensor([atom_to_token[i].item() in kept_set for i in range(len(atom_to_token))])

    # Remap token indices
    old_to_new = torch.full((features["n_tokens"],), -1, dtype=torch.long)
    for new_idx, old_idx in enumerate(token_indices):
        old_to_new[old_idx] = new_idx

    new_atom_to_token = old_to_new[atom_to_token[atom_mask]]

    result = {
        "token_types": features["token_types"][token_indices],
        "chain_indices": features["chain_indices"][token_indices],
        "res_indices": features["res_indices"][token_indices],
        "rel_pos": features["rel_pos"][token_indices],
        "token_index": features["token_index"][token_indices],
        "entity_id": features["entity_id"][token_indices],
        "sym_id": features["sym_id"][token_indices],
        "atom_coords": features["atom_coords"][atom_mask],
        "ref_coords": features["ref_coords"][atom_mask],
        "atom_to_token": new_atom_to_token,
        "atoms_per_token": features["atoms_per_token"][token_indices],
        "atom_element_idx": features["atom_element_idx"][atom_mask],
        "chain_same": features["chain_same"][token_indices][:, token_indices],
        "n_tokens": n_new,
        "n_atoms": atom_mask.sum().item(),
    }

    return result


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class HelicoDataset(Dataset):
    """PyTorch dataset for Helico training.

    Loads pre-processed tokenized structures and applies cropping.
    """

    def __init__(
        self,
        structures: list[TokenizedStructure],
        crop_size: int = 384,
        crop_mode: str = "spatial",  # "spatial" or "contiguous"
        msa_features: dict[str, MSAFeatures] | None = None,
    ):
        self.structures = structures
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.msa_features = msa_features or {}

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ts = self.structures[idx]
        features = ts.to_features()

        # Apply cropping
        if self.crop_mode == "spatial":
            features = spatial_crop(features, self.crop_size)
        else:
            features = contiguous_crop(features, self.crop_size)

        # Add MSA features if available
        msa_key = ts.pdb_id
        if msa_key in self.msa_features:
            msa_feat = self.msa_features[msa_key]
            features["msa_profile"] = torch.tensor(msa_feat.profile, dtype=torch.float32)
            features["cluster_msa"] = torch.tensor(msa_feat.cluster_msa, dtype=torch.long)
            features["cluster_profile"] = torch.tensor(msa_feat.cluster_profile, dtype=torch.float32)
            features["has_msa"] = torch.tensor(1)
        else:
            # Placeholder MSA features
            n_tok = features["n_tokens"]
            features["msa_profile"] = torch.zeros(n_tok, 22)
            features["cluster_msa"] = torch.zeros(1, n_tok, dtype=torch.long)
            features["cluster_profile"] = torch.zeros(1, n_tok, 22)
            features["has_msa"] = torch.tensor(0)

        return features


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function that pads variable-length tensors to form a batch."""
    if not batch:
        return {}

    max_tokens = max(b["n_tokens"] for b in batch)
    max_atoms = max(b["n_atoms"] for b in batch)
    B = len(batch)

    result: dict[str, torch.Tensor] = {}

    # Token-level tensors: pad to max_tokens
    for key in ["token_types", "chain_indices", "res_indices", "rel_pos", "token_index", "entity_id", "sym_id", "atoms_per_token"]:
        tensors = []
        for b in batch:
            t = b[key]
            pad_size = max_tokens - len(t)
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, pad_size))
            tensors.append(t)
        result[key] = torch.stack(tensors)

    # Pair tensor: chain_same
    chain_same_list = []
    for b in batch:
        t = b["chain_same"]
        n = t.shape[0]
        pad = max_tokens - n
        if pad > 0:
            t = torch.nn.functional.pad(t, (0, pad, 0, pad))
        chain_same_list.append(t)
    result["chain_same"] = torch.stack(chain_same_list)

    # Atom-level tensors: pad to max_atoms
    for key in ["atom_coords", "ref_coords"]:
        tensors = []
        for b in batch:
            t = b[key]
            pad_size = max_atoms - len(t)
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, 0, 0, pad_size))
            tensors.append(t)
        result[key] = torch.stack(tensors)

    for key in ["atom_to_token", "atom_element_idx"]:
        tensors = []
        for b in batch:
            t = b[key]
            pad_size = max_atoms - len(t)
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, pad_size))
            tensors.append(t)
        result[key] = torch.stack(tensors)

    # Scalar values
    result["n_tokens"] = torch.tensor([b["n_tokens"] for b in batch])
    result["n_atoms"] = torch.tensor([b["n_atoms"] for b in batch])

    # Token mask (1 for real tokens, 0 for padding)
    token_mask = torch.zeros(B, max_tokens, dtype=torch.bool)
    for i, b in enumerate(batch):
        token_mask[i, :b["n_tokens"]] = True
    result["token_mask"] = token_mask

    # Atom mask
    atom_mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    for i, b in enumerate(batch):
        atom_mask[i, :b["n_atoms"]] = True
    result["atom_mask"] = atom_mask

    # MSA features (pad to max_tokens along sequence dimension)
    max_msa_len = max(b["msa_profile"].shape[0] for b in batch)
    max_clusters = max(b["cluster_msa"].shape[0] for b in batch)

    msa_profiles = []
    cluster_msas = []
    cluster_profiles = []
    for b in batch:
        mp = b["msa_profile"]
        pad_l = max_tokens - mp.shape[0]
        if pad_l > 0:
            mp = torch.nn.functional.pad(mp, (0, 0, 0, pad_l))
        msa_profiles.append(mp[:max_tokens])

        cm = b["cluster_msa"]
        pad_c = max_clusters - cm.shape[0]
        pad_l = max_tokens - cm.shape[1]
        if pad_c > 0 or pad_l > 0:
            cm = torch.nn.functional.pad(cm, (0, max(0, pad_l), 0, max(0, pad_c)))
        cluster_msas.append(cm[:max_clusters, :max_tokens])

        cp = b["cluster_profile"]
        pad_c = max_clusters - cp.shape[0]
        pad_l = max_tokens - cp.shape[1]
        if pad_c > 0 or pad_l > 0:
            cp = torch.nn.functional.pad(cp, (0, 0, 0, max(0, pad_l), 0, max(0, pad_c)))
        cluster_profiles.append(cp[:max_clusters, :max_tokens])

    result["msa_profile"] = torch.stack(msa_profiles)
    result["cluster_msa"] = torch.stack(cluster_msas)
    result["cluster_profile"] = torch.stack(cluster_profiles)
    result["has_msa"] = torch.tensor([b["has_msa"].item() for b in batch])

    return result


def make_dataloader(
    structures: list[TokenizedStructure],
    crop_size: int = 384,
    batch_size: int = 1,
    num_workers: int = 4,
    msa_features: dict[str, MSAFeatures] | None = None,
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = HelicoDataset(
        structures=structures,
        crop_size=crop_size,
        msa_features=msa_features,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


# ============================================================================
# Synthetic data for testing
# ============================================================================

def make_synthetic_structure(
    n_residues: int = 50,
    n_chains: int = 1,
    include_ligand: bool = False,
) -> Structure:
    """Create a synthetic protein structure for testing."""
    chains = []
    for ci in range(n_chains):
        residues = []
        aa_names = ["ALA", "GLY", "VAL", "LEU", "ILE", "PHE", "TRP", "MET",
                     "PRO", "SER", "THR", "CYS", "TYR", "ASN", "GLN",
                     "ASP", "GLU", "LYS", "ARG", "HIS"]
        for ri in range(n_residues):
            name = aa_names[ri % len(aa_names)]
            # Simple helix-like coords
            theta = ri * 1.745  # ~100 degrees
            x = 2.3 * np.cos(theta)
            y = 2.3 * np.sin(theta)
            z = ri * 1.5 + ci * 30.0  # offset chains

            atoms = [
                Atom("N", "N", np.array([x - 0.5, y, z], dtype=np.float32), 20.0, 1.0),
                Atom("CA", "C", np.array([x, y, z], dtype=np.float32), 20.0, 1.0),
                Atom("C", "C", np.array([x + 0.5, y, z + 0.3], dtype=np.float32), 20.0, 1.0),
                Atom("O", "O", np.array([x + 0.5, y + 1.0, z + 0.3], dtype=np.float32), 20.0, 1.0),
                Atom("CB", "C", np.array([x, y + 1.2, z - 0.5], dtype=np.float32), 20.0, 1.0),
            ]
            residues.append(Residue(name=name, seq_id=ri + 1, atoms=atoms, is_standard=True))

        chain = Chain(
            chain_id=chr(65 + ci),
            entity_type="polymer",
            polymer_type="polypeptide(L)",
            residues=residues,
        )
        chains.append(chain)

    if include_ligand:
        # Add a small ligand
        lig_atoms = [
            Atom("C1", "C", np.array([5.0, 5.0, 5.0], dtype=np.float32), 20.0, 1.0),
            Atom("C2", "C", np.array([6.5, 5.0, 5.0], dtype=np.float32), 20.0, 1.0),
            Atom("O1", "O", np.array([5.75, 6.0, 5.0], dtype=np.float32), 20.0, 1.0),
            Atom("N1", "N", np.array([5.75, 4.0, 5.0], dtype=np.float32), 20.0, 1.0),
        ]
        lig_res = Residue(name="LIG", seq_id=1, atoms=lig_atoms, is_standard=False)
        lig_chain = Chain(
            chain_id="Z",
            entity_type="non-polymer",
            polymer_type="",
            residues=[lig_res],
        )
        chains.append(lig_chain)

    return Structure(
        pdb_id="SYNTH",
        chains=chains,
        resolution=2.0,
        release_date="2024-01-01",
        method="SYNTHETIC",
    )


def make_synthetic_batch(
    n_tokens: int = 32,
    n_atoms_per_token: int = 5,
    batch_size: int = 1,
    has_msa: bool = True,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create a synthetic feature batch for model testing."""
    n_atoms = n_tokens * n_atoms_per_token

    batch = {
        "token_types": torch.randint(0, NUM_TOKEN_TYPES, (batch_size, n_tokens), device=device),
        "chain_indices": torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        "res_indices": torch.arange(n_tokens, device=device).unsqueeze(0).expand(batch_size, -1),
        "rel_pos": torch.arange(n_tokens, device=device).unsqueeze(0).expand(batch_size, -1),
        "token_index": torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        "entity_id": torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        "sym_id": torch.zeros(batch_size, n_tokens, dtype=torch.long, device=device),
        "atom_coords": torch.randn(batch_size, n_atoms, 3, device=device),
        "ref_coords": torch.randn(batch_size, n_atoms, 3, device=device),
        "atom_to_token": torch.arange(n_tokens, device=device).repeat_interleave(n_atoms_per_token).unsqueeze(0).expand(batch_size, -1),
        "atoms_per_token": torch.full((batch_size, n_tokens), n_atoms_per_token, dtype=torch.long, device=device),
        "atom_element_idx": torch.randint(0, len(ELEMENTS), (batch_size, n_atoms), device=device),
        "chain_same": torch.ones(batch_size, n_tokens, n_tokens, dtype=torch.long, device=device),
        "token_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device),
        "atom_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device),
        "n_tokens": torch.tensor([n_tokens] * batch_size, device=device),
        "n_atoms": torch.tensor([n_atoms] * batch_size, device=device),
    }

    if has_msa:
        batch["msa_profile"] = torch.randn(batch_size, n_tokens, 22, device=device).softmax(dim=-1)
        batch["cluster_msa"] = torch.randint(0, 22, (batch_size, 4, n_tokens), device=device)
        batch["cluster_profile"] = torch.randn(batch_size, 4, n_tokens, 22, device=device).softmax(dim=-1)
        batch["has_msa"] = torch.ones(batch_size, device=device)
    else:
        batch["msa_profile"] = torch.zeros(batch_size, n_tokens, 22, device=device)
        batch["cluster_msa"] = torch.zeros(batch_size, 1, n_tokens, dtype=torch.long, device=device)
        batch["cluster_profile"] = torch.zeros(batch_size, 1, n_tokens, 22, device=device)
        batch["has_msa"] = torch.zeros(batch_size, device=device)

    return batch


# ============================================================================
# Preprocessing Pipeline
# ============================================================================

@dataclass
class StructureMetadata:
    """Metadata for a preprocessed structure."""
    pdb_id: str
    pickle_path: str       # relative to PROCESSED_DIR
    n_tokens: int
    n_atoms: int
    n_chains: int
    resolution: float
    release_date: str
    method: str
    entity_types: list[str]
    chain_ids: list[str]
    chain_sequences: dict[str, str] = field(default_factory=dict)  # chain_id -> sequence


def discover_mmcif_files(mmcif_dir: Path) -> list[Path]:
    """Find all .cif.gz files under the mmCIF directory."""
    return sorted(mmcif_dir.glob("**/*.cif.gz"))


# Module-level CCD cache for fork-inherited sharing across workers
_CCD_CACHE: dict | None = None


def _init_worker(ccd: dict) -> None:
    """Multiprocessing worker initializer: set the global CCD cache."""
    global _CCD_CACHE
    _CCD_CACHE = ccd


def _process_single_structure(args: tuple) -> StructureMetadata | None:
    """Process a single mmCIF file: parse → tokenize → pickle → return metadata.

    Args is a tuple of (cif_path, output_dir, max_resolution) to work with Pool.imap.
    """
    cif_path, output_dir, max_resolution = args
    global _CCD_CACHE

    try:
        structure = parse_mmcif(cif_path, max_resolution=max_resolution)
        if structure is None:
            return None

        tokenized = tokenize_structure(structure, ccd=_CCD_CACHE)
        if tokenized.n_tokens == 0:
            return None

        # Determine output path: structures/{subdir}/{pdb_id}.pkl
        pdb_id = structure.pdb_id.lower()
        subdir = pdb_id[1:3]
        rel_path = f"structures/{subdir}/{pdb_id}.pkl"
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)

        return StructureMetadata(
            pdb_id=structure.pdb_id,
            pickle_path=rel_path,
            n_tokens=tokenized.n_tokens,
            n_atoms=tokenized.n_atoms,
            n_chains=len(structure.chains),
            resolution=structure.resolution,
            release_date=structure.release_date,
            method=structure.method,
            entity_types=list(set(tokenized.entity_types)),
            chain_ids=tokenized.chain_ids,
            chain_sequences={c.chain_id: c.sequence for c in structure.chains if c.sequence},
        )
    except Exception as e:
        logger.warning(f"Failed to process {cif_path}: {e}")
        return None


def preprocess_structures(
    mmcif_dir: Path,
    output_dir: Path,
    ccd_cache_path: Path | None = None,
    max_resolution: float = 9.0,
    n_workers: int | None = None,
    skip_existing: bool = True,
) -> dict[str, StructureMetadata]:
    """Process all mmCIF files into pickled TokenizedStructures.

    Uses multiprocessing with fork to share the CCD cache read-only via COW.
    Returns dict mapping pdb_id -> StructureMetadata.
    """
    if n_workers is None:
        n_workers = min(32, os.cpu_count() or 4)

    # Load CCD
    logger.info("Loading CCD...")
    ccd = parse_ccd(cache_path=ccd_cache_path)
    logger.info(f"CCD loaded with {len(ccd)} components")

    # Discover files
    cif_files = discover_mmcif_files(mmcif_dir)
    logger.info(f"Found {len(cif_files)} mmCIF files")

    # Filter out already-processed structures if resuming
    if skip_existing:
        structures_dir = output_dir / "structures"
        if structures_dir.exists():
            existing = set()
            for pkl_path in structures_dir.glob("**/*.pkl"):
                existing.add(pkl_path.stem)
            before = len(cif_files)
            cif_files = [p for p in cif_files if p.name.split(".")[0].lower() not in existing]
            logger.info(f"Skipping {before - len(cif_files)} already-processed structures, {len(cif_files)} remaining")

    if not cif_files:
        logger.info("No files to process")
        # Load existing manifest if available
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            return load_manifest(manifest_path)
        return {}

    # Prepare args
    args_list = [(p, output_dir, max_resolution) for p in cif_files]

    # Process with multiprocessing
    metadata: dict[str, StructureMetadata] = {}
    ctx = multiprocessing.get_context("fork")
    n_done = 0
    n_total = len(args_list)

    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(ccd,)) as pool:
        for result in pool.imap_unordered(_process_single_structure, args_list, chunksize=16):
            n_done += 1
            if result is not None:
                metadata[result.pdb_id] = result
            if n_done % 1000 == 0:
                logger.info(f"Processed {n_done}/{n_total} files, {len(metadata)} structures kept")

    logger.info(f"Preprocessing complete: {len(metadata)} structures from {n_total} files")
    return metadata


def build_manifest(metadata: dict[str, StructureMetadata], output_path: Path | None = None) -> None:
    """Save structure metadata as a JSON manifest."""
    if output_path is None:
        output_path = _require_processed_dir() / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for pdb_id, m in metadata.items():
        data[pdb_id] = {
            "pdb_id": m.pdb_id,
            "pickle_path": m.pickle_path,
            "n_tokens": m.n_tokens,
            "n_atoms": m.n_atoms,
            "n_chains": m.n_chains,
            "resolution": m.resolution,
            "release_date": m.release_date,
            "method": m.method,
            "entity_types": m.entity_types,
            "chain_ids": m.chain_ids,
            "chain_sequences": m.chain_sequences,
        }

    with open(output_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Manifest saved to {output_path} ({len(data)} entries)")


def load_manifest(path: Path | None = None) -> dict[str, StructureMetadata]:
    """Load a manifest JSON into StructureMetadata dict."""
    if path is None:
        path = _require_processed_dir() / "manifest.json"

    with open(path, "r") as f:
        data = json.load(f)

    metadata = {}
    for pdb_id, d in data.items():
        metadata[pdb_id] = StructureMetadata(
            pdb_id=d["pdb_id"],
            pickle_path=d["pickle_path"],
            n_tokens=d["n_tokens"],
            n_atoms=d["n_atoms"],
            n_chains=d["n_chains"],
            resolution=d["resolution"],
            release_date=d["release_date"],
            method=d["method"],
            entity_types=d["entity_types"],
            chain_ids=d["chain_ids"],
            chain_sequences=d.get("chain_sequences", {}),
        )

    return metadata


# ============================================================================
# Lazy Dataset (for real data training)
# ============================================================================

class LazyHelicoDataset(Dataset):
    """PyTorch dataset that loads pickled structures on demand.

    Unlike HelicoDataset which holds all structures in memory, this loads
    each structure from disk when accessed. Suitable for 249K+ structures.
    """

    def __init__(
        self,
        manifest: dict[str, StructureMetadata],
        processed_dir: Path,
        crop_size: int = 384,
        crop_mode: str = "spatial",
        msa_tar_indices: list[TarIndex] | None = None,
        msa_dir: Path | None = None,
        filter_fn: callable | None = None,
    ):
        self.processed_dir = processed_dir
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.msa_tar_indices = msa_tar_indices or []
        self.msa_dir = msa_dir

        if filter_fn is not None:
            self.entries = [(pid, m) for pid, m in manifest.items() if filter_fn(m)]
        else:
            self.entries = list(manifest.items())

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pdb_id, meta = self.entries[idx]

        # Load pickled TokenizedStructure
        pkl_path = self.processed_dir / meta.pickle_path
        with open(pkl_path, "rb") as f:
            tokenized: TokenizedStructure = pickle.load(f)

        # Backward compat: old pickles may lack chain_sequences
        if not hasattr(tokenized, "chain_sequences"):
            tokenized.chain_sequences = {}

        # Use sequences from metadata (always up-to-date) with pickle as fallback
        chain_seqs = meta.chain_sequences or tokenized.chain_sequences

        features = tokenized.to_features()

        # Apply cropping
        if self.crop_mode == "spatial":
            features = spatial_crop(features, self.crop_size)
        else:
            features = contiguous_crop(features, self.crop_size)

        # Load MSA features for the first polymer chain
        msa_feat = None
        seen_chains = set()
        for chain_id in tokenized.chain_ids:
            if chain_id in seen_chains:
                continue
            seen_chains.add(chain_id)
            seq = chain_seqs.get(chain_id, "")
            for tar_idx in self.msa_tar_indices:
                msa_feat = load_msa_for_chain(
                    pdb_id, chain_id,
                    sequence=seq,
                    msa_dir=self.msa_dir,
                    tar_index=tar_idx,
                )
                if msa_feat is not None:
                    break
            if msa_feat is None and self.msa_dir:
                msa_feat = load_msa_for_chain(pdb_id, chain_id, sequence=seq, msa_dir=self.msa_dir)
            if msa_feat is not None:
                break

        # Add MSA features
        if msa_feat is not None:
            features["msa_profile"] = torch.tensor(msa_feat.profile, dtype=torch.float32)
            features["cluster_msa"] = torch.tensor(msa_feat.cluster_msa, dtype=torch.long)
            features["cluster_profile"] = torch.tensor(msa_feat.cluster_profile, dtype=torch.float32)
            features["has_msa"] = torch.tensor(1)
        else:
            n_tok = features["n_tokens"]
            features["msa_profile"] = torch.zeros(n_tok, 22)
            features["cluster_msa"] = torch.zeros(1, n_tok, dtype=torch.long)
            features["cluster_profile"] = torch.zeros(1, n_tok, 22)
            features["has_msa"] = torch.tensor(0)

        return features


# ============================================================================
# Preprocessing CLI
# ============================================================================

def preprocess_main():
    """CLI entry point for helico-preprocess."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Helico Preprocessing Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Preprocessing subcommand")

    # structures subcommand
    sp_struct = subparsers.add_parser("structures", help="Process mmCIF files into pickled structures")
    sp_struct.add_argument("--mmcif-dir", type=Path, default=None, help="Path to mmCIF dir (default: $HELICO_RAW_DIR/mmCIF)")
    sp_struct.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: $HELICO_PROCESSED_DIR)")
    sp_struct.add_argument("--ccd-cache", type=Path, default=None, help="CCD cache path (default: $HELICO_PROCESSED_DIR/ccd_cache.pkl)")
    sp_struct.add_argument("--max-resolution", type=float, default=9.0)
    sp_struct.add_argument("--n-workers", type=int, default=None)
    sp_struct.add_argument("--no-skip-existing", action="store_true")

    # msa-index subcommand
    sp_msa = subparsers.add_parser("msa-index", help="Build tar index for MSA archives")
    sp_msa.add_argument("--tar-path", type=Path, required=True)
    sp_msa.add_argument("--output", type=Path, required=True)

    # all subcommand
    sp_all = subparsers.add_parser("all", help="Run full preprocessing pipeline")
    sp_all.add_argument("--mmcif-dir", type=Path, default=None, help="Path to mmCIF dir (default: $HELICO_RAW_DIR/mmCIF)")
    sp_all.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: $HELICO_PROCESSED_DIR)")
    sp_all.add_argument("--ccd-cache", type=Path, default=None, help="CCD cache path (default: $HELICO_PROCESSED_DIR/ccd_cache.pkl)")
    sp_all.add_argument("--max-resolution", type=float, default=9.0)
    sp_all.add_argument("--n-workers", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Resolve defaults from env vars
    def _resolve(val: Path | None, env_var: str, suffix: str = "") -> Path:
        if val is not None:
            return val
        base = os.environ.get(env_var)
        if base is None:
            raise EnvironmentError(f"Set {env_var} env var or pass the path explicitly")
        return Path(base) / suffix if suffix else Path(base)

    if args.command == "structures":
        mmcif_dir = _resolve(args.mmcif_dir, "HELICO_RAW_DIR", "mmCIF")
        output_dir = _resolve(args.output_dir, "HELICO_PROCESSED_DIR")
        ccd_cache = _resolve(args.ccd_cache, "HELICO_PROCESSED_DIR", "ccd_cache.pkl")
        metadata = preprocess_structures(
            mmcif_dir=mmcif_dir,
            output_dir=output_dir,
            ccd_cache_path=ccd_cache,
            max_resolution=args.max_resolution,
            n_workers=args.n_workers,
            skip_existing=not args.no_skip_existing,
        )
        build_manifest(metadata, output_dir / "manifest.json")

    elif args.command == "msa-index":
        logger.info(f"Building tar index for {args.tar_path}...")
        index = build_tar_index(args.tar_path)
        save_tar_index(index, args.output)
        logger.info(f"Index saved to {args.output} ({len(index.entries)} entries)")

    elif args.command == "all":
        raw_dir = _resolve(None, "HELICO_RAW_DIR")
        mmcif_dir = _resolve(args.mmcif_dir, "HELICO_RAW_DIR", "mmCIF")
        output_dir = _resolve(args.output_dir, "HELICO_PROCESSED_DIR")
        ccd_cache = _resolve(args.ccd_cache, "HELICO_PROCESSED_DIR", "ccd_cache.pkl")

        # Step 1: Process structures
        logger.info("=== Step 1: Processing structures ===")
        metadata = preprocess_structures(
            mmcif_dir=mmcif_dir,
            output_dir=output_dir,
            ccd_cache_path=ccd_cache,
            max_resolution=args.max_resolution,
            n_workers=args.n_workers,
        )
        build_manifest(metadata, output_dir / "manifest.json")

        # Step 2: Build MSA tar indices
        for name, tar_path in [
            ("rcsb_raw_msa", raw_dir / "rcsb_raw_msa.tar"),
            ("openfold_raw_msa", raw_dir / "openfold_raw_msa.tar"),
        ]:
            if tar_path.exists():
                logger.info(f"=== Step 2: Building tar index for {name} ===")
                index = build_tar_index(tar_path)
                out = output_dir / f"{name}_index.pkl"
                save_tar_index(index, out)
                logger.info(f"Index saved to {out} ({len(index.entries)} entries)")

        logger.info("=== Preprocessing complete ===")
