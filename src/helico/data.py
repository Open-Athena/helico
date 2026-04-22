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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Paths — data auto-downloads from HuggingFace to ~/.cache/helico/data/
# ---------------------------------------------------------------------------
HF_REPO = "timodonnell/helico-data"


def _default_data_dir() -> Path:
    return Path(os.environ.get("HELICO_DATA_DIR", "~/.cache/helico/data")).expanduser()


def _raw_dir() -> Path:
    return _default_data_dir() / "raw"


def _processed_dir() -> Path:
    return _default_data_dir() / "processed"


# Module-level constants for backward compat with test imports
RAW_DIR = _raw_dir()
PROCESSED_DIR = _processed_dir()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
UNK_AA_IDX = len(STANDARD_AAS)  # 20

NUCLEOTIDES = "ACGTU"
NUC_TO_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}
UNK_NUC_IDX = len(NUCLEOTIDES)

# Element types for ligand token classification
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Se", "B", "Si", "Fe", "Zn",
            "Mg", "Ca", "Mn", "Cu", "Co", "Ni", "Mo", "Na", "K"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
UNK_ELEM_IDX = len(ELEMENTS)

# Atomic numbers for ref_element one-hot (Protenix convention: index = atomic_number - 1)
ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Xe": 54, "W": 74, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
}
UNK_ATOMIC_IDX = 127  # last slot in 128-class one-hot

# Token type offsets
TOKEN_PROTEIN = 0        # 0-20 (20 AAs + UNK)
TOKEN_NUCLEOTIDE = 21    # 21-26 (5 nucleotides + UNK)
TOKEN_LIGAND_ATOM = 27   # ligand atoms start here

NUM_TOKEN_TYPES = 28 + UNK_ELEM_IDX + 1  # protein + nucleotide + per-element ligand tokens

# Protenix 32-class MSA encoding (matches Protenix checkpoint expectations)
PROTENIX_MSA_AAS = "ARNDCQEGHILKMFPSTWYV"  # 3-letter alphabetical order
PROTENIX_MSA_AA_MAP = {aa: i for i, aa in enumerate(PROTENIX_MSA_AAS)}
PROTENIX_MSA_UNK_PROTEIN = 20
PROTENIX_MSA_RNA = {c: 21 + i for i, c in enumerate("AGCU")}  # 21-24 (purines first)
PROTENIX_MSA_UNK_RNA = 25
PROTENIX_MSA_DNA = {c: 26 + i for i, c in enumerate("AGCT")}  # 26-29 (purines first)
PROTENIX_MSA_UNK_DNA = 30
PROTENIX_MSA_GAP = 31
PROTENIX_NUM_MSA_CLASSES = 32

# Map Helico token_types indices to Protenix restype indices
# token_types 0-19 = ACDEFGHIKLMNPQRSTVWY (1-letter alpha)
# Protenix restype 0-19 = ARNDCQEGHILKMFPSTWYV (3-letter alpha)
_HELICO_AA_TO_PROTENIX = [PROTENIX_MSA_AA_MAP[aa] for aa in STANDARD_AAS]
TOKEN_TYPE_TO_RESTYPE = _HELICO_AA_TO_PROTENIX + [PROTENIX_MSA_UNK_PROTEIN]  # idx 20 = UNK
# For nucleotides (token_types 21-26) and ligands (27+), map to gap (31) as placeholder
TOKEN_TYPE_TO_RESTYPE += [PROTENIX_MSA_GAP] * (NUM_TOKEN_TYPES - 21)

# CCD res_name -> Protenix restype index for nucleotides.
# TOKEN_TYPE_TO_RESTYPE can't distinguish RNA from DNA (both map to gap), so we
# fix up nucleotide restypes using the token's res_name (CCD code) instead.
RES_NAME_TO_RESTYPE: dict[str, int] = {
    # RNA — purines first: A=21, G=22, C=23, U=24 (matching Protenix)
    "A": 21, "G": 22, "C": 23, "U": 24,
    # DNA — purines first: DA=26, DG=27, DC=28, DT=29 (matching Protenix)
    "DA": 26, "DG": 27, "DC": 28, "DT": 29,
}

# Amino acid 3-letter <-> 1-letter mappings
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
RNA_ONE_TO_CCD = {"A": "A", "C": "C", "G": "G", "U": "U", "N": "N"}
DNA_ONE_TO_CCD = {"A": "DA", "C": "DC", "G": "DG", "T": "DT", "N": "DN"}

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

    @property
    def non_leaving_heavy_atom_mask(self) -> list[bool]:
        """Mask for heavy atoms that are NOT leaving atoms (e.g. OP3, OXT)."""
        if self.atom_leaving:
            return [e != "H" and not lv for e, lv in zip(self.atom_elements, self.atom_leaving)]
        return self.heavy_atom_mask


def parse_ccd(cif_path: Path | None = None, cache_path: Path | None = None) -> dict[str, CCDComponent]:
    """Parse the Chemical Component Dictionary into a lookup table.

    Returns dict mapping 3-letter code -> CCDComponent.
    Caches result as pickle for fast reload.
    Auto-downloads from HuggingFace if not found locally.
    """
    if cache_path is None:
        cache_path = _processed_dir() / "ccd_cache.pkl"
    if cif_path is None:
        cif_path = _raw_dir() / "components.cif"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            components = pickle.load(f)
        # Strip CIF quoting from atom names (e.g. "C3'" -> C3')
        for comp in components.values():
            comp.atom_names = [n.strip('"') for n in comp.atom_names]
            comp.bonds = [(a1.strip('"'), a2.strip('"'), o) for a1, a2, o in comp.bonds]
        return components

    # If components.cif is available, parse it directly
    if cif_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Try auto-download from HuggingFace
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            logger.info("CCD cache not found locally, downloading from HuggingFace...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename="processed/ccd_cache.pkl",
                repo_type="dataset",
                local_dir=_default_data_dir(),
            )
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not download CCD cache from HuggingFace: {e}")

        raise FileNotFoundError(
            f"CCD cache not found at {cache_path} and components.cif not found at {cif_path}. "
            f"Run 'helico-download --subset ccd-only' to download, or provide paths explicitly."
        )

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
        if current_comp is not None:
            # Single-atom CCD entries (CL, NA, MG, ZN, ...) use the mmCIF
            # single-value format, not a loop_ block. Materialize the
            # accumulated row here if atom_names is still empty.
            if not current_comp.atom_names and hasattr(current_comp, "_single_atom_row"):
                row = current_comp._single_atom_row
                atom_id = row.get("atom_id", "").strip('"')
                type_symbol = row.get("type_symbol", "")
                if atom_id and type_symbol:
                    current_comp.atom_names.append(atom_id)
                    current_comp.atom_elements.append(type_symbol)
                    try:
                        current_comp.atom_charges.append(
                            int(row.get("charge", "0")) if row.get("charge", "0") not in ("?", ".", "") else 0
                        )
                    except (ValueError, TypeError):
                        current_comp.atom_charges.append(0)
                    current_comp.atom_leaving.append(
                        row.get("pdbx_leaving_atom_flag", "N") == "Y"
                    )
                    try:
                        x = float(row.get("pdbx_model_Cartn_x_ideal", "?"))
                        y = float(row.get("pdbx_model_Cartn_y_ideal", "?"))
                        z = float(row.get("pdbx_model_Cartn_z_ideal", "?"))
                        coords_x.append(x); coords_y.append(y); coords_z.append(z)
                    except (ValueError, TypeError):
                        coords_x.append(0.0); coords_y.append(0.0); coords_z.append(0.0)
                delattr(current_comp, "_single_atom_row")
            if coords_x:
                current_comp.ideal_coords = np.array(
                    list(zip(coords_x, coords_y, coords_z)), dtype=np.float32
                )
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

            # Atom loop fields (or single-value atom row if followed by a value
            # on the same line — used for single-atom CCD entries like CL, NA).
            elif line_stripped.startswith("_chem_comp_atom."):
                # Split into field and optional value. If there's a value on
                # the same line, this is a single-atom compound using the
                # single-value mmCIF format.
                full = line_stripped[len("_chem_comp_atom."):]
                parts = full.split(None, 1)
                field_name = parts[0].strip()
                if len(parts) > 1:
                    # Single-value: accumulate into `_single_atom_row` buffer.
                    if not hasattr(current_comp, "_single_atom_row"):
                        current_comp._single_atom_row = {}
                    current_comp._single_atom_row[field_name] = parts[1].strip().strip('"')
                else:
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
                current_comp.atom_names.append(row.get("atom_id", "").strip('"'))
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
                    row.get("atom_id_1", "").strip('"'),
                    row.get("atom_id_2", "").strip('"'),
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
    # Covalent connections from _struct_conn: list of (asym1, seq_id1, comp1, atom1, asym2, seq_id2, comp2, atom2)
    struct_conn: list[tuple[str, str, str, str, str, str, str, str]] = field(default_factory=list)

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

    # Parse _struct_conn for covalent bonds (COVALE, METALC, DISULF)
    struct_conn = []
    conn_types = mmcif_dict.get("_struct_conn.conn_type_id", [])
    conn_asym1 = mmcif_dict.get("_struct_conn.ptnr1_label_asym_id", [])
    conn_seq1 = mmcif_dict.get("_struct_conn.ptnr1_label_seq_id", [])
    conn_comp1 = mmcif_dict.get("_struct_conn.ptnr1_label_comp_id", [])
    conn_atom1 = mmcif_dict.get("_struct_conn.ptnr1_label_atom_id", [])
    conn_asym2 = mmcif_dict.get("_struct_conn.ptnr2_label_asym_id", [])
    conn_seq2 = mmcif_dict.get("_struct_conn.ptnr2_label_seq_id", [])
    conn_comp2 = mmcif_dict.get("_struct_conn.ptnr2_label_comp_id", [])
    conn_atom2 = mmcif_dict.get("_struct_conn.ptnr2_label_atom_id", [])
    for i in range(len(conn_types)):
        ctype = conn_types[i].upper() if i < len(conn_types) else ""
        if ctype not in ("COVALE", "METALC", "DISULF"):
            continue
        struct_conn.append((
            conn_asym1[i] if i < len(conn_asym1) else "",
            conn_seq1[i] if i < len(conn_seq1) else "",
            conn_comp1[i] if i < len(conn_comp1) else "",
            conn_atom1[i] if i < len(conn_atom1) else "",
            conn_asym2[i] if i < len(conn_asym2) else "",
            conn_seq2[i] if i < len(conn_seq2) else "",
            conn_comp2[i] if i < len(conn_comp2) else "",
            conn_atom2[i] if i < len(conn_atom2) else "",
        ))

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
        struct_conn=struct_conn,
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
    ref_space_uid: int = -1   # residue group ID for atom pair masking (same for all atoms in a residue)
    atom_charges: list[int] | None = None  # per-atom formal charges from CCD


def _encode_atom_name_chars(atom_names: list[str]) -> torch.Tensor:
    """Encode atom names as 4-char × 64-class one-hot (Protenix convention).

    Each atom name is left-justified and padded to 4 characters.
    Each character is mapped to index ord(c) - 32, then one-hot encoded to 64 dims.
    Returns: (N_atoms, 256) float tensor.
    """
    n = len(atom_names)
    if n == 0:
        return torch.zeros(0, 256)
    indices = torch.zeros(n, 4, dtype=torch.long)
    for i, name in enumerate(atom_names):
        padded = name.ljust(4)[:4]
        for j, c in enumerate(padded):
            indices[i, j] = max(0, min(63, ord(c) - 32))
    onehot = F.one_hot(indices, 64).float()  # (N, 4, 64)
    return onehot.reshape(n, 256)


@dataclass
class TokenizedStructure:
    """Tokenized representation of a structure."""
    pdb_id: str
    tokens: list[Token]
    chain_ids: list[str]          # chain_id per token
    entity_types: list[str]       # entity type per token
    chain_sequences: dict[str, str] = field(default_factory=dict)  # chain_id -> sequence
    # Sparse edge list of (ti, tj) covalent/metallic/disulfide + intra-ligand CCD bonds.
    # Stored sparse because a dense (N_tok, N_tok) tensor is quadratic in n_tokens and
    # blew up preprocess workers to 200+GB RSS on ribosomes/capsids. Materialized to
    # dense (N_tok, N_tok) in to_features(). Older pickles may store a Tensor directly.
    token_bonds: list[tuple[int, int]] | torch.Tensor | None = None

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

        # Restype: Protenix 32-class index per token.
        # TOKEN_TYPE_TO_RESTYPE maps nucleotides and ligands to gap (31) as a
        # placeholder; fix that up here using res_name and entity_type.
        # IMPORTANT: restype 31 is the MSA-gap sentinel — if a ligand token is
        # left at 31 the restype embedding treats it as "MSA gap" rather than
        # a real residue, which silently corrupts every ligand token.
        # Protenix uses UNK (20) for ligand tokens and modified residues,
        # and RNA_UNK (25) / DNA_UNK (30) for unknown nucleotides.
        restype = [TOKEN_TYPE_TO_RESTYPE[t.token_type] for t in self.tokens]
        for i, token in enumerate(self.tokens):
            if restype[i] != PROTENIX_MSA_GAP:
                continue
            mapped = RES_NAME_TO_RESTYPE.get(token.res_name)
            if mapped is not None:
                restype[i] = mapped
                continue
            etype = self.entity_types[i]
            if etype == "nucleotide":
                restype[i] = (
                    PROTENIX_MSA_UNK_DNA
                    if token.res_name.startswith("D")
                    else PROTENIX_MSA_UNK_RNA
                )
            else:
                # Ligand, modified residue, or anything we couldn't classify
                restype[i] = PROTENIX_MSA_UNK_PROTEIN
        restype = torch.tensor(restype, dtype=torch.long)

        # Relative position encoding (within each chain)
        rel_pos = torch.zeros(n_tok, dtype=torch.long)
        for i, tok in enumerate(self.tokens):
            rel_pos[i] = tok.res_idx

        # Atom-level features: flatten all atoms across tokens
        all_atom_coords = []
        all_atom_elements = []
        all_ref_coords = []
        all_ref_space_uid = []
        all_ref_charge = []
        atom_to_token = []  # maps each atom to its token index
        atoms_per_token = []
        rep_atom_idx = []  # representative atom index per token (CB/CA/C4/C2/single)
        has_frame = []  # whether token has a backbone frame (for pTM/ipTM filtering)

        # Nucleotide representative atom names
        _PURINE_RES = {"A", "G", "DA", "DG"}
        _PYRIMIDINE_RES = {"C", "U", "DC", "DT"}

        atom_offset = 0
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
            # ref_space_uid: group atoms by residue (same for all atoms in same residue)
            # If not set (-1), fall back to token index
            uid = tok.ref_space_uid if tok.ref_space_uid >= 0 else ti
            all_ref_space_uid.extend([uid] * n_atoms)
            # ref_charge: formal charge per atom from CCD
            if tok.atom_charges is not None and len(tok.atom_charges) == n_atoms:
                all_ref_charge.extend(tok.atom_charges)
            else:
                all_ref_charge.extend([0] * n_atoms)

            # Representative atom: CB for proteins (CA for GLY), C4/C2 for nucleotides, atom 0 for ligands
            rep_offset = 0  # default: first atom
            if n_atoms == 1:
                rep_offset = 0  # ligand single-atom tokens
            elif tok.res_name in _PURINE_RES:
                rep_offset = tok.atom_names.index("C4") if "C4" in tok.atom_names else 0
            elif tok.res_name in _PYRIMIDINE_RES:
                rep_offset = tok.atom_names.index("C2") if "C2" in tok.atom_names else 0
            elif tok.res_name == "GLY":
                rep_offset = tok.atom_names.index("CA") if "CA" in tok.atom_names else 0
            elif "CB" in tok.atom_names:
                rep_offset = tok.atom_names.index("CB")
            elif "CA" in tok.atom_names:
                rep_offset = tok.atom_names.index("CA")
            rep_atom_idx.append(atom_offset + rep_offset)

            # has_frame: token has a rigid backbone frame (for pTM / ipTM masking).
            # Protein needs N, CA, C present. Nucleic acid needs C1', C3', C4'.
            # Ligand: ≥3 atoms. Previously this was unconditionally True for
            # protein/nucleotide which inflated pTM for tokens with partial
            # backbone atoms (e.g. disordered termini) and skewed best-of-N
            # sample selection.
            etype = self.entity_types[ti] if ti < len(self.entity_types) else "ligand"
            if etype == "protein":
                names = set(tok.atom_names)
                has_frame.append({"N", "CA", "C"}.issubset(names))
            elif etype == "nucleotide":
                names = set(tok.atom_names)
                has_frame.append({"C1'", "C3'", "C4'"}.issubset(names))
            else:
                has_frame.append(n_atoms >= 3)

            atom_offset += n_atoms

        total_atoms = sum(atoms_per_token)
        atom_coords = torch.tensor(np.concatenate(all_atom_coords, axis=0), dtype=torch.float32) if total_atoms > 0 else torch.zeros(0, 3)

        # Centralize ref_coords per ref_space_uid group (matches Protenix convention).
        # This must be done across tokens sharing the same uid — per-token centralization
        # would zero out single-atom ligand tokens.
        if total_atoms > 0:
            ref_flat = np.concatenate(all_ref_coords, axis=0)
            uid_arr = np.array(all_ref_space_uid)
            for uid in np.unique(uid_arr):
                mask = uid_arr == uid
                ref_flat[mask] -= ref_flat[mask].mean(axis=0, keepdims=True)
            ref_coords = torch.tensor(ref_flat, dtype=torch.float32)
        else:
            ref_coords = torch.zeros(0, 3)
        atom_to_token_idx = torch.tensor(atom_to_token, dtype=torch.long)
        atoms_per_token_t = torch.tensor(atoms_per_token, dtype=torch.long)
        ref_space_uid = torch.tensor(all_ref_space_uid, dtype=torch.long)
        ref_charge = torch.tensor(all_ref_charge, dtype=torch.float32)

        # Element indices for atoms (atomic_number - 1, matching Protenix ref_element)
        atom_element_idx = torch.tensor(
            [ATOMIC_NUMBER[e] - 1 if e in ATOMIC_NUMBER else UNK_ATOMIC_IDX
             for e in all_atom_elements],
            dtype=torch.long,
        )

        # Atom name chars: 4 chars × 64-class one-hot = 256 features per atom
        # Encoding: ord(c) - 32 for each character, left-justified padded to 4 chars
        all_atom_names: list[str] = []
        for tok in self.tokens:
            all_atom_names.extend(tok.atom_names)
        atom_name_chars = _encode_atom_name_chars(all_atom_names)

        # Chain-pair mask (same chain = 1, different chain = 0)
        chain_same = (chain_indices.unsqueeze(0) == chain_indices.unsqueeze(1)).long()

        # Per-token relpe features. token_index is a global monotonic index
        # (Protenix convention); the per-token stored value is ignored. Within
        # one residue (a ligand), atom tokens are consecutive in the token list
        # so their token_index differences encode their relative order.
        token_index = torch.arange(n_tok, dtype=torch.long)
        entity_id = torch.tensor([t.entity_id for t in self.tokens], dtype=torch.long)
        sym_id = torch.tensor([t.sym_id for t in self.tokens], dtype=torch.long)

        result = {
            "token_types": token_types,            # (N_tok,)
            "restype": restype,                      # (N_tok,) Protenix 32-class index
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
            "atom_name_chars": atom_name_chars,      # (N_atoms, 256)
            "ref_space_uid": ref_space_uid,          # (N_atoms,)
            "ref_charge": ref_charge,                # (N_atoms,)
            "rep_atom_idx": torch.tensor(rep_atom_idx, dtype=torch.long),  # (N_tok,)
            "has_frame": torch.tensor(has_frame, dtype=torch.bool),  # (N_tok,)
            "chain_same": chain_same,                # (N_tok, N_tok)
            "n_tokens": n_tok,
            "n_atoms": total_atoms,
        }
        if self.token_bonds is not None:
            if isinstance(self.token_bonds, torch.Tensor):
                # Legacy pickles stored a dense (N_tok, N_tok) tensor.
                result["token_bonds"] = self.token_bonds
            elif self.token_bonds:
                bonds = torch.zeros(n_tok, n_tok)
                for ti, tj in self.token_bonds:
                    bonds[ti, tj] = 1.0
                    bonds[tj, ti] = 1.0
                result["token_bonds"] = bonds
        return result


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
    residue_uid_counter = 0  # groups atoms by residue for ref_space_uid

    # Mapping for token_bonds: (asym_id, seq_id_str) → token index for polymers,
    # (asym_id, seq_id_str, atom_name) → token index for ligands (one token per atom)
    residue_to_token: dict[tuple[str, str], int] = {}
    atom_to_token_bond: dict[tuple[str, str, str], int] = {}
    ligand_bond_edges: list[tuple[int, int]] = []

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
                atom_charges = None
                if ccd and res.name in ccd:
                    comp = ccd[res.name]
                    ref_c = comp.ideal_coords
                    if ref_c is not None:
                        # Match by heavy atoms only
                        mask = comp.heavy_atom_mask
                        ref_c = ref_c[mask] if len(mask) == len(ref_c) else None
                        if ref_c is not None and len(ref_c) != len(heavy_atoms):
                            ref_c = None  # shape mismatch, skip
                    if comp.atom_charges:
                        mask = comp.heavy_atom_mask
                        atom_charges = [c for c, m in zip(comp.atom_charges, mask) if m]
                        if len(atom_charges) != len(heavy_atoms):
                            atom_charges = None

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
                    ref_space_uid=residue_uid_counter,
                    atom_charges=atom_charges,
                ))
                chain_ids.append(chain.chain_id)
                entity_types.append("protein")
                tok_idx = len(tokens) - 1
                residue_to_token[(chain.chain_id, str(res.seq_id))] = tok_idx
                for aname in atom_names:
                    atom_to_token_bond[(chain.chain_id, str(res.seq_id), aname)] = tok_idx
                res_counter += 1
                residue_uid_counter += 1

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
                atom_charges = None
                # Match CCD ideal coords to structure atoms by NAME, not by
                # position. Positional matching used to require the non-leaving
                # heavy-atom list to exactly equal the structure's atom set; a
                # single mismatch (e.g. a 5' OP3 present or missing) wiped the
                # entire ref_c to None.
                if ccd and res.name in ccd:
                    comp = ccd[res.name]
                    if comp.ideal_coords is not None:
                        mask = comp.heavy_atom_mask
                        if len(mask) == len(comp.ideal_coords):
                            heavy_names = comp.heavy_atom_names
                            heavy_coords = comp.ideal_coords[mask]
                            name_to_ref = {n: c for n, c in zip(heavy_names, heavy_coords)}
                            matched = [
                                name_to_ref.get(n, np.zeros(3, dtype=np.float32))
                                for n in atom_names
                            ]
                            ref_c = np.stack(matched).astype(np.float32)
                    if comp.atom_charges:
                        mask = comp.heavy_atom_mask
                        heavy_names_c = comp.heavy_atom_names
                        heavy_charges = [c for c, m in zip(comp.atom_charges, mask) if m]
                        name_to_charge = dict(zip(heavy_names_c, heavy_charges))
                        atom_charges = [name_to_charge.get(n, 0) for n in atom_names]

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
                    ref_space_uid=residue_uid_counter,
                    atom_charges=atom_charges,
                ))
                chain_ids.append(chain.chain_id)
                entity_types.append("nucleotide")
                tok_idx = len(tokens) - 1
                residue_to_token[(chain.chain_id, str(res.seq_id))] = tok_idx
                for aname in atom_names:
                    atom_to_token_bond[(chain.chain_id, str(res.seq_id), aname)] = tok_idx
                res_counter += 1
                residue_uid_counter += 1

            else:
                # Ligand / modified residue: one token per heavy atom
                # Look up CCD for ref_coords and charges
                lig_ref_coords = None
                lig_charges = None
                if ccd and res.name in ccd:
                    comp = ccd[res.name]
                    if comp.ideal_coords is not None:
                        mask = comp.heavy_atom_mask
                        if len(mask) == len(comp.ideal_coords):
                            ccd_heavy_names = comp.heavy_atom_names
                            ccd_heavy_coords = comp.ideal_coords[mask]
                            # Build name->coord map for matching
                            ccd_name_to_coord = {n: c for n, c in zip(ccd_heavy_names, ccd_heavy_coords)}
                            lig_ref_coords = ccd_name_to_coord
                    if comp.atom_charges:
                        mask = comp.heavy_atom_mask
                        ccd_heavy_charges = [c for c, m in zip(comp.atom_charges, mask) if m]
                        ccd_heavy_names_c = comp.heavy_atom_names
                        lig_charges = {n: c for n, c in zip(ccd_heavy_names_c, ccd_heavy_charges)}

                # All atoms of this ligand residue share the same ref_space_uid
                lig_uid = residue_uid_counter
                residue_uid_counter += 1

                # All atoms of a ligand residue must share res_idx so that
                # RelPE's `same_res` is True between them — needed for
                # `a_rel_token` to carry intra-ligand relative positions.
                lig_res_idx = res_counter
                lig_name_to_tok: dict[str, int] = {}
                for atom_idx, atom in enumerate(heavy_atoms):
                    elem_idx = ELEM_TO_IDX.get(atom.element, UNK_ELEM_IDX)
                    token_type = TOKEN_LIGAND_ATOM + elem_idx

                    # Get ref_coords from CCD by atom name
                    ref_c = None
                    if lig_ref_coords is not None and atom.name in lig_ref_coords:
                        ref_c = lig_ref_coords[atom.name].reshape(1, 3)

                    # Get charge from CCD by atom name
                    charge = [lig_charges.get(atom.name, 0)] if lig_charges else None

                    tokens.append(Token(
                        token_type=token_type,
                        chain_idx=chain_idx,
                        res_idx=lig_res_idx,
                        atom_names=[atom.name],
                        atom_elements=[atom.element],
                        atom_coords=atom.coords.reshape(1, 3),
                        ref_coords=ref_c,
                        res_name=res.name,
                        entity_id=chain_to_entity[chain.chain_id],
                        sym_id=0,
                        token_index=0,  # overwritten to global arange in to_features
                        ref_space_uid=lig_uid,
                        atom_charges=charge,
                    ))
                    chain_ids.append(chain.chain_id)
                    entity_types.append("ligand")
                    tok_idx = len(tokens) - 1
                    atom_to_token_bond[(chain.chain_id, str(res.seq_id), atom.name)] = tok_idx
                    lig_name_to_tok[atom.name] = tok_idx
                    # Also map residue for ligands (first atom wins; others reachable via atom key)
                    if (chain.chain_id, str(res.seq_id)) not in residue_to_token:
                        residue_to_token[(chain.chain_id, str(res.seq_id))] = tok_idx

                # Emit intra-ligand CCD bonds as token_bonds (Bug 3). Without
                # these, multi-atom ligands lose their chemistry graph and the
                # model has no bond signal between ligand atoms.
                if ccd and res.name in ccd:
                    for a1, a2, _order in ccd[res.name].bonds:
                        ti = lig_name_to_tok.get(a1)
                        tj = lig_name_to_tok.get(a2)
                        if ti is not None and tj is not None and ti != tj:
                            ligand_bond_edges.append((ti, tj))
                res_counter += 1

    # Compute sym_id: per-entity counter so homo-multimer chains get distinct sym_ids
    entity_sym_counter: dict[int, int] = {}
    chain_sym: dict[str, int] = {}
    for token, cid in zip(tokens, chain_ids):
        if cid not in chain_sym:
            eid = token.entity_id
            chain_sym[cid] = entity_sym_counter.get(eid, 0)
            entity_sym_counter[eid] = entity_sym_counter.get(eid, 0) + 1
        token.sym_id = chain_sym[cid]

    # Compute token_bonds from _struct_conn (covalent/metallic/disulfide bonds)
    # plus intra-ligand CCD bonds (Bug 3). Emit sparse edge list — see note on
    # TokenizedStructure.token_bonds.
    token_bond_edges: list[tuple[int, int]] = []
    if structure.struct_conn or ligand_bond_edges:
        for asym1, seq1, _comp1, atom1, asym2, seq2, _comp2, atom2 in structure.struct_conn:
            # Try atom-level match first (needed for ligand per-atom tokens)
            ti = atom_to_token_bond.get((asym1, seq1, atom1))
            tj = atom_to_token_bond.get((asym2, seq2, atom2))
            if ti is None:
                ti = residue_to_token.get((asym1, seq1))
            if tj is None:
                tj = residue_to_token.get((asym2, seq2))
            if ti is not None and tj is not None and ti != tj:
                token_bond_edges.append((ti, tj))
        token_bond_edges.extend(ligand_bond_edges)

    chain_sequences = {c.chain_id: c.sequence for c in structure.chains if c.sequence}

    return TokenizedStructure(
        pdb_id=structure.pdb_id,
        tokens=tokens,
        chain_ids=chain_ids,
        entity_types=entity_types,
        chain_sequences=chain_sequences,
        token_bonds=token_bond_edges if token_bond_edges else None,
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
    # Ligand intra-residue bond edges, collected as we build tokens.
    ligand_bond_edges: list[tuple[int, int]] = []
    res_counter = 0
    residue_uid_counter = 0  # groups atoms by residue for ref_space_uid

    # Build entity_id mapping. Chains with identical sequence/ccd trivially
    # share entity_id; for proteins we also merge chains whose sequences are
    # >95% identical (handles ragged-termini homodimers like 7y0l where
    # chain B has one extra residue vs chain A). Without this, near-identical
    # chains get distinct entity_ids — killing the RelPE "same_entity +
    # different sym_id" signal that Protenix relies on for homodimer
    # geometry. 7y0l full complex on Protenix v1 weights: 0.18 LDDT (chains
    # collapse onto each other) WITHOUT merge; 0.69 LDDT WITH merge.
    from difflib import SequenceMatcher
    SIMILARITY_THRESHOLD = 0.95
    entity_reps: list[tuple[str, str]] = []  # ordered list of (ctype, rep_seq_or_ccd)
    chain_to_entity: dict[str, int] = {}
    for chain in chains:
        ctype = chain["type"]
        key_str = chain.get("sequence", chain.get("ccd", ""))
        matched = None
        for idx, (rep_ctype, rep_key) in enumerate(entity_reps):
            if rep_ctype != ctype:
                continue
            if rep_key == key_str:
                matched = idx
                break
            # Approximate match for proteins only — RNA/DNA are short and
            # exact-match is appropriate; ligands use CCD codes.
            if ctype == "protein" and rep_key and key_str:
                if SequenceMatcher(None, rep_key, key_str).ratio() >= SIMILARITY_THRESHOLD:
                    matched = idx
                    break
        if matched is None:
            entity_reps.append((ctype, key_str))
            matched = len(entity_reps) - 1
        chain_to_entity[chain["id"]] = matched

    for chain_idx, chain in enumerate(chains):
        chain_id = chain["id"]
        ctype = chain["type"]
        entity_id = chain_to_entity[chain_id]

        if ctype == "protein":
            sequence = chain["sequence"]
            chain_sequences[chain_id] = sequence
            seq_len = len(sequence)
            for char_idx, char in enumerate(sequence):
                three_code = ONE_TO_THREE.get(char, "UNK")
                token_type = AA_TO_IDX.get(char, UNK_AA_IDX)
                is_c_terminal = (char_idx == seq_len - 1)

                # Look up CCD for atom info and ideal coords
                if three_code in ccd:
                    comp = ccd[three_code]
                    mask = comp.heavy_atom_mask
                    atom_names = comp.heavy_atom_names
                    atom_elements = [e for e, m in zip(comp.atom_elements, mask) if m]
                    atom_charges = [c for c, m in zip(comp.atom_charges, mask) if m] if comp.atom_charges else None
                    if comp.ideal_coords is not None and len(mask) == len(comp.ideal_coords):
                        ref_c = comp.ideal_coords[mask]
                    else:
                        ref_c = np.zeros((len(atom_names), 3), dtype=np.float32)

                    # Remove OXT for non-C-terminal residues (matches Protenix)
                    if not is_c_terminal and "OXT" in atom_names:
                        oxt_idx = atom_names.index("OXT")
                        atom_names = atom_names[:oxt_idx] + atom_names[oxt_idx + 1:]
                        atom_elements = atom_elements[:oxt_idx] + atom_elements[oxt_idx + 1:]
                        if atom_charges is not None:
                            atom_charges = atom_charges[:oxt_idx] + atom_charges[oxt_idx + 1:]
                        ref_c = np.delete(ref_c, oxt_idx, axis=0)
                else:
                    # Fallback: minimal backbone
                    atom_names = ["N", "CA", "C", "O"]
                    atom_elements = ["N", "C", "C", "O"]
                    atom_charges = None
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
                    ref_space_uid=residue_uid_counter,
                    atom_charges=atom_charges,
                ))
                chain_ids.append(chain_id)
                entity_types.append("protein")
                res_counter += 1
                residue_uid_counter += 1

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
                    # Use non_leaving mask for nucleotides (removes OP3 etc.)
                    mask = comp.non_leaving_heavy_atom_mask
                    atom_names = [n for n, m in zip(comp.atom_names, mask) if m]
                    atom_elements = [e for e, m in zip(comp.atom_elements, mask) if m]
                    atom_charges = [c for c, m in zip(comp.atom_charges, mask) if m] if comp.atom_charges else None
                    if comp.ideal_coords is not None and len(mask) == len(comp.ideal_coords):
                        ref_c = comp.ideal_coords[mask]
                    else:
                        ref_c = np.zeros((len(atom_names), 3), dtype=np.float32)
                else:
                    atom_names = [char]
                    atom_elements = ["C"]
                    atom_charges = None
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
                    ref_space_uid=residue_uid_counter,
                    atom_charges=atom_charges,
                ))
                chain_ids.append(chain_id)
                entity_types.append("nucleotide")
                res_counter += 1
                residue_uid_counter += 1

        elif ctype == "ligand":
            ccd_code = chain["ccd"]
            if ccd_code not in ccd:
                logger.warning(f"CCD code {ccd_code} not found, skipping ligand chain {chain_id}")
                continue
            comp = ccd[ccd_code]
            mask = comp.heavy_atom_mask
            heavy_names = comp.heavy_atom_names
            heavy_elements = [e for e, m in zip(comp.atom_elements, mask) if m]
            heavy_charges = [c for c, m in zip(comp.atom_charges, mask) if m] if comp.atom_charges else None
            if comp.ideal_coords is not None and len(mask) == len(comp.ideal_coords):
                heavy_coords = comp.ideal_coords[mask]
            else:
                heavy_coords = np.zeros((len(heavy_names), 3), dtype=np.float32)

            # All atoms of this ligand residue share the same ref_space_uid
            # and res_idx (same_res is needed for RelPE a_rel_token).
            lig_uid = residue_uid_counter
            residue_uid_counter += 1
            lig_res_idx = res_counter
            lig_name_to_tok: dict[str, int] = {}

            for atom_idx, (aname, aelem) in enumerate(zip(heavy_names, heavy_elements)):
                elem_idx = ELEM_TO_IDX.get(aelem, UNK_ELEM_IDX)
                token_type = TOKEN_LIGAND_ATOM + elem_idx
                coord = heavy_coords[atom_idx].reshape(1, 3)

                tokens.append(Token(
                    token_type=token_type,
                    chain_idx=chain_idx,
                    res_idx=lig_res_idx,
                    atom_names=[aname],
                    atom_elements=[aelem],
                    atom_coords=coord.copy(),
                    ref_coords=coord,
                    res_name=ccd_code,
                    entity_id=entity_id,
                    sym_id=0,
                    token_index=0,  # overwritten to global arange in to_features
                    ref_space_uid=lig_uid,
                    atom_charges=[heavy_charges[atom_idx]] if heavy_charges else None,
                ))
                chain_ids.append(chain_id)
                entity_types.append("ligand")
                lig_name_to_tok[aname] = len(tokens) - 1

            # Emit intra-ligand covalent bonds from the CCD for this residue.
            # Without these, multi-atom ligands lose their chemistry graph.
            for a1, a2, _order in comp.bonds:
                ti = lig_name_to_tok.get(a1)
                tj = lig_name_to_tok.get(a2)
                if ti is not None and tj is not None and ti != tj:
                    ligand_bond_edges.append((ti, tj))
            res_counter += 1

    # Compute sym_id: per-entity counter so homo-multimer chains get distinct sym_ids
    entity_sym_counter: dict[int, int] = {}
    chain_sym: dict[str, int] = {}
    for token, cid in zip(tokens, chain_ids):
        if cid not in chain_sym:
            eid = token.entity_id
            chain_sym[cid] = entity_sym_counter.get(eid, 0)
            entity_sym_counter[eid] = entity_sym_counter.get(eid, 0) + 1
        token.sym_id = chain_sym[cid]

    # Build token_bonds from ligand intra-residue bonds (Bug 3). Sparse edge
    # list — see note on TokenizedStructure.token_bonds.
    return TokenizedStructure(
        pdb_id="PREDICT",
        tokens=tokens,
        chain_ids=chain_ids,
        entity_types=entity_types,
        chain_sequences=chain_sequences,
        token_bonds=list(ligand_bond_edges) if ligand_bond_edges else None,
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
    msa: np.ndarray             # (N_seqs, L) integer-encoded MSA (Protenix 32-class)
    deletion_matrix: np.ndarray  # (N_seqs, L) deletion counts
    profile: np.ndarray          # (L, 32) residue frequency profile (Protenix 32-class)
    deletion_mean: np.ndarray    # (L,) mean deletion value per position
    cluster_msa: np.ndarray      # (N_clusters, L) clustered MSA
    cluster_profile: np.ndarray  # (N_clusters, L, 32) cluster profiles
    cluster_deletion_mean: np.ndarray  # (N_clusters, L) mean deletions per cluster
    n_seqs: int
    length: int


# Species-ID regexes from Protenix (msa_utils.py). Protein MSA descriptions
# carry species info via UniProt (>tr|ACCESSION|GENE_SPEC) or UniRef100 format.
_UNIPROT_REGEX = re.compile(
    r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)
_UNIREF_REGEX = re.compile(r"^UniRef100_[^_]+_([^_/]+)")


def get_species_ids(descriptions: list[str]) -> list[str]:
    """Extract species identifiers from MSA description lines.

    Returns one ID per description (empty string if no species identifier
    was recognizable). Follows Protenix's regex exactly.
    """
    ids = []
    for d in descriptions:
        d = d.strip()
        m = _UNIPROT_REGEX.match(d) or _UNIREF_REGEX.match(d)
        if m:
            if "SpeciesId" in m.groupdict():
                ids.append(m.group("SpeciesId"))
            else:
                ids.append(m.group(1))
        else:
            ids.append("")
    return ids


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
        msa: (N_seqs, L) integer encoded (Protenix 32-class: 0-19=AA, 20=unk, 31=gap)
        deletions: (N_seqs, L) deletion counts
    """
    if not seqs:
        return np.zeros((0, 0), dtype=np.int8), np.zeros((0, 0), dtype=np.int8)

    # Process query to get alignment length
    query = seqs[0]
    query_aligned = "".join(c for c in query if not c.islower())
    L = len(query_aligned)

    aa_map = PROTENIX_MSA_AA_MAP  # ARNDCQEGHILKMFPSTWYV -> 0-19
    gap_idx = PROTENIX_MSA_GAP    # 31
    unk_idx = PROTENIX_MSA_UNK_PROTEIN  # 20

    n_seqs = len(seqs)
    msa = np.full((n_seqs, L), gap_idx, dtype=np.int8)
    deletions = np.zeros((n_seqs, L), dtype=np.int16)

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

    # Compute profile (residue frequencies at each position, Protenix 32-class)
    n_classes = PROTENIX_NUM_MSA_CLASSES  # 32
    profile = np.zeros((L, n_classes), dtype=np.float32)
    for pos in range(L):
        counts = np.bincount(msa[:, pos].astype(np.int32), minlength=n_classes)[:n_classes]
        total = counts.sum()
        if total > 0:
            profile[pos] = counts / total

    # Compute deletion_mean across all sequences
    deletion_mean = deletions.astype(np.float32).mean(axis=0)  # (L,)

    # Simple clustering by sequence identity to query
    query = msa[0]
    n_clust = min(n_clusters, n_seqs)

    if n_seqs <= n_clust:
        cluster_msa = msa
        cluster_profile = np.zeros((n_seqs, L, n_classes), dtype=np.float32)
        cluster_deletion_mean = np.zeros((n_seqs, L), dtype=np.float32)
        for ci in range(n_seqs):
            for pos in range(L):
                cluster_profile[ci, pos, msa[ci, pos]] = 1.0
            cluster_deletion_mean[ci] = deletions[ci].astype(np.float32)
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
        cluster_deletion_mean = np.zeros((len(centers), L), dtype=np.float32)
        for ci in range(len(centers)):
            members_mask = assigned == ci
            members = msa[members_mask]
            member_dels = deletions[members_mask]
            for pos in range(L):
                counts = np.bincount(members[:, pos].astype(np.int32), minlength=n_classes)[:n_classes]
                total = counts.sum()
                if total > 0:
                    cluster_profile[ci, pos] = counts / total
            cluster_deletion_mean[ci] = member_dels.astype(np.float32).mean(axis=0)

    return MSAFeatures(
        msa=msa,
        deletion_matrix=deletions,
        profile=profile,
        deletion_mean=deletion_mean,
        cluster_msa=cluster_msa,
        cluster_profile=cluster_profile,
        cluster_deletion_mean=cluster_deletion_mean,
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


@dataclass
class RawChainMSA:
    """Per-chain raw MSA, pre-feature-computation. Used as input to
    `assemble_complex_msa_features` for species pairing across chains."""
    msa: np.ndarray             # (N_seqs, L) int8, Protenix 32-class
    deletion_matrix: np.ndarray  # (N_seqs, L) int16
    species_ids: list[str]       # per-row species ID ("" if unidentified)

    @property
    def n_seqs(self) -> int:
        return int(self.msa.shape[0])

    @property
    def length(self) -> int:
        return int(self.msa.shape[1])


def _raw_msa_from_a3m(content: str) -> RawChainMSA | None:
    seqs, descs = parse_a3m(content)
    if not seqs:
        return None
    # Deduplicate identical sequences (Protenix's RawMsa.from_a3m passes
    # dedup=True for unpaired MSAs). Keep query (row 0) and first occurrence
    # of each unique sequence. Identical sequences add no information and
    # Protenix measurably has fewer rows (e.g. 13837 vs our 13906 for 8tuz).
    seen: set[str] = set()
    keep_seqs, keep_descs = [], []
    for s, d in zip(seqs, descs):
        if s in seen:
            continue
        seen.add(s)
        keep_seqs.append(s)
        keep_descs.append(d)
    seqs, descs = keep_seqs, keep_descs
    msa, dels = a3m_to_msa_matrix(seqs)
    species = get_species_ids(descs)
    return RawChainMSA(msa=msa, deletion_matrix=dels, species_ids=species)


def load_raw_msa_for_chain(
    pdb_id: str,
    chain_id: str,
    sequence: str = "",
    msa_dir: Path | None = None,
    msa_tar_path: Path | None = None,
    tar_index: TarIndex | None = None,
) -> RawChainMSA | None:
    """Load raw MSA (with species IDs) for a given chain. Same lookup
    strategy as `load_msa_for_chain` but returns the pre-featurization
    matrix + species IDs so multi-chain pairing can be performed."""
    seq_hash = hashlib.sha256((sequence + "\n").encode()).hexdigest() if sequence else ""

    if msa_dir and msa_dir.exists():
        for hash_name in [seq_hash] if seq_hash else []:
            msa_path = msa_dir / f"{hash_name}.a3m.gz"
            if msa_path.exists():
                with gzip.open(msa_path, "rt") as f:
                    content = f.read()
                return _raw_msa_from_a3m(content)

    if tar_index is not None and seq_hash:
        target = f"rcsb_raw_msa/{seq_hash}.a3m.gz"
        if target in tar_index.entries:
            offset, size = tar_index.entries[target]
            raw = read_tar_member(tar_index.tar_path, offset, size)
            return _raw_msa_from_a3m(gzip.decompress(raw).decode())

    if msa_tar_path is not None and msa_tar_path.exists() and seq_hash:
        target = f"rcsb_raw_msa/{seq_hash}.a3m.gz"
        try:
            with tarfile.open(msa_tar_path, "r") as tar:
                member = tar.getmember(target)
                f = tar.extractfile(member)
                if f:
                    return _raw_msa_from_a3m(gzip.decompress(f.read()).decode())
        except (KeyError, tarfile.TarError):
            pass

    return None


def _pair_rows_across_chains(
    chain_species: list[list[str]],
    max_paired: int,
    max_per_species: int = 1,
) -> np.ndarray:
    """Protenix-style species pairing. Returns an (S_paired, n_chains) int32
    matrix where entry [s, i] is the row index into chain i's MSA for the
    s-th paired row; -1 means "chain i has no row for this species —
    gap-fill."

    Always includes row 0 (query) as the first paired row."""
    n_chains = len(chain_species)
    chain_species_map: list[dict[str, np.ndarray]] = []
    all_species_counts: dict[str, int] = {}
    species_min_hits: dict[str, int] = {}

    for ids in chain_species:
        if len(ids) == 0 or (len(ids) == 1 and not ids[0]):
            chain_species_map.append({})
            continue
        ids_arr = np.array(ids, dtype=object)
        row_indices = np.arange(len(ids_arr))
        sort = ids_arr.argsort(kind="stable")
        ids_arr = ids_arr[sort]
        row_indices = row_indices[sort]
        species, unique_first = np.unique(ids_arr, return_index=True)
        grouped = np.split(row_indices, unique_first[1:])
        mapping = dict(zip(species, grouped))
        chain_species_map.append(mapping)
        for s in species:
            if not s:
                continue
            all_species_counts[s] = all_species_counts.get(s, 0) + 1
        for s, idxs in mapping.items():
            if not s:
                continue
            species_min_hits[s] = min(
                species_min_hits.get(s, max_per_species), len(idxs)
            )

    # Rank species by chain count (descending); skip singletons.
    ranked: dict[int, list[str]] = {}
    for s, count in all_species_counts.items():
        if not s or count <= 1:
            continue
        ranked.setdefault(count, []).append(s)

    pair_blocks: list[np.ndarray] = [np.zeros((1, n_chains), dtype=np.int32)]
    total = 0
    for count in sorted(ranked.keys(), reverse=True):
        for species in ranked[count]:
            min_hits = species_min_hits[species]
            chain_rows = []
            for mapping in chain_species_map:
                if species in mapping:
                    rows = mapping[species][:min_hits]
                else:
                    rows = np.full(min_hits, -1, dtype=np.int32)
                chain_rows.append(rows.astype(np.int32))
            block = np.stack(chain_rows, axis=1)  # (min_hits, n_chains)
            pair_blocks.append(block)
            total += block.shape[0]
            if total >= max_paired:
                break
        if total >= max_paired:
            break

    return np.concatenate(pair_blocks, axis=0)[:max_paired]


def assemble_complex_msa_features(
    chain_raws: list[RawChainMSA],
    max_paired: int = 8192,
    max_seqs: int = 512,
    n_clusters: int = 64,
    max_per_species: int = 1,
) -> MSAFeatures:
    """Combine per-chain raw MSAs into a single complex-level feature set.

    Rows are laid out as: [paired rows (rows span all chains via species
    matching), then per-chain unpaired rows (block-diagonal, one chain's
    columns populated, others gap)]. The column axis is the concatenation
    of per-chain lengths.

    Profile and clustering are computed on the combined matrix.
    """
    gap = PROTENIX_MSA_GAP
    n_chains = len(chain_raws)
    lengths = [c.length for c in chain_raws]
    L_total = sum(lengths)
    col_offsets = np.cumsum([0] + lengths).tolist()

    # Species-pair row indices (S_paired, n_chains), -1 means missing-for-chain.
    chain_species = [c.species_ids for c in chain_raws]
    pair_idxs = _pair_rows_across_chains(
        chain_species, max_paired=max_paired, max_per_species=max_per_species
    )
    S_paired = pair_idxs.shape[0]

    # Paired block: fill per-chain columns using chain i's msa[pair_idxs[:, i]].
    paired_msa = np.full((S_paired, L_total), gap, dtype=np.int8)
    paired_dels = np.zeros((S_paired, L_total), dtype=np.int16)
    for i, c in enumerate(chain_raws):
        col0, col1 = col_offsets[i], col_offsets[i + 1]
        sel = pair_idxs[:, i]
        valid = sel >= 0
        if valid.any():
            paired_msa[valid, col0:col1] = c.msa[sel[valid]]
            paired_dels[valid, col0:col1] = c.deletion_matrix[sel[valid]]

    # Unpaired blocks: one per chain; drop rows already used in pairing.
    # (Protenix's cleanup_unpaired_features does byte-equality on whole rows;
    # we approximate by dropping rows whose indices were chosen for pairing.)
    unpaired_parts = []
    unpaired_dels_parts = []
    for i, c in enumerate(chain_raws):
        used = set(int(x) for x in pair_idxs[:, i] if x >= 0)
        keep = np.array([r for r in range(c.n_seqs) if r not in used], dtype=np.int64)
        if keep.size == 0:
            continue
        col0, col1 = col_offsets[i], col_offsets[i + 1]
        block_msa = np.full((keep.size, L_total), gap, dtype=np.int8)
        block_dels = np.zeros((keep.size, L_total), dtype=np.int16)
        block_msa[:, col0:col1] = c.msa[keep]
        block_dels[:, col0:col1] = c.deletion_matrix[keep]
        unpaired_parts.append(block_msa)
        unpaired_dels_parts.append(block_dels)

    if unpaired_parts:
        msa = np.concatenate([paired_msa] + unpaired_parts, axis=0)
        dels = np.concatenate([paired_dels] + unpaired_dels_parts, axis=0)
    else:
        msa = paired_msa
        dels = paired_dels

    # Cap at max_seqs (keep query + paired first, then unpaired).
    if msa.shape[0] > max_seqs:
        msa = msa[:max_seqs]
        dels = dels[:max_seqs]

    return compute_msa_features(msa, dels, max_seqs=max_seqs, n_clusters=n_clusters)


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
        path = _raw_dir() / "pdb_seqres.txt.gz"
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
        "restype": features["restype"][token_indices],
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
        "atom_name_chars": features["atom_name_chars"][atom_mask],
        "chain_same": features["chain_same"][token_indices][:, token_indices],
        "n_tokens": n_new,
        "n_atoms": atom_mask.sum().item(),
    }

    # token_bonds: (N_tok, N_tok) pair tensor
    if "token_bonds" in features:
        result["token_bonds"] = features["token_bonds"][token_indices][:, token_indices]

    # Atom-level optional features
    if "ref_space_uid" in features:
        result["ref_space_uid"] = features["ref_space_uid"][atom_mask]
    if "ref_charge" in features:
        result["ref_charge"] = features["ref_charge"][atom_mask]

    # rep_atom_idx: remap old atom indices to new atom indices
    if "rep_atom_idx" in features:
        old_to_new_atom = torch.full((len(atom_mask),), -1, dtype=torch.long)
        new_idx = 0
        for i in range(len(atom_mask)):
            if atom_mask[i]:
                old_to_new_atom[i] = new_idx
                new_idx += 1
        old_rep = features["rep_atom_idx"][token_indices]
        result["rep_atom_idx"] = old_to_new_atom[old_rep]

    if "has_frame" in features:
        result["has_frame"] = features["has_frame"][token_indices]

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
            features["deletion_mean"] = torch.tensor(msa_feat.deletion_mean, dtype=torch.float32)
            features["cluster_deletion_mean"] = torch.tensor(msa_feat.cluster_deletion_mean, dtype=torch.float32)
            features["has_msa"] = torch.tensor(1)
        else:
            # Placeholder MSA features
            n_tok = features["n_tokens"]
            features["msa_profile"] = torch.zeros(n_tok, PROTENIX_NUM_MSA_CLASSES)
            features["cluster_msa"] = torch.zeros(1, n_tok, dtype=torch.long)
            features["cluster_profile"] = torch.zeros(1, n_tok, PROTENIX_NUM_MSA_CLASSES)
            features["deletion_mean"] = torch.zeros(n_tok)
            features["cluster_deletion_mean"] = torch.zeros(1, n_tok)
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
    for key in ["token_types", "restype", "chain_indices", "res_indices", "rel_pos", "token_index", "entity_id", "sym_id", "atoms_per_token", "rep_atom_idx", "has_frame"]:
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

    # token_bonds: (N_tok, N_tok) pair tensor, optional
    if any("token_bonds" in b for b in batch):
        tb_list = []
        for b in batch:
            if "token_bonds" in b:
                t = b["token_bonds"]
            else:
                t = torch.zeros(b["n_tokens"], b["n_tokens"])
            n = t.shape[0]
            pad = max_tokens - n
            if pad > 0:
                t = torch.nn.functional.pad(t, (0, pad, 0, pad))
            tb_list.append(t)
        result["token_bonds"] = torch.stack(tb_list)

    # Atom-level tensors: pad to max_atoms
    for key in ["atom_coords", "ref_coords", "atom_name_chars"]:
        tensors = []
        for b in batch:
            t = b[key]
            pad_size = max_atoms - len(t)
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, 0, 0, pad_size))
            tensors.append(t)
        result[key] = torch.stack(tensors)

    for key in ["atom_to_token", "atom_element_idx", "ref_space_uid"]:
        if key not in batch[0]:
            continue
        tensors = []
        pad_val = -1 if key == "ref_space_uid" else 0
        for b in batch:
            t = b[key]
            pad_size = max_atoms - len(t)
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, pad_size), value=pad_val)
            tensors.append(t)
        result[key] = torch.stack(tensors)

    # ref_charge: (N_atoms,) float
    if "ref_charge" in batch[0]:
        tensors = []
        for b in batch:
            t = b["ref_charge"]
            pad_size = max_atoms - len(t)
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, pad_size))
            tensors.append(t)
        result["ref_charge"] = torch.stack(tensors)

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
    deletion_means = []
    cluster_deletion_means = []
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

        # deletion_mean: (L,) -> pad to max_tokens
        dm = b["deletion_mean"]
        pad_l = max_tokens - dm.shape[0]
        if pad_l > 0:
            dm = torch.nn.functional.pad(dm, (0, pad_l))
        deletion_means.append(dm[:max_tokens])

        # cluster_deletion_mean: (N_clusters, L) -> pad both dims
        cdm = b["cluster_deletion_mean"]
        pad_c = max_clusters - cdm.shape[0]
        pad_l = max_tokens - cdm.shape[1]
        if pad_c > 0 or pad_l > 0:
            cdm = torch.nn.functional.pad(cdm, (0, max(0, pad_l), 0, max(0, pad_c)))
        cluster_deletion_means.append(cdm[:max_clusters, :max_tokens])

    result["msa_profile"] = torch.stack(msa_profiles)
    result["cluster_msa"] = torch.stack(cluster_msas)
    result["cluster_profile"] = torch.stack(cluster_profiles)
    result["deletion_mean"] = torch.stack(deletion_means)
    result["cluster_deletion_mean"] = torch.stack(cluster_deletion_means)
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
        "restype": torch.randint(0, PROTENIX_NUM_MSA_CLASSES, (batch_size, n_tokens), device=device),
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
        "atom_name_chars": torch.zeros(batch_size, n_atoms, 256, device=device),
        "ref_space_uid": torch.arange(n_tokens, device=device).repeat_interleave(n_atoms_per_token).unsqueeze(0).expand(batch_size, -1),
        "ref_charge": torch.zeros(batch_size, n_atoms, device=device),
        "rep_atom_idx": torch.arange(0, n_atoms, n_atoms_per_token, device=device).unsqueeze(0).expand(batch_size, -1),
        "has_frame": torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device),
        "chain_same": torch.ones(batch_size, n_tokens, n_tokens, dtype=torch.long, device=device),
        "token_mask": torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device),
        "atom_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device),
        "n_tokens": torch.tensor([n_tokens] * batch_size, device=device),
        "n_atoms": torch.tensor([n_atoms] * batch_size, device=device),
    }

    if has_msa:
        batch["msa_profile"] = torch.randn(batch_size, n_tokens, PROTENIX_NUM_MSA_CLASSES, device=device).softmax(dim=-1)
        batch["cluster_msa"] = torch.randint(0, PROTENIX_NUM_MSA_CLASSES, (batch_size, 4, n_tokens), device=device)
        batch["cluster_profile"] = torch.randn(batch_size, 4, n_tokens, PROTENIX_NUM_MSA_CLASSES, device=device).softmax(dim=-1)
        batch["deletion_mean"] = torch.rand(batch_size, n_tokens, device=device)
        batch["cluster_deletion_mean"] = torch.rand(batch_size, 4, n_tokens, device=device)
        batch["has_msa"] = torch.ones(batch_size, device=device)
    else:
        batch["msa_profile"] = torch.zeros(batch_size, n_tokens, PROTENIX_NUM_MSA_CLASSES, device=device)
        batch["cluster_msa"] = torch.zeros(batch_size, 1, n_tokens, dtype=torch.long, device=device)
        batch["cluster_profile"] = torch.zeros(batch_size, 1, n_tokens, PROTENIX_NUM_MSA_CLASSES, device=device)
        batch["deletion_mean"] = torch.zeros(batch_size, n_tokens, device=device)
        batch["cluster_deletion_mean"] = torch.zeros(batch_size, 1, n_tokens, device=device)
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
        # Conservative cap: each worker can balloon to tens of GB on large
        # assemblies (ribosomes, capsids), and workers leak memory across tasks.
        # With ~500GB RAM this leaves ~30GB headroom per worker before swap.
        n_workers = min(16, os.cpu_count() or 4)

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

    # maxtasksperchild recycles workers periodically so leaked memory from
    # parse_mmcif/tokenize_structure doesn't accumulate (the pre-2026-04 runs
    # OOM-killed the machine when workers grew to 200-300GB RSS).
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(ccd,), maxtasksperchild=25) as pool:
        for result in pool.imap_unordered(_process_single_structure, args_list, chunksize=8):
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
        output_path = _processed_dir() / "manifest.json"
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
        path = _processed_dir() / "manifest.json"

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
            features["deletion_mean"] = torch.tensor(msa_feat.deletion_mean, dtype=torch.float32)
            features["cluster_deletion_mean"] = torch.tensor(msa_feat.cluster_deletion_mean, dtype=torch.float32)
            features["has_msa"] = torch.tensor(1)
        else:
            n_tok = features["n_tokens"]
            features["msa_profile"] = torch.zeros(n_tok, PROTENIX_NUM_MSA_CLASSES)
            features["cluster_msa"] = torch.zeros(1, n_tok, dtype=torch.long)
            features["cluster_profile"] = torch.zeros(1, n_tok, PROTENIX_NUM_MSA_CLASSES)
            features["deletion_mean"] = torch.zeros(n_tok)
            features["cluster_deletion_mean"] = torch.zeros(1, n_tok)
            features["has_msa"] = torch.tensor(0)

        return features


# ============================================================================
# Preprocessing CLI
# ============================================================================

def preprocess_main():
    """CLI entry point for helico-preprocess.

    Usage:
        helico-preprocess ccd <raw-dir> <processed-dir>
        helico-preprocess structures <raw-dir> <processed-dir>
        helico-preprocess msa-index --tar-path PATH --output PATH
        helico-preprocess all <raw-dir> <processed-dir>
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Helico Preprocessing Pipeline",
        usage="helico-preprocess {ccd,structures,msa-index,all} ...",
    )
    subparsers = parser.add_subparsers(dest="command", help="Preprocessing subcommand")

    # ccd subcommand
    sp_ccd = subparsers.add_parser("ccd", help="Parse components.cif and write ccd_cache.pkl")
    sp_ccd.add_argument("raw_dir", type=Path, help="Directory containing components.cif")
    sp_ccd.add_argument("processed_dir", type=Path, help="Output directory for ccd_cache.pkl")

    # structures subcommand
    sp_struct = subparsers.add_parser("structures", help="Process mmCIF files into pickled structures")
    sp_struct.add_argument("raw_dir", type=Path, help="Raw data directory (must contain mmCIF/)")
    sp_struct.add_argument("processed_dir", type=Path, help="Output directory for pickles and manifest")
    sp_struct.add_argument("--max-resolution", type=float, default=9.0)
    sp_struct.add_argument("--n-workers", type=int, default=None)
    sp_struct.add_argument("--no-skip-existing", action="store_true")

    # msa-index subcommand
    sp_msa = subparsers.add_parser("msa-index", help="Build tar index for MSA archives")
    sp_msa.add_argument("--tar-path", type=Path, required=True)
    sp_msa.add_argument("--output", type=Path, required=True)

    # all subcommand
    sp_all = subparsers.add_parser("all", help="Run full preprocessing pipeline (ccd + structures + msa indices)")
    sp_all.add_argument("raw_dir", type=Path, help="Raw data directory")
    sp_all.add_argument("processed_dir", type=Path, help="Output directory for all processed data")
    sp_all.add_argument("--max-resolution", type=float, default=9.0)
    sp_all.add_argument("--n-workers", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "ccd":
        cif_path = args.raw_dir / "components.cif"
        cache_path = args.processed_dir / "ccd_cache.pkl"
        logger.info(f"Parsing CCD from {cif_path}...")
        ccd = parse_ccd(cif_path=cif_path, cache_path=cache_path)
        logger.info(f"CCD cache written to {cache_path} ({len(ccd)} components)")

    elif args.command == "structures":
        raw_dir = args.raw_dir
        output_dir = args.processed_dir
        ccd_cache = output_dir / "ccd_cache.pkl"
        metadata = preprocess_structures(
            mmcif_dir=raw_dir / "mmCIF",
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
        raw_dir = args.raw_dir
        output_dir = args.processed_dir

        # Step 1: Parse CCD
        cif_path = raw_dir / "components.cif"
        ccd_cache = output_dir / "ccd_cache.pkl"
        logger.info("=== Step 1: Parsing CCD ===")
        ccd = parse_ccd(cif_path=cif_path, cache_path=ccd_cache)
        logger.info(f"CCD: {len(ccd)} components")

        # Step 2: Process structures
        logger.info("=== Step 2: Processing structures ===")
        metadata = preprocess_structures(
            mmcif_dir=raw_dir / "mmCIF",
            output_dir=output_dir,
            ccd_cache_path=ccd_cache,
            max_resolution=args.max_resolution,
            n_workers=args.n_workers,
        )
        build_manifest(metadata, output_dir / "manifest.json")

        # Step 3: Build MSA tar indices
        for name, tar_path in [
            ("rcsb_raw_msa", raw_dir / "rcsb_raw_msa.tar"),
            ("openfold_raw_msa", raw_dir / "openfold_raw_msa.tar"),
        ]:
            if tar_path.exists():
                logger.info(f"=== Step 3: Building tar index for {name} ===")
                index = build_tar_index(tar_path)
                out = output_dir / f"{name}_index.pkl"
                save_tar_index(index, out)
                logger.info(f"Index saved to {out} ({len(index.entries)} entries)")

        logger.info("=== Preprocessing complete ===")


# ============================================================================
# Download from HuggingFace
# ============================================================================

# Small files that can be downloaded individually
_PROCESSED_FILES = [
    "processed/ccd_cache.pkl",
    "processed/manifest.json.gz",
    "processed/rcsb_raw_msa_index.pkl",
    "processed/openfold_raw_msa_index.pkl",
]

# Split tar prefix for large directories
_PROCESSED_SPLIT_TARS = ["processed/structures.tar"]


def _reassemble_split_tar(data_dir: Path, prefix: str, extract: bool = True) -> None:
    """Reassemble split tar parts (prefix.00, prefix.01, ...) and optionally extract."""
    import glob as glob_mod

    parts = sorted(glob_mod.glob(str(data_dir / f"{prefix}.*")))
    parts = [p for p in parts if re.match(r".*\.\d{2}$", p)]
    if not parts:
        logger.warning(f"No split parts found for {prefix}")
        return

    tar_path = data_dir / prefix
    logger.info(f"Reassembling {len(parts)} parts into {tar_path}...")
    with open(tar_path, "wb") as out_f:
        for part in parts:
            with open(part, "rb") as in_f:
                while True:
                    chunk = in_f.read(64 * 1024 * 1024)  # 64MB chunks
                    if not chunk:
                        break
                    out_f.write(chunk)

    if extract:
        # Determine extraction directory from tar path
        extract_dir = tar_path.parent
        logger.info(f"Extracting {tar_path} to {extract_dir}...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_dir)
        logger.info(f"Extraction complete, removing tar and parts...")
        tar_path.unlink()

    # Clean up split parts
    for part in parts:
        Path(part).unlink()


def _decompress_manifest(data_dir: Path) -> None:
    """Decompress manifest.json.gz to manifest.json if needed."""
    gz_path = data_dir / "processed" / "manifest.json.gz"
    json_path = data_dir / "processed" / "manifest.json"
    if gz_path.exists() and not json_path.exists():
        logger.info("Decompressing manifest.json.gz...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(json_path, "wb") as f_out:
                while True:
                    chunk = f_in.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
        gz_path.unlink()
        logger.info(f"Manifest decompressed to {json_path}")


def download_main() -> None:
    """CLI entry point: download Helico data from HuggingFace."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Download Helico training data from HuggingFace"
    )
    parser.add_argument(
        "--subset",
        choices=["all", "ccd-only"],
        default="all",
        help="What to download (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Target directory (default: {_default_data_dir()})",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract split tars after downloading",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download, list_repo_files

    def _download(filename: str) -> None:
        logger.info(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir,
        )

    if args.subset == "ccd-only":
        _download("processed/ccd_cache.pkl")
        logger.info("Done. CCD cache downloaded.")
        return

    # Download individual files
    for f in _PROCESSED_FILES:
        _download(f)

    # Download split tar parts
    all_repo_files = list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    for prefix in _PROCESSED_SPLIT_TARS:
        parts = sorted(f for f in all_repo_files if f.startswith(prefix + "."))
        if not parts:
            logger.warning(f"No split parts found for {prefix} in HF repo")
            continue
        for part in parts:
            _download(part)
        if not args.no_extract:
            _reassemble_split_tar(data_dir, prefix)

    # Decompress manifest
    _decompress_manifest(data_dir)

    logger.info(f"Download complete. Data is at {data_dir}")
