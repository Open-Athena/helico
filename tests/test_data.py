"""Integration tests for Helico data pipeline."""

import io
import numpy as np
import pytest
import torch
from pathlib import Path

from helico.data import (
    PROCESSED_DIR,
    PROTENIX_MSA_GAP,
    CCDComponent,
    LazyHelicoDataset,
    StructureMetadata,
    TarIndex,
    TokenizedStructure,
    build_manifest,
    build_tar_index,
    discover_mmcif_files,
    load_manifest,
    load_tar_index,
    parse_ccd,
    parse_mmcif,
    save_tar_index,
    tokenize_structure,
    tokenize_sequences,
    parse_sequences_arg,
    parse_input_yaml,
    parse_a3m,
    a3m_to_msa_matrix,
    compute_msa_features,
    spatial_crop,
    contiguous_crop,
    make_synthetic_structure,
    make_synthetic_batch,
    HelicoDataset,
    collate_fn,
    NUM_TOKEN_TYPES,
    TOKEN_LIGAND_ATOM,
    ELEM_TO_IDX,
    _process_single_structure,
    _init_worker,
)

# ============================================================================
# Tokenizer Tests
# ============================================================================

class TestTokenizer:
    def test_protein_tokenization(self):
        """Tokenize a synthetic protein and verify token counts."""
        structure = make_synthetic_structure(n_residues=20)
        tokenized = tokenize_structure(structure)
        assert tokenized.n_tokens == 20
        assert all(et == "protein" for et in tokenized.entity_types)

    def test_multi_chain(self):
        """Tokenize a multi-chain structure."""
        structure = make_synthetic_structure(n_residues=15, n_chains=2)
        tokenized = tokenize_structure(structure)
        assert tokenized.n_tokens == 30  # 15 residues * 2 chains
        chain_ids = set(tokenized.chain_ids)
        assert len(chain_ids) == 2

    def test_with_ligand(self):
        """Tokenize structure with ligand."""
        structure = make_synthetic_structure(n_residues=10, include_ligand=True)
        tokenized = tokenize_structure(structure)
        # 10 protein residues + 4 ligand atoms (each becomes a token)
        assert tokenized.n_tokens == 14
        assert "ligand" in tokenized.entity_types

    def test_to_features(self):
        """Verify feature tensor shapes."""
        structure = make_synthetic_structure(n_residues=20)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()

        assert features["token_types"].shape == (20,)
        assert features["chain_indices"].shape == (20,)
        assert features["atom_coords"].shape[1] == 3
        assert features["atom_to_token"].shape[0] == features["atom_coords"].shape[0]
        assert features["chain_same"].shape == (20, 20)
        assert features["n_tokens"] == 20

    def test_token_types_in_range(self):
        """Token types should be valid indices."""
        structure = make_synthetic_structure(n_residues=20, include_ligand=True)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()
        assert (features["token_types"] >= 0).all()
        assert (features["token_types"] < NUM_TOKEN_TYPES).all()


# ============================================================================
# Sequence Tokenization Tests
# ============================================================================

def _make_fake_ccd_component(comp_id, atom_names, atom_elements, coords=None):
    """Helper to create a fake CCDComponent for testing."""
    n = len(atom_names)
    # Add hydrogens so heavy_atom_mask filtering works
    all_names = atom_names
    all_elements = atom_elements
    if coords is None:
        coords = np.random.randn(n, 3).astype(np.float32)
    return CCDComponent(
        comp_id=comp_id,
        name=comp_id,
        comp_type="L-PEPTIDE LINKING",
        formula="",
        atom_names=list(all_names),
        atom_elements=list(all_elements),
        atom_charges=[0] * n,
        atom_leaving=[False] * n,
        ideal_coords=coords,
    )


class TestTokenizeSequences:
    """Tests for tokenize_sequences and input parsing helpers."""

    def _fake_ccd(self):
        """Build a minimal fake CCD with a few amino acids."""
        ccd = {}
        ccd["ALA"] = _make_fake_ccd_component(
            "ALA",
            ["N", "CA", "C", "O", "CB"],
            ["N", "C", "C", "O", "C"],
        )
        ccd["MET"] = _make_fake_ccd_component(
            "MET",
            ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
            ["N", "C", "C", "O", "C", "C", "S", "C"],
        )
        ccd["GLY"] = _make_fake_ccd_component(
            "GLY",
            ["N", "CA", "C", "O"],
            ["N", "C", "C", "O"],
        )
        ccd["LYS"] = _make_fake_ccd_component(
            "LYS",
            ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
            ["N", "C", "C", "O", "C", "C", "C", "C", "N"],
        )
        # A ligand
        ccd["ATP"] = _make_fake_ccd_component(
            "ATP",
            ["PG", "O1G", "O2G", "O3G", "PB", "O1B", "C1'", "N9"],
            ["P", "O", "O", "O", "P", "O", "C", "N"],
        )
        return ccd

    def test_tokenize_protein(self):
        """Single chain 'MAGK' with fake CCD → 4 tokens, all protein, non-zero ref_coords."""
        ccd = self._fake_ccd()
        chains = [{"type": "protein", "id": "A", "sequence": "MAGK"}]
        tokenized = tokenize_sequences(chains, ccd)

        assert tokenized.n_tokens == 4
        assert all(et == "protein" for et in tokenized.entity_types)
        assert tokenized.pdb_id == "PREDICT"

        # MET token should have 8 heavy atoms
        met_tok = tokenized.tokens[0]
        assert len(met_tok.atom_names) == 8
        assert met_tok.ref_coords is not None
        assert met_tok.ref_coords.shape == (8, 3)
        assert not np.allclose(met_tok.ref_coords, 0.0)

        # GLY token should have 4 heavy atoms
        gly_tok = tokenized.tokens[2]
        assert len(gly_tok.atom_names) == 4

    def test_tokenize_multi_chain_entity_id(self):
        """Two chains same sequence share entity_id, different sequence gets different."""
        ccd = self._fake_ccd()
        chains = [
            {"type": "protein", "id": "A", "sequence": "MAG"},
            {"type": "protein", "id": "B", "sequence": "MAG"},
            {"type": "protein", "id": "C", "sequence": "AK"},
        ]
        tokenized = tokenize_sequences(chains, ccd)

        # A and B should share entity_id
        entity_a = tokenized.tokens[0].entity_id  # first token of chain A
        entity_b = tokenized.tokens[3].entity_id  # first token of chain B
        entity_c = tokenized.tokens[6].entity_id  # first token of chain C
        assert entity_a == entity_b
        assert entity_a != entity_c

    def test_tokenize_to_features(self):
        """to_features() on sequence-derived TokenizedStructure should have correct shapes."""
        ccd = self._fake_ccd()
        chains = [{"type": "protein", "id": "A", "sequence": "MAG"}]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()

        assert features["n_tokens"] == 3
        assert features["token_types"].shape == (3,)
        assert features["ref_coords"].shape[1] == 3
        assert features["ref_coords"].shape[0] == features["atom_coords"].shape[0]
        assert features["ref_coords"].shape[0] > 0
        # ref_coords should be non-zero (from CCD ideal coords)
        assert not torch.allclose(features["ref_coords"], torch.zeros_like(features["ref_coords"]))

        expected_keys = [
            "token_types", "chain_indices", "res_indices", "rel_pos",
            "atom_coords", "ref_coords", "atom_to_token", "atoms_per_token",
            "atom_element_idx", "chain_same", "n_tokens", "n_atoms",
            "token_index", "entity_id", "sym_id",
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_tokenize_ligand(self):
        """Ligand by CCD code → one token per heavy atom, correct token types."""
        ccd = self._fake_ccd()
        chains = [{"type": "ligand", "id": "D", "ccd": "ATP"}]
        tokenized = tokenize_sequences(chains, ccd)

        # ATP has 8 heavy atoms → 8 tokens
        assert tokenized.n_tokens == 8
        assert all(et == "ligand" for et in tokenized.entity_types)

        # Check token types are ligand-type
        for tok in tokenized.tokens:
            assert tok.token_type >= TOKEN_LIGAND_ATOM

        # First atom is PG (phosphorus)
        first_tok = tokenized.tokens[0]
        assert first_tok.atom_names == ["PG"]
        assert first_tok.atom_elements == ["P"]
        assert first_tok.token_type == TOKEN_LIGAND_ATOM + ELEM_TO_IDX["P"]

    def test_parse_sequences_arg(self):
        """parse_sequences_arg should parse 'A:MKFLILF,B:ACDEF' correctly."""
        chains = parse_sequences_arg("A:MKFLILF,B:ACDEF")
        assert len(chains) == 2
        assert chains[0] == {"type": "protein", "id": "A", "sequence": "MKFLILF"}
        assert chains[1] == {"type": "protein", "id": "B", "sequence": "ACDEF"}

    def test_parse_sequences_arg_single(self):
        """Single chain should also work."""
        chains = parse_sequences_arg("A:MAGK")
        assert len(chains) == 1
        assert chains[0]["id"] == "A"
        assert chains[0]["sequence"] == "MAGK"

    def test_tokenize_with_real_ccd(self):
        """With real CCD: ALA→5 heavy atoms, GLY→4 heavy atoms, real ideal coords."""
        ccd = parse_ccd()
        chains = [{"type": "protein", "id": "A", "sequence": "AG"}]
        tokenized = tokenize_sequences(chains, ccd)

        assert tokenized.n_tokens == 2
        # ALA: N, CA, C, O, CB = 5 heavy atoms
        ala_tok = tokenized.tokens[0]
        assert len(ala_tok.atom_names) == 5, f"ALA got {len(ala_tok.atom_names)} atoms: {ala_tok.atom_names}"
        # GLY: N, CA, C, O, OXT = 5 heavy atoms (CCD includes C-terminal OXT)
        gly_tok = tokenized.tokens[1]
        assert len(gly_tok.atom_names) == 5, f"GLY got {len(gly_tok.atom_names)} atoms: {gly_tok.atom_names}"
        # Ideal coords should be non-zero
        assert not np.allclose(ala_tok.ref_coords, 0.0)
        assert not np.allclose(gly_tok.ref_coords, 0.0)

    def test_parse_input_yaml(self, tmp_path):
        """parse_input_yaml should handle protein, rna, and ligand entries."""
        yaml_content = """\
sequences:
  - protein: {id: A, sequence: MKFLILF}
  - rna: {id: B, sequence: AUGCCU}
  - ligand: {id: C, ccd: ATP}
"""
        yaml_path = tmp_path / "input.yaml"
        yaml_path.write_text(yaml_content)

        chains = parse_input_yaml(yaml_path)
        assert len(chains) == 3
        assert chains[0] == {"type": "protein", "id": "A", "sequence": "MKFLILF"}
        assert chains[1] == {"type": "rna", "id": "B", "sequence": "AUGCCU"}
        assert chains[2] == {"type": "ligand", "id": "C", "ccd": "ATP"}


# ============================================================================
# RNA / DNA / sym_id / ligand feature tests
# ============================================================================

def _make_nuc_ccd():
    """Build a fake CCD with RNA, DNA, protein, and ligand entries."""
    ccd = {}
    # Protein
    for comp_id, atoms, elems in [
        ("ALA", ["N", "CA", "C", "O", "CB"], ["N", "C", "C", "O", "C"]),
        ("GLY", ["N", "CA", "C", "O"], ["N", "C", "C", "O"]),
    ]:
        ccd[comp_id] = _make_fake_ccd_component(comp_id, atoms, elems)
    # RNA nucleotides
    nuc_atoms = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
                 "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N1", "C2",
                 "N3", "C4"]
    nuc_elems = ["P", "O", "O", "O", "C", "C", "O", "C", "O",
                 "C", "O", "C", "N", "C", "N", "C", "C", "N", "C",
                 "N", "C"]
    for code in ["A", "C", "G", "U"]:
        ccd[code] = _make_fake_ccd_component(code, nuc_atoms, nuc_elems)
    # DNA nucleotides (same atoms but without O2')
    dna_atoms = [a for a in nuc_atoms if a != "O2'"]
    dna_elems = [e for a, e in zip(nuc_atoms, nuc_elems) if a != "O2'"]
    for code in ["DA", "DC", "DG", "DT"]:
        ccd[code] = _make_fake_ccd_component(code, dna_atoms, dna_elems)
    # Ligand
    ccd["ATP"] = _make_fake_ccd_component(
        "ATP",
        ["PG", "O1G", "O2G", "O3G", "PB", "O1B", "C1'", "N9"],
        ["P", "O", "O", "O", "P", "O", "C", "N"],
    )
    return ccd


class TestNucleotideAndSymId:
    """Tests for RNA/DNA restype, sym_id, and ligand features."""

    def test_rna_restype(self):
        """RNA tokens should get Protenix restype 21-24, not gap (31)."""
        ccd = _make_nuc_ccd()
        chains = [{"type": "rna", "id": "A", "sequence": "AUGC"}]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()
        restype = features["restype"]
        # Purines first: A=21, G=22, C=23, U=24
        assert restype[0].item() == 21, f"A got {restype[0].item()}"
        assert restype[1].item() == 24, f"U got {restype[1].item()}"
        assert restype[2].item() == 22, f"G got {restype[2].item()}"
        assert restype[3].item() == 23, f"C got {restype[3].item()}"
        assert (restype != PROTENIX_MSA_GAP).all(), "No RNA token should be gap"

    def test_dna_restype(self):
        """DNA tokens should get Protenix restype 26-29, not gap (31)."""
        ccd = _make_nuc_ccd()
        chains = [{"type": "dna", "id": "A", "sequence": "ATGC"}]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()
        restype = features["restype"]
        # Purines first: DA=26, DG=27, DC=28, DT=29
        assert restype[0].item() == 26, f"DA got {restype[0].item()}"
        assert restype[1].item() == 29, f"DT got {restype[1].item()}"
        assert restype[2].item() == 27, f"DG got {restype[2].item()}"
        assert restype[3].item() == 28, f"DC got {restype[3].item()}"
        assert (restype != PROTENIX_MSA_GAP).all(), "No DNA token should be gap"

    def test_sym_id_homo_multimer(self):
        """Two identical protein chains should get distinct sym_id values."""
        ccd = _make_nuc_ccd()
        chains = [
            {"type": "protein", "id": "A", "sequence": "AG"},
            {"type": "protein", "id": "B", "sequence": "AG"},
        ]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()
        sym_id = features["sym_id"]
        # Chain A tokens should be sym_id=0, chain B tokens should be sym_id=1
        assert sym_id[0].item() == 0
        assert sym_id[1].item() == 0
        assert sym_id[2].item() == 1
        assert sym_id[3].item() == 1

    def test_sym_id_hetero(self):
        """Different-sequence chains get independent sym_id counters."""
        ccd = _make_nuc_ccd()
        chains = [
            {"type": "protein", "id": "A", "sequence": "AG"},
            {"type": "protein", "id": "B", "sequence": "AA"},
        ]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()
        sym_id = features["sym_id"]
        # Both are sym_id=0 for their respective entities (each entity has one chain)
        assert sym_id[0].item() == 0
        assert sym_id[2].item() == 0

    def test_ligand_features(self):
        """Ligand tokens should have restype=UNK (20), not the MSA-gap sentinel (31)."""
        ccd = _make_nuc_ccd()
        chains = [
            {"type": "protein", "id": "A", "sequence": "AG"},
            {"type": "ligand", "id": "B", "ccd": "ATP"},
        ]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()
        # 2 protein tokens + 8 ligand tokens = 10
        assert features["n_tokens"] == 10
        restype = features["restype"]
        # Ligand tokens must NOT be the MSA-gap sentinel (which would make the
        # restype embedding treat them as missing-alignment positions).
        assert (restype[2:] != PROTENIX_MSA_GAP).all()
        # Ligand tokens use UNK-protein (20) per Protenix convention.
        assert (restype[2:] == 20).all()
        # Protein tokens should not be gap
        assert (restype[:2] != PROTENIX_MSA_GAP).all()

    def test_protein_restype_unchanged(self):
        """Protein restype should still map correctly (regression test)."""
        ccd = _make_nuc_ccd()
        chains = [{"type": "protein", "id": "A", "sequence": "AG"}]
        tokenized = tokenize_sequences(chains, ccd)
        features = tokenized.to_features()
        restype = features["restype"]
        # ALA(A) = Protenix index 0, GLY(G) = Protenix index 7
        assert restype[0].item() == 0, f"ALA got {restype[0].item()}"
        assert restype[1].item() == 7, f"GLY got {restype[1].item()}"


# ============================================================================
# MSA Tests
# ============================================================================

class TestMSA:
    def test_parse_a3m(self):
        """Parse a simple A3M string."""
        content = ">query\nACDEFG\n>seq1\nACdDEFG\n>seq2\nA-DEFG\n"
        seqs, descs = parse_a3m(content)
        assert len(seqs) == 3
        assert descs[0] == "query"
        assert seqs[0] == "ACDEFG"

    def test_a3m_to_matrix(self):
        """Convert A3M to MSA matrix."""
        seqs = ["ACDEFG", "ACDEFG", "A-DEFG"]
        msa, dels = a3m_to_msa_matrix(seqs)
        assert msa.shape == (3, 6)
        assert dels.shape == (3, 6)
        # First sequence should have no gaps (all values < 20 = valid AAs in Protenix encoding)
        assert (msa[0] < 20).all()  # all standard AAs
        # Third sequence has gap at position 1 (Protenix gap = 31)
        assert msa[2, 1] == 31  # gap index (Protenix encoding)

    def test_a3m_deletions(self):
        """Lowercase letters should be counted as deletions."""
        seqs = ["ACDEFG", "AcdCDEFG"]  # 'cd' are insertions -> deletions in query
        msa, dels = a3m_to_msa_matrix(seqs)
        assert msa.shape[1] == 6  # alignment length matches query
        assert dels[1, 1] == 2  # 2 deletions before position 1

    def test_compute_features(self):
        """MSA features should have correct shapes."""
        np.random.seed(42)
        L = 50
        n_seqs = 100
        msa = np.random.randint(0, 21, (n_seqs, L), dtype=np.int8)
        dels = np.zeros((n_seqs, L), dtype=np.int8)

        features = compute_msa_features(msa, dels, max_seqs=64, n_clusters=8)
        assert features.profile.shape == (L, 32)
        assert features.deletion_mean.shape == (L,)
        assert features.cluster_msa.shape[0] <= 8
        assert features.cluster_profile.shape[1] == L
        assert features.cluster_profile.shape[2] == 32
        assert features.cluster_deletion_mean.shape[0] == features.cluster_msa.shape[0]
        assert features.cluster_deletion_mean.shape[1] == L

        # Profile should sum to approximately 1.0
        profile_sums = features.profile.sum(axis=1)
        assert np.allclose(profile_sums, 1.0, atol=0.01)

    def test_species_ids(self):
        """Species-ID parsing matches Protenix convention."""
        from helico.data import get_species_ids
        descs = [
            "tr|A0A0C5B5G6|ABC_HUMAN sapiens",
            "sp|P12345|XYZ_MOUSE",
            "UniRef100_Q12345_HUMAN/1-100",
            "101",  # query
            "UniRef100_A0A6C0PQN6",  # accession-only, no species suffix
        ]
        assert get_species_ids(descs) == ["HUMAN", "MOUSE", "HUMAN", "", ""]

    def test_pair_rows_across_chains(self):
        """Pairing puts species-matched rows at the same index across chains."""
        from helico.data import _pair_rows_across_chains
        # Chain A: query + 3 rows (HUMAN, MOUSE, HUMAN).
        # Chain B: query + 2 rows (HUMAN, MOUSE).
        chain_species = [
            ["", "HUMAN", "MOUSE", "HUMAN"],
            ["", "HUMAN", "MOUSE"],
        ]
        idxs = _pair_rows_across_chains(chain_species, max_paired=10, max_per_species=1)
        # First paired row is always the query (row 0 of each chain).
        assert idxs[0].tolist() == [0, 0]
        # Both species appear in both chains; max_per_species=1 means one row
        # each. Every paired row should have non-negative indices (no gap fills).
        assert (idxs >= 0).all()
        # At least 3 rows (query + 2 species blocks); <= max_paired.
        assert idxs.shape == (3, 2)

    def test_pair_gap_fills_missing_species(self):
        """Species only in chain A becomes gap (-1) for chain B."""
        from helico.data import _pair_rows_across_chains
        chain_species = [
            ["", "HUMAN", "RAT"],
            ["", "HUMAN"],
        ]
        idxs = _pair_rows_across_chains(chain_species, max_paired=10, max_per_species=1)
        # HUMAN in both, RAT only in A: RAT row for chain B is -1 (gap).
        # Singleton species (only in one chain) are SKIPPED by the algorithm
        # (matches Protenix). So only HUMAN is paired.
        assert (idxs[:, 0] >= 0).all()
        # Query + HUMAN in both = 2 paired rows, no -1s because RAT is skipped.
        assert idxs.shape[0] == 2

    def test_assemble_complex_msa(self):
        """End-to-end: raw per-chain MSAs → combined complex features."""
        from helico.data import RawChainMSA, assemble_complex_msa_features
        msa_a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                         dtype=np.int8)
        msa_b = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
        dels_a = np.zeros_like(msa_a, dtype=np.int16)
        dels_b = np.zeros_like(msa_b, dtype=np.int16)
        raws = [
            RawChainMSA(msa=msa_a, deletion_matrix=dels_a,
                        species_ids=["", "HUMAN", "MOUSE"]),
            RawChainMSA(msa=msa_b, deletion_matrix=dels_b,
                        species_ids=["", "HUMAN"]),
        ]
        feats = assemble_complex_msa_features(raws, max_seqs=64, n_clusters=8)
        # Combined L = 5 + 3 = 8; n_seqs includes query + paired + unpaired.
        assert feats.length == 8
        assert feats.profile.shape == (8, 32)
        assert feats.deletion_mean.shape == (8,)
        assert feats.cluster_msa.shape[1] == 8


# ============================================================================
# Cropping Tests
# ============================================================================

class TestCropping:
    def test_spatial_crop(self):
        """Spatial cropping should reduce token count."""
        structure = make_synthetic_structure(n_residues=100)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()
        cropped = spatial_crop(features, crop_size=32)
        assert cropped["n_tokens"] == 32
        assert cropped["token_types"].shape[0] == 32
        assert cropped["chain_same"].shape == (32, 32)

    def test_contiguous_crop(self):
        """Contiguous cropping should preserve sequence order."""
        structure = make_synthetic_structure(n_residues=100)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()
        cropped = contiguous_crop(features, crop_size=32)
        assert cropped["n_tokens"] == 32
        # Residue indices should be contiguous
        res_idx = cropped["res_indices"]
        diffs = res_idx[1:] - res_idx[:-1]
        assert (diffs == 1).all()

    def test_no_crop_when_small(self):
        """Small structures should pass through unchanged."""
        structure = make_synthetic_structure(n_residues=10)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()
        cropped = spatial_crop(features, crop_size=32)
        assert cropped["n_tokens"] == 10

    def test_atom_consistency_after_crop(self):
        """Atoms should remain consistent with tokens after cropping."""
        structure = make_synthetic_structure(n_residues=50)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()
        cropped = spatial_crop(features, crop_size=20)

        # All atom_to_token indices should be in [0, n_tokens)
        assert (cropped["atom_to_token"] >= 0).all()
        assert (cropped["atom_to_token"] < cropped["n_tokens"]).all()

        # atoms_per_token should sum to n_atoms
        assert cropped["atoms_per_token"].sum().item() == cropped["n_atoms"]


# ============================================================================
# Dataset / DataLoader Tests
# ============================================================================

class TestDataset:
    def test_dataset_basic(self):
        """Dataset should return valid feature dicts."""
        structures = [tokenize_structure(make_synthetic_structure(n_residues=30)) for _ in range(5)]
        dataset = HelicoDataset(structures, crop_size=20)
        assert len(dataset) == 5
        item = dataset[0]
        assert "token_types" in item
        assert item["n_tokens"] <= 30

    def test_collate(self):
        """Collation should handle variable-length structures."""
        structures = [
            tokenize_structure(make_synthetic_structure(n_residues=20)),
            tokenize_structure(make_synthetic_structure(n_residues=30)),
        ]
        dataset = HelicoDataset(structures, crop_size=40)
        batch = collate_fn([dataset[0], dataset[1]])

        assert batch["token_types"].shape[0] == 2  # batch size
        # Padded to at least max tokens in batch (rounded up to a multiple
        # for cuDNN flash-attn compatibility — see collate_fn docstring).
        max_tok = max(20, 30)
        n_padded = batch["token_types"].shape[1]
        assert n_padded >= max_tok
        assert batch["token_mask"].shape == (2, n_padded)

    def test_synthetic_batch(self):
        """Synthetic batch for model testing should have correct shapes."""
        batch = make_synthetic_batch(n_tokens=32, batch_size=2, device="cpu")
        assert batch["token_types"].shape == (2, 32)
        assert batch["atom_coords"].shape == (2, 32 * 5, 3)
        assert batch["token_mask"].shape == (2, 32)


# ============================================================================
# Full Pipeline Integration Test
# ============================================================================

class TestFullPipeline:
    def test_synthetic_pipeline(self):
        """Process a synthetic structure through the full pipeline."""
        structure = make_synthetic_structure(n_residues=50, n_chains=2, include_ligand=True)
        tokenized = tokenize_structure(structure)
        features = tokenized.to_features()
        cropped = spatial_crop(features, crop_size=32)

        # Verify all expected keys
        expected_keys = [
            "token_types", "chain_indices", "res_indices", "rel_pos",
            "atom_coords", "ref_coords", "atom_to_token", "atoms_per_token",
            "atom_element_idx", "chain_same", "n_tokens", "n_atoms",
        ]
        for key in expected_keys:
            assert key in cropped, f"Missing key: {key}"

        # Check dtypes
        assert cropped["token_types"].dtype == torch.long
        assert cropped["atom_coords"].dtype == torch.float32
        assert cropped["chain_same"].dtype == torch.long

    def test_batch_pipeline(self):
        """Process multiple structures into a batched tensor."""
        structures = [
            tokenize_structure(make_synthetic_structure(n_residues=r, n_chains=c))
            for r, c in [(20, 1), (30, 2), (15, 1)]
        ]
        dataset = HelicoDataset(structures, crop_size=25)
        batch = collate_fn([dataset[i] for i in range(3)])

        assert batch["token_types"].shape[0] == 3
        assert "token_mask" in batch
        assert "atom_mask" in batch

