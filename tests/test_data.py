"""Integration tests for Helico data pipeline."""

import io
import numpy as np
import pytest
import torch
from pathlib import Path

from helico.data import (
    RAW_DIR,
    PROCESSED_DIR,
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
    load_pdb_seqres,
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

# Derived paths — may be None if env vars not set
COMPONENTS_CIF = RAW_DIR / "components.cif" if RAW_DIR else None
PDB_SEQRES = RAW_DIR / "pdb_seqres.txt.gz" if RAW_DIR else None


# ============================================================================
# CCD Parser Tests
# ============================================================================

class TestCCDParser:
    """Tests for Chemical Component Dictionary parsing."""

    @pytest.fixture(scope="class")
    def ccd(self):
        """Parse CCD once for all tests in this class."""
        if COMPONENTS_CIF is None or not COMPONENTS_CIF.exists():
            pytest.skip("HELICO_RAW_DIR not set or components.cif not found")
        cache_path = PROCESSED_DIR / "ccd_cache_test.pkl"
        return parse_ccd(COMPONENTS_CIF, cache_path=cache_path)

    def test_alanine(self, ccd):
        """ALA should have known atom count and elements."""
        assert "ALA" in ccd
        ala = ccd["ALA"]
        assert isinstance(ala, CCDComponent)
        assert ala.comp_id == "ALA"
        # ALA has 5 heavy atoms: N, CA, C, O, CB
        assert ala.n_heavy_atoms >= 5
        heavy_elements = [e for e in ala.atom_elements if e != "H"]
        assert "N" in heavy_elements
        assert "C" in heavy_elements
        assert "O" in heavy_elements

    def test_glycine(self, ccd):
        """GLY should have fewer atoms than ALA (no CB)."""
        assert "GLY" in ccd
        gly = ccd["GLY"]
        assert gly.n_heavy_atoms >= 4  # N, CA, C, O
        ala = ccd["ALA"]
        assert gly.n_heavy_atoms < ala.n_heavy_atoms

    def test_atp(self, ccd):
        """ATP should be present with many atoms and bonds."""
        assert "ATP" in ccd
        atp = ccd["ATP"]
        assert atp.n_heavy_atoms > 20  # ATP is a large molecule
        assert len(atp.bonds) > 20
        assert "P" in atp.atom_elements  # ATP has phosphorus

    def test_hem(self, ccd):
        """HEM (heme) should have iron."""
        assert "HEM" in ccd
        hem = ccd["HEM"]
        assert "Fe" in hem.atom_elements or "FE" in [e.upper() for e in hem.atom_elements]
        assert hem.n_heavy_atoms > 30

    def test_ideal_coords(self, ccd):
        """Components should have ideal coordinates."""
        ala = ccd["ALA"]
        assert ala.ideal_coords is not None
        assert ala.ideal_coords.shape[0] == len(ala.atom_names)
        assert ala.ideal_coords.shape[1] == 3
        # Coords should be finite
        assert np.all(np.isfinite(ala.ideal_coords))

    def test_component_type(self, ccd):
        """Check component types are parsed."""
        ala = ccd["ALA"]
        assert "PEPTIDE" in ala.comp_type.upper() or "L-PEPTIDE" in ala.comp_type.upper()

    def test_many_components_parsed(self, ccd):
        """Should have parsed thousands of components."""
        assert len(ccd) > 1000


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
        if PROCESSED_DIR is None:
            pytest.skip("HELICO_PROCESSED_DIR not set")
        ccd_cache = PROCESSED_DIR / "ccd_cache.pkl"
        if not ccd_cache.exists():
            ccd_cache = PROCESSED_DIR / "ccd_cache_test.pkl"
        if not ccd_cache.exists():
            pytest.skip("CCD cache not found")

        ccd = parse_ccd(cache_path=ccd_cache)
        chains = [{"type": "protein", "id": "A", "sequence": "AG"}]
        tokenized = tokenize_sequences(chains, ccd)

        assert tokenized.n_tokens == 2
        # ALA: N, CA, C, O, CB = 5 heavy atoms
        ala_tok = tokenized.tokens[0]
        assert len(ala_tok.atom_names) == 5, f"ALA got {len(ala_tok.atom_names)} atoms: {ala_tok.atom_names}"
        # GLY: N, CA, C, O = 4 heavy atoms
        gly_tok = tokenized.tokens[1]
        assert len(gly_tok.atom_names) == 4, f"GLY got {len(gly_tok.atom_names)} atoms: {gly_tok.atom_names}"
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
        # First sequence should have no gaps
        assert (msa[0] < 20).all()  # all standard AAs
        # Third sequence has gap at position 1
        assert msa[2, 1] == 20  # gap index

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
        assert features.profile.shape == (L, 22)
        assert features.cluster_msa.shape[0] <= 8
        assert features.cluster_profile.shape[1] == L
        assert features.cluster_profile.shape[2] == 22

        # Profile should sum to approximately 1.0
        profile_sums = features.profile.sum(axis=1)
        assert np.allclose(profile_sums, 1.0, atol=0.01)


# ============================================================================
# PDB Seqres Tests
# ============================================================================

class TestSeqres:
    def test_load_seqres(self):
        """Load pdb_seqres.txt.gz and verify structure."""
        if PDB_SEQRES is None or not PDB_SEQRES.exists():
            pytest.skip("HELICO_RAW_DIR not set or pdb_seqres.txt.gz not found")
        seqres = load_pdb_seqres(PDB_SEQRES)
        assert len(seqres) > 1000  # should have many PDB entries
        # Check a known entry
        for pdb_id, chains in list(seqres.items())[:5]:
            assert len(pdb_id) == 4
            for chain_id, seq in chains.items():
                assert len(seq) > 0


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
        # Padded to max tokens in batch
        max_tok = max(20, 30)
        assert batch["token_types"].shape[1] == max_tok
        assert batch["token_mask"].shape == (2, max_tok)

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


# ============================================================================
# Preprocessing Pipeline Tests
# ============================================================================

class TestPreprocessing:
    """Tests for the preprocessing pipeline."""

    @property
    def MMCIF_DIR(self) -> Path | None:
        return RAW_DIR / "mmCIF" if RAW_DIR else None

    def _skip_without_raw(self):
        if self.MMCIF_DIR is None or not self.MMCIF_DIR.exists():
            pytest.skip("HELICO_RAW_DIR not set or mmCIF directory not found")

    def _find_a_cif_gz(self) -> Path:
        """Find a single .cif.gz file for testing."""
        for subdir in sorted(self.MMCIF_DIR.iterdir()):
            if subdir.is_dir():
                for f in sorted(subdir.iterdir()):
                    if f.name.endswith(".cif.gz"):
                        return f
        pytest.skip("No .cif.gz files found in mmCIF dir")

    def test_parse_mmcif_gzipped(self):
        """parse_mmcif should handle .cif.gz files."""
        self._skip_without_raw()
        cif_path = self._find_a_cif_gz()
        structure = parse_mmcif(cif_path)
        # Structure may be None if filtered, but should not raise
        if structure is not None:
            assert structure.pdb_id != ""
            assert len(structure.chains) > 0

    def test_discover_mmcif_files(self):
        """discover_mmcif_files should find files in the mmCIF directory."""
        self._skip_without_raw()
        files = discover_mmcif_files(self.MMCIF_DIR)
        assert len(files) > 0
        assert all(f.name.endswith(".cif.gz") for f in files)

    def test_process_single_structure(self, tmp_path):
        """End-to-end: parse + tokenize + pickle one real structure."""
        self._skip_without_raw()
        if PROCESSED_DIR is None:
            pytest.skip("HELICO_PROCESSED_DIR not set")

        # Accept either the main cache or the test cache
        ccd_cache = PROCESSED_DIR / "ccd_cache.pkl"
        if not ccd_cache.exists():
            ccd_cache = PROCESSED_DIR / "ccd_cache_test.pkl"
        if not ccd_cache.exists():
            pytest.skip("CCD cache not found")
        ccd = parse_ccd(cache_path=ccd_cache)
        _init_worker(ccd)

        # Find a structure that passes filters
        cif_path = self._find_a_cif_gz()
        result = _process_single_structure((cif_path, tmp_path, 9.0))
        # Result may be None if structure was filtered; try a few more
        if result is None:
            files = discover_mmcif_files(self.MMCIF_DIR)[:20]
            for f in files:
                result = _process_single_structure((f, tmp_path, 9.0))
                if result is not None:
                    break

        if result is None:
            pytest.skip("No structures passed filters in first 20 files")

        assert isinstance(result, StructureMetadata)
        assert result.n_tokens > 0
        # Verify pickle was written
        pkl_path = tmp_path / result.pickle_path
        assert pkl_path.exists()
        import pickle
        with open(pkl_path, "rb") as f:
            ts = pickle.load(f)
        assert isinstance(ts, TokenizedStructure)
        assert ts.n_tokens == result.n_tokens

    def test_manifest_round_trip(self, tmp_path):
        """Save and load a manifest, verifying data preservation."""
        metadata = {
            "1ABC": StructureMetadata(
                pdb_id="1ABC",
                pickle_path="structures/ab/1abc.pkl",
                n_tokens=100,
                n_atoms=500,
                n_chains=2,
                resolution=2.5,
                release_date="2020-01-15",
                method="X-RAY DIFFRACTION",
                entity_types=["protein", "ligand"],
                chain_ids=["A", "B"],
            ),
            "2XYZ": StructureMetadata(
                pdb_id="2XYZ",
                pickle_path="structures/xy/2xyz.pkl",
                n_tokens=50,
                n_atoms=200,
                n_chains=1,
                resolution=1.8,
                release_date="2021-06-01",
                method="X-RAY DIFFRACTION",
                entity_types=["protein"],
                chain_ids=["A"],
            ),
        }

        manifest_path = tmp_path / "manifest.json"
        build_manifest(metadata, manifest_path)
        assert manifest_path.exists()

        loaded = load_manifest(manifest_path)
        assert len(loaded) == 2
        assert "1ABC" in loaded
        assert loaded["1ABC"].n_tokens == 100
        assert loaded["1ABC"].resolution == 2.5
        assert loaded["1ABC"].release_date == "2020-01-15"
        assert loaded["2XYZ"].pickle_path == "structures/xy/2xyz.pkl"

    def test_lazy_dataset(self, tmp_path):
        """Create small set of pickles and verify LazyHelicoDataset works."""
        import pickle

        # Create synthetic structures and save as pickles
        metadata = {}
        for i, n_res in enumerate([20, 30, 25]):
            pdb_id = f"TST{i}"
            structure = make_synthetic_structure(n_residues=n_res)
            structure = structure.__class__(
                pdb_id=pdb_id,
                chains=structure.chains,
                resolution=structure.resolution,
                release_date=f"2020-0{i+1}-01",
                method=structure.method,
            )
            tokenized = tokenize_structure(structure)
            rel_path = f"structures/{pdb_id.lower()}.pkl"
            pkl_path = tmp_path / rel_path
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump(tokenized, f)

            metadata[pdb_id] = StructureMetadata(
                pdb_id=pdb_id,
                pickle_path=rel_path,
                n_tokens=tokenized.n_tokens,
                n_atoms=tokenized.n_atoms,
                n_chains=1,
                resolution=2.0,
                release_date=f"2020-0{i+1}-01",
                method="SYNTHETIC",
                entity_types=["protein"],
                chain_ids=["A"],
            )

        dataset = LazyHelicoDataset(
            manifest=metadata,
            processed_dir=tmp_path,
            crop_size=15,
        )
        assert len(dataset) == 3

        item = dataset[0]
        assert "token_types" in item
        assert item["n_tokens"] <= 20
        assert "msa_profile" in item
        assert "has_msa" in item

        # Test with filter
        filtered = LazyHelicoDataset(
            manifest=metadata,
            processed_dir=tmp_path,
            crop_size=15,
            filter_fn=lambda m: m.release_date < "2020-02-01",
        )
        assert len(filtered) == 1

    def test_tar_index(self, tmp_path):
        """Build a tar index from a small synthetic tar and verify it."""
        import tarfile as tf

        # Create a small test tar
        tar_path = tmp_path / "test.tar"
        with tf.open(tar_path, "w") as tar:
            for i in range(5):
                data = f"content of file {i}".encode()
                info = tf.TarInfo(name=f"dir/file{i}.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        index = build_tar_index(tar_path)
        assert len(index.entries) == 5
        assert index.tar_path == tar_path

        # Verify we can read back via the index
        from helico.data import read_tar_member
        for i in range(5):
            name = f"dir/file{i}.txt"
            assert name in index.entries
            offset, size = index.entries[name]
            assert offset >= 0
            assert size > 0
            content = read_tar_member(tar_path, offset, size)
            assert content == f"content of file {i}".encode()

    def test_tar_index_real(self):
        """Build index from real rcsb_raw_msa.tar if available (slow test)."""
        if RAW_DIR is None:
            pytest.skip("HELICO_RAW_DIR not set")
        tar_path = RAW_DIR / "rcsb_raw_msa.tar"
        if not tar_path.exists():
            pytest.skip("rcsb_raw_msa.tar not found")
        # Only verify the tar is openable and has entries; full indexing is too slow for CI
        import tarfile as tf
        with tf.open(tar_path, "r") as tar:
            members = []
            for i, member in enumerate(tar):
                members.append(member)
                if i >= 9:
                    break
        assert len(members) > 0
        assert all(m.name.endswith(".a3m.gz") or "/" in m.name for m in members)

    def test_tar_index_save_load(self, tmp_path):
        """TarIndex round-trip through pickle."""
        index = TarIndex(
            tar_path=Path("/fake/path.tar"),
            entries={"file1.txt": (512, 100), "file2.txt": (1024, 200)},
        )
        save_path = tmp_path / "test_index.pkl"
        save_tar_index(index, save_path)
        loaded = load_tar_index(save_path)
        assert loaded.tar_path == index.tar_path
        assert loaded.entries == index.entries
