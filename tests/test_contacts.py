"""Integration tests for residue/residue contact conditioning.

Real pyconfind runs against real PDB structures — no stubs. Structures are
downloaded from RCSB on first use and cached under the pytest tmp dir's parent
so repeated runs are cheap.

Covers:
  - the token -> pyconfind index mapping (helico.contacts)
  - densification into the 3-state matrix (TokenizedStructure.to_features)
  - crop and collate behaviour
  - the conditioning sampler's invariants
"""

import os
import pickle
import urllib.request
from pathlib import Path

import pytest
import torch

from helico.contacts import (
    MIN_SEQ_SEPARATION,
    build_gemmi,
    compute_contacts,
    load_rotamer_library,
    sample_conditioning,
)
from helico.data import (
    CONTACT_ABSENT,
    CONTACT_PRESENT,
    CONTACT_UNKNOWN,
    _subset_features,
    collate_fn,
    make_synthetic_batch,
    parse_mmcif,
    tokenize_structure,
)

# 1UBQ  small single-chain globular protein
# 4HHB  hetero-tetramer with HEM ligands -> inter-chain contacts + ineligible tokens
# 1BNA  DNA only -> no protein at all
PDB_IDS = ["1UBQ", "4HHB", "1BNA"]

_env_dir = os.environ.get("HELICO_TEST_PDB_DIR")
# Note: Path("") is Path("."), which is truthy — so check the string, not the Path.
_CACHE = Path(_env_dir) if _env_dir else Path(__file__).parent / ".pdb_cache"


def _pdb_path(pdb_id: str) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{pdb_id}.cif"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"could not download {pdb_id}: {e}")
    return path


# `ccd` and `rotamer_library` come from tests/conftest.py.


def _tokenize(pdb_id, ccd, rotamer_library):
    structure = parse_mmcif(_pdb_path(pdb_id))
    assert structure is not None, f"{pdb_id} failed to parse"
    ts = tokenize_structure(structure, ccd=ccd)
    ts.contact_edges, ts.contact_eligible = compute_contacts(ts.tokens, rotamer_library)
    return ts


class TestTokenMapping:
    """The token -> pyconfind Position index correspondence."""

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    def test_slot_count_matches_positions(self, pdb_id, ccd, rotamer_library):
        """compute_contacts asserts this internally; here we check it doesn't raise."""
        ts = _tokenize(pdb_id, ccd, rotamer_library)
        assert isinstance(ts.contact_edges, list)
        assert isinstance(ts.contact_eligible, list)

    @pytest.mark.parametrize("pdb_id", ["1UBQ", "4HHB"])
    def test_residue_identity_roundtrips(self, pdb_id, ccd, rotamer_library):
        """Every emitted gemmi residue must carry the token's own residue name.

        A mismatch means the grouping drifted and contacts would land on the
        wrong tokens.
        """
        from pyconfind import analyze
        from helico.contacts import PYCONFIND_KWARGS

        structure = parse_mmcif(_pdb_path(pdb_id))
        ts = tokenize_structure(structure, ccd=ccd)
        st, slot_to_token = build_gemmi(ts.tokens)
        analysis = analyze(st, rotamer_library=rotamer_library, **PYCONFIND_KWARGS)
        assert len(analysis.positions) == len(slot_to_token)
        for slot, token_idx in enumerate(slot_to_token):
            if token_idx < 0:
                continue
            assert (
                analysis.positions[slot].position.resname
                == ts.tokens[token_idx].res_name
            )

    def test_eligible_tokens_are_protein(self, ccd, rotamer_library):
        """4HHB's HEM ligand atoms must not be eligible."""
        ts = _tokenize("4HHB", ccd, rotamer_library)
        for token_idx in ts.contact_eligible:
            assert ts.tokens[token_idx].token_type <= 20
        assert len(ts.contact_eligible) < len(ts.tokens), "expected ineligible ligand tokens"

    def test_dna_only_structure_has_no_contacts(self, ccd, rotamer_library):
        ts = _tokenize("1BNA", ccd, rotamer_library)
        assert ts.contact_edges == []
        assert ts.contact_eligible == []

    def test_inter_chain_contacts_present(self, ccd, rotamer_library):
        """pyconfind handles complexes natively; we must not drop cross-chain pairs."""
        ts = _tokenize("4HHB", ccd, rotamer_library)
        chain_of = {i: t.chain_idx for i, t in enumerate(ts.tokens)}
        inter = [(i, j) for i, j in ts.contact_edges if chain_of[i] != chain_of[j]]
        assert len(inter) > 0, "4HHB is a tetramer; expected inter-chain contacts"


class TestContactStateMatrix:
    """Densification into the 3-state matrix."""

    @pytest.mark.parametrize("pdb_id", ["1UBQ", "4HHB"])
    def test_symmetric_with_unknown_diagonal(self, pdb_id, ccd, rotamer_library):
        cs = _tokenize(pdb_id, ccd, rotamer_library).to_features()["contact_state"]
        assert torch.equal(cs, cs.T)
        assert bool(cs.diagonal().eq(CONTACT_UNKNOWN).all())
        assert cs.dtype == torch.uint8

    @pytest.mark.parametrize("pdb_id", ["1UBQ", "4HHB"])
    def test_present_respects_seq_separation(self, pdb_id, ccd, rotamer_library):
        """Contacts closer than MIN_SEQ_SEPARATION within a chain must be UNKNOWN.

        MarinFold filters those out before emitting, so training on them would
        teach the model to expect a signal no contact predictor supplies.
        """
        f = _tokenize(pdb_id, ccd, rotamer_library).to_features()
        cs, chain, res = f["contact_state"], f["chain_indices"], f["res_indices"]
        i, j = (cs == CONTACT_PRESENT).nonzero(as_tuple=True)
        ok = (chain[i] != chain[j]) | ((res[i] - res[j]).abs() >= MIN_SEQ_SEPARATION)
        assert bool(ok.all())

    def test_ineligible_tokens_are_unknown(self, ccd, rotamer_library):
        """Ligand tokens can never be ABSENT — we simply don't know about them."""
        ts = _tokenize("4HHB", ccd, rotamer_library)
        cs = ts.to_features()["contact_state"]
        eligible = torch.zeros(len(ts.tokens), dtype=torch.bool)
        eligible[torch.tensor(ts.contact_eligible)] = True
        assert bool((cs[~eligible] == CONTACT_UNKNOWN).all())

    def test_contact_density_is_about_one_per_residue(self, ccd, rotamer_library):
        """Sanity band on the measured 0.6-1.2 contacts/residue."""
        ts = _tokenize("1UBQ", ccd, rotamer_library)
        cs = ts.to_features()["contact_state"]
        n_contacts = int((cs == CONTACT_PRESENT).sum()) // 2
        assert 0.4 < n_contacts / len(ts.contact_eligible) < 1.6

    def test_absent_state_exists(self, ccd, rotamer_library):
        """A fully-specified matrix is mostly ABSENT — that is the bulk of the signal."""
        cs = _tokenize("1UBQ", ccd, rotamer_library).to_features()["contact_state"]
        assert int((cs == CONTACT_ABSENT).sum()) > int((cs == CONTACT_PRESENT).sum())

    def test_absent_from_legacy_pickle(self, ccd, rotamer_library):
        """Pickles written before contacts existed must still produce features."""
        structure = parse_mmcif(_pdb_path("1UBQ"))
        ts = tokenize_structure(structure, ccd=ccd)
        ts.__dict__.pop("contact_eligible", None)
        ts.__dict__.pop("contact_edges", None)
        assert "contact_state" not in ts.to_features()


class TestCropAndCollate:
    def test_crop_commutes_with_contact_computation(self, ccd, rotamer_library):
        """Contacts of a crop == crop of the contacts.

        _subset_features is a whitelist; if contact_state were left out it would
        silently vanish, so this is the regression guard for that.
        """
        f = _tokenize("4HHB", ccd, rotamer_library).to_features()
        idx = torch.arange(0, f["n_tokens"], 3)
        sub = _subset_features(f, idx)
        assert "contact_state" in sub
        assert torch.equal(sub["contact_state"], f["contact_state"][idx][:, idx])

    def test_collate_pads_with_unknown(self, ccd, rotamer_library):
        feats = []
        for pdb_id in ["1UBQ", "4HHB"]:
            f = _tokenize(pdb_id, ccd, rotamer_library).to_features()
            n = f["n_tokens"]
            # collate_fn requires the MSA keys the datasets normally attach
            f["msa_profile"] = torch.zeros(n, 32)
            f["cluster_msa"] = torch.zeros(1, n, dtype=torch.long)
            f["cluster_profile"] = torch.zeros(1, n, 32)
            f["deletion_mean"] = torch.zeros(n)
            f["cluster_deletion_mean"] = torch.zeros(1, n)
            f["has_msa"] = torch.tensor(0)
            feats.append(f)
        batch = collate_fn(feats)
        cs = batch["contact_state"]
        n_small = min(f["n_tokens"] for f in feats)
        assert cs.shape[0] == 2 and cs.shape[1] == cs.shape[2]
        # the padded rows/cols of the smaller item assert nothing
        small = 0 if feats[0]["n_tokens"] == n_small else 1
        assert bool((cs[small, n_small:, :] == CONTACT_UNKNOWN).all())
        assert bool((cs[small, :, n_small:] == CONTACT_UNKNOWN).all())

    def test_synthetic_batch_has_symmetric_contacts(self):
        batch = make_synthetic_batch(n_tokens=16, device="cpu")
        cs = batch["contact_state"]
        assert cs.shape == (1, 16, 16) and cs.dtype == torch.uint8
        assert torch.equal(cs, cs.transpose(-1, -2))

    def test_synthetic_batch_without_contacts(self):
        cs = make_synthetic_batch(n_tokens=16, has_contacts=False, device="cpu")["contact_state"]
        assert bool((cs == CONTACT_UNKNOWN).all())


class TestPreprocessResume:
    # These call preprocess_structures, which loads the rotamer library itself.
    # Requesting the fixture makes them skip when it cannot be fetched, instead
    # of hard-failing the suite on a network blip the way sibling tests do not.
    """A partially-resumed preprocess must not truncate the manifest.

    Real end-to-end runs of ``preprocess_structures`` against gzipped mmCIFs —
    the bug this guards was invisible to every unit-level test because it only
    appears when *some* files are skipped and *some* are processed.
    """

    @staticmethod
    def _stage(tmp_path, ccd, pdb_ids):
        """Lay out raw/ + processed/ the way preprocess expects."""
        import gzip
        import shutil

        from helico.data import _processed_dir

        raw = tmp_path / "raw" / "mmCIF" / "xx"
        raw.mkdir(parents=True, exist_ok=True)
        processed = tmp_path / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        cache = processed / "ccd_cache.pkl"
        if not cache.exists():
            shutil.copy(_processed_dir() / "ccd_cache.pkl", cache)
        for pdb_id in pdb_ids:
            dest = raw / f"{pdb_id.lower()}.cif.gz"
            if not dest.exists():
                with open(_pdb_path(pdb_id), "rb") as src, gzip.open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        return tmp_path / "raw", processed

    def test_resume_keeps_previously_processed_structures(self, tmp_path, ccd, rotamer_library):
        from helico.data import build_manifest, load_manifest, preprocess_structures

        # First pass: two structures.
        raw, processed = self._stage(tmp_path, ccd, ["1UBQ", "1CRN"])
        first = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
        )
        build_manifest(first, processed / "manifest.json")
        assert len(first) == 2, first.keys()

        # Second pass: two more arrive, the original two are skipped.
        self._stage(tmp_path, ccd, ["1UBQ", "1CRN", "3HTB", "1PGB"])
        second = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
            skip_existing=True,
        )
        build_manifest(second, processed / "manifest.json")

        final = load_manifest(processed / "manifest.json")
        assert set(final) == set(first) | set(second), (
            f"manifest truncated: {sorted(final)} vs expected {sorted(set(first) | set(second))}"
        )
        assert len(final) == 4

    def test_all_skipped_returns_full_manifest(self, tmp_path, ccd, rotamer_library):
        from helico.data import build_manifest, load_manifest, preprocess_structures

        raw, processed = self._stage(tmp_path, ccd, ["1UBQ", "1CRN"])
        first = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
        )
        build_manifest(first, processed / "manifest.json")

        again = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
            skip_existing=True,
        )
        build_manifest(again, processed / "manifest.json")
        assert set(load_manifest(processed / "manifest.json")) == set(first)

    def test_reprocessed_entries_win_over_carried(self, tmp_path, ccd):
        """A structure redone in this run must use the fresh metadata."""
        from helico.data import build_manifest, load_manifest, preprocess_structures

        raw, processed = self._stage(tmp_path, ccd, ["1UBQ", "1CRN"])
        first = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
        )
        # Corrupt one manifest entry, then force a full reprocess.
        pdb_id = sorted(first)[0]
        first[pdb_id].n_tokens = -999
        build_manifest(first, processed / "manifest.json")

        redone = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
            skip_existing=False,
        )
        build_manifest(redone, processed / "manifest.json")
        assert load_manifest(processed / "manifest.json")[pdb_id].n_tokens > 0

    def test_require_contacts_reprocesses_only_what_is_missing(self, tmp_path, ccd):
        """A contacts migration must be resumable.

        Plain skip_existing keys off the pickle merely existing, so after an
        earlier contact-free preprocess it skips everything and the migration
        silently does nothing. This is the guard for that — it is exactly the
        situation a partially-completed migration leaves on the volume.
        """
        import pickle as _pickle

        from helico.data import build_manifest, load_manifest, preprocess_structures

        raw, processed = self._stage(tmp_path, ccd, ["1UBQ", "1CRN", "1PGB"])
        kwargs = dict(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
        )
        first = preprocess_structures(**kwargs, skip_existing=False)
        build_manifest(first, processed / "manifest.json")

        # Strip contacts from one pickle, as if the run died before reaching it.
        victim = sorted(processed.glob("structures/**/*.pkl"))[0]
        ts = _pickle.load(open(victim, "rb"))
        stripped_id = ts.pdb_id
        ts.__dict__.pop("contact_edges", None)
        ts.__dict__.pop("contact_eligible", None)
        with open(victim, "wb") as f:
            _pickle.dump(ts, f, protocol=_pickle.HIGHEST_PROTOCOL)

        # Plain skip_existing is a no-op — it cannot see the missing contacts.
        plain = preprocess_structures(**kwargs, skip_existing=True)
        assert getattr(
            _pickle.load(open(victim, "rb")), "contact_eligible", None
        ) is None, "plain skip_existing unexpectedly reprocessed"
        assert len(plain) == len(first), "manifest lost entries on the no-op pass"

        # require_contacts repairs exactly the gap.
        repaired = preprocess_structures(**kwargs, skip_existing=True, require_contacts=True)
        build_manifest(repaired, processed / "manifest.json")
        assert getattr(
            _pickle.load(open(victim, "rb")), "contact_eligible", None
        ) is not None, f"{stripped_id} still missing contacts"
        for p in processed.glob("structures/**/*.pkl"):
            assert getattr(_pickle.load(open(p, "rb")), "contact_eligible", None) is not None
        assert len(load_manifest(processed / "manifest.json")) == len(first)

    def test_stems_with_contacts_detects_correctly(self, tmp_path, ccd):
        from helico.data import _stems_with_contacts, preprocess_structures

        raw, processed = self._stage(tmp_path, ccd, ["1UBQ", "1CRN"])
        preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
            skip_existing=False,
        )
        pickles = sorted(processed.glob("structures/**/*.pkl"))
        assert _stems_with_contacts(pickles) == {p.stem for p in pickles}

    def test_contacts_survive_a_resume(self, tmp_path, ccd):
        """Carried-forward structures keep the contacts from their pickle."""
        import pickle as _pickle

        from helico.data import build_manifest, preprocess_structures

        raw, processed = self._stage(tmp_path, ccd, ["1UBQ"])
        first = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
        )
        build_manifest(first, processed / "manifest.json")

        self._stage(tmp_path, ccd, ["1UBQ", "1CRN"])
        second = preprocess_structures(
            mmcif_dir=raw / "mmCIF", output_dir=processed,
            ccd_cache_path=processed / "ccd_cache.pkl", n_workers=1,
            skip_existing=True,
        )
        build_manifest(second, processed / "manifest.json")

        for meta in second.values():
            ts = _pickle.load(open(processed / meta.pickle_path, "rb"))
            assert ts.contact_eligible is not None, f"{meta.pdb_id} lost contacts"


class TestDatasetIntegration:
    """The dataset must apply conditioning after cropping, per its spec."""

    @staticmethod
    def _dataset(ccd, rotamer_library, spec, crop_size=64):
        from helico.data import HelicoDataset

        ts = _tokenize("1AKE", ccd, rotamer_library)
        return HelicoDataset(
            structures=[ts], crop_size=crop_size, contact_conditioning=spec
        )

    def test_oracle_spec_passes_matrix_through(self, ccd, rotamer_library):
        """spec=None must leave the ground-truth matrix untouched."""
        ds = self._dataset(ccd, rotamer_library, None)
        item = ds[0]
        assert "contact_state" in item
        assert int((item["contact_state"] == CONTACT_PRESENT).sum()) > 0

    def test_sampled_spec_varies_between_draws(self, ccd, rotamer_library):
        """Conditioning is sampled per __getitem__, not fixed per dataset."""
        ds = self._dataset(ccd, rotamer_library, "sampled")
        torch.manual_seed(0)
        seen = {int((ds[0]["contact_state"] != CONTACT_UNKNOWN).sum()) for _ in range(12)}
        assert len(seen) > 1, "conditioning level never varied across draws"

    def test_fixed_spec_is_reproducible(self, ccd, rotamer_library):
        """A pinned spec must give the same matrix every time — this is what
        makes the validation conditioning curve comparable across steps.

        Uses a crop larger than the structure: spatial_crop picks a random
        centre, so with real cropping two draws differ by *crop*, which would
        mask what this test is actually about.
        """
        ds = self._dataset(
            ccd, rotamer_library,
            {"mode": "full", "eps_fp": 0.0, "eps_fn": 0.0},
            crop_size=4096,
        )
        a, b = ds[0]["contact_state"], ds[0]["contact_state"]
        assert torch.equal(a, b)

    def test_none_spec_yields_all_unknown(self, ccd, rotamer_library):
        ds = self._dataset(ccd, rotamer_library, {"mode": "none"})
        assert bool((ds[0]["contact_state"] == CONTACT_UNKNOWN).all())

    def test_conditioning_applies_after_cropping(self, ccd, rotamer_library):
        """The matrix must be crop-sized, so the level describes what the model
        actually sees rather than the uncropped structure."""
        crop = 48
        ds = self._dataset(ccd, rotamer_library, "sampled", crop_size=crop)
        item = ds[0]
        assert item["contact_state"].shape == (crop, crop)
        assert item["n_tokens"] == crop

    def test_collates_into_a_training_batch(self, ccd, rotamer_library):
        """Full path: structure -> contacts -> crop -> condition -> collate."""
        ds = self._dataset(ccd, rotamer_library, "sampled", crop_size=48)
        batch = collate_fn([ds[0], ds[0]])
        cs = batch["contact_state"]
        assert cs.shape == (2, 48, 48) and cs.dtype == torch.uint8
        assert torch.equal(cs, cs.transpose(-1, -2))


class TestOracleContacts:
    """bench.oracle_contact_state: GT contacts re-indexed onto predicted tokens."""

    @pytest.mark.parametrize("pdb_id", ["1UBQ", "4HHB"])
    def test_matches_direct_computation(self, pdb_id, ccd, rotamer_library):
        """Re-indexing must preserve every contact.

        The predicted tokenization comes from sequences derived from the GT
        structure, so the positional mapping should be exact — any loss here
        means the chain pairing drifted.
        """
        from helico.bench import oracle_contact_state, structure_to_chains
        from helico.data import tokenize_sequences

        gt = parse_mmcif(_pdb_path(pdb_id))
        predicted = tokenize_sequences(structure_to_chains(gt), ccd)
        state = oracle_contact_state(gt, predicted, rotamer_library)
        assert state is not None

        direct = _tokenize(pdb_id, ccd, rotamer_library).to_features()["contact_state"]
        n_mapped = int((state == CONTACT_PRESENT).sum()) // 2
        n_direct = int((direct == CONTACT_PRESENT).sum()) // 2
        assert n_mapped == n_direct, f"lost contacts: {n_mapped} vs {n_direct}"

    def test_shape_and_symmetry(self, ccd, rotamer_library):
        from helico.bench import oracle_contact_state, structure_to_chains
        from helico.data import tokenize_sequences

        gt = parse_mmcif(_pdb_path("1AKE"))
        predicted = tokenize_sequences(structure_to_chains(gt), ccd)
        state = oracle_contact_state(gt, predicted, rotamer_library)
        assert state.shape == (predicted.n_tokens, predicted.n_tokens)
        assert torch.equal(state, state.T)

    def test_returns_none_for_protein_free_structure(self, ccd, rotamer_library):
        from helico.bench import oracle_contact_state, structure_to_chains
        from helico.data import tokenize_sequences

        gt = parse_mmcif(_pdb_path("1BNA"))
        predicted = tokenize_sequences(structure_to_chains(gt), ccd)
        assert oracle_contact_state(gt, predicted, rotamer_library) is None


@pytest.fixture(scope="module")
def contact_state(ccd, rotamer_library):
    return _tokenize("4HHB", ccd, rotamer_library).to_features()["contact_state"]


class TestConditioningSampler:

    @pytest.mark.parametrize(
        "mode", ["none", "full", "pair-subset", "contact-list"]
    )
    def test_output_is_symmetric(self, contact_state, mode):
        g = torch.Generator().manual_seed(0)
        out = sample_conditioning(contact_state, generator=g, mode=mode, reveal=0.5)
        assert torch.equal(out, out.T)

    def test_none_mode_reveals_nothing(self, contact_state):
        out = sample_conditioning(contact_state, mode="none")
        assert bool((out == CONTACT_UNKNOWN).all())

    def test_full_clean_is_identity(self, contact_state):
        out = sample_conditioning(
            contact_state, mode="full", reveal=1.0, eps_fp=0.0, eps_fn=0.0
        )
        assert torch.equal(out, contact_state)

    def test_clean_conditioning_invents_no_contacts(self, contact_state):
        """Without corruption, every asserted contact must be a real one."""
        for mode in ["full", "pair-subset", "contact-list"]:
            g = torch.Generator().manual_seed(3)
            out = sample_conditioning(
                contact_state, generator=g, mode=mode, reveal=0.5, eps_fp=0.0, eps_fn=0.0
            )
            invented = (out == CONTACT_PRESENT) & (contact_state != CONTACT_PRESENT)
            assert not bool(invented.any()), f"{mode} invented a contact"

    def test_reveal_fraction_is_monotone(self, contact_state):
        counts = []
        for reveal in [0.1, 0.25, 0.5, 1.0]:
            g = torch.Generator().manual_seed(11)
            out = sample_conditioning(
                contact_state, generator=g, mode="pair-subset",
                reveal=reveal, eps_fp=0.0, eps_fn=0.0,
            )
            counts.append(int((out != CONTACT_UNKNOWN).sum()))
        assert counts == sorted(counts), f"not monotone in reveal: {counts}"

    def test_false_negatives_reduce_recall(self, contact_state):
        n_true = int((contact_state == CONTACT_PRESENT).sum()) // 2
        g = torch.Generator().manual_seed(5)
        out = sample_conditioning(
            contact_state, generator=g, mode="full", reveal=1.0, eps_fp=0.0, eps_fn=0.3
        )
        kept = int(((out == CONTACT_PRESENT) & (contact_state == CONTACT_PRESENT)).sum()) // 2
        assert 0.5 < kept / n_true < 0.9, f"recall {kept / n_true:.2f} outside expected band"

    def test_false_positives_reduce_precision(self, contact_state):
        g = torch.Generator().manual_seed(5)
        out = sample_conditioning(
            contact_state, generator=g, mode="full", reveal=1.0, eps_fp=0.3, eps_fn=0.0
        )
        asserted = int((out == CONTACT_PRESENT).sum()) // 2
        true_positive = int(
            ((out == CONTACT_PRESENT) & (contact_state == CONTACT_PRESENT)).sum()
        ) // 2
        assert asserted > true_positive, "no false positives injected"
        assert 0.6 < true_positive / asserted < 0.95

    def test_corruption_keeps_symmetry(self, contact_state):
        g = torch.Generator().manual_seed(9)
        out = sample_conditioning(
            contact_state, generator=g, mode="full", reveal=1.0, eps_fp=0.3, eps_fn=0.3
        )
        assert torch.equal(out, out.T)

    def test_seeded_determinism(self, contact_state):
        a = sample_conditioning(contact_state, generator=torch.Generator().manual_seed(7))
        b = sample_conditioning(contact_state, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a, b)

    @pytest.mark.parametrize("eps_fp", [0.0, 0.3, 1.5])
    def test_never_asserts_outside_known_region(self, contact_state, eps_fp):
        """Conditioning may only ever weaken knowledge, never invent a knowable pair.

        Parametrised over eps_fp because the original version pinned it to 0 and
        so only ever exercised the clean path — while the bug it was meant to
        catch lived in the false-positive injector, which sampled the whole
        upper triangle. ~40% of injected contacts landed on non-protein tokens
        or at |i-j| < MIN_SEQ_SEPARATION, where no real predictor can emit one.
        """
        unknown = contact_state == CONTACT_UNKNOWN
        for mode in ["full", "pair-subset", "contact-list"]:
            g = torch.Generator().manual_seed(13)
            out = sample_conditioning(
                contact_state, generator=g, mode=mode, reveal=0.7,
                eps_fp=eps_fp, eps_fn=0.0,
            )
            leaked = out[unknown] != CONTACT_UNKNOWN
            assert not bool(leaked.any()), (
                f"{mode} asserted {int(leaked.sum())} pairs outside the eligible "
                f"region at eps_fp={eps_fp}"
            )

    def test_mode_mixture_includes_both_extremes(self, contact_state):
        """The sampled default must reach all-unknown and near-full conditioning."""
        small = contact_state[:80, :80].contiguous()
        g = torch.Generator().manual_seed(17)
        saw_none = saw_rich = False
        for _ in range(200):
            out = sample_conditioning(small, generator=g)
            known = int((out != CONTACT_UNKNOWN).sum())
            if known == 0:
                saw_none = True
            elif known > int((small != CONTACT_UNKNOWN).sum()) * 0.9:
                saw_rich = True
        assert saw_none, "never sampled the ab initio level"
        assert saw_rich, "never sampled a near-fully-specified level"


class TestSingleSequenceMode:
    """bench.single_sequence_msa: depth-1 MSA whose one row is the query.

    Three different things get called "no MSA" in this codebase and they are
    not interchangeable:

      1. ``single_sequence_msa`` — depth-1 MSA, row 0 = query. The MSA module
         runs. This is the fair no-alignments baseline.
      2. ``empty_msa`` — depth-1 MSA of *gaps*. A sequence of nothing.
      3. ``HelicoConfig.use_msa=False`` — the MSA module never runs. A lesion.

    Reporting (3) as a single-sequence baseline understated Protenix badly and
    made a fine-tuned model look like it beat a baseline it was never compared
    against. These tests pin the distinction.
    """

    def _restype(self, pdb_id, ccd):
        from helico.bench import structure_to_chains
        from helico.data import tokenize_sequences

        gt = parse_mmcif(_pdb_path(pdb_id))
        tokenized = tokenize_sequences(structure_to_chains(gt), ccd)
        return tokenized.to_features()["restype"].unsqueeze(0)

    def test_query_row_is_the_query(self, ccd):
        """The single MSA row must BE the query sequence, not gaps."""
        from helico.bench import single_sequence_msa
        from helico.data import AF3_MSA_GAP

        restype = self._restype("1UBQ", ccd)
        feats = single_sequence_msa(restype)

        assert feats["msa"].shape == (1, 1, restype.shape[1]), "depth must be 1"
        assert torch.equal(feats["msa"][0, 0], restype[0])
        # A protein chain must not come out all-gap — that is failure mode (2).
        assert not (feats["msa"][0, 0] == AF3_MSA_GAP).all()
        assert float(feats["has_msa"][0]) == 1.0

    def test_profile_is_query_one_hot(self, ccd):
        from helico.bench import single_sequence_msa

        restype = self._restype("1UBQ", ccd)
        prof = single_sequence_msa(restype)["msa_profile"][0]
        assert torch.allclose(prof.sum(-1), torch.ones(prof.shape[0]))
        assert torch.equal(prof.argmax(-1), restype[0])

    def test_differs_from_empty_msa(self, ccd):
        """The two depth-1 paths must not produce the same features."""
        from helico.bench import empty_msa, single_sequence_msa
        from helico.data import AF3_MSA_GAP

        restype = self._restype("1UBQ", ccd)
        single = single_sequence_msa(restype)
        empty = empty_msa(restype.shape[1])

        assert (empty["msa"][0, 0] == AF3_MSA_GAP).all()
        assert not torch.equal(single["msa"], empty["msa"])
        assert float(empty["has_msa"][0]) == 0.0
        assert not torch.equal(single["msa_profile"], empty["msa_profile"])

    def test_deletion_features_are_zero(self, ccd):
        """No homologs means no insertions relative to the query."""
        from helico.bench import single_sequence_msa

        feats = single_sequence_msa(self._restype("1UBQ", ccd))
        assert float(feats["deletion_matrix"].abs().max()) == 0.0
        assert float(feats["deletion_mean"].abs().max()) == 0.0

    def test_rejects_unbatched_restype(self):
        """Shape confusion here would silently mis-scatter the query row."""
        from helico.bench import single_sequence_msa

        with pytest.raises(ValueError, match="shape"):
            single_sequence_msa(torch.tensor([0, 5, 12]))


class TestPredictorOperatingPoint:
    """Conditioning must be able to reproduce MarinFold's actual output.

    MarinFold emits a truncated top-k contact list at roughly 60% precision and
    60% recall (2026-08). Two things follow, and both were previously wrong:

    - An unlisted pair means "did not make the cut", not "not a contact", so
      the sampler must never assert ABSENT in this mode.
    - 60% precision needs eps_fp = (1-p)/p = 0.667, far outside the old
      U(0, 0.3) sampling range — the model was never trained near the
      operating point it has to work at.
    """

    def _measure(self, contact_state, **kwargs):
        n_true = int((contact_state == CONTACT_PRESENT).sum()) // 2
        precisions, recalls = [], []
        for seed in range(60):
            g = torch.Generator().manual_seed(4000 + seed)
            out = sample_conditioning(contact_state, generator=g, **kwargs)
            asserted = out == CONTACT_PRESENT
            n_asserted = int(asserted.sum()) // 2
            if not n_asserted:
                continue
            tp = int((asserted & (contact_state == CONTACT_PRESENT)).sum()) // 2
            precisions.append(tp / n_asserted)
            recalls.append(tp / n_true)
        return sum(precisions) / len(precisions), sum(recalls) / len(recalls)

    @pytest.mark.parametrize("precision,recall", [(0.6, 0.6), (0.8, 0.5), (1.0, 1.0)])
    def test_realises_the_requested_operating_point(self, contact_state, precision, recall):
        got_p, got_r = self._measure(contact_state, mode="contact-list",
                                     precision=precision, recall=recall)
        assert abs(got_p - precision) < 0.06, f"precision {got_p:.3f} != {precision}"
        assert abs(got_r - recall) < 0.06, f"recall {got_r:.3f} != {recall}"

    def test_top_k_list_never_asserts_absence(self, contact_state):
        """Unlisted pairs are unknown, not absent — the list is truncated."""
        for seed in range(30):
            g = torch.Generator().manual_seed(seed)
            out = sample_conditioning(contact_state, generator=g, mode="contact-list")
            assert not bool((out == CONTACT_ABSENT).any()), (
                "contact-list asserted an absence; a truncated top-k list "
                "cannot distinguish 'not a contact' from 'did not fit'"
            )

    def test_sampled_precision_covers_marinfold(self, contact_state):
        """The default sampling range must include the real operating point."""
        from helico.contacts import MARINFOLD_PRECISION

        seen_at_or_below = False
        for seed in range(200):
            g = torch.Generator().manual_seed(9000 + seed)
            out = sample_conditioning(contact_state, generator=g, mode="contact-list")
            asserted = out == CONTACT_PRESENT
            n_asserted = int(asserted.sum()) // 2
            if not n_asserted:
                continue
            tp = int((asserted & (contact_state == CONTACT_PRESENT)).sum()) // 2
            if tp / n_asserted <= MARINFOLD_PRECISION + 0.02:
                seen_at_or_below = True
                break
        assert seen_at_or_below, (
            f"never sampled a conditioning as noisy as precision "
            f"{MARINFOLD_PRECISION}; the model would not be trained for it"
        )

    def test_precision_recall_roundtrip(self):
        from helico.contacts import conditioning_from_precision_recall

        reveal, eps_fp, eps_fn = conditioning_from_precision_recall(0.6, 0.6)
        assert reveal == pytest.approx(0.6)
        assert eps_fp == pytest.approx(2 / 3, abs=1e-6)
        assert eps_fn == 0.0
        with pytest.raises(ValueError, match="precision"):
            conditioning_from_precision_recall(0.0, 0.5)
        with pytest.raises(ValueError, match="recall"):
            conditioning_from_precision_recall(0.5, 1.5)


class TestValidationLevels:
    """Every pinned validation level must be a valid conditioning spec.

    The levels are splatted straight into sample_conditioning as kwargs, so a
    typo'd key is a TypeError thousands of steps into a run rather than at
    import. This catches it at test time.
    """

    def test_all_levels_are_accepted(self, contact_state):
        from helico.train import VAL_CONTACT_LEVELS

        for name, spec in VAL_CONTACT_LEVELS.items():
            g = torch.Generator().manual_seed(3)
            out = sample_conditioning(contact_state, generator=g, **spec)
            assert out.shape == contact_state.shape, name
            assert torch.equal(out, out.T), f"{name} broke symmetry"

    def test_marinfold_level_matches_the_constants(self, contact_state):
        from helico.contacts import MARINFOLD_PRECISION, MARINFOLD_RECALL
        from helico.train import VAL_CONTACT_LEVELS

        spec = VAL_CONTACT_LEVELS["@contactsMarinFold"]
        assert spec["precision"] == MARINFOLD_PRECISION
        assert spec["recall"] == MARINFOLD_RECALL

        n_true = int((contact_state == CONTACT_PRESENT).sum()) // 2
        ps, rs = [], []
        for seed in range(40):
            g = torch.Generator().manual_seed(seed)
            out = sample_conditioning(contact_state, generator=g, **spec)
            asserted = out == CONTACT_PRESENT
            n = int(asserted.sum()) // 2
            if not n:
                continue
            tp = int((asserted & (contact_state == CONTACT_PRESENT)).sum()) // 2
            ps.append(tp / n)
            rs.append(tp / n_true)
        assert abs(sum(ps) / len(ps) - MARINFOLD_PRECISION) < 0.06
        assert abs(sum(rs) / len(rs) - MARINFOLD_RECALL) < 0.06


class TestNoMsaSkipsMsaLoading:
    """--no-msa must not even load alignment data.

    The model gates every MSA-derived feature (see TestNoMSALeak), but a run
    that still loads MSAs is one weakened gate away from leaking again -- which
    is exactly how the profile leak survived: `use_msa=False` was assumed to
    mean "no MSAs anywhere" while the loader ignored it entirely.
    """

    def test_arg_gates_the_loader(self):
        """The MSA index loop must be inside the `else` of the no_msa branch."""
        import inspect

        from helico import train as train_mod

        src = inspect.getsource(train_mod)
        head, _, tail = src.partition("if args.no_msa:")
        assert tail, "expected a --no-msa branch guarding MSA index loading"
        # The loader must appear after the branch opens, not before it.
        assert "Loading MSA tar index" not in head, (
            "MSA indices are loaded before the --no-msa check, so an MSA-free "
            "run still reads alignment data"
        )
        guarded, _, rest = tail.partition("\n        # Create training dataset")
        assert "Loading MSA tar index" in guarded, (
            "MSA index loading moved out of the branch it should sit in"
        )


class TestBenchContactDegradation:
    """bench can score a predictor operating point, not just perfect contacts.

    Without this the benchmark can only measure the ceiling (oracle contacts),
    which is not the deployment condition. The corruption must also be *paired*
    across checkpoints: a given target has to receive the same degraded map in
    every arm, or a checkpoint-to-checkpoint comparison is unpaired and the
    differences are swamped by resampling noise.
    """

    def _state(self, ccd, rotamer_library):
        return _tokenize("4HHB", ccd, rotamer_library).to_features()["contact_state"]

    def test_degradation_is_deterministic_per_target(self, ccd, rotamer_library):
        """Same target name -> identical corruption, so arms stay paired."""
        import hashlib

        state = self._state(ccd, rotamer_library)

        def degrade(name):
            gen = torch.Generator().manual_seed(
                int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            )
            return sample_conditioning(state, generator=gen, mode="contact-list",
                                       precision=0.6, recall=0.6)

        assert torch.equal(degrade("8ycm"), degrade("8ycm"))
        assert not torch.equal(degrade("8ycm"), degrade("7z65")), (
            "different targets should not share a corruption draw"
        )

    def test_degraded_map_hits_the_operating_point(self, ccd, rotamer_library):
        import hashlib

        state = self._state(ccd, rotamer_library)
        n_true = int((state == CONTACT_PRESENT).sum()) // 2
        ps, rs = [], []
        for name in [f"t{i}" for i in range(40)]:
            gen = torch.Generator().manual_seed(
                int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            )
            out = sample_conditioning(state, generator=gen, mode="contact-list",
                                      precision=0.6, recall=0.6)
            asserted = out == CONTACT_PRESENT
            n = int(asserted.sum()) // 2
            if not n:
                continue
            tp = int((asserted & (state == CONTACT_PRESENT)).sum()) // 2
            ps.append(tp / n)
            rs.append(tp / n_true)
        assert abs(sum(ps) / len(ps) - 0.6) < 0.06, f"precision {sum(ps)/len(ps):.3f}"
        assert abs(sum(rs) / len(rs) - 0.6) < 0.06, f"recall {sum(rs)/len(rs):.3f}"

    def test_predict_target_accepts_the_knobs(self):
        """Signature guard: the plumbing must reach predict_target."""
        import inspect

        from helico.bench import predict_target, run_benchmark

        for fn in (predict_target, run_benchmark):
            params = inspect.signature(fn).parameters
            assert "contact_precision" in params, fn.__name__
            assert "contact_recall" in params, fn.__name__
