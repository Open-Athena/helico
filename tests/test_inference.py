"""Tests for the contact-conditioned inference API (helico.inference).

The CPU-only tests cover contact-map construction, which is where callers are
most likely to get it wrong: residue positions vs token indices, 0- vs
1-indexing, and the top-k list semantics that say an unlisted pair is *unknown*
rather than *absent*.

The GPU tests fold real sequences. They need cuEquivariance kernels, so they are
skipped without CUDA and normally run on Modal (`modal run modal/ci.py`).
"""

import pytest
import torch

from helico.contacts import MIN_SEQ_SEPARATION
from helico.data import CONTACT_ABSENT, CONTACT_PRESENT, CONTACT_UNKNOWN
from helico.inference import contacts_from_pairs

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SEP = MIN_SEQ_SEPARATION


class TestContactsFromPairs:
    def test_listed_pairs_are_present_and_symmetric(self):
        state = contacts_from_pairs([(0, 20), (5, 40)], seq_len=60)
        assert state.shape == (60, 60)
        assert torch.equal(state, state.T)
        for i, j in ((0, 20), (5, 40)):
            assert state[i, j] == CONTACT_PRESENT
            assert state[j, i] == CONTACT_PRESENT

    def test_unlisted_pairs_are_unknown_not_absent(self):
        """A truncated top-n list cannot assert a non-contact."""
        state = contacts_from_pairs([(0, 20)], seq_len=60)
        assert not (state == CONTACT_ABSENT).any(), (
            "unlisted pairs must stay UNKNOWN: the predictor never claimed they "
            "are non-contacts, it just did not rank them"
        )
        assert int((state == CONTACT_PRESENT).sum()) == 2  # one pair, both triangles
        assert int((state == CONTACT_UNKNOWN).sum()) == 60 * 60 - 2

    def test_short_range_pairs_are_dropped(self):
        """pyconfind filters |i-j| < MIN_SEQ_SEPARATION, so the model never saw one."""
        state = contacts_from_pairs([(10, 10 + SEP - 1), (10, 10 + SEP)], seq_len=60)
        assert state[10, 10 + SEP - 1] == CONTACT_UNKNOWN
        assert state[10, 10 + SEP] == CONTACT_PRESENT

    def test_strict_raises_on_short_range(self):
        with pytest.raises(ValueError, match="MIN_SEQ_SEPARATION"):
            contacts_from_pairs([(10, 11)], seq_len=60, strict=True)

    def test_strict_raises_out_of_range(self):
        with pytest.raises(ValueError, match="outside the input"):
            contacts_from_pairs([(0, 999)], seq_len=60, strict=True)

    def test_out_of_range_dropped_when_not_strict(self):
        state = contacts_from_pairs([(0, 999), (0, 20)], seq_len=60)
        assert int((state == CONTACT_PRESENT).sum()) == 2

    def test_one_indexed_offset(self):
        zero = contacts_from_pairs([(0, 20)], seq_len=60)
        one = contacts_from_pairs([(1, 21)], seq_len=60, one_indexed=True)
        assert torch.equal(zero, one)

    def test_rejects_malformed_pair(self):
        with pytest.raises(ValueError, match="pair must be"):
            contacts_from_pairs([(1, 2, 3)], seq_len=60)


class TestMultiChainMapping:
    """Residue positions are per chain; token indices are global."""

    @pytest.fixture(scope="class")
    def tokenized(self, ccd):
        from helico.data import tokenize_sequences

        return tokenize_sequences(
            [{"type": "protein", "id": "A", "sequence": "M" * 30},
             {"type": "protein", "id": "B", "sequence": "A" * 25}],
            ccd,
        )

    def test_second_chain_positions_are_offset(self, tokenized):
        """B:0 must not land on token 0 -- that is chain A's first residue."""
        state = contacts_from_pairs([("B", 0, "B", 20)], tokenized=tokenized)
        assert state[0, 20] == CONTACT_UNKNOWN, "B:0 was mapped onto chain A"
        assert state[30, 50] == CONTACT_PRESENT

    def test_interchain_pair(self, tokenized):
        """Inter-chain contacts have no sequence-separation constraint."""
        state = contacts_from_pairs([("A", 3, "B", 1)], tokenized=tokenized)
        assert state[3, 31] == CONTACT_PRESENT
        assert torch.equal(state, state.T)

    def test_matches_tokenized_size(self, tokenized):
        state = contacts_from_pairs([("A", 0, "A", 20)], tokenized=tokenized)
        assert state.shape == (len(tokenized.tokens), len(tokenized.tokens))


@cuda_only
class TestFold:
    """End-to-end folding. Oracle contacts must beat no contacts."""

    @pytest.fixture(scope="class")
    def model(self):
        from helico.inference import load_model

        return load_model()

    @pytest.fixture(scope="class")
    def ubq(self, ccd, pdb_path):
        from helico.bench import structure_to_chains
        from helico.data import parse_mmcif, tokenize_sequences

        gt = parse_mmcif(pdb_path("1UBQ"))
        chains = structure_to_chains(gt)
        return gt, chains, tokenize_sequences(chains, ccd)

    def test_fold_with_oracle_contacts_beats_no_contacts(self, model, ubq, ccd):
        from helico.inference import contacts_from_structure, fold

        gt, chains, tok = ubq
        seq = chains[0]["sequence"]
        oracle = contacts_from_structure(tok, gt)

        with_c = fold({"A": seq}, contacts=oracle, model=model, n_samples=2, ccd=ccd, seed=0)
        without = fold({"A": seq}, contacts=None, model=model, n_samples=2, ccd=ccd, seed=0)

        assert with_c.coords.shape == without.coords.shape
        assert with_c.mean_plddt > without.mean_plddt, (
            f"contacts did not help: {with_c.mean_plddt:.1f} vs {without.mean_plddt:.1f}"
        )

    def test_pdb_output_is_wellformed(self, model, ubq, ccd):
        from helico.inference import contacts_from_structure, fold

        gt, chains, tok = ubq
        res = fold({"A": chains[0]["sequence"]},
                   contacts=contacts_from_structure(tok, gt),
                   model=model, n_samples=1, ccd=ccd, seed=0)
        pdb = res.pdb
        atom_lines = [ln for ln in pdb.splitlines() if ln.startswith("ATOM")]
        assert len(atom_lines) == res.coords.shape[0]
        assert 0.0 <= res.mean_plddt <= 100.0

    def test_contacts_shape_is_validated(self, model, ubq, ccd):
        """A mismatched matrix is a silent disaster; it must raise."""
        from helico.inference import fold

        _gt, chains, _tok = ubq
        with pytest.raises(ValueError, match="tokenizes to"):
            fold({"A": chains[0]["sequence"]},
                 contacts=torch.zeros(5, 5, dtype=torch.uint8),
                 model=model, n_samples=1, ccd=ccd)
