"""The MarinFold->Helico index map must be exact, or the experiment is nonsense.

MarinFold indexes residues into the published prompt (full SEQRES); Helico's
bench derives its sequence from the *resolved* residues of the ground-truth
structure. For 83 of 100 FoldBench monomers those differ, so a naive
identity map shifts contacts on most targets -- and a shifted map looks
exactly like "real predicted contacts do not help", which is the conclusion
this experiment exists to test.

The check: take exp211's ground-truth contacts, push them through the map, and
compare against what helico computes itself with oracle_contact_state. They must
agree almost exactly. Requires the exp211 artifacts, so it skips without them.
"""

import json
from pathlib import Path

import pytest
import torch

EXP211 = Path("/home/bizon/git/MarinFold/.claude/worktrees/contact-consistency-exp199-9d394e"
              "/experiments/exp211_evals_contact_set_3d_self_consistency")
ARMS = Path(__file__).parent.parent / "experiments/marinfold_contacts/arms"

pytestmark = pytest.mark.skipif(
    not (ARMS / "gt_from_exp211.json").exists(),
    reason="exp211 export not present (run experiments/marinfold_contacts/export_contacts.py)",
)


@pytest.fixture(scope="module")
def exported_gt():
    return json.loads((ARMS / "gt_from_exp211.json").read_text())


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 1.0


class TestIndexMap:
    def test_roundtrip_reproduces_oracle_contacts(self, exported_gt, ccd, rotamer_library):
        """Mapped MarinFold ground truth == helico's own oracle contact set."""
        from helico.bench import (
            _find_gt_path, oracle_contact_state, structure_to_chains,
        )
        from helico.data import CONTACT_PRESENT, parse_mmcif, tokenize_sequences

        gt_dir = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths"
        if not gt_dir.exists():
            pytest.skip("FoldBench ground truths not downloaded")

        scores, checked = [], 0
        for pdb_id, pairs in sorted(exported_gt.items()):
            if checked >= 8:
                break
            try:
                st = parse_mmcif(_find_gt_path(gt_dir, pdb_id), max_resolution=float("inf"))
            except FileNotFoundError:
                continue
            if st is None:
                continue
            tok = tokenize_sequences(structure_to_chains(st), ccd)
            mine = oracle_contact_state(st, tok, rotamer_library)
            if mine is None:      # helico declined this target; nothing to compare
                continue
            mine_set = {(int(a), int(b))
                        for a, b in torch.triu(mine == CONTACT_PRESENT, 1).nonzero()}
            scores.append(_jaccard(mine_set, {(a, b) for a, b in pairs}))
            checked += 1

        assert checked >= 3, f"only {checked} targets comparable"
        assert min(scores) > 0.95, (
            f"index map is off: min Jaccard {min(scores):.3f} across {checked} targets. "
            f"A shifted map presents as 'real contacts do not help'."
        )

    def test_arms_are_within_token_range(self, exported_gt):
        """Every exported pair must index a real token."""
        import csv

        sizes = {r["pdb_id"]: int(r["L"])
                 for r in csv.DictReader((ARMS / "targets.csv").open())}
        for arm in sorted(ARMS.glob("*_L*.json")):
            data = json.loads(arm.read_text())
            for pdb_id, pairs in data.items():
                L = sizes[pdb_id]
                for i, j in pairs:
                    assert 0 <= i < L and 0 <= j < L, f"{arm.name} {pdb_id}: ({i},{j}) vs L={L}"
                    assert i < j, f"{arm.name} {pdb_id}: pair not ordered"

    def test_top_n_arms_are_nested(self, exported_gt):
        """L/5 must be a prefix of L/2, which must be a prefix of L."""
        for recipe in ("rollout", "single"):
            a5, a2, aL = (json.loads((ARMS / f"{recipe}_{s}.json").read_text())
                          for s in ("L5", "L2", "L"))
            for pdb_id in a5:
                s5, s2, sL = ({tuple(p) for p in a[pdb_id]} for a in (a5, a2, aL))
                assert s5 <= s2 <= sL, f"{recipe} {pdb_id}: top-n sets not nested"
