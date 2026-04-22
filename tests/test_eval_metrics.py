"""Tests for the GPU-batched eval metrics in helico.eval_metrics."""

import numpy as np
import pytest
import torch

from helico.eval_metrics import (
    gdt_ts,
    hard_lddt,
    kabsch_align,
    mean_plddt,
    rmsd_after_kabsch,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_coords(B: int, N: int, scale: float = 10.0) -> torch.Tensor:
    return torch.from_numpy(_rng().normal(size=(B, N, 3)) * scale).float()


# ---------------------------------------------------------------------------
# hard_lddt
# ---------------------------------------------------------------------------

class TestHardLDDT:
    def test_identity_is_one(self):
        coords = _random_coords(2, 30)
        score = hard_lddt(coords, coords)
        assert torch.allclose(score, torch.ones(2), atol=1e-6)

    def test_translation_invariant(self):
        coords = _random_coords(1, 50)
        translated = coords + torch.tensor([5.0, -3.0, 2.0])
        # LDDT only looks at pairwise distances → translation doesn't affect it.
        score = hard_lddt(translated, coords)
        assert torch.allclose(score, torch.ones(1), atol=1e-6)

    def test_rotation_invariant(self):
        coords = _random_coords(1, 50)
        # Random rotation
        angle = torch.tensor(0.7)
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotated = coords @ R.T
        score = hard_lddt(rotated, coords)
        assert torch.allclose(score, torch.ones(1), atol=1e-6)

    def test_perturbation_lowers_score(self):
        coords = _random_coords(1, 60)
        noisy = coords + torch.randn_like(coords) * 1.0  # large noise
        score = hard_lddt(noisy, coords)
        assert score.item() < 0.95
        # And gets worse with more noise
        worse = coords + torch.randn_like(coords) * 5.0
        worse_score = hard_lddt(worse, coords)
        assert worse_score.item() < score.item()

    def test_mask_is_respected(self):
        # Build coords where the first half perfectly matches and the second
        # half is garbage. Masking out the second half should give LDDT=1.
        gt = _random_coords(1, 40)
        pred = gt.clone()
        pred[:, 20:] = pred[:, 20:] + torch.randn_like(pred[:, 20:]) * 5.0
        full_mask = torch.ones(1, 40)
        partial_mask = torch.tensor([[1.0] * 20 + [0.0] * 20])

        full_score = hard_lddt(pred, gt, full_mask)
        partial_score = hard_lddt(pred, gt, partial_mask)
        assert partial_score.item() == pytest.approx(1.0, abs=1e-6)
        assert partial_score.item() > full_score.item()

    def test_batch_independence(self):
        # Two batches: identity + heavily perturbed. Scores should differ.
        gt = _random_coords(2, 30)
        pred = gt.clone()
        pred[1] = pred[1] + torch.randn_like(pred[1]) * 5.0
        scores = hard_lddt(pred, gt)
        assert scores.shape == (2,)
        assert torch.allclose(scores[0], torch.tensor(1.0), atol=1e-6)
        assert scores[1].item() < 0.5

    def test_matches_numpy_reference(self):
        # Cross-check vs the offline numpy implementation in bench.py.
        from helico.bench import compute_lddt
        rng = _rng(7)
        gt = torch.from_numpy(rng.normal(size=(35, 3)) * 6.0).float()
        pred = gt + torch.from_numpy(rng.normal(size=(35, 3)) * 0.5).float()
        np_score = compute_lddt(pred.numpy(), gt.numpy())
        torch_score = hard_lddt(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        assert torch_score == pytest.approx(np_score, abs=1e-5)


# ---------------------------------------------------------------------------
# kabsch_align / rmsd / gdt_ts
# ---------------------------------------------------------------------------

class TestKabsch:
    def test_aligns_translated(self):
        coords = _random_coords(1, 25)
        translated = coords + torch.tensor([10.0, -5.0, 3.0])
        aligned = kabsch_align(translated, coords)
        assert torch.allclose(aligned, coords, atol=1e-4)

    def test_aligns_rotated(self):
        coords = _random_coords(1, 25)
        # 30 deg rotation around z
        angle = torch.tensor(0.523599)
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotated = coords @ R.T
        aligned = kabsch_align(rotated, coords)
        assert torch.allclose(aligned, coords, atol=1e-4)

    def test_handles_reflection(self):
        # If we mirror the coords, Kabsch should still find the closest
        # right-handed rotation rather than returning a reflection.
        coords = _random_coords(1, 30)
        mirrored = coords * torch.tensor([1.0, 1.0, -1.0])
        aligned = kabsch_align(mirrored, coords)
        # We don't expect a perfect match (reflection isn't a rotation), but
        # the determinant of the recovered transform should be +1, i.e. the
        # alignment should not return the trivial -z flip.
        rmsd = rmsd_after_kabsch(mirrored, coords)
        assert rmsd.item() > 0.0  # Mirror cannot be perfectly aligned by rotation


class TestRMSDAfterKabsch:
    def test_identity_zero(self):
        coords = _random_coords(2, 30)
        rmsd = rmsd_after_kabsch(coords, coords)
        assert torch.allclose(rmsd, torch.zeros(2), atol=1e-5)

    def test_translation_zero(self):
        coords = _random_coords(1, 25)
        translated = coords + torch.tensor([7.0, 2.0, -1.0])
        rmsd = rmsd_after_kabsch(translated, coords)
        assert rmsd.item() == pytest.approx(0.0, abs=1e-4)

    def test_known_displacement(self):
        # Each atom shifted by exactly d in x. After Kabsch this becomes 0
        # because the centroid translation is removed.
        coords = _random_coords(1, 20)
        shifted = coords + torch.tensor([3.0, 0.0, 0.0])
        rmsd = rmsd_after_kabsch(shifted, coords)
        assert rmsd.item() == pytest.approx(0.0, abs=1e-4)

    def test_matches_numpy_reference(self):
        # Compare to a direct numpy reference implementation. Note: we
        # deliberately do NOT compare to helico.bench.compute_rmsd — that
        # function divides by N*3 instead of N (per-coord vs per-atom RMSD,
        # off by sqrt(3)). Should be fixed there separately.
        from helico.bench import _kabsch_superpose
        rng = _rng(11)
        gt = torch.from_numpy(rng.normal(size=(40, 3)) * 8.0).float()
        pred = gt + torch.from_numpy(rng.normal(size=(40, 3)) * 1.5).float()
        aligned_np, _ = _kabsch_superpose(pred.numpy(), gt.numpy())
        # Standard RMSD = sqrt(mean over atoms of squared atom distance)
        sq_dist = ((aligned_np - gt.numpy()) ** 2).sum(axis=-1)
        np_rmsd = float(np.sqrt(sq_dist.mean()))
        torch_rmsd = rmsd_after_kabsch(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        assert torch_rmsd == pytest.approx(np_rmsd, abs=1e-4)


class TestGDT:
    def test_identity_one(self):
        coords = _random_coords(2, 25)
        score = gdt_ts(coords, coords)
        assert torch.allclose(score, torch.ones(2), atol=1e-6)

    def test_far_offset_zero(self):
        # All atoms 100Å away → all thresholds (max 8Å) miss.
        coords = _random_coords(1, 20)
        far = coords + torch.tensor([100.0, 0.0, 0.0])
        # Kabsch will translate, so 100Å translation yields RMSD≈0; instead
        # construct a "spread" perturbation that Kabsch cannot remove.
        rng = _rng(2)
        far = coords + torch.from_numpy(rng.uniform(50, 100, size=(1, 20, 3))).float()
        score = gdt_ts(far, coords)
        assert score.item() < 0.05

    def test_in_range(self):
        # Small noise → most atoms within 4Å → score should be high.
        coords = _random_coords(1, 25)
        pred = coords + torch.randn_like(coords) * 0.5
        score = gdt_ts(pred, coords)
        assert 0.5 < score.item() <= 1.0

    def test_matches_numpy_reference(self):
        from helico.bench import compute_gdt_ts
        rng = _rng(17)
        gt = torch.from_numpy(rng.normal(size=(50, 3)) * 7.0).float()
        pred = gt + torch.from_numpy(rng.normal(size=(50, 3)) * 1.0).float()
        np_gdt = compute_gdt_ts(pred.numpy(), gt.numpy())
        torch_gdt = gdt_ts(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        assert torch_gdt == pytest.approx(np_gdt, abs=1e-4)


# ---------------------------------------------------------------------------
# mean_plddt
# ---------------------------------------------------------------------------

class TestMeanPLDDT:
    def test_basic_mean(self):
        plddt = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        m = mean_plddt(plddt)
        assert m.item() == pytest.approx(25.0)

    def test_mask_excludes_atoms(self):
        plddt = torch.tensor([[10.0, 90.0, 90.0, 90.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        m = mean_plddt(plddt, mask)
        assert m.item() == pytest.approx(50.0)

    def test_batched(self):
        plddt = torch.tensor([[50.0, 50.0], [80.0, 80.0]])
        m = mean_plddt(plddt)
        assert m.shape == (2,)
        assert torch.allclose(m, torch.tensor([50.0, 80.0]))
