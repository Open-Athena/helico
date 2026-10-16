"""Rigid coordinate augmentation must remain rigid under mixed precision."""
import pytest
import torch

from helico.model.diffusion import _centre_random_augmentation


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mixed_precision_augmentation_preserves_bond_distances(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(2301)
    coords = torch.randn(2, 128, 3, generator=generator) * 30
    coords[:, 1::2] = coords[:, ::2] + torch.tensor([1.5, 0., 0.])
    coords = coords.to(device)
    before = torch.linalg.vector_norm(coords[:, ::2] - coords[:, 1::2], dim=-1)
    mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=device)
    torch.manual_seed(2302)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        transformed = _centre_random_augmentation(coords, mask)
    after = torch.linalg.vector_norm(transformed[:, ::2] - transformed[:, 1::2], dim=-1)
    assert transformed.dtype == torch.float32
    torch.testing.assert_close(after, before, atol=5e-5, rtol=0)
    torch.manual_seed(2302)
    reference = _centre_random_augmentation(coords, mask)
    torch.testing.assert_close(transformed, reference, atol=0, rtol=0)
