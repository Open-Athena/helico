"""Padding must neither affect real denoising outputs nor carry loss gradients."""
import pytest
import torch

from helico.model.diffusion import DiffusionAttentionPairBias
from helico.model.triangle import TriangleMultiplicativeUpdate


def test_diffusion_attention_ignores_padding_in_forward_and_backward():
    torch.manual_seed(7)
    model = DiffusionAttentionPairBias(16, 16, 8, 4, 4)
    a = torch.randn(1, 5, 16, requires_grad=True)
    s = torch.randn(1, 5, 16, requires_grad=True)
    z = torch.randn(1, 5, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]])
    actual = model(a, s, z, pad_mask=mask)[:, :3]
    expected = model(a[:, :3], s[:, :3], z[:, :3, :3])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    for value in (a, s, z):
        assert value.grad.isfinite().all()
        assert not value.grad[:, 3:].any()
    assert not z.grad[:, :, 3:].any()


def test_diffusion_attention_empty_mask_has_finite_zero_gradient():
    model = DiffusionAttentionPairBias(16, 16, 8, 4, 4)
    a = torch.randn(1, 5, 16, requires_grad=True)
    s = torch.randn_like(a)
    z = torch.randn(1, 5, 5, 8)
    out = model(a, s, z, pad_mask=torch.zeros(1, 5, dtype=torch.bool))
    assert not out.any()
    out.sum().backward()
    assert a.grad.isfinite().all() and not a.grad.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused triangle update needs CUDA")
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_triangle_update_masks_padded_output_and_gradients(direction):
    model = TriangleMultiplicativeUpdate(32, direction).cuda()
    with torch.no_grad():
        model.layer_norm_out.bias.fill_(1.)
    x = torch.randn(1, 32, 32, 32, device="cuda", requires_grad=True)
    valid = torch.arange(32, device="cuda") < 21
    mask = (valid[:, None] & valid[None, :])[None]
    out = model(x, mask)
    assert not out[~mask].any()
    out[~mask].sum().backward()
    assert x.grad.isfinite().all() and not x.grad.any()
