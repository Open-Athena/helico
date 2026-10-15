import pytest
import torch
from torch.nn import functional as F

from helico.model.template import _TemplateTriAtt, _TemplateTriMul
from helico.model.diffusion import DiffusionAttentionPairBias, _partition_to_windows


@pytest.mark.parametrize("mode", ["starting", "ending"])
@pytest.mark.parametrize("masked", [False, True])
def test_chunked_template_attention_preserves_outputs_and_gradients(mode, masked):
    torch.manual_seed(23)
    module = _TemplateTriAtt(16, 2, 8, mode).double()
    full = torch.randn(2, 7, 7, 16, dtype=torch.float64, requires_grad=True)
    chunked = full.detach().clone().requires_grad_()
    mask = None
    if masked:
        token_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
        mask = token_mask[:, :, None] * token_mask[:, None, :]
    expected = module(full, mask)
    actual = module(chunked, mask, anchor_chunk_size=3)
    torch.testing.assert_close(actual, expected)
    variables = tuple(module.parameters())
    expected_grad = torch.autograd.grad(expected.square().sum(), (full, *variables))
    actual_grad = torch.autograd.grad(actual.square().sum(), (chunked, *variables))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("mode", ["starting", "ending"])
def test_padded_template_attention_preserves_real_pairs_and_finite_backward(mode):
    torch.manual_seed(7)
    module = _TemplateTriAtt(16, 2, 8, mode)
    z = torch.randn(1, 3, 3, 16)
    reference = module(z, torch.ones(1, 3, 3))
    padded = F.pad(z, (0, 0, 0, 1, 0, 1)).requires_grad_()
    mask = F.pad(torch.ones(1, 3, 3), (0, 1, 0, 1))
    result = module(padded, mask)
    torch.testing.assert_close(result[:, :3, :3], reference)
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    assert torch.isfinite(padded.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_template_multiplication_ignores_padded_pairs(direction):
    torch.manual_seed(7)
    module = _TemplateTriMul(16, 32, direction)
    # A nonzero learned normalization bias exposes the mask bug.
    with torch.no_grad(): module.layer_norm_in.bias.fill_(.5)
    z = torch.randn(1, 3, 3, 16)
    reference = module(z, torch.ones(1, 3, 3))
    padded = F.pad(z, (0, 0, 0, 1, 0, 1))
    mask = F.pad(torch.ones(1, 3, 3), (0, 1, 0, 1))
    torch.testing.assert_close(module(padded, mask)[:, :3, :3], reference)


def test_partial_atom_attention_window_has_finite_parameter_gradients():
    torch.manual_seed(7)
    module = DiffusionAttentionPairBias(16, 16, 8, n_heads=2, head_dim=8)
    a = torch.randn(1, 5, 16, requires_grad=True)
    s = torch.randn_like(a)
    _, _, mask, blocks, _ = _partition_to_windows(a, 4, 8)
    z = torch.randn(1, blocks, 4, 8, 8)
    out = module(a, s, z, n_queries=4, n_keys=8, pad_mask=mask)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert torch.isfinite(a.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
