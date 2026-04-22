"""Reproduces the cuDNN flash-attn shape rejection observed during val sweeps.

On the Modal training image, calling `model(batch)` in `eval()` + `no_grad()`
mode at n_tokens >= 128 crashes with:

    RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error:
        No valid execution plans built.

The same shapes pass in `train()` + grad-enabled mode (which is why the
proof-v1 run with crop_size=256 worked end-to-end), and they pass on the
local H100 in any mode. The crash is environment-specific to whatever
cuDNN/PyTorch combination Modal's image ships with — see gh#3 for the
backend-selection details.

This test will FAIL on environments that hit the bug (e.g. Modal CI).
That's the point: when we land a clean fix, this test should go green.
"""

import pytest
import torch

from helico.data import make_synthetic_batch
from helico.model import Helico, HelicoConfig


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)


@cuda_only
@pytest.mark.parametrize("n_tokens", [128, 256])
def test_eval_forward_at_problematic_shape(n_tokens: int):
    """Forward pass in eval+no_grad at shapes seen during val sweeps.

    Shapes >= 128 trigger the cuDNN flash-attn rejection on the Modal
    training image. Smaller shapes (<= 100) pass fine. The same shapes
    pass in train mode with grad enabled.

    See gh#3.
    """
    cfg = HelicoConfig(n_pairformer_blocks=2, n_diffusion_token_blocks=2)
    model = Helico(cfg).cuda().eval()

    batch = make_synthetic_batch(n_tokens=n_tokens, device="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(batch)

    # Sanity: no NaNs in any output tensor.
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any().item(), f"NaN in output[{k!r}]"
