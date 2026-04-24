"""Generic neural-net building blocks used throughout the model.

Grouped here because they have no AF3 algorithm of their own — they're
the primitive layers (LayerNorm, SwiGLU transition, Fourier embedding,
FiLM-style adaptive LayerNorm) composed into larger modules elsewhere.

AF3-spec provenance noted per class where relevant; most of these are
standard transformer primitives.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm that handles mixed precision (casts weights to input dtype)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.shape[-1],),
            self.weight.to(x.dtype), self.bias.to(x.dtype),
            self.eps,
        )


class AdaptiveLayerNorm(nn.Module):
    """FiLM-style modulated normalization.

    Used throughout the diffusion module (Algorithms 20-25 in AF3 SI §3.7):
    normalizes ``a`` without affine, then applies a scale+shift predicted
    from a separate conditioning signal ``s``. Scale is sigmoid-gated so
    initial weights produce the identity (sigmoid(0) = 0.5, and the shift
    is also zero-initialized), letting training gently learn modulation.
    """

    def __init__(self, d_a: int, d_s: int):
        super().__init__()
        self.norm_a = nn.LayerNorm(d_a, elementwise_affine=False)
        self.norm_s = LayerNorm(d_s)
        self.scale_proj = nn.Linear(d_s, d_a, bias=True)
        self.shift_proj = nn.Linear(d_s, d_a, bias=False)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.norm_s(s)
        return (
            torch.sigmoid(self.scale_proj(s_norm)) * self.norm_a(a)
            + self.shift_proj(s_norm)
        )


# ---------------------------------------------------------------------------
# Linear-layer factories
# ---------------------------------------------------------------------------

def linear_no_bias(d_in: int, d_out: int, zeros_init: bool = False) -> nn.Linear:
    """Factory for Linear(bias=False) with optional zero weight init.

    Matches AF3 SI shorthand `LinearNoBias`. Zero init is used for gating
    residuals so stacking new blocks is a no-op before training — cheap
    identity initialization.
    """
    lin = nn.Linear(d_in, d_out, bias=False)
    if zeros_init:
        nn.init.zeros_(lin.weight)
    return lin


class BiasInitLinear(nn.Module):
    """Linear with weight=0, bias=constant. Used for conditioning gates.

    `sigmoid(b_init)` picks the initial gate value — e.g., ``b_init=-2``
    gives gate ≈ 0.12, biasing towards "mostly closed" so stacked residual
    updates start close to zero and ramp up during training.
    """

    def __init__(self, d_in: int, d_out: int, bias_init: float = -2.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# MLP blocks
# ---------------------------------------------------------------------------

class Transition(nn.Module):
    """SwiGLU transition block (AF3 SI Algorithm 11).

    LayerNorm → gated MLP with SwiGLU activation: ``SiLU(Wa·h) * Wb·h``
    followed by a projection back to ``d``. The gated variant replaced
    AlphaFold 2's ReLU transitions and gives a measurable quality bump.
    """

    def __init__(self, d: int, factor: int = 4):
        super().__init__()
        self.norm = LayerNorm(d)
        self.linear_a = nn.Linear(d, d * factor, bias=False)
        self.linear_b = nn.Linear(d, d * factor, bias=False)
        self.linear_out = nn.Linear(d * factor, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.linear_out(F.silu(self.linear_a(h)) * self.linear_b(h))


class ConditionedTransitionBlock(nn.Module):
    """AdaLN + SwiGLU + s-gate (AF3 SI Algorithm 25).

    Diffusion-specific transition: the SwiGLU MLP acts on an input ``a``
    conditioned on ``s`` via AdaptiveLayerNorm, and the output is
    element-wise gated by sigmoid(linear(s)). All gating paths zero-init
    so the block is a no-op at initialization.
    """

    def __init__(self, d_a: int, d_s: int, factor: int = 2):
        super().__init__()
        self.ada_ln = AdaptiveLayerNorm(d_a, d_s)
        self.linear_a = nn.Linear(d_a, d_a * factor, bias=False)
        self.linear_b = nn.Linear(d_a, d_a * factor, bias=False)
        self.linear_out = nn.Linear(d_a * factor, d_a, bias=False)
        self.s_gate = BiasInitLinear(d_s, d_a, bias_init=-2.0)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.ada_ln(a, s)
        return (
            torch.sigmoid(self.s_gate(s))
            * self.linear_out(F.silu(self.linear_a(h)) * self.linear_b(h))
        )


# ---------------------------------------------------------------------------
# Noise-level embedding
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Random Fourier features for the diffusion noise level σ.

    AF3 SI Algorithm 22: σ → cos(2π * (σ·w + b)) with fixed random w, b.
    Gives the diffusion conditioning a sinusoidal "time code" smooth in σ
    but not unboundedly differentiable at any particular σ, which helps
    the model generalize across noise scales.
    """

    def __init__(self, d: int = 256, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(d, generator=gen))
        self.register_buffer("b", torch.randn(d, generator=gen))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar per batch element. Returns (B, d)."""
        return torch.cos(2 * math.pi * (t.unsqueeze(-1) * self.w + self.b))
