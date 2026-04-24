"""Helico model — PyTorch implementation of AlphaFold 3.

Structure (in rough order of AF3 SI section):

- ``config.py``        HelicoConfig (all hyperparameters)
- ``blocks.py``        LayerNorm, SwiGLU Transition (Alg 11), AdaLN (Alg 26),
                       FourierEmbedding (Alg 22), ConditionedTransitionBlock
                       (Alg 25), linear_no_bias / BiasInitLinear factories
- ``triangle.py``      TriangleMultiplicativeUpdate (Alg 12/13),
                       TriangleAttention (Alg 14/15)  [cuEquivariance kernels]
- ``pairformer.py``    SingleAttentionWithPairBias, PairformerBlock/Stack
                       (Alg 17), RelativePositionEncoding (Alg 3)
- ``msa.py``           OuterProductMean (Alg 9), MSAPairWeightedAveraging
                       (Alg 10), MSABlock / MSAModule (Alg 8, §3.3)
- ``diffusion.py``     DiffusionAttentionPairBias (Alg 24),
                       DiffusionTransformer (Alg 23),
                       AtomAttentionEncoder (Alg 5), AtomAttentionDecoder
                       (Alg 6), DiffusionConditioning (Alg 21),
                       _centre_random_augmentation (Alg 19), DiffusionModule
                       (Alg 20, ``sample`` = Alg 18)
- ``template.py``      TemplateEmbedder (Alg 16) + the pure-PyTorch
                       template-specific triangle ops (hidden ≠ d variants)
- ``input_embedder.py`` InputFeatureEmbedder (Alg 2)
- ``heads.py``         DistogramHead (§4.4), ConfidenceHead (§4.3)
- ``losses.py``        diffusion_loss (Eq. 3), smooth_lddt_loss (Alg 27),
                       distogram_loss, violation_loss
- ``metrics.py``       compute_plddt / pae / ptm / iptm / clash /
                       ranking_score — post-processing of head logits
- ``dumper.py``        _maybe_build_dumper — pipeline-diff instrumentation
- ``helico.py``        Helico top-level nn.Module (Alg 1)

The public API is a small set of classes re-exported below. Importing
``helico.model`` gets you ``Helico`` and ``HelicoConfig`` + everything
tests and external callers need.
"""

from __future__ import annotations

from helico.model.config import HelicoConfig
from helico.model.blocks import (
    LayerNorm,
    Transition,
    AdaptiveLayerNorm,
    linear_no_bias,
    BiasInitLinear,
    FourierEmbedding,
    ConditionedTransitionBlock,
)
from helico.model.triangle import (
    TriangleMultiplicativeUpdate,
    TriangleAttention,
)
from helico.model.pairformer import (
    SingleAttentionWithPairBias,
    PairformerBlock,
    Pairformer,
    RelativePositionEncoding,
)
from helico.model.msa import (
    OuterProductMean,
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
)
from helico.model.diffusion import (
    DiffusionAttentionPairBias,
    DiffusionTransformerBlock,
    DiffusionTransformer,
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    DiffusionConditioning,
    DiffusionModule,
    _centre_random_augmentation,
    _partition_to_windows,
    _unpartition_from_windows,
)
from helico.model.input_embedder import InputFeatureEmbedder
from helico.model.template import TemplateEmbedder
from helico.model.heads import ConfidenceHead, DistogramHead
from helico.model.losses import (
    diffusion_loss,
    smooth_lddt_loss,
    distogram_loss,
    violation_loss,
)
from helico.model.metrics import (
    compute_plddt,
    compute_pae,
    compute_ptm,
    compute_iptm,
    compute_clash,
    compute_ranking_score,
    _flatten_plddt,
)
from helico.model.dumper import _maybe_build_dumper
from helico.model.helico import Helico


__all__ = [
    # Config
    "HelicoConfig",
    # Building blocks
    "LayerNorm", "Transition", "AdaptiveLayerNorm",
    "linear_no_bias", "BiasInitLinear",
    "FourierEmbedding", "ConditionedTransitionBlock",
    # Triangle
    "TriangleMultiplicativeUpdate", "TriangleAttention",
    # Pairformer
    "SingleAttentionWithPairBias", "PairformerBlock", "Pairformer",
    "RelativePositionEncoding",
    # MSA
    "OuterProductMean", "MSAPairWeightedAveraging",
    "MSAStack", "MSABlock", "MSAModule",
    # Diffusion
    "DiffusionAttentionPairBias", "DiffusionTransformerBlock",
    "DiffusionTransformer",
    "AtomAttentionEncoder", "AtomAttentionDecoder",
    "DiffusionConditioning", "DiffusionModule",
    # Input + template
    "InputFeatureEmbedder", "TemplateEmbedder",
    # Heads
    "DistogramHead", "ConfidenceHead",
    # Losses
    "diffusion_loss", "smooth_lddt_loss",
    "distogram_loss", "violation_loss",
    # Metrics
    "compute_plddt", "compute_pae",
    "compute_ptm", "compute_iptm",
    "compute_clash", "compute_ranking_score",
    # Top-level
    "Helico",
]
