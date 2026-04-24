"""Protenix-v1.0.0 checkpoint compatibility shims.

Helico loads the upstream Protenix v1.0.0 checkpoint (~368 M params) as
its only pretrained source — there's no independently trained Helico
checkpoint yet. A handful of behaviors in the forward pass are not in
the AF3 SI but *are* required for Protenix's weights to produce their
trained predictions. Those behaviors live here so they're easy to find
(and easy to rip out if we ever train our own weights).

Currently captured:

- ``dummy_template_features_v1_0_0`` — the specific per-slot aatype
  pattern Protenix's InferenceTemplateFeaturizer produces when
  ``use_template=False``. See
  ``src/helico/model/template.py::TemplateEmbedder.forward`` for where
  it's consumed and empirical verification.

Not captured here (AF3 SI spec, not Protenix-specific):

- Per-cycle random MSA subsampling (AF3 SI §3.5)
- EDM preconditioning + power-law σ schedule (AF3 SI Eq. 7)
- 32-class MSA encoding / AA ordering (AF3 SI Table 13) — lives in data.py
- Per-atom reference features (AF3 SI §2.8 Table 5) — lives in features.py

Not captured here (one-shot glue, not forward-pass behavior):

- State-dict name mapping (Protenix checkpoint → Helico module paths)
  lives in ``load_protenix.py`` — it's a pure data transform and
  touches no forward code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DummyTemplateSpec:
    """Shape + per-slot content of Protenix v1.0.0's dummy templates.

    Protenix's InferenceTemplateFeaturizer always pads template features
    to ``num_templates=4``; when no real templates are provided (the
    default ``use_template=False`` path), slot 0 gets ``aatype=31``
    (gap) and slots 1-3 get ``aatype=0`` (zero-padding), with every
    other template feature (distogram, unit_vector, pseudo_beta_mask,
    backbone_frame_mask) zeroed.

    The v1.0.0 checkpoint was trained to expect this path, so reproducing
    it exactly is how we get parity between Helico and Protenix on the
    same weights. Empirically verified via pipeline-diff dumps on
    8t59-assembly1 / 8v52-assembly1 (see the TemplateEmbedder fix in
    ``72f10e6`` for the story).
    """
    num_templates: int = 4
    aatype_per_slot: tuple[int, ...] = (31, 0, 0, 0)


DUMMY_TEMPLATE_SPEC_V1_0_0 = DummyTemplateSpec()
