"""Batch → submodule-arg feature adapters.

These free functions translate the batch dict produced by
``helico.data`` (see ``TokenizedStructure.to_features``) into the
per-submodule tensor shapes that ``InputFeatureEmbedder``,
``RelativePositionEncoding``, and ``MSAModule`` expect.

They're pure — no ``nn.Module`` state — and live outside the ``Helico``
class so the root module stays focused on orchestration rather than
feature plumbing. Each helper has a snapshot test pin in
``tests/test_snapshots.py::TestBuildHelpersSnapshot``.

Feature-naming crosswalk
------------------------

  batch key          →  downstream role (AF3 SI §2.8 / Table 5)
  -----------------     -----------------------------------------
  rel_pos            →  residue_index (Alg 3)
  token_index        →  token_index    (Alg 3)
  chain_indices      →  asym_id        (Alg 3)
  entity_id, sym_id  →  (Alg 3)
  msa                →  f^msa          (Alg 8)
  deletion_matrix    →  f^has_deletion, f^deletion_value (SI §2.8)
  atom_element_idx   →  f^ref_element (one-hot)
  atom_name_chars    →  f^ref_atom_name_chars
  ref_charge         →  f^ref_charge (arcsinh-transformed)
  atom_mask          →  f^ref_mask
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_ref_features(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-atom reference features for the AtomAttentionEncoder.

    Returns ``(ref_charge, ref_features)``:
      - ``ref_charge``:   (B, N_atoms, 1) — arcsinh-transformed formal charges
      - ``ref_features``: (B, N_atoms, 385) — mask(1) + element_onehot(128)
                          + atom_name_chars(256) concatenated

    The 128-wide element one-hot covers atomic numbers up to 128 (AF3 SI
    Table 5). The 256-wide atom_name_chars encodes each atom's CCD name
    as 4 characters × 64 classes (SI Table 5 again) — the data pipeline
    pads names < 4 chars and encodes them as ``ord(c) - 32``.
    """
    B, N_atoms = batch["atom_element_idx"].shape
    device = batch["atom_element_idx"].device
    dtype = batch.get("ref_coords", batch["atom_coords"]).dtype

    if "ref_charge" in batch:
        # arcsinh (per AF3 SI §2.8): compresses large charges but preserves sign
        ref_charge = torch.arcsinh(batch["ref_charge"].to(dtype)).unsqueeze(-1)
    else:
        ref_charge = torch.zeros(B, N_atoms, 1, device=device, dtype=dtype)

    elem_onehot = F.one_hot(batch["atom_element_idx"].clamp(max=127), 128).to(dtype)

    atom_mask = batch.get("atom_mask")
    if atom_mask is not None:
        mask_feat = atom_mask.unsqueeze(-1).to(dtype)
    else:
        mask_feat = torch.ones(B, N_atoms, 1, device=device, dtype=dtype)

    name_chars = batch.get("atom_name_chars")
    if name_chars is None:
        name_chars = torch.zeros(B, N_atoms, 256, device=device, dtype=dtype)
    else:
        name_chars = name_chars.to(device=device, dtype=dtype)

    ref_features = torch.cat([mask_feat, elem_onehot, name_chars], dim=-1)
    return ref_charge, ref_features


def build_relpe_feats(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the five per-token integer features RelativePositionEncoding needs.

    AF3 Alg 3 takes (residue_index, token_index, asym_id, entity_id, sym_id).
    In Helico's batch those live under slightly different names
    (``rel_pos``, ``chain_indices``) — this tiny adapter renames them.
    """
    return {
        "residue_index": batch["rel_pos"],
        "token_index": batch["token_index"],
        "asym_id": batch["chain_indices"],
        "entity_id": batch["entity_id"],
        "sym_id": batch["sym_id"],
    }


def build_s_inputs(
    input_embedder,
    batch: dict[str, torch.Tensor],
    ref_charge: torch.Tensor,
    ref_features: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Run InputFeatureEmbedder (AF3 Alg 2) on the batch.

    Extracts the per-token scalars (restype one-hot, MSA profile,
    deletion_mean) and concatenates with the atom-encoded per-token
    features. Returns (B, N_tok, c_s_inputs=449).
    """
    B, N_tok = batch["token_types"].shape
    device = batch["token_types"].device
    dtype = ref_charge.dtype

    # restype one-hot — precomputed in the data pipeline using the AF3 SI
    # Table 13 32-class encoding (handles RNA/DNA correctly)
    restype = F.one_hot(batch["restype"], 32).to(dtype)

    profile = batch.get("msa_profile",
                        torch.zeros(B, N_tok, 32, device=device, dtype=dtype))

    deletion_mean = batch.get("deletion_mean",
                              torch.zeros(B, N_tok, 1, device=device, dtype=dtype))
    if deletion_mean.dim() == 2:
        deletion_mean = deletion_mean.unsqueeze(-1)

    return input_embedder(
        ref_pos=batch["ref_coords"],
        ref_charge=ref_charge,
        ref_features=ref_features,
        atom_to_token=batch["atom_to_token"],
        atom_mask=atom_mask,
        n_tokens=N_tok,
        restype=restype,
        profile=profile,
        deletion_mean=deletion_mean,
        ref_space_uid=batch.get("ref_space_uid"),
    )


def build_msa_raw(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Assemble the 34-dim per-(row, token) MSA feature fed to MSAModule.

    AF3 SI §2.8 / Alg 8 line 1: ``m_Si = concat(f^msa, f^has_deletion, f^deletion_value)``
    with 32 + 1 + 1 = 34 channels. ``has_deletion = clip(deletion_matrix, 0, 1)``
    and ``deletion_value = (2/π) · arctan(deletion_matrix / 3)``.

    Falls back to zeros when no MSA is provided (single-sequence / test paths).
    """
    B, N_tok = batch["token_types"].shape
    device = batch["token_types"].device
    dtype = batch.get("ref_coords", batch["atom_coords"]).dtype

    msa_int = batch.get("msa")
    if msa_int is not None:
        del_raw = batch.get(
            "deletion_matrix",
            torch.zeros(B, msa_int.shape[1], N_tok, device=device, dtype=dtype),
        )
    else:
        # Back-compat path — older batches carried clustered MSA
        msa_int = batch.get("cluster_msa")
        del_raw = batch.get("cluster_deletion_mean")
    if msa_int is None:
        # No MSA at all: feed a single-row all-gap dummy so downstream linears
        # don't see empty shapes.
        msa_raw = torch.zeros(B, 1, N_tok, 34, device=device, dtype=dtype)
        return msa_raw, None

    N_msa = msa_int.shape[1]
    if del_raw is None:
        del_raw = torch.zeros(B, N_msa, N_tok, device=device, dtype=dtype)

    # 32-class one-hot — MSA entries already use AF3 SI Table 13 indices
    msa_onehot = F.one_hot(msa_int.clamp(max=31), 32).to(dtype)

    # AF3 SI §2.8: has_deletion + deletion_value transforms (SI Table 5)
    del_raw = del_raw.to(dtype)
    has_del = del_raw.clamp(0, 1).unsqueeze(-1)
    del_val = (torch.arctan(del_raw / 3.0) * (2.0 / math.pi)).unsqueeze(-1)

    msa_raw = torch.cat([msa_onehot, has_del, del_val], dim=-1)
    return msa_raw, None
