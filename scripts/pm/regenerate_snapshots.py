"""Regenerate numerical-regression snapshots under tests/data/snapshots/.

Each snapshot is a golden output tensor (or dict of tensors) produced by
the *current* code with Protenix v1.0.0 weights and a fixed torch seed.
The matching tests in tests/test_snapshots.py assert that future runs
reproduce these outputs bit-for-bit (within bf16 tolerance).

When we intentionally change numerics, re-run this script, commit the
updated .npz files alongside the code change — the diff in PR review
tells reviewers which snapshot moved and by how much.

Run:
    uv run python scripts/pm/regenerate_snapshots.py [--force]

Requires:
    - checkpoints/protenix_base_default_v1.0.0.pt (1.4 GB)
    - CUDA GPU (tested on A5000 / H100)
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from helico.data import make_synthetic_batch
from helico.model import Helico, HelicoConfig
from helico.load_protenix import load_protenix_state_dict

SNAPSHOTS = REPO / "tests" / "data" / "snapshots"
CHECKPOINT = REPO / "checkpoints" / "protenix_base_default_v1.0.0.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16
SEED = 42


def _load_protenix_model() -> Helico:
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    sd = OrderedDict((k.removeprefix("module."), v) for k, v in sd.items())
    model = Helico(HelicoConfig())
    load_protenix_state_dict(sd, model)
    # Keep model fp32; inference uses autocast to bf16 on compute-heavy ops
    return model.to(device=DEVICE).eval()


def _canonical_batch(n_tokens: int = 16, n_atoms_per_token: int = 4, n_msa: int = 8):
    """Small synthetic batch on DEVICE, BF16 where appropriate.

    Output: the same dict make_synthetic_batch returns, plus MSA fields.
    Deterministic: the synthetic-batch function uses torch.randn, so we
    wrap the call in a manual_seed.
    """
    torch.manual_seed(SEED)
    batch = make_synthetic_batch(
        n_tokens=n_tokens,
        n_atoms_per_token=n_atoms_per_token,
        batch_size=1,
        device=DEVICE,
    )
    # Add MSA fields (make_synthetic_batch supplies small defaults; we
    # ensure N_msa=n_msa so TestMSASubsample's path is exercised)
    B, _ = batch["token_types"].shape
    N_tok = n_tokens
    batch["msa"] = torch.randint(0, 32, (B, n_msa, N_tok), device=DEVICE, dtype=torch.long)
    batch["msa"][:, 0, :] = batch["restype"]  # row 0 is query
    batch["deletion_matrix"] = torch.zeros(B, n_msa, N_tok, device=DEVICE, dtype=torch.float32)
    batch["deletion_mean"] = torch.zeros(B, N_tok, device=DEVICE, dtype=torch.float32)
    batch["msa_profile"] = torch.zeros(B, N_tok, 32, device=DEVICE, dtype=torch.float32)
    batch["has_msa"] = torch.ones(B, device=DEVICE, dtype=torch.float32)
    return batch


@torch.no_grad()
@torch.amp.autocast("cuda", dtype=torch.bfloat16)
def generate_trunk_snapshot(model: Helico, batch: dict) -> dict[str, np.ndarray]:
    """Snapshot trunk activations: s_inputs, s_init, z_init, post-cycle s/z.

    Runs 2 recycles (not the default 10) to keep the snapshot small and
    the test fast. The fix for the TemplateEmbedder and MSA-subsample bugs
    still exercises: both fire in cycle 0 already.
    """
    torch.manual_seed(SEED)
    # We manually replicate the predict() trunk path so we can capture
    # tensors at each boundary without depending on the dumper format.
    mask = batch.get("token_mask")
    pair_mask = None
    if mask is not None:
        pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

    ref_charge, ref_features = model._build_ref_features(batch)
    atom_mask = batch.get("atom_mask", torch.ones(1, batch["atom_coords"].shape[1], device=DEVICE))
    atom_mask = atom_mask.float()
    s_inputs = model._build_s_inputs(batch, ref_charge, ref_features, atom_mask)

    s_init = model.linear_sinit(s_inputs)
    z_init = (
        model.linear_zinit1(s_init).unsqueeze(2)
        + model.linear_zinit2(s_init).unsqueeze(1)
    )
    relpe_feats = model._build_relpe_feats(batch)
    z_init = z_init + model.trunk_relpe(**relpe_feats)

    msa_raw, msa_mask = model._build_msa_raw(batch)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    N_CYCLES = 2
    for _ in range(N_CYCLES):
        z = z_init + model.linear_z_cycle(model.layernorm_z_cycle(z))
        z = z + model.template_embedder(batch, z)
        z = model.msa_module(msa_raw, z, s_inputs, msa_mask, pair_mask,
                              msa_chunk_size=2048)
        s = s_init + model.linear_s(model.layernorm_s(s))
        s, z = model.pairformer(s, z, mask=mask, pair_mask=pair_mask)

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().to(dtype=torch.float32, device="cpu").numpy()

    return {
        "s_inputs": _np(s_inputs),
        "s_init": _np(s_init),
        "z_init": _np(z_init),
        "s_post_recycle": _np(s),
        "z_post_recycle": _np(z),
        "ref_charge": _np(ref_charge),
        "ref_features": _np(ref_features),
        "msa_raw": _np(msa_raw),
    }


@torch.no_grad()
@torch.amp.autocast("cuda", dtype=torch.bfloat16)
def generate_diffusion_snapshot(model: Helico, batch: dict) -> dict[str, np.ndarray]:
    """Snapshot diffusion trajectory: intermediate x after N=5 steps."""
    # Shrink n_diffusion_steps for the snapshot run
    orig_steps = model.diffusion.n_steps
    model.diffusion.n_steps = 5

    torch.manual_seed(SEED)
    # Need trunk state first; reuse generate_trunk_snapshot's path, but
    # skip dumps
    mask = batch.get("token_mask")
    pair_mask = None
    if mask is not None:
        pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

    ref_charge, ref_features = model._build_ref_features(batch)
    atom_mask = batch.get("atom_mask", torch.ones(1, batch["atom_coords"].shape[1], device=DEVICE))
    atom_mask = atom_mask.float()
    s_inputs = model._build_s_inputs(batch, ref_charge, ref_features, atom_mask)
    s_init = model.linear_sinit(s_inputs)
    z_init = (
        model.linear_zinit1(s_init).unsqueeze(2)
        + model.linear_zinit2(s_init).unsqueeze(1)
    )
    relpe_feats = model._build_relpe_feats(batch)
    z_init = z_init + model.trunk_relpe(**relpe_feats)
    msa_raw, msa_mask = model._build_msa_raw(batch)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    for _ in range(2):
        z = z_init + model.linear_z_cycle(model.layernorm_z_cycle(z))
        z = z + model.template_embedder(batch, z)
        z = model.msa_module(msa_raw, z, s_inputs, msa_mask, pair_mask,
                              msa_chunk_size=2048)
        s = s_init + model.linear_s(model.layernorm_s(s))
        s, z = model.pairformer(s, z, mask=mask, pair_mask=pair_mask)

    # Fresh seed for diffusion noise, independent of trunk
    torch.manual_seed(SEED)
    coords = model.diffusion.sample(
        ref_pos=batch["ref_coords"],
        ref_charge=ref_charge,
        ref_features=ref_features,
        atom_to_token=batch["atom_to_token"],
        atom_mask=atom_mask,
        s_trunk=s,
        z_trunk=z,
        s_inputs=s_inputs,
        relpe_feats=relpe_feats,
        ref_space_uid=batch.get("ref_space_uid"),
    )
    model.diffusion.n_steps = orig_steps

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().to(dtype=torch.float32, device="cpu").numpy()

    return {"coords": _np(coords)}


@torch.no_grad()
@torch.amp.autocast("cuda", dtype=torch.bfloat16)
def generate_build_helpers_snapshot(model: Helico, batch: dict) -> dict[str, np.ndarray]:
    """Snapshot outputs of Helico._build_* methods (pre-extraction pins)."""
    ref_charge, ref_features = model._build_ref_features(batch)
    atom_mask = batch.get("atom_mask", torch.ones(1, batch["atom_coords"].shape[1], device=DEVICE))
    atom_mask = atom_mask.float()
    s_inputs = model._build_s_inputs(batch, ref_charge, ref_features, atom_mask)
    relpe_feats = model._build_relpe_feats(batch)
    msa_raw, msa_mask = model._build_msa_raw(batch)

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().to(dtype=torch.float32, device="cpu").numpy()

    out = {
        "ref_charge": _np(ref_charge),
        "ref_features": _np(ref_features),
        "s_inputs": _np(s_inputs),
        "msa_raw": _np(msa_raw),
    }
    for k, v in relpe_feats.items():
        out[f"relpe_{k}"] = _np(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite existing snapshots")
    args = ap.parse_args()

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)

    targets = {
        "trunk.npz": generate_trunk_snapshot,
        "diffusion.npz": generate_diffusion_snapshot,
        "build_helpers.npz": generate_build_helpers_snapshot,
    }

    for fname, fn in targets.items():
        path = SNAPSHOTS / fname
        if path.exists() and not args.force:
            print(f"[skip] {path} exists (use --force to overwrite)")
            continue
        print(f"[generating] {path}")
        model = _load_protenix_model()
        batch = _canonical_batch()
        arrs = fn(model, batch)
        np.savez_compressed(path, **arrs)
        print(f"  wrote {path} ({sum(a.nbytes for a in arrs.values()) / 1e6:.2f} MB)")
        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
