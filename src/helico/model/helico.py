"""Top-level Helico model — AF3 SI Algorithm 1 (MainInferenceLoop).

Ties together the building blocks:
  1. InputFeatureEmbedder (Alg 2)       → s_inputs
  2. Linear init (Alg 1 lines 2-5)      → s_init, z_init  (+ RelPE, token_bonds)
  3. Recycling loop (Alg 1 lines 7-14)  → s, z via TemplateEmbedder, MSAModule,
                                          PairformerStack
  4. DiffusionModule.sample (Alg 18)    → coordinate
  5. ConfidenceHead (§4.3)              → pae/pde/plddt/resolved
  6. DistogramHead (§4.4)               → distogram logits
  7. Ranking (§5.9.3)                   → best-of-N sample selection

``forward`` runs the training path (trunk + one diffusion step + heads).
``predict`` runs the full inference path (trunk + n_samples diffusion
rollouts + confidence + ranking).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm, linear_no_bias
from .config import HelicoConfig
from .pairformer import Pairformer, RelativePositionEncoding
from .msa import MSAModule
from .diffusion import DiffusionModule
from .template import TemplateEmbedder
from .input_embedder import InputFeatureEmbedder
from .heads import ConfidenceHead, DistogramHead
from .losses import diffusion_loss, smooth_lddt_loss, distogram_loss, violation_loss
from .metrics import (
    compute_plddt, compute_pae, compute_ptm, compute_iptm,
    compute_clash, compute_ranking_score, _flatten_plddt,
)
from .features import (
    build_ref_features,
    build_relpe_feats,
    build_s_inputs,
    build_msa_raw,
)

class Helico(nn.Module):
    """Complete Helico model."""

    def __init__(self, config: HelicoConfig | None = None):
        super().__init__()
        if config is None:
            config = HelicoConfig()
        self.config = config

        # Input embedding (AtomAttentionEncoder without coords)
        self.input_embedder = InputFeatureEmbedder(config)

        # Trunk initialization
        self.linear_sinit = linear_no_bias(config.c_s_inputs, config.d_single)     # 449->384
        self.linear_zinit1 = linear_no_bias(config.d_single, config.d_pair)        # 384->128
        self.linear_zinit2 = linear_no_bias(config.d_single, config.d_pair)        # 384->128
        self.trunk_relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=config.d_pair)
        self.linear_token_bond = linear_no_bias(1, config.d_pair)                  # 1->128

        # Recycling (zero-initialized)
        self.layernorm_s = LayerNorm(config.d_single)
        self.linear_s = linear_no_bias(config.d_single, config.d_single, zeros_init=True)
        self.layernorm_z_cycle = LayerNorm(config.d_pair)
        self.linear_z_cycle = linear_no_bias(config.d_pair, config.d_pair, zeros_init=True)

        # Template embedder
        self.template_embedder = TemplateEmbedder(config)

        # MSA module
        self.msa_module = MSAModule(config)

        # Pairformer trunk
        self.pairformer = Pairformer(config)

        # Diffusion module
        self.diffusion = DiffusionModule(config)

        # Heads
        self.confidence_head = ConfidenceHead(config)
        self.distogram_head = DistogramHead(config)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        compute_confidence: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training."""
        mask = batch.get("token_mask")
        pair_mask = None
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device

        # Build reference features and atom mask
        ref_charge, ref_features = build_ref_features(batch)
        atom_mask = batch.get("atom_mask", torch.ones(B, batch["atom_coords"].shape[1], device=device))
        atom_mask = atom_mask.float()

        # 1. Input embedding -> s_inputs (B, N_tok, 449)
        s_inputs = build_s_inputs(self.input_embedder, batch, ref_charge, ref_features, atom_mask)

        # 2. Trunk initialization
        s_init = self.linear_sinit(s_inputs)
        z_init = self.linear_zinit1(s_init).unsqueeze(2) + self.linear_zinit2(s_init).unsqueeze(1)

        relpe_feats = build_relpe_feats(batch)
        z_init = z_init + self.trunk_relpe(**relpe_feats)

        # Token bonds
        token_bonds = batch.get("token_bonds")
        if token_bonds is not None:
            z_init = z_init + self.linear_token_bond(token_bonds.unsqueeze(-1).to(z_init.dtype))

        # 3. Recycling loop
        msa_raw, msa_mask = build_msa_raw(batch)
        n_cycles = self.config.n_cycles

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for cycle in range(n_cycles):
            z = z_init + self.linear_z_cycle(self.layernorm_z_cycle(z))
            z = z + self.template_embedder(batch, z)
            z = self.msa_module(
                msa_raw, z, s_inputs, msa_mask, pair_mask,
                msa_chunk_size=(None if self.training else 2048),
            )
            s = s_init + self.linear_s(self.layernorm_s(s))
            s, z = self.pairformer(s, z, mask=mask, pair_mask=pair_mask)

        results = {"single": s, "pair": z}

        # 4a. Distogram (always computed; needs to be available *before*
        # diffusion when diffusion_pair_source == "distogram_logits" so the
        # diffusion module can read from it. distogram_head is itself a
        # single Linear (z → 64-bin logits, symmetrized).
        distogram_logits = self.distogram_head(z)
        results["distogram_logits"] = distogram_logits

        # 4b. Diffusion — s_inputs is already (B, N_tok, 449 = d_single + 65)
        # n_diffusion_samples > 1 amortizes the expensive trunk over several
        # denoising passes per batch entry (gh#6). Outputs are (B*N_d, ...).
        # gh#9: when configured, swap z_trunk for the trunk's distogram
        # output (information bottleneck). detach() ensures the trunk
        # graph isn't pinned through the diffusion backward when the
        # trunk is frozen — saves activation memory.
        if self.config.diffusion_pair_source == "distogram_logits":
            z_for_diffusion = distogram_logits.detach()
        else:
            z_for_diffusion = z
        n_d = max(1, self.config.n_diffusion_samples)
        x_denoised, gt_coords, sigma = self.diffusion.forward_training(
            gt_coords=batch["atom_coords"],
            ref_pos=batch["ref_coords"],
            ref_charge=ref_charge,
            ref_features=ref_features,
            atom_to_token=batch["atom_to_token"],
            atom_mask=atom_mask,
            s_trunk=s,
            z_trunk=z_for_diffusion,
            s_inputs=s_inputs,
            relpe_feats=relpe_feats,
            n_samples=n_d,
        )

        results["x_denoised"] = x_denoised
        results["sigma"] = sigma
        # diffusion_loss averages over all B*N_d samples — atom_mask must
        # match the expanded batch.
        atom_mask_d = atom_mask.repeat_interleave(n_d, dim=0) if n_d > 1 else atom_mask
        results["diffusion_loss"] = diffusion_loss(x_denoised, gt_coords, sigma, atom_mask_d)

        # 6. Confidence head (uses pred_coords from diffusion). Use only
        # the first denoising sample per batch entry — the head expects
        # (B, N_atoms, 3), not (B*N_d, ...).
        if compute_confidence:
            x_for_conf = x_denoised[::n_d] if n_d > 1 else x_denoised
            confidence = self.confidence_head(
                s_trunk=s, z_trunk=z, s_inputs=s_inputs,
                pred_coords=x_for_conf.detach(),
                atom_to_token=batch["atom_to_token"],
                atom_mask=atom_mask,
                mask=mask, pair_mask=pair_mask,
                rep_atom_idx=batch.get("rep_atom_idx"),
            )
            results.update(confidence)

            token_centers = self._get_token_centers(batch)
            results["distogram_loss"] = distogram_loss(
                distogram_logits, token_centers, mask,
            )

        return results

    def _get_token_centers(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute token center coordinates (mean of atom coords per token)."""
        B = batch["atom_coords"].shape[0]
        N = batch["token_types"].shape[1]
        device = batch["atom_coords"].device

        centers = torch.zeros(B, N, 3, device=device, dtype=batch["atom_coords"].dtype)
        counts = torch.zeros(B, N, 1, device=device, dtype=batch["atom_coords"].dtype)

        for b in range(B):
            n_atoms = batch["n_atoms"][b]
            idx = batch["atom_to_token"][b, :n_atoms]
            centers[b].scatter_add_(0, idx.unsqueeze(1).expand(-1, 3), batch["atom_coords"][b, :n_atoms])
            counts[b].scatter_add_(0, idx.unsqueeze(1), torch.ones(n_atoms, 1, device=device, dtype=batch["atom_coords"].dtype))

        counts = counts.clamp(min=1)
        return centers / counts

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int = 5,
        n_cycles: int | None = None,
        verbose_timing: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run inference: generate structure predictions.

        Args:
            batch: Input feature dict.
            n_samples: Number of diffusion samples per input.
            n_cycles: Override number of recycling cycles (default: self.config.n_cycles).
            verbose_timing: Print detailed timing breakdown for each phase.
        """
        self.eval()

        def _sync_time():
            if verbose_timing:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.0

        t_overall_start = _sync_time()

        mask = batch.get("token_mask")
        pair_mask = None
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()

        B, N_tok = batch["token_types"].shape
        device = batch["token_types"].device

        t0 = _sync_time()
        ref_charge, ref_features = build_ref_features(batch)
        atom_mask = batch.get("atom_mask", torch.ones(B, batch["ref_coords"].shape[1], device=device))
        atom_mask = atom_mask.float()

        # Build s_inputs
        s_inputs = build_s_inputs(self.input_embedder, batch, ref_charge, ref_features, atom_mask)

        # Trunk init
        s_init = self.linear_sinit(s_inputs)
        z_init = self.linear_zinit1(s_init).unsqueeze(2) + self.linear_zinit2(s_init).unsqueeze(1)
        relpe_feats = build_relpe_feats(batch)
        z_init = z_init + self.trunk_relpe(**relpe_feats)
        token_bonds = batch.get("token_bonds")
        if token_bonds is not None:
            z_init = z_init + self.linear_token_bond(token_bonds.unsqueeze(-1).to(z_init.dtype))
        t_embed = _sync_time() - t0

        # Recycling
        msa_raw, msa_mask = build_msa_raw(batch)
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        actual_cycles = n_cycles if n_cycles is not None else self.config.n_cycles
        t_recycle_start = _sync_time()
        cycle_times = []

        for cycle in range(actual_cycles):
            t_c0 = _sync_time()
            z = z_init + self.linear_z_cycle(self.layernorm_z_cycle(z))
            z = z + self.template_embedder(batch, z)
            z = self.msa_module(
                msa_raw, z, s_inputs, msa_mask, pair_mask,
                msa_chunk_size=(None if self.training else 2048),
            )
            s = s_init + self.linear_s(self.layernorm_s(s))
            s, z = self.pairformer(s, z, mask=mask, pair_mask=pair_mask)
            cycle_times.append(_sync_time() - t_c0)
        t_recycle = _sync_time() - t_recycle_start

        # Generate all samples in one batched call: expand (B, ...) → (B*n_samples, ...)
        def _expand(t):
            return t.unsqueeze(1).expand(-1, n_samples, *[-1] * (t.dim() - 1)).reshape(B * n_samples, *t.shape[1:])

        ref_space_uid = batch.get("ref_space_uid")
        # gh#9: same swap as in forward — at inference, when the diffusion
        # module is configured to read from the distogram, run the head
        # before sampling and feed those logits in place of z.
        if self.config.diffusion_pair_source == "distogram_logits":
            z_for_diffusion = self.distogram_head(z)
        else:
            z_for_diffusion = z
        t_diffusion_start = _sync_time()
        batched_coords = self.diffusion.sample(
            ref_pos=_expand(batch["ref_coords"]),
            ref_charge=_expand(ref_charge),
            ref_features=_expand(ref_features),
            atom_to_token=_expand(batch["atom_to_token"]),
            atom_mask=_expand(atom_mask),
            s_trunk=_expand(s),
            z_trunk=_expand(z_for_diffusion),
            s_inputs=_expand(s_inputs),
            relpe_feats={k: _expand(v) for k, v in relpe_feats.items()},
            ref_space_uid=_expand(ref_space_uid) if ref_space_uid is not None else None,
        )  # (B*n_samples, N_atoms, 3)
        all_coords = batched_coords.reshape(B, n_samples, *batched_coords.shape[1:])
        t_diffusion = _sync_time() - t_diffusion_start

        # Score all samples and pick the best by ranking_score
        best_ranking = None
        best_idx = torch.zeros(B, dtype=torch.long, device=device)
        best_confidence = None
        best_plddt = None
        best_pae = None
        best_ptm = None
        best_iptm = None

        # Per-sample stats for downstream re-ranking from saved predictions.
        all_ptm = torch.zeros(B, n_samples, device=device)
        all_iptm = torch.zeros(B, n_samples, device=device)
        all_ranking = torch.zeros(B, n_samples, device=device)
        all_has_clash = torch.zeros(B, n_samples, device=device)

        chain_indices = batch.get("chain_indices")

        rep_atom_idx = batch.get("rep_atom_idx")
        has_frame = batch.get("has_frame")
        t_confidence_start = _sync_time()
        conf_times = []
        for si in range(n_samples):
            t_ci = _sync_time()
            confidence = self.confidence_head(
                s_trunk=s, z_trunk=z, s_inputs=s_inputs,
                pred_coords=all_coords[:, si],
                atom_to_token=batch["atom_to_token"],
                atom_mask=atom_mask,
                mask=mask, pair_mask=pair_mask,
                rep_atom_idx=rep_atom_idx,
            )

            plddt = compute_plddt(confidence["plddt_logits"])
            pae = compute_pae(confidence["pae_logits"])
            ptm = compute_ptm(confidence["pae_logits"], mask=mask, has_frame=has_frame)

            if chain_indices is not None:
                iptm = compute_iptm(confidence["pae_logits"], chain_indices, mask=mask, has_frame=has_frame)
                unique_counts = []
                for b in range(B):
                    ci = chain_indices[b]
                    if mask is not None:
                        ci = ci[mask[b]]
                    unique_counts.append(ci.unique().numel())
                has_interface = torch.tensor([c > 1 for c in unique_counts], device=device, dtype=torch.bool)
            else:
                iptm = ptm.clone()
                has_interface = torch.zeros(B, device=device, dtype=torch.bool)

            # Compute clash penalty
            if chain_indices is not None:
                has_clash = compute_clash(
                    all_coords[:, si], chain_indices,
                    batch["atom_to_token"], atom_mask,
                )
            else:
                has_clash = torch.zeros(B, device=device)

            ranking = compute_ranking_score(ptm, iptm, has_interface, has_clash=has_clash)
            conf_times.append(_sync_time() - t_ci)

            all_ptm[:, si] = ptm
            all_iptm[:, si] = iptm
            all_ranking[:, si] = ranking
            all_has_clash[:, si] = has_clash

            if best_ranking is None:
                best_ranking = ranking.clone()
                best_confidence = confidence
                best_plddt = plddt
                best_pae = pae
                best_ptm = ptm
                best_iptm = iptm
            else:
                # Update best per batch element
                better = ranking > best_ranking
                if better.any():
                    best_ranking = torch.where(better, ranking, best_ranking)
                    best_ptm = torch.where(better, ptm, best_ptm)
                    best_iptm = torch.where(better, iptm, best_iptm)
                    for b in range(B):
                        if better[b]:
                            best_idx[b] = si
                            best_plddt[b] = plddt[b]
                            best_pae[b] = pae[b]
                            best_confidence = {k: v.clone() for k, v in confidence.items()}  # save logits
        t_confidence = _sync_time() - t_confidence_start

        # Gather best coords per batch element
        best_coords = torch.stack([all_coords[b, best_idx[b]] for b in range(B)])

        # Flatten pLDDT to per-atom
        plddt_flat = _flatten_plddt(
            best_plddt, batch["atom_to_token"], batch["atoms_per_token"], atom_mask,
        )

        t_overall = _sync_time() - t_overall_start

        if verbose_timing:
            n_steps = self.diffusion.n_steps
            print(f"\n{'='*60}")
            print(f"  Helico predict() timing  (N_tok={N_tok}, B={B})")
            print(f"{'='*60}")
            print(f"  Input embedding:      {t_embed:8.2f}s")
            print(f"  Recycling ({actual_cycles} cycles):  {t_recycle:8.2f}s")
            for i, ct in enumerate(cycle_times):
                print(f"    cycle {i:2d}:            {ct:8.2f}s")
            print(f"  Diffusion ({n_samples} samples): {t_diffusion:8.2f}s  (batched, B*S={B*n_samples})")
            print(f"    avg per step:       {t_diffusion/n_steps:8.3f}s  ({n_steps} steps)")
            print(f"  Confidence ({n_samples} samples):{t_confidence:8.2f}s")
            for i, ct in enumerate(conf_times):
                print(f"    sample {i}:           {ct:8.2f}s")
            print(f"  {'─'*40}")
            print(f"  Total wall time:      {t_overall:8.2f}s")
            print(f"{'='*60}\n")

        return {
            "coords": best_coords,               # (B, N_atoms, 3)
            "all_coords": all_coords,             # (B, n_samples, N_atoms, 3)
            "plddt": plddt_flat,                  # (B, N_atoms) 0-100 scale
            "pae": best_pae,                      # (B, N_tok, N_tok) Angstroms
            "ptm": best_ptm,                      # (B,)
            "iptm": best_iptm,                    # (B,)
            "ranking_score": best_ranking,        # (B,)
            # Per-sample arrays so downstream code can re-rank without
            # re-running diffusion.
            "all_ptm": all_ptm,                   # (B, n_samples)
            "all_iptm": all_iptm,                 # (B, n_samples)
            "all_ranking_score": all_ranking,     # (B, n_samples)
            "all_has_clash": all_has_clash,       # (B, n_samples)
            # Raw logits for downstream use
            "pae_logits": best_confidence["pae_logits"],
            "plddt_logits": best_confidence["plddt_logits"],
            "pde_logits": best_confidence["pde_logits"],
        }
