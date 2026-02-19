"""Helico training loop, DDP, checkpointing, multi-stage schedule, and inference."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from helico.data import (
    HelicoDataset,
    LazyHelicoDataset,
    TarIndex,
    TokenizedStructure,
    MSAFeatures,
    collate_fn,
    load_manifest,
    load_tar_index,
    make_synthetic_batch,
    make_synthetic_structure,
    parse_ccd,
    parse_mmcif,
    parse_sequences_arg,
    parse_input_yaml,
    tokenize_sequences,
    tokenize_structure,
    PROCESSED_DIR,
)
from helico.model import Helico, HelicoConfig, diffusion_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    n_pairformer_blocks: int = 48
    n_diffusion_token_blocks: int = 24

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Schedule
    lr_schedule: str = "cosine"  # "cosine" or "constant"

    # Data
    crop_size: int = 384
    batch_size: int = 1
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    log_every: int = 10

    # EMA
    ema_decay: float = 0.999

    # Mixed precision
    dtype: str = "bfloat16"

    # DDP
    distributed: bool = False

    def get_torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]


@dataclass
class StageConfig:
    """Multi-stage training schedule."""
    stages: list[dict] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                {"name": "stage1", "crop_size": 384, "lr": 1e-3, "steps": 50_000},
                {"name": "stage2", "crop_size": 640, "lr": 5e-4, "steps": 30_000},
                {"name": "stage3", "crop_size": 768, "lr": 1e-4, "steps": 20_000},
            ]

    def get_stage(self, step: int) -> dict:
        cumulative = 0
        for stage in self.stages:
            cumulative += stage["steps"]
            if step < cumulative:
                return stage
        return self.stages[-1]


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(step: int, config: TrainConfig, stage_lr: float | None = None) -> float:
    """Compute learning rate with warmup and cosine decay."""
    base_lr = stage_lr or config.lr

    # Warmup
    if step < config.warmup_steps:
        return base_lr * step / max(1, config.warmup_steps)

    # Cosine decay
    if config.lr_schedule == "cosine":
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return base_lr * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * min(progress, 1.0)))

    return base_lr


# ============================================================================
# EMA
# ============================================================================

class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (for inference)."""
        self.original = {name: param.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after inference."""
        for name, param in model.named_parameters():
            param.data.copy_(self.original[name])


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    ema: EMAModel | None = None,
    path: str | None = None,
):
    if path is None:
        path = os.path.join(config.checkpoint_dir, f"step_{step}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }
    if ema is not None:
        state["ema_shadow"] = ema.shadow

    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMAModel | None = None,
) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(model, DDP):
        model.module.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if ema is not None and "ema_shadow" in state:
        ema.shadow = state["ema_shadow"]

    step = state.get("step", 0)
    logger.info(f"Loaded checkpoint from {path} at step {step}")
    return step


# ============================================================================
# Training Loop
# ============================================================================

def train(
    model: Helico,
    train_data: list[TokenizedStructure] | None = None,
    config: TrainConfig = None,
    model_config: HelicoConfig | None = None,
    msa_features: dict[str, MSAFeatures] | None = None,
    resume_path: str | None = None,
    dataset: torch.utils.data.Dataset | None = None,
):
    """Main training loop.

    Accepts either train_data (list of TokenizedStructure) or a pre-built dataset.
    If dataset is provided, train_data is ignored.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = config.get_torch_dtype()
    rank = 0

    # DDP setup
    if config.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    model = model.to(device)

    if config.distributed:
        model = DDP(model, device_ids=[device])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # EMA
    base_model = model.module if isinstance(model, DDP) else model
    ema = EMAModel(base_model, decay=config.ema_decay)

    # Stage config
    stage_config = StageConfig()

    # Resume
    start_step = 0
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer, ema)

    # DataLoader
    if dataset is None:
        dataset = HelicoDataset(
            structures=train_data,
            crop_size=config.crop_size,
            msa_features=msa_features,
        )

    sampler = DistributedSampler(dataset) if config.distributed else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Training
    model.train()
    step = start_step
    epoch = 0
    running_loss = 0.0
    tokens_processed = 0
    t_start = time.time()

    while step < config.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            if step >= config.max_steps:
                break

            # Update stage-specific settings
            stage = stage_config.get_stage(step)
            current_lr = get_lr(step, config, stage.get("lr"))
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Update crop size if changed
            if stage.get("crop_size") != dataset.crop_size:
                dataset.crop_size = stage["crop_size"]

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model(batch)
                loss = outputs["diffusion_loss"]

                if "distogram_loss" in outputs:
                    loss = loss + 0.1 * outputs["distogram_loss"]

                loss = loss / config.grad_accum_steps

            # Backward
            loss.backward()

            # Gradient accumulation
            if (step + 1) % config.grad_accum_steps == 0:
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(base_model)

            # Logging
            running_loss += loss.item() * config.grad_accum_steps
            tokens_processed += batch["n_tokens"].sum().item()

            if step % config.log_every == 0 and rank == 0:
                elapsed = time.time() - t_start
                avg_loss = running_loss / max(1, config.log_every)
                throughput = tokens_processed / max(1e-6, elapsed)
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Stage: {stage['name']} | Tokens/s: {throughput:.0f}"
                )
                running_loss = 0.0
                tokens_processed = 0
                t_start = time.time()

            # Checkpointing
            if step % config.save_every == 0 and step > 0 and rank == 0:
                save_checkpoint(model, optimizer, step, config, ema)

            step += 1

        epoch += 1

    # Final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, step, config, ema, path=os.path.join(config.checkpoint_dir, "final.pt"))

    if config.distributed:
        dist.destroy_process_group()


# ============================================================================
# Inference
# ============================================================================

def run_inference(
    model: Helico,
    batch: dict[str, torch.Tensor],
    n_samples: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Run inference on a batch.

    Returns predicted coordinates, confidence scores, etc.
    """
    model.eval()
    model = model.to(device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        results = model.predict(batch, n_samples=n_samples)

    return results


def coords_to_pdb(
    coords: torch.Tensor,
    plddt: torch.Tensor,
    tokenized: TokenizedStructure,
) -> str:
    """Convert predicted coordinates to PDB format string.

    Args:
        coords: (N_atoms, 3) predicted atomic coordinates
        plddt: (N_atoms,) per-atom pLDDT scores (0-100 scale) for B-factor column
        tokenized: TokenizedStructure with real atom names, residue names, chain IDs
    """
    lines = []
    atom_serial = 0
    coords_np = coords.cpu().float().numpy()
    plddt_np = plddt.cpu().float().numpy()

    prev_chain_id = None
    res_serial = 0

    for tok_idx, token in enumerate(tokenized.tokens):
        chain_id = tokenized.chain_ids[tok_idx]
        if chain_id != prev_chain_id:
            if prev_chain_id is not None:
                # TER record between chains
                lines.append(
                    f"TER   {atom_serial + 1:5d}      "
                    f"{tokenized.tokens[tok_idx - 1].res_name:>3s} "
                    f"{prev_chain_id:1s}{res_serial:4d}"
                )
            prev_chain_id = chain_id
            res_serial = 0

        res_serial += 1
        res_name = token.res_name if token.res_name else "UNK"

        for ai, atom_name in enumerate(token.atom_names):
            if atom_serial >= len(coords_np):
                break

            x, y, z = coords_np[atom_serial]
            bfactor = float(plddt_np[atom_serial]) if atom_serial < len(plddt_np) else 0.0
            element = token.atom_elements[ai] if ai < len(token.atom_elements) else "  "

            # PDB atom name formatting: 4-char names left-justified, <=3-char right-justified
            if len(atom_name) == 4:
                atom_name_fmt = atom_name
            else:
                atom_name_fmt = f" {atom_name:<3s}"

            # Element right-justified in columns 77-78
            element_fmt = f"{element:>2s}"

            lines.append(
                f"ATOM  {(atom_serial + 1) % 100000:5d} {atom_name_fmt:4s} "
                f"{res_name:>3s} {chain_id:1s}{res_serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}          {element_fmt:2s}  "
            )
            atom_serial += 1

    lines.append("END")
    return "\n".join(lines)


# ============================================================================
# Entry Points
# ============================================================================

def main():
    """Training entry point."""
    parser = argparse.ArgumentParser(description="Helico Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--n-blocks", type=int, default=48, help="Number of Pairformer blocks")
    parser.add_argument("--n-diffusion-token-blocks", type=int, default=24, help="Number of diffusion token transformer blocks")
    parser.add_argument("--crop-size", type=int, default=384, help="Initial crop size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--distributed", action="store_true", help="Use DDP")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")

    # Real data args
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.json")
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to processed data directory")
    parser.add_argument("--val-date-cutoff", type=str, default="2022-01-01", help="Date cutoff for train/val split")
    parser.add_argument("--msa-dir", type=str, default=None, help="Path to extracted MSA directory")

    args = parser.parse_args()

    # Build configs
    train_config = TrainConfig(
        n_pairformer_blocks=args.n_blocks,
        n_diffusion_token_blocks=args.n_diffusion_token_blocks,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        distributed=args.distributed,
    )

    model_config = HelicoConfig(
        n_pairformer_blocks=args.n_blocks,
        n_diffusion_token_blocks=args.n_diffusion_token_blocks,
    )

    model = Helico(model_config)

    # Data
    if args.synthetic:
        structures = [tokenize_structure(make_synthetic_structure(n_residues=50)) for _ in range(10)]
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        train(model, structures, train_config, model_config, resume_path=args.resume)
    else:
        # Load real data
        if args.processed_dir:
            processed_dir = Path(args.processed_dir)
        elif PROCESSED_DIR is not None:
            processed_dir = PROCESSED_DIR
        else:
            logger.error("Must set --processed-dir or HELICO_PROCESSED_DIR env var")
            return
        manifest_path = Path(args.manifest) if args.manifest else processed_dir / "manifest.json"

        if not manifest_path.exists():
            logger.error(f"Manifest not found at {manifest_path}. Run helico-preprocess first.")
            return

        logger.info(f"Loading manifest from {manifest_path}...")
        manifest = load_manifest(manifest_path)
        logger.info(f"Manifest has {len(manifest)} structures")

        # Load MSA tar indices if available
        msa_tar_indices: list[TarIndex] = []
        for idx_name in ["rcsb_msa_index.pkl", "rcsb_raw_msa_index.pkl",
                         "openfold_msa_index.pkl", "openfold_raw_msa_index.pkl"]:
            idx_path = processed_dir / idx_name
            if idx_path.exists():
                logger.info(f"Loading MSA tar index from {idx_path}...")
                msa_tar_indices.append(load_tar_index(idx_path))

        msa_dir = Path(args.msa_dir) if args.msa_dir else None

        # Create training dataset with date-based filter
        cutoff = args.val_date_cutoff
        train_dataset = LazyHelicoDataset(
            manifest=manifest,
            processed_dir=processed_dir,
            crop_size=train_config.crop_size,
            msa_tar_indices=msa_tar_indices,
            msa_dir=msa_dir,
            filter_fn=lambda m: m.release_date < cutoff if m.release_date else True,
        )
        logger.info(f"Training dataset: {len(train_dataset)} structures (release_date < {cutoff})")

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        train(model, config=train_config, model_config=model_config,
              resume_path=args.resume, dataset=train_dataset)


def infer_main():
    """Inference entry point."""
    parser = argparse.ArgumentParser(description="Helico Inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Helico checkpoint")
    parser.add_argument("--protenix", type=str, default=None, help="Path to Protenix checkpoint (.pt)")
    parser.add_argument("--input", type=str, default=None, help="Path to input mmCIF file")
    parser.add_argument("--sequences", type=str, default=None, help="Comma-separated chain:seq pairs, e.g. 'A:MKFLILF,B:ACDEF'")
    parser.add_argument("--yaml", type=str, default=None, help="Path to YAML input file (Boltz2-style)")
    parser.add_argument("--ccd", type=str, default=None, help="Path to CCD cache pickle (optional, falls back to env vars)")
    parser.add_argument("--output", type=str, default="output.pdb", help="Output PDB file")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    if args.checkpoint is None and args.protenix is None:
        parser.error("Must specify either --checkpoint or --protenix")

    if args.input is None and args.sequences is None and args.yaml is None:
        parser.error("Must specify at least one of --input, --sequences, or --yaml")

    # Load model
    if args.protenix is not None:
        from collections import OrderedDict
        from scripts.load_protenix import load_protenix_state_dict
        config = HelicoConfig()  # default = Protenix dimensions
        model = Helico(config)
        ckpt = torch.load(args.protenix, map_location="cpu", weights_only=False)
        ptx_sd = ckpt["model"]
        ptx_sd = OrderedDict((k.removeprefix("module."), v) for k, v in ptx_sd.items())
        stats = load_protenix_state_dict(ptx_sd, model)
        logger.info(f"Loaded Protenix checkpoint: {stats['n_transferred']} params transferred")
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = HelicoConfig(**{k: v for k, v in state.get("config", {}).items() if hasattr(HelicoConfig, k)})
        model = Helico(config)
        model.load_state_dict(state["model_state_dict"])

    # Load CCD
    ccd_cache = Path(args.ccd) if args.ccd else None
    ccd = parse_ccd(cache_path=ccd_cache)
    logger.info(f"CCD loaded with {len(ccd)} components")

    # Build TokenizedStructure based on input mode
    if args.sequences:
        chains = parse_sequences_arg(args.sequences)
        tokenized = tokenize_sequences(chains, ccd)
    elif args.yaml:
        chains = parse_input_yaml(args.yaml)
        tokenized = tokenize_sequences(chains, ccd)
    else:
        structure = parse_mmcif(args.input)
        if structure is None:
            logger.error("Failed to parse input structure")
            return
        tokenized = tokenize_structure(structure, ccd=ccd)

    logger.info(f"Tokenized: {tokenized.n_tokens} tokens, {tokenized.n_atoms} atoms")
    features = tokenized.to_features()

    # Add batch dimension
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    # Wrap scalars
    for key in ["n_tokens", "n_atoms"]:
        if key in batch and not isinstance(batch[key], torch.Tensor):
            batch[key] = torch.tensor([batch[key]])

    # Add masks
    n_tok = features["n_tokens"]
    n_atoms = features["n_atoms"]
    batch["token_mask"] = torch.ones(1, n_tok, dtype=torch.bool)
    batch["atom_mask"] = torch.ones(1, n_atoms, dtype=torch.bool)
    # Add empty MSA features if not present
    if "msa_profile" not in batch:
        batch["msa_profile"] = torch.zeros(1, n_tok, 22)
        batch["cluster_msa"] = torch.zeros(1, 1, n_tok, dtype=torch.long)
        batch["cluster_profile"] = torch.zeros(1, 1, n_tok, 22)
        batch["has_msa"] = torch.zeros(1)

    # Run inference
    results = run_inference(model, batch, n_samples=args.n_samples)

    # Write output
    pdb_str = coords_to_pdb(results["coords"][0], results["plddt"][0], tokenized)
    with open(args.output, "w") as f:
        f.write(pdb_str)

    logger.info(f"Predicted structure written to {args.output}")
    logger.info(f"Mean pLDDT: {results['plddt'][0].mean():.1f}")
    logger.info(f"pTM: {results['ptm'][0]:.3f}")
    logger.info(f"ipTM: {results['iptm'][0]:.3f}")
    logger.info(f"Ranking score: {results['ranking_score'][0]:.3f}")


if __name__ == "__main__":
    main()
