"""Full-data, single-node DDP masked-contact fine-tuning from a frozen data lock."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from helico.contact_diffusion import absorbing_mask
from helico.contact_pilot import sampled_time
from helico.datasets import record_run_data, write_json
from helico.load_protenix import load_protenix_checkpoint
from helico.model import Helico, HelicoConfig
from helico.model.diffusion import _centre_random_augmentation
from helico.protenix_data import load_datasets, SeededCrops, batch_example, adapt_example


def first(items):
    return items[0]


def weighted_draws(weights, steps, accumulation, world, rank, start, seed):
    """One global draw stream; resuming consumes precisely its remaining suffix."""
    draws = torch.multinomial(torch.as_tensor(weights, dtype=torch.double),
                             steps * accumulation * world, replacement=True,
                             generator=torch.Generator().manual_seed(seed))
    return [(int(draws[i]), i) for i in range(start * accumulation * world + rank,
                                             len(draws), world)]


def structure_losses(result, batch, sigma_data=16.):
    """Use the pinned upstream aligned, molecule-weighted MSE and sparse smooth lDDT."""
    from protenix.model.loss import MSELoss, SmoothLDDTLoss
    prediction = result["x_denoised"].float()
    truth = batch["atom_coords"][0].float()
    mask = batch["coordinate_mask"][0]
    sigma = result["sigma"].float()
    scale = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
    with torch.autocast("cuda", enabled=False):
        mse = MSELoss()(prediction, truth, mask,
            batch["is_dna"][0], batch["is_rna"][0], batch["is_ligand"][0], scale)
        with torch.no_grad():
            distance = torch.cdist(truth, truth)
            radius = torch.where(batch["is_dna"][0] | batch["is_rna"][0], 30., 15.)
            close = (distance < radius[:, None]) & mask[:, None] & mask[None, :]
            close.fill_diagonal_(False)
        smooth = (SmoothLDDTLoss().sparse_forward(prediction, truth, close, diffusion_chunk_size=1)
                  if close.any() else prediction.sum() * 0)
    return mse, smooth


def make_batch(features, draw, seed, device, *, validation=False, msa_depth=512):
    rng = random.Random(seed + draw * 37)
    t = 1000 if validation else sampled_time(rng)
    state = absorbing_mask(features["contact_target"], features["contact_valid"], t,
                           generator=torch.Generator().manual_seed(seed + draw))
    if len(features["msa"]) > msa_depth:
        generator = torch.Generator().manual_seed(seed + draw)
        rows = torch.cat([torch.zeros(1, dtype=torch.long),
            torch.randperm(len(features["msa"]) - 1, generator=generator)[:msa_depth - 1] + 1])
        features = {**features, **{k: features[k][rows]
            for k in ("msa", "has_deletion", "deletion_value")}}
    batch = batch_example(features, state, device)
    batch["atom_coords"] = _centre_random_augmentation(batch["atom_coords"], batch["coordinate_mask"])
    batch["atom_coords"] *= batch["coordinate_mask"].unsqueeze(-1)
    return batch, (True if validation else rng.random() >= .2), t


@torch.no_grad()
def validate(model, datasets, count, rank, world, seed):
    # Fixed validation crops/seeds; FoldBench remains untouched during training.
    model.eval(); model.config.use_msa = True
    values = torch.zeros(5, device="cuda", dtype=torch.float64)
    for dataset in datasets.values():
        for i in range(rank, min(count, len(dataset)), world):
            torch.manual_seed(seed + i); np.random.seed(seed + i); random.seed(seed + i)
            feature = adapt_example(dataset[i])
            batch, _, _ = make_batch(feature, i, seed, "cuda", validation=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch, compute_confidence=False)
            mse, smooth = structure_losses(out, batch)
            values += torch.stack([out["contact_loss"], out["distogram_loss"], mse, smooth,
                                   torch.ones((), device="cuda")]).double()
    dist.all_reduce(values)
    metrics = dict(zip(["contact_bce", "distogram", "mse", "smooth_lddt"],
                       (values[:4] / values[4].clamp_min(1)).tolist()))
    metrics["examples"] = int(values[4])
    model.train()
    return metrics


def save_checkpoint(output, model, optimizer, ema, step, config, lock, *, final=False):
    from helico.coreweave_worker import filesystem
    path = output / ("final.pt" if final else f"step-{step:06d}.pt")
    tmp = path.with_suffix(".pending")
    torch.save({"model_state_dict": model.state_dict(), "ema_state_dict": ema,
                "model_config": {**asdict(model.config), "use_msa": True}, "optimizer_state_dict": optimizer.state_dict(),
                "step": step, "training_config": config, "data_lock": lock,
                "data_lock_sha256": lock["lock_sha256"], "git_sha": os.environ["HELICO_CODE_SHA"],
                "sampler": "global-weighted-draw-seed-v1"}, tmp)
    tmp.replace(path)
    fs = filesystem()
    uri = config["output_uri"] + "/" + path.name
    fs.put_file(str(path), uri)
    if fs.size(uri) != path.stat().st_size:
        raise RuntimeError("Checkpoint upload size mismatch")
    with fs.open(config["output_uri"] + "/latest.json", "w") as f:
        json.dump({"step": step, "checkpoint": uri, "data_lock_sha256": lock["lock_sha256"]}, f)
    # Only local scratch is reclaimed; all uploaded snapshots remain durable.
    for old in output.glob("step-*.pt"):
        if old != path:
            old.unlink()
    print(json.dumps({"checkpoint": uri, "step": step, "final": final}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    lock = json.loads(Path(cfg["data_lock"]).read_text())
    bundle = json.loads(args.bundle.read_text())
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(2); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    if world != cfg["gpus"]:
        raise ValueError("World size changed: global batch/draw order would change")
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        record_run_data(lock, args.output)
    torch.manual_seed(cfg["seed"])
    model_cfg = HelicoConfig(predict_contacts=True, n_diffusion_samples=cfg["diffusion_samples"],
                            msa_sample_cutoff=cfg["msa_depth"], msa_sample_min_train=32,
                            msa_sample_min_eval=cfg["msa_depth"])
    model = Helico(model_cfg)
    stats = load_protenix_checkpoint(args.checkpoint, model)
    if stats["shape_mismatches"]:
        raise ValueError(f"Checkpoint mismatch: {stats['shape_mismatches']}")
    torch.nn.init.zeros_(model.contact_head.weight)
    torch.nn.init.constant_(model.contact_head.bias, -2.)
    for name, parameter in model.named_parameters():
        if name.startswith(("confidence_head.", "template_embedder.")):
            parameter.requires_grad_(False)
    model.cuda().train()
    new_parameters, pretrained_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = new_parameters if name.startswith(("contact_head.", "linear_contact.")) else pretrained_parameters
            group.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": pretrained_parameters, "lr_scale": 1.},
        {"params": new_parameters, "lr_scale": cfg["contact_lr"] / cfg["lr"]}],
        lr=cfg["lr"], betas=(.9, .95), weight_decay=.01)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["data_lock_sha256"] != lock["lock_sha256"] or checkpoint["training_config"] != cfg:
            raise ValueError("Resume configuration/data mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ema = {k: v.cuda() for k, v in checkpoint["ema_state_dict"].items()}
        start = checkpoint["step"]
        del checkpoint
    train_data, validation = load_datasets(lock, bundle, cfg["crop_size"], args.output / "data-errors")
    # Validation uses fixed 384-token crops for online diagnostics. The original
    # 384-entry validation membership is unchanged; final FoldBench uses full inputs.
    for dataset in validation.values():
        dataset.cropping_configs["crop_size"] = cfg["crop_size"]
    draws = weighted_draws(train_data.merged_datapoint_weights, cfg["steps"], cfg["accumulation"],
                          world, rank, start, cfg["seed"])
    loader = DataLoader(SeededCrops(train_data, cfg["seed"]), sampler=draws,
                        batch_size=1, collate_fn=first, num_workers=cfg["workers"],
                        persistent_workers=cfg["workers"] > 0, pin_memory=True)
    iterator = iter(loader)
    wrapped = DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                      find_unused_parameters=True)
    run = None
    if rank == 0:
        import wandb
        run = wandb.init(project="helico", entity="timodonnell", name=cfg["run_name"],
                         id=cfg["run_name"], resume="allow", config={**cfg,
                         "data_lock_sha256": lock["lock_sha256"], "git_sha": os.environ["HELICO_CODE_SHA"],
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "training_records": len(train_data), "data_lock": lock})
        print(json.dumps({"wandb": run.url, "records": len(train_data), "start_step": start}), flush=True)
    began = time.monotonic()
    step = start
    for step in range(start + 1, cfg["steps"] + 1):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        lr = cfg["lr"] * min(1., step / cfg["warmup_steps"]) * (
            .1 + .9 * .5 * (1 + math.cos(math.pi * max(0, step - cfg["warmup_steps"]) /
                                       max(1, cfg["steps"] - cfg["warmup_steps"]))))
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        totals = torch.zeros(8, device="cuda", dtype=torch.float64)
        for micro in range(cfg["accumulation"]):
            feature, pdb_id, draw = next(iterator)
            torch.manual_seed(cfg["seed"] + draw); torch.cuda.manual_seed_all(cfg["seed"] + draw)
            batch, use_msa, t = make_batch(feature, draw, cfg["seed"], "cuda", msa_depth=cfg["msa_depth"])
            model.config.use_msa = use_msa
            sync = nullcontext() if micro == cfg["accumulation"] - 1 else wrapped.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = wrapped(batch, compute_confidence=False)
                mse, smooth = structure_losses(out, batch)
                loss = 4 * (mse + smooth) + .03 * out["distogram_loss"] + .1 * out["contact_loss"] + .01 * out["contact_observed_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step={step}, PDB={pdb_id}, draw={draw}")
                (loss / cfg["accumulation"]).backward()
            totals += torch.tensor([loss.detach(), mse.detach(), smooth.detach(),
                out["distogram_loss"].detach(), out["contact_loss"].detach(),
                out["contact_observed_loss"].detach(), float(use_msa), t], device="cuda")
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.values()), list(model.state_dict().values()), .001)
        dist.all_reduce(totals); totals /= world * cfg["accumulation"]
        if rank == 0 and (step <= 5 or step % 10 == 0):
            metrics = dict(zip(["train/loss", "train/mse", "train/smooth_lddt", "train/distogram",
                "train/contact_bce", "train/observed_bce", "train/msa_fraction", "train/contact_time"], totals.tolist()))
            metrics.update({"train/lr": lr, "train/grad_norm": float(grad),
                            "train/step_seconds": time.monotonic() - tick,
                            "train/examples_seen": step * world * cfg["accumulation"],
                            "train/gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(json.dumps({"step": step, **metrics}), flush=True); run.log(metrics, step=step)
        if step % cfg["validate_every"] == 0:
            metrics = validate(model, validation, cfg["validation_samples"], rank, world, cfg["seed"] + 10000000)
            if rank == 0:
                run.log({f"validation/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps({"step": step, "validation": metrics}), flush=True)
        if step == 1 or step % cfg["save_every"] == 0:
            if rank == 0:
                save_checkpoint(args.output, model, optimizer, ema, step, cfg, lock)
            dist.barrier()
        stop = torch.tensor(int(time.monotonic() - began > cfg["train_seconds"]), device="cuda")
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            break
    if rank == 0:
        save_checkpoint(args.output, model, optimizer, ema, step, cfg, lock, final=True)
        write_json(args.output / "result.json", {"step": step, "requested_steps": cfg["steps"],
            "finished_steps": step == cfg["steps"], "wall_seconds": time.monotonic() - began})
        run.finish()
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
