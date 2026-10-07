"""Bounded paired fine-tuning and structural-control pilot on preallocated GPUs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F

from helico.contact_diffusion import absorbing_mask, heavy_atom_contacts
from helico.data import collate_fn
from helico.eval_metrics import hard_lddt
from helico.load_protenix import load_protenix_checkpoint
from helico.model import Helico, HelicoConfig
from helico.model.diffusion import _centre_random_augmentation


def batch_for(features, state, device):
    """Pad token dimensions to 32 for cuDNN's fused attention kernels."""
    batch = collate_fn([{**features, "contact_state": state}])
    batch["contact_target"] = features["contact_target"][None]
    batch["contact_valid"] = features["contact_valid"][None]
    n = features["n_tokens"]; padding = (-n) % 32
    token_keys = {"token_types", "restype", "chain_indices", "res_indices", "rel_pos",
                  "token_index", "entity_id", "sym_id", "atoms_per_token", "rep_atom_idx",
                  "has_frame", "token_mask", "deletion_mean"}
    pair_keys = {"chain_same", "token_bonds", "contact_state", "contact_target", "contact_valid"}
    for key, value in batch.items():
        if key in token_keys:
            value = F.pad(value, (0, padding))
        elif key in pair_keys:
            value = F.pad(value, (0, padding, 0, padding))
        elif key == "msa_profile":
            value = F.pad(value, (0, 0, 0, padding))
        elif key in {"cluster_msa", "cluster_deletion_mean"}:
            value = F.pad(value, (0, padding))
        elif key == "cluster_profile":
            value = F.pad(value, (0, 0, 0, padding))
        if key == "cluster_msa":
            value = value.long()
        batch[key] = value.to(device)
    return batch


def sampled_time(rng):
    """Oversample endpoints and sparse times; revelation remains value-blind."""
    u = rng.random()
    if u < .2: return 1000
    if u < .3: return 0
    if u < .55: return rng.randint(995, 999)
    return rng.randint(1, 999)


def average_precision(labels, probs):
    labels = labels.astype(bool)
    if not labels.any(): return float("nan")
    order = np.argsort(-probs, kind="stable")
    y = labels[order]
    # Group equal scores (including the constant untrained head) at a single
    # threshold, so AP does not depend on token order within ties.
    ends = np.r_[np.flatnonzero(np.diff(probs[order])), len(y) - 1]
    tp = np.cumsum(y)[ends]
    increments = np.diff(np.r_[0, tp])
    return float(((tp / (ends + 1)) * increments).sum() / y.sum())


def measure(features, state, result, base_contacts=None):
    n = features["n_tokens"]
    coords = result["coords"][0].float().cpu()
    predicted = heavy_atom_contacts(coords, features["atom_to_token"], n)
    target = features["contact_target"]
    valid = torch.triu(features["contact_valid"], 1)
    logits = result["contact_logits"][0, :n, :n].float().cpu()
    probs = logits.sigmoid()
    p = features["protein_token"]
    groups = {"all": valid,
              "pp": valid & ~features["chain_same"].bool() & p[:, None] & p[None, :],
              "pl": valid & (p[:, None] ^ p[None, :])}
    metrics = {}
    for group, mask in groups.items():
        tp = int((predicted & target & mask).sum())
        npred = int((predicted & mask).sum()); ntrue = int((target & mask).sum())
        metrics[f"{group}_true"] = ntrue
        metrics[f"{group}_recall"] = tp / ntrue if ntrue else float("nan")
        metrics[f"{group}_precision"] = tp / npred if npred else float("nan")
        hidden = mask & (state == 0)
        metrics[f"{group}_masked_pairs"] = int(hidden.sum())
        metrics[f"{group}_bce"] = float(F.binary_cross_entropy_with_logits(logits[hidden], target[hidden].float())) if hidden.any() else float("nan")
        metrics[f"{group}_ap"] = average_precision(target[hidden].numpy(), probs[hidden].numpy()) if hidden.any() else float("nan")
        metrics[f"{group}_brier"] = float(((probs[hidden] - target[hidden].float()) ** 2).mean()) if hidden.any() else float("nan")
    requested = (state == 2) & valid
    metrics["requested_contacts"] = int(requested.sum())
    metrics["request_precision"] = float(target[requested].float().mean()) if requested.any() else float("nan")
    metrics["request_satisfaction"] = float(predicted[requested].float().mean()) if requested.any() else float("nan")
    metrics["revealed_pairs"] = int(((state != 0) & valid).sum())
    metrics["lddt"] = float(hard_lddt(coords[None], features["atom_coords"][None]))
    if base_contacts is not None:
        metrics["changed_contacts"] = int(((predicted != base_contacts) & valid).sum())
    else:
        metrics["changed_contacts"] = 0
    return metrics, predicted


@torch.no_grad()
def evaluate(model, examples, output, rank, world, arm, phase="final"):
    model.eval(); model.config.use_msa = True
    records = []
    predictions = output / "predictions"; predictions.mkdir(exist_ok=True)
    for idx, example in enumerate(examples):
        if idx % world != rank: continue
        features = example["features"]
        n = features["n_tokens"]
        for seed in (101, 202):
            empty = torch.zeros_like(features["contact_target"], dtype=torch.uint8)
            base_contacts = None; base_probs = None
            modes = ("none",) if phase == "initial" else ("none", "sparse", "full", "search4", "oracle4")
            for mode in modes:
                state = empty.clone()
                if mode == "sparse":
                    state = absorbing_mask(features["contact_target"], features["contact_valid"], 995,
                                           generator=torch.Generator().manual_seed(seed))
                elif mode == "full":
                    state = absorbing_mask(features["contact_target"], features["contact_valid"], 0)
                elif mode in {"search4", "oracle4"}:
                    candidate = torch.triu(features["contact_valid"] & ~base_contacts, 1)
                    inter = ~features["chain_same"].bool()
                    if (candidate & inter).any(): candidate &= inter
                    if mode == "oracle4": candidate &= features["contact_target"]
                    ij = candidate.nonzero()
                    if len(ij):
                        selected = ij[base_probs[candidate].argsort(descending=True)[:4]]
                        state[selected[:, 0], selected[:, 1]] = 2
                        state = torch.maximum(state, state.T)
                batch = batch_for(features, state, "cuda")
                # Common random numbers isolate the conditioning intervention.
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.predict(batch, n_samples=1, n_cycles=1)
                metrics, contacts = measure(features, state, result, base_contacts)
                if mode == "none":
                    base_contacts = contacts
                    base_probs = result["contact_probs"][0, :n, :n].float().cpu()
                record = dict(arm=arm, phase=phase, pdb_id=example["id"], kind=example["kind"],
                              seed=seed, mode=mode, **metrics)
                records.append(record)
                torch.save({"coords": result["coords"].float().cpu(), "contact_state": state,
                            "contact_probs": result["contact_probs"][0, :n, :n].float().cpu()},
                           predictions / f"{phase}-{example['id']}-{seed}-{mode}.pt")
                print(json.dumps({k: record[k] for k in ("arm", "phase", "pdb_id", "mode", "lddt", "all_recall", "request_satisfaction")}), flush=True)
                write_csv(output / f"{phase}-rank{rank}.csv", records)
    return records


def write_csv(path, records):
    if not records: return
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--arm", choices=["masked", "unknown"], required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--train-seconds", type=int, default=3300)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(2); torch.cuda.set_device(local_rank)
    if world > 1: dist.init_process_group("nccl")
    output = Path(os.environ["HELICO_TRAIN_OUTPUT"]); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(20); np.random.seed(20); random.seed(20)
    config = HelicoConfig(predict_contacts=True, n_diffusion_samples=1,
                          n_diffusion_steps=50, msa_sample_cutoff=64,
                          msa_sample_min_eval=64, msa_sample_min_train=32)
    model = Helico(config)
    stats = load_protenix_checkpoint(args.assets / "protenix_base_default_v1.0.0.pt", model)
    if stats["shape_mismatches"]:
        raise RuntimeError(f"Checkpoint shape mismatch: {stats['shape_mismatches']}")
    torch.nn.init.zeros_(model.contact_head.weight)
    torch.nn.init.constant_(model.contact_head.bias, -2.0)
    # Confidence and dummy-template weights are frozen in both arms.
    for name, parameter in model.named_parameters():
        if name.startswith(("confidence_head.", "template_embedder.")):
            parameter.requires_grad_(False)
    model.cuda()
    dataset = torch.load(args.assets / "pilot.pt", weights_only=False)
    if args.smoke:
        torch.autograd.set_detect_anomaly(True)
        feature = max(dataset["train"], key=lambda x: x["features"]["n_tokens"])["features"]
        state = absorbing_mask(feature["contact_target"], feature["contact_valid"], 500)
        batch = batch_for(feature, state, "cuda")
        model.train()
        def check_finite(name):
            def check(module, inputs, result):
                tensors = result.values() if isinstance(result, dict) else result if isinstance(result, tuple) else [result]
                for value in tensors:
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite forward at {name}: {tuple(value.shape)}")
            return check
        handles = [module.register_forward_hook(check_finite(name))
                   for name, module in model.named_modules() if not list(module.children())]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, compute_confidence=False)
            loss = out["diffusion_loss"] + .1 * out["distogram_loss"] + .1 * out["contact_loss"]
        print({key: float(value.detach()) for key, value in out.items() if key.endswith("loss")}, flush=True)
        loss.backward()
        for handle in handles: handle.remove()
        if not torch.isfinite(loss) or not torch.isfinite(model.linear_contact.weight.grad).all():
            raise FloatingPointError("Nonfinite smoke loss or contact gradient")
        print(dict(smoke_loss=float(loss.detach()), max_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                   projection_grad=float(model.linear_contact.weight.grad.norm())), flush=True)
        return
    import wandb
    run = None
    if rank == 0:
        run = wandb.init(entity="timodonnell", project="helico", name=f"exp20-clean-contact-{args.arm}",
                         config={**asdict(config), "arm": args.arm, "steps": args.steps,
                                 "world_size": world, "loss": "diffusion + .1 distogram + .1 masked BCE + .01 observed BCE",
                                 "base_lr": 1e-5, "new_lr": 1e-3,
                                 "CUEQ_TRIMUL_FALLBACK_THRESHOLD": os.environ.get("CUEQ_TRIMUL_FALLBACK_THRESHOLD"),
                                 "CUEQ_TRIATTN_FALLBACK_THRESHOLD": os.environ.get("CUEQ_TRIATTN_FALLBACK_THRESHOLD"), "contact_threshold_angstrom": 5,
                                 "data_sha256": hashlib.sha256((args.assets / "pilot.pt").read_bytes()).hexdigest()},
                         dir=str(output), tags=["exp20", "clean-contact-pilot"])
        (output / "wandb.json").write_text(json.dumps({"url": run.url, "id": run.id}, indent=2))
    evaluate(model, dataset["validation"], output, rank, world, args.arm, "initial")
    if world > 1: dist.barrier()
    wrapped = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    groups = []
    for new in (False, True):
        params = [p for name, p in model.named_parameters() if p.requires_grad and
                  name.startswith(("linear_contact.", "contact_head.")) == new]
        groups.append({"params": params, "lr": 1e-3 if new else 1e-5, "base_lr": 1e-3 if new else 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    logs = []; training_start = time.monotonic()
    for step in range(args.steps):
        stop = torch.tensor(int(time.monotonic() - training_start >= args.train_seconds), device="cuda")
        if world > 1: dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item(): break
        rng = random.Random(200000 + step * world + rank)
        feature = dataset["train"][rng.randrange(len(dataset["train"]))]["features"]
        t = sampled_time(rng)
        if args.arm == "unknown": t = 1000
        state = absorbing_mask(feature["contact_target"], feature["contact_valid"], t,
                               generator=torch.Generator().manual_seed(300000 + step * world + rank))
        batch = batch_for(feature, state, "cuda")
        model.config.use_msa = random.Random(400000 + step).random() >= .1
        torch.manual_seed(500000 + step * world + rank)
        torch.cuda.manual_seed_all(500000 + step * world + rank)
        batch["atom_coords"] = _centre_random_augmentation(batch["atom_coords"], batch["atom_mask"])
        wrapped.train(); optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * min(1., (step + 1) / 20)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = wrapped(batch, compute_confidence=False)
            loss = out["diffusion_loss"] + .1 * out["distogram_loss"] + .1 * out["contact_loss"] + .01 * out["contact_observed_loss"]
        if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        values = torch.stack([loss.detach(), out["diffusion_loss"].detach(), out["distogram_loss"].detach(),
                              out["contact_loss"].detach(), out["contact_observed_loss"].detach(), norm.detach()]).float()
        if world > 1: dist.all_reduce(values); values /= world
        if rank == 0:
            metrics = dict(zip(("loss", "diffusion_loss", "distogram_loss", "contact_loss", "contact_observed_loss", "grad_norm"), values.cpu().tolist()))
            metrics.update(step=step + 1, train_seconds=time.monotonic()-training_start,
                           projection_norm=float(model.linear_contact.weight.norm()), msa_on=int(model.config.use_msa))
            logs.append(metrics); run.log(metrics, step=step + 1)
            if step % 10 == 0:
                print(json.dumps(metrics), flush=True); write_csv(output / "training.csv", logs)
    actual_steps = step + 1 if not stop.item() else step
    model.config.use_msa = True
    if rank == 0:
        write_csv(output / "training.csv", logs)
        torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(config),
                    "step": actual_steps, "arm": args.arm}, output / "final.pt")
    if world > 1: dist.barrier()
    evaluate(model, dataset["validation"], output, rank, world, args.arm)
    if world > 1: dist.barrier()
    if rank == 0:
        records = []
        for phase in ("initial", "final"):
            for path in sorted(output.glob(f"{phase}-rank*.csv")):
                with path.open() as f: records.extend(csv.DictReader(f))
        write_csv(output / "evaluation.csv", records)
        summary = {"steps": actual_steps, "wall_seconds": time.monotonic()-started,
                   "gpu_hours": (time.monotonic()-started)*world/3600,
                   "train_examples": len(dataset["train"]), "validation_examples": len(dataset["validation"]),
                   "wandb_url": run.url}
        (output / "result.json").write_text(json.dumps(summary, indent=2))
        run.summary.update(summary); run.finish()
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
