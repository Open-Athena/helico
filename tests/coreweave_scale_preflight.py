"""Run inside the allocated training job while its full source data stages.

torchrun --standalone --nproc-per-node=8 tests/coreweave_scale_preflight.py FEATURES CHECKPOINT
This checks the actual full-size model/optimizer and dynamic MSA DDP branches;
it discards updates and is not a separate research/control training run.
"""
import json
import argparse
import faulthandler
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from helico.model import Helico, HelicoConfig
from helico.load_protenix import load_protenix_checkpoint
from helico.train_contacts import make_batch, structure_losses

p = argparse.ArgumentParser()
p.add_argument("features")
p.add_argument("checkpoint")
p.add_argument("--steps", type=int, default=2)
p.add_argument("--first-draw", type=int, default=0)
p.add_argument("--absolute-draw-files", action="store_true")
args = p.parse_args()
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
assert 32 % world == 0
accumulation = 32 // world
faulthandler.enable(); faulthandler.dump_traceback_later(180, repeat=True)
torch.set_num_threads(2); torch.cuda.set_device(rank)
dist.init_process_group("nccl")
torch.manual_seed(2201)
model = Helico(HelicoConfig(predict_contacts=True, n_diffusion_samples=4,
    msa_sample_cutoff=512, msa_sample_min_train=32, msa_sample_min_eval=512))
stats = load_protenix_checkpoint(args.checkpoint, model)
assert not stats["shape_mismatches"], stats
torch.nn.init.zeros_(model.contact_head.weight)
torch.nn.init.constant_(model.contact_head.bias, -2.)
for name, parameter in model.named_parameters():
    if name.startswith(("confidence_head.", "template_embedder.")):
        parameter.requires_grad_(False)
model.cuda().train()
wrapped = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
feature_path = Path(args.features)
features = None if feature_path.is_dir() else torch.load(feature_path, weights_only=False)
for step in range(args.steps):
    started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    for micro in range(accumulation):
        draw = args.first_draw + step * 32 + micro * world + rank
        if feature_path.is_dir():
            key = draw if args.absolute_draw_files else draw % 32
            record = torch.load(feature_path / f"draw-{key:02d}.pt", weights_only=False)
            features = record["feature"]
            print(json.dumps({"rank": rank, "draw": draw, "pdb_id": record.get("pdb_id"), "stage": "forward"}), flush=True)
        torch.manual_seed(2201 + draw); torch.cuda.manual_seed_all(2201 + draw)
        batch, use_msa, _ = make_batch(features, draw, 2201, "cuda")
        model.config.use_msa = use_msa if feature_path.is_dir() else (rank + micro + step) % 3 != 0
        with (nullcontext() if micro == accumulation - 1 else wrapped.no_sync()):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = wrapped(batch, compute_confidence=False)
            mse, smooth = structure_losses(result, batch)
            loss = 4*(mse+smooth) + .03*result["distogram_loss"] + .1*result["contact_loss"] + .01*result["contact_observed_loss"]
            assert torch.isfinite(loss), loss
            print(json.dumps({"rank": rank, "draw": draw, "stage": "backward"}), flush=True)
            (loss/accumulation).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    print(json.dumps({"rank": rank, "preflight_step": step + 1, "loss": float(loss.detach()),
        "grad_norm": float(norm), "seconds": time.monotonic()-started,
        "gpu_gb": torch.cuda.max_memory_allocated()/1e9}), flush=True)
    faulthandler.dump_traceback_later(180, repeat=True)
dist.barrier(); dist.destroy_process_group()
faulthandler.cancel_dump_traceback_later()
