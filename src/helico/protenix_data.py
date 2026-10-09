"""Upstream Protenix complex features -> Helico, without re-tokenizing structures."""
from __future__ import annotations

from copy import deepcopy
import faulthandler
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from helico.contact_diffusion import heavy_atom_contacts
from helico.datasets import protenix_config


def load_datasets(lock, bundle, crop_size, error_dir):
    overrides = protenix_config(lock, bundle, crop_size)
    # CCD paths are captured at upstream module import, so set this first.
    os.environ["PROTENIX_ROOT_DIR"] = overrides["PROTENIX_ROOT_DIR"]
    from configs.configs_base import configs as base
    from configs.configs_data import data_configs
    from ml_collections import ConfigDict
    from protenix.config.config import ConfigManager
    from protenix.data.pipeline.dataset import get_datasets

    raw = deepcopy(base)
    raw["data"] = deepcopy(data_configs)
    defaults = ConfigManager(raw, fill_required_with_null=True).default_configs
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    merge(defaults["data"], overrides["data"])
    # Helico currently implements the checkpoint's dummy-template path.
    # Explicitly disable retrieval instead of silently discarding templates.
    defaults["data"]["template"]["enable_prot_template"] = False
    return get_datasets(ConfigDict(defaults), error_dir=str(error_dir))


def adapt_example(example):
    """Keep input/reference masks distinct from observed-coordinate loss masks."""
    f, labels = example["input_feature_dict"], example["label_dict"]
    index = f["atom_to_token_idx"].long()
    n, a = len(f["token_index"]), len(index)
    observed = labels["coordinate_mask"].bool()
    # Upstream ref_element is zero-based: hydrogen=0, carbon=5, etc.
    element = f["ref_element"].argmax(-1)
    heavy = element != 0
    target = heavy_atom_contacts(labels["coordinate"], index, n,
                                 atom_mask=observed & heavy)
    missing = torch.zeros(n, dtype=torch.long).scatter_add_(0, index, (~observed & heavy).long())
    total = torch.bincount(index[heavy], minlength=n)
    complete = (missing == 0) & (total > 0)
    protein = torch.zeros(n, dtype=torch.long).scatter_add_(0, index, f["is_protein"].long()) > 0
    same = f["asym_id"][:, None] == f["asym_id"][None, :]
    separation = (f["residue_index"][:, None] - f["residue_index"][None, :]).abs()
    eligible = (protein[:, None] | protein[None, :]) & ~(
        same & protein[:, None] & protein[None, :] & (separation < 6))
    eligible.fill_diagonal_(False)
    valid = eligible & (target | (complete[:, None] & complete[None, :]))
    representative = f["distogram_rep_atom_mask"].bool().nonzero().flatten()
    if len(representative) != n or not torch.equal(index[representative], torch.arange(n)):
        raise ValueError("Upstream distogram representative atoms must align with tokens")
    out = {
        "token_types": f["restype"].argmax(-1), "restype": f["restype"].argmax(-1),
        "rel_pos": f["residue_index"], "token_index": f["token_index"],
        "chain_indices": f["asym_id"], "entity_id": f["entity_id"], "sym_id": f["sym_id"],
        "chain_same": same, "token_mask": torch.ones(n, dtype=torch.bool),
        "token_bonds": f["token_bonds"], "rep_atom_idx": representative,
        "atom_coords": labels["coordinate"].float(), "ref_coords": f["ref_pos"].float(),
        "atom_to_token": index, "atom_element_idx": element,
        "atom_name_chars": f["ref_atom_name_chars"].flatten(-2).float(),
        "ref_charge": f["ref_charge"].float(), "ref_mask": f["ref_mask"].bool(),
        "ref_space_uid": f["ref_space_uid"], "atom_mask": torch.ones(a, dtype=torch.bool),
        "coordinate_mask": observed, "distogram_mask": observed[representative],
        "msa": f["msa"].long(), "msa_profile": f["profile"].float(),
        "has_deletion": f["has_deletion"].float(), "deletion_value": f["deletion_value"].float(),
        "deletion_mean": f["deletion_mean"].float(),
        "contact_target": target, "contact_valid": valid,
        "is_ligand": f["is_ligand"].bool(), "is_dna": f["is_dna"].bool(),
        "is_rna": f["is_rna"].bool(),
        "n_tokens": n, "n_atoms": a,
    }
    if not torch.isfinite(out["atom_coords"][observed]).all():
        raise ValueError("Nonfinite resolved coordinates")
    out["atom_coords"][~observed] = 0
    return out


TOKEN_KEYS = {"token_types", "restype", "rel_pos", "token_index", "chain_indices",
              "entity_id", "sym_id", "token_mask", "rep_atom_idx", "distogram_mask", "deletion_mean"}
PAIR_KEYS = {"token_bonds", "chain_same", "contact_target", "contact_valid", "contact_state"}
MSA_KEYS = {"msa", "has_deletion", "deletion_value"}


def batch_example(features, state, device="cpu"):
    """One crop per device; token padding is required by fused cuDNN attention."""
    padding = (-features["n_tokens"]) % 32
    batch = {}
    for key, value in {**features, "contact_state": state}.items():
        if key in {"n_tokens", "n_atoms"}:
            batch[key] = torch.tensor([value], device=device)
            continue
        if key in TOKEN_KEYS or key in MSA_KEYS:
            value = F.pad(value, (0, padding), value=31 if key == "msa" else 0)
        elif key in PAIR_KEYS:
            value = F.pad(value, (0, padding, 0, padding))
        elif key == "msa_profile":
            value = F.pad(value, (0, 0, 0, padding))
        batch[key] = value.unsqueeze(0).to(device)
    return batch


class SeededCrops(torch.utils.data.Dataset):
    """Deterministic draw seeds make prefetch and resume independent of worker timing."""
    def __init__(self, dataset, seed, diagnostics_dir=None):
        self.dataset, self.seed = dataset, seed
        self.diagnostics_dir = diagnostics_dir
        self.trace_file = None

    def progress(self, stage, index, draw):
        if self.diagnostics_dir is None:
            return
        folder = Path(self.diagnostics_dir)
        folder.mkdir(parents=True, exist_ok=True)
        if self.trace_file is None:
            self.trace_file = (folder / f"worker-{os.getpid()}.trace").open("a")
            faulthandler.enable(file=self.trace_file)
        path = folder / f"worker-{os.getpid()}.json"
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(dict(stage=stage, index=index, draw=draw, timestamp=time.time())))
        pending.replace(path)
        if stage == "loading":
            faulthandler.dump_traceback_later(180, repeat=True, file=self.trace_file)
        else:
            faulthandler.cancel_dump_traceback_later()

    def __getitem__(self, key):
        index, draw = key
        self.progress("loading", index, draw)
        seed = (self.seed + draw * 1000003) % (2**32)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        example = self.dataset[index]
        result = adapt_example(example), str(example["basic"]["pdb_id"]), draw
        self.progress("ready", index, draw)
        return result

    def __len__(self):
        return len(self.dataset)
