"""Explicit Protenix dataset subclass for the pinned feature pipeline.

Import only after setting PROTENIX_ROOT_DIR. Dataset construction follows
Protenix 85767b811c40ed46e73a9b39519cf6bfca8701ba's public constructors;
upstream modules and callables are never replaced at runtime.
"""
from __future__ import annotations

import numpy as np
import torch

from protenix.data.pipeline.dataset import (
    BaseSingleDataset, WeightedMultiDataset, get_msa_featurizer,
    get_sample_weights, get_template_featurizer,
)
from protenix.utils.cropping import CropData

from helico.protenix_data import chain_contiguous_token_order


class ChainContiguousDataset(BaseSingleDataset):
    """Normalize shuffled chain fragments before upstream crop selection."""

    def crop(self, sample_indice, bioassembly_dict, **kwargs):
        atoms, tokens = bioassembly_dict["atom_array"], bioassembly_dict["token_array"]
        centres = tokens.get_annotation("centre_atom_index")
        order = chain_contiguous_token_order(atoms.asym_id_int[centres], atoms.res_id[centres])
        if not np.array_equal(order, np.arange(len(tokens))):
            # This is a permutation, preserving all atoms, tokens and bonds.
            # Update the full assembly too, keeping later label mappings aligned.
            tokens, atoms = CropData.select_by_token_indices(
                tokens, atoms, torch.as_tensor(order, dtype=torch.long))
            bioassembly_dict.update(token_array=tokens, atom_array=atoms)
        return super().crop(sample_indice, bioassembly_dict, **kwargs)


def get_datasets(configs, error_dir):
    """Build the upstream weighted mixture with the explicit crop subclass.

    Keep the pinned upstream configuration, sampler and test grouping intact.
    Its factory has no dataset-class argument, so construct through the same
    public APIs here rather than substituting an imported class or function.
    """
    data = configs.data

    def dataset(name, stage):
        config = data[name].to_dict()
        return ChainContiguousDataset(
            name=name, **config["base_info"],
            cropping_configs=config["cropping_configs"], error_dir=error_dir,
            msa_featurizer=get_msa_featurizer(configs, name, stage),
            template_featurizer=get_template_featurizer(configs, name, stage),
            lig_atom_rename=config.get("lig_atom_rename", False),
            shuffle_mols=config.get("shuffle_mols", False),
            shuffle_sym_ids=config.get("shuffle_sym_ids", False),
            constraint=config.get("constraint", {}),
            ref_pos_augment=data.get(f"{stage}_ref_pos_augment", True),
            limits=data.get("limits", -1) if stage == "train" else -1,
        )

    assert len(data.train_sets) == len(data.train_sampler.train_sample_weights)
    training = [dataset(name, "train") for name in data.train_sets]
    weights = [get_sample_weights(**data[name]["sampler_configs"], indices_df=ds.indices_list)
               for name, ds in zip(data.train_sets, training, strict=True)]
    mixture = WeightedMultiDataset(training, data.train_sets, weights,
                                   data.train_sampler.train_sample_weights)
    return mixture, {name: dataset(name, "test") for name in data.test_sets}
