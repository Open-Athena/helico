"""Contact diagnostics and proposals with an explicit inference/label boundary."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F


def input_pair_groups(features, protein, ligand):
    """Masks from input identities only; no coordinates/observation labels."""
    n = features['n_tokens']
    same = features['chain_same'].bool()
    separation = (features['rel_pos'][:, None] - features['rel_pos'][None, :]).abs()
    pp = protein[:, None] & protein[None, :]
    eligible = (protein[:, None] | protein[None, :]) & ~(same & pp & (separation < 6))
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    return dict(all=upper & eligible,
        pp=upper & pp & ~same,
        pl=upper & ((protein[:, None] & ligand[None, :]) | (ligand[:, None] & protein[None, :])),
        intra=upper & pp & same & (separation >= 6))


def average_precision(labels, probabilities):
    labels, probabilities = np.asarray(labels, dtype=bool), np.asarray(probabilities)
    if not labels.any():
        return float('nan')
    order = np.argsort(-probabilities, kind='stable')
    y = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(probabilities[order])), len(y) - 1]
    tp = np.cumsum(y)[ends]
    return float((((tp / (ends + 1)) * np.diff(np.r_[0, tp])).sum()) / y.sum())


def contact_metrics(logits, target, valid, group_masks, state=None):
    logits, target = logits.float().cpu(), target.bool().cpu()
    probabilities = logits.sigmoid()
    rows, bins = [], []
    for group, candidates in group_masks.items():
        mask = candidates & valid.bool()
        if state is not None:
            mask &= state == 0
        p, y, l = probabilities[mask], target[mask], logits[mask]
        row = dict(group=group, pairs=len(y), positives=int(y.sum()))
        if len(y):
            row.update(prevalence=float(y.float().mean()),
                bce=float(F.binary_cross_entropy_with_logits(l, y.float())),
                brier=float(((p - y.float()) ** 2).mean()),
                ap=average_precision(y.numpy(), p.numpy()))
            order = p.argsort(descending=True, stable=True)
            for k in (1, 4, 16, 64):
                row[f'precision_at_{k}'] = float(y[order[:k]].float().mean())
                row[f'top_{k}_count'] = min(k, len(y))
            for b in range(10):
                selected = (p >= b / 10) & ((p < (b + 1) / 10) if b < 9 else p <= 1)
                bins.append(dict(group=group, bin=b, count=int(selected.sum()),
                    probability_sum=float(p[selected].sum()), positive_sum=int(y[selected].sum())))
        rows.append(row)
    return rows, bins


def contact_distances(coords, atom_to_token, pairs, heavy=None):
    """Exact minimum distances for a small selected token-pair list."""
    xyz = torch.as_tensor(coords).float().cpu()
    index = atom_to_token.cpu()
    heavy = torch.ones(len(index), dtype=torch.bool) if heavy is None else heavy.cpu().bool()
    distances = []
    for i, j in pairs.tolist():
        a, b = xyz[(index == i) & heavy], xyz[(index == j) & heavy]
        distances.append(float(torch.cdist(a, b).min()) if len(a) and len(b) else float('nan'))
    return distances


def discrepancy_pairs(probabilities, generated_contacts, eligible, state, count=4, offset=0):
    """Rank missing positive hypotheses without consulting native labels."""
    mask = torch.triu(eligible.bool(), 1) & ~generated_contacts.bool() & (state == 0)
    pairs = mask.nonzero()
    if not len(pairs):
        return pairs
    order = probabilities[mask].argsort(descending=True, stable=True)
    return pairs[order[offset:offset + count]]


def reveal_pairs(state, pairs, values):
    result = state.clone()
    if len(pairs):
        result[pairs[:, 0], pairs[:, 1]] = values
        result[pairs[:, 1], pairs[:, 0]] = values
    return result


def reverse_mask_step(state, probabilities, eligible, current_t, next_t, generator):
    """Ancestral absorbing transition: reveal fraction (t-s)/t, sample labels.

    Already revealed values remain fixed; unordered pairs share one draw.
    Multiple revealed labels use the model's factorized conditional marginals.
    This is an approximate learned reverse process, not a joint consistency claim.
    """
    if not 0 <= next_t < current_t <= 1000:
        raise ValueError('Require 0 <= next_t < current_t <= 1000')
    mask = torch.triu(eligible.bool(), 1) & (state == 0)
    reveal = torch.rand(state.shape, generator=generator) < ((current_t - next_t) / current_t)
    pairs = (mask & reveal).nonzero()
    labels = (torch.rand((len(pairs),), generator=generator)
              < probabilities[pairs[:, 0], pairs[:, 1]]).to(state.dtype) + 1
    return reveal_pairs(state, pairs, labels)


def oracle_state(target, valid, eligible, count, positive, generator):
    """Native-label diagnostic only; never use for deployable search."""
    mask = torch.triu(valid & eligible & (target if positive else ~target), 1)
    pairs = mask.nonzero()
    pairs = pairs[torch.randperm(len(pairs), generator=generator)[:count]]
    return reveal_pairs(torch.zeros_like(target, dtype=torch.uint8), pairs, 2 if positive else 1)


def sparse_lddt(coords, native, observed, atom_chain=None):
    """Observed-heavy-atom pair-weighted lDDT diagnostic, without dense A² arrays.

    Fixed atom/chain correspondence; official symmetry-aware FoldBench scores
    are reported separately. Also report the inter-chain distance subset.
    """
    pred, truth = np.asarray(coords), np.asarray(native)
    keep = np.asarray(observed, dtype=bool)
    index = np.flatnonzero(keep)
    pairs = cKDTree(truth[keep]).query_pairs(15., output_type='ndarray')
    if not len(pairs):
        return dict(observed_lddt=float('nan'), interface_lddt=float('nan'), lddt_pairs=0)
    pairs = index[pairs]
    native_distance = np.linalg.norm(truth[pairs[:, 0]] - truth[pairs[:, 1]], axis=-1)
    pairs, native_distance = pairs[native_distance > .01], native_distance[native_distance > .01]
    predicted_distance = np.linalg.norm(pred[pairs[:, 0]] - pred[pairs[:, 1]], axis=-1)
    delta = np.abs(predicted_distance - native_distance)
    scores = (delta[:, None] < np.array([.5, 1., 2., 4.])).mean(axis=-1)
    inter = (np.asarray(atom_chain)[pairs[:, 0]] != np.asarray(atom_chain)[pairs[:, 1]]) if atom_chain is not None else np.zeros(len(pairs), bool)
    return dict(observed_lddt=float(scores.mean()),
        interface_lddt=float(scores[inter].mean()) if inter.any() else float('nan'), lddt_pairs=len(pairs))


def geometry_contacts(coords, features):
    from helico.contact_diffusion import heavy_atom_contacts
    return heavy_atom_contacts(coords, features['atom_to_token'], features['n_tokens'],
                               atom_mask=features['atom_element_idx'] != 0)


def structure_metrics(coords, features, groups, state):
    contacts = geometry_contacts(coords, features)
    target, valid = features['contact_target'], features['contact_valid']
    index = features['atom_to_token']
    chain = features['chain_indices'][index]
    observed = features['coordinate_mask'] & (features['atom_element_idx'] != 0)
    metrics = sparse_lddt(coords.numpy(), features['atom_coords'].numpy(), observed.numpy(), chain.numpy())
    for name, eligible in groups.items():
        mask = eligible & valid
        truth, predicted = target & mask, contacts & mask
        tp = int((truth & predicted).sum())
        metrics.update({name + '_contact_recall': tp / int(truth.sum()) if truth.any() else float('nan'),
            name + '_contact_precision': tp / int(predicted.sum()) if predicted.any() else float('nan')})
        hidden = mask & (state == 0)
        htruth, hpred = target & hidden, contacts & hidden
        metrics[name + '_unrequested_recall'] = int((htruth & hpred).sum()) / int(htruth.sum()) if htruth.any() else float('nan')
    for value, label in [(2, 'contact'), (1, 'no_contact')]:
        requested = groups['all'] & (state == value)
        metrics['requested_' + label + '_count'] = int(requested.sum())
        metrics['requested_' + label + '_satisfaction'] = float((contacts[requested] == (value == 2)).float().mean()) if requested.any() else float('nan')
    heavy_indices = (features['atom_element_idx'] != 0).nonzero().flatten().numpy()
    pairs = cKDTree(coords.numpy()[heavy_indices]).query_pairs(1.1, output_type='ndarray')
    if len(pairs):
        pairs = heavy_indices[pairs]
        inter = chain[pairs[:, 0]] != chain[pairs[:, 1]]
        metrics['interchain_close_atom_pairs'] = int(inter.sum())
    else:
        metrics['interchain_close_atom_pairs'] = 0
    return metrics, contacts
