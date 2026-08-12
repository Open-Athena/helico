"""Run a contact-conditioned Helico model on sequences plus a contact map.

The point of this model is that contacts replace the MSA, so the API is built
around getting a contact map in. Three ways to supply one, matching the three
places contacts come from in practice:

``contacts_from_pairs``      a ranked list of residue pairs, i.e. what a contact
                             predictor such as MarinFold emits
``contacts_from_structure``  derived from a reference structure with pyconfind
                             (the oracle condition -- it uses the answer)
``None``                     no contacts; the model still runs, badly

Nothing here touches MSAs. The published checkpoint has ``use_msa=False``, which
zeroes the alignment-derived features in ``s_inputs`` as well as skipping the
MSA module, so no alignment is fetched or used at any point.

Example::

    from helico.inference import fold, contacts_from_pairs, load_model

    model = load_model()                      # pulls from the Hub
    contacts = contacts_from_pairs(seq_len=len(seq), pairs=predicted_pairs)
    result = fold({"A": seq}, contacts=contacts, model=model)
    open("pred.pdb", "w").write(result.pdb)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

logger = logging.getLogger(__name__)

DEFAULT_REPO = "timodonnell/helico"
DEFAULT_FILENAME = "contacts-msafree-01-step6000.pt"

# Contacts closer than this in sequence are excluded from the contact map --
# they are trivially implied by the chain and pyconfind filters them out, so the
# model never saw one asserted. Mirrors helico.contacts.MIN_SEQ_SEPARATION.
from helico.contacts import MIN_SEQ_SEPARATION  # noqa: E402
from helico.data import (  # noqa: E402
    CONTACT_PRESENT,
    CONTACT_UNKNOWN,
    parse_ccd,
    tokenize_sequences,
)


@dataclass
class FoldResult:
    """One prediction, with the coordinates and the model's own confidence."""

    coords: torch.Tensor          # (N_atoms, 3)
    plddt: torch.Tensor           # (N_atoms,) predicted lDDT, 0-100
    tokenized: Any                # TokenizedStructure, for writing PDB
    ranking_score: float | None = None

    @property
    def mean_plddt(self) -> float:
        return float(self.plddt.mean())

    @property
    def pdb(self) -> str:
        from helico.train import coords_to_pdb

        return coords_to_pdb(self.coords, self.plddt, self.tokenized)

    def write_pdb(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.pdb)
        return path


def load_model(
    checkpoint: str | Path | None = None,
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    device: str | None = None,
    dtype: torch.dtype | None = None,
):
    """Load a Helico checkpoint, pulling from the Hub if no path is given.

    ``dtype`` defaults to bfloat16 on Ampere and newer and float32 below it --
    bf16 predates neither the kernels nor the model, but Turing (sm75, e.g. a
    Colab T4) has no native bf16 and emulating it is slower than fp32.
    """
    from helico.model import Helico, HelicoConfig

    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info("Loaded checkpoint from %s/%s", repo_id, filename)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise ValueError(
            f"{checkpoint} is not a Helico checkpoint "
            f"(no 'model_state_dict'; keys={sorted(state)[:6]})"
        )
    cfg = HelicoConfig(**{k: v for k, v in state.get("config", {}).items()
                          if k in HelicoConfig.__dataclass_fields__})
    model = Helico(cfg)
    model.load_state_dict(state["model_state_dict"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    model._helico_device, model._helico_dtype = device, dtype  # noqa: SLF001
    logger.info("Model on %s in %s (step %s)", device, dtype, state.get("step"))
    return model


def _token_positions(tokenized) -> dict[str, list[int]]:
    """{chain_id: [token index of residue 0, residue 1, ...]} for protein chains.

    Callers think in sequence positions; the model thinks in token indices, and
    the two differ as soon as there is more than one chain or any non-protein
    entity. Everything that accepts residue positions goes through this.
    """
    out: dict[str, list[int]] = {}
    for idx in range(len(tokenized.tokens)):
        if tokenized.entity_types[idx] != "protein":
            continue
        # chain_ids is per-token, not a list of distinct chains: indexing it by
        # token.chain_idx silently returns the first chain for every token, so
        # every chain-B contact would land on chain A.
        out.setdefault(tokenized.chain_ids[idx], []).append(idx)
    return out


def contacts_from_pairs(
    pairs: Iterable[tuple],
    tokenized=None,
    seq_len: int | None = None,
    chain: str = "A",
    one_indexed: bool = False,
    strict: bool = False,
) -> torch.Tensor:
    """Build a contact matrix from a list of residue pairs.

    This is the deployment path: a contact predictor emits a ranked list, you
    take the top n, and hand them over.

    ``pairs`` entries are either ``(i, j)`` residue positions within ``chain``,
    or ``(chain_a, i, chain_b, j)`` for multi-chain inputs.

    Unlisted pairs are left ``UNKNOWN``, never ``ABSENT``. A truncated top-n
    list cannot distinguish "not a contact" from "did not make the cut", and the
    model was trained on exactly that semantics -- marking the remainder absent
    would assert millions of true negatives the predictor never claimed.

    Pairs closer than ``MIN_SEQ_SEPARATION`` in sequence are dropped: pyconfind
    filters them from the ground truth, so the model has never seen one asserted
    and its response is undefined. Set ``strict`` to raise instead of dropping.
    """
    if tokenized is not None:
        positions = _token_positions(tokenized)
        n_tok = len(tokenized.tokens)
    elif seq_len is not None:
        positions = {chain: list(range(seq_len))}
        n_tok = seq_len
    else:
        raise ValueError("pass either `tokenized` or `seq_len`")

    state = torch.full((n_tok, n_tok), CONTACT_UNKNOWN, dtype=torch.uint8)
    off = 1 if one_indexed else 0
    n_kept = n_short = n_oob = 0

    for pair in pairs:
        if len(pair) == 2:
            ca, ra, cb, rb = chain, pair[0], chain, pair[1]
        elif len(pair) == 4:
            ca, ra, cb, rb = pair
        else:
            raise ValueError(f"pair must be (i, j) or (chain_a, i, chain_b, j), got {pair!r}")
        ra, rb = int(ra) - off, int(rb) - off

        try:
            ti, tj = positions[ca][ra], positions[cb][rb]
        except (KeyError, IndexError):
            n_oob += 1
            if strict:
                raise ValueError(f"pair {pair!r} is outside the input") from None
            continue

        if ca == cb and abs(ra - rb) < MIN_SEQ_SEPARATION:
            n_short += 1
            if strict:
                raise ValueError(
                    f"pair {pair!r} is closer than MIN_SEQ_SEPARATION={MIN_SEQ_SEPARATION}; "
                    "pyconfind filters these out so the model never saw one asserted"
                )
            continue

        state[ti, tj] = CONTACT_PRESENT
        state[tj, ti] = CONTACT_PRESENT
        n_kept += 1

    if n_short or n_oob:
        logger.warning(
            "contacts_from_pairs: kept %d, dropped %d closer than %d residues, "
            "%d out of range", n_kept, n_short, MIN_SEQ_SEPARATION, n_oob,
        )
    return state


def contacts_from_structure(tokenized, structure, min_degree: float | None = None) -> torch.Tensor:
    """Contact map derived from a reference structure via pyconfind.

    This is the oracle condition: it uses the answer. Useful for reproducing the
    reported ceiling and for sanity checks, not for prediction.
    """
    from helico.bench import oracle_contact_state
    from helico.contacts import load_rotamer_library

    state = oracle_contact_state(structure, tokenized, load_rotamer_library())
    if state is None:
        raise ValueError("no protein contacts could be derived from that structure")
    return state


def fold(
    sequences: dict[str, str] | str | Sequence[dict],
    contacts: torch.Tensor | None = None,
    model=None,
    n_samples: int = 5,
    ccd: dict | None = None,
    seed: int | None = None,
) -> FoldResult:
    """Fold ``sequences``, optionally conditioned on a contact map.

    ``sequences`` is ``{chain_id: sequence}``, or a bare string for a monomer.
    ``contacts`` is an ``(N_tok, N_tok)`` uint8 matrix from one of the
    ``contacts_from_*`` helpers; ``None`` runs the model with no contact
    information, which is the weak baseline rather than a normal mode of use.

    Returns the best-ranked of ``n_samples`` diffusion samples.
    """
    from helico.train import run_inference

    if model is None:
        model = load_model()
    device = getattr(model, "_helico_device", "cuda" if torch.cuda.is_available() else "cpu")
    # run_inference defaults to bf16 autocast; honour what load_model chose so a
    # pre-Ampere GPU does not get bf16 forced on it.
    dtype = getattr(model, "_helico_dtype", torch.bfloat16)

    if isinstance(sequences, str):
        sequences = {"A": sequences}
    if isinstance(sequences, dict):
        # tokenize_sequences expects {"type", "id", "sequence"}.
        chains = [{"type": "protein", "id": c, "sequence": s}
                  for c, s in sequences.items()]
    else:
        chains = list(sequences)

    if ccd is None:
        ccd = parse_ccd()
    tokenized = tokenize_sequences(chains, ccd)
    features = tokenized.to_features()

    if contacts is not None:
        n_tok = len(tokenized.tokens)
        if tuple(contacts.shape) != (n_tok, n_tok):
            raise ValueError(
                f"contacts is {tuple(contacts.shape)} but the input tokenizes to "
                f"{n_tok} tokens; build it with the same `tokenized` you fold"
            )
        features["contact_state"] = contacts.to(torch.uint8)

    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
             for k, v in features.items()}
    for key in ("n_tokens", "n_atoms"):
        if key in batch and not isinstance(batch[key], torch.Tensor):
            batch[key] = torch.tensor([batch[key]])
    batch["token_mask"] = torch.ones(1, features["n_tokens"], dtype=torch.bool)
    batch["atom_mask"] = torch.ones(1, features["n_atoms"], dtype=torch.bool)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    results = run_inference(model, batch, n_samples=n_samples, device=device, dtype=dtype)
    rs = results.get("ranking_score")
    return FoldResult(
        coords=results["coords"][0].float().cpu(),
        plddt=results["plddt"][0].float().cpu(),
        tokenized=tokenized,
        ranking_score=float(rs[0]) if rs is not None else None,
    )
