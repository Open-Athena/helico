"""Residue/residue contacts from pyconfind, indexed by Helico token.

Contacts are side-chain contact degrees computed by `pyconfind
<https://github.com/timodonnell/pyconfind>`_ with the same parameters MarinFold
uses for its ``contacts-v1`` document type, so a MarinFold contact-prediction
model can drive Helico at inference time.

The one non-obvious part is the index mapping. pyconfind numbers positions by
input order, and Helico parses ``label_asym_id``/``label_seq_id`` while
pyconfind reads author numbering — so aligning a file-parsed pyconfind run
against Helico tokens would need a sequence alignment. We sidestep that
entirely by building the ``gemmi.Structure`` ourselves from the tokens, which
makes ``Position.index`` a direct index into our own ordering.

Two further wrinkles, both handled by :func:`build_gemmi`:

* Helico atom-tokenizes modified residues (MSE, SEP, TPO, …) because they miss
  the ``THREE_TO_ONE`` fast path in ``tokenize_structure``, while pyconfind
  treats them as one residue. Tokens are grouped by ``(chain_idx, res_idx)`` so
  the geometry pyconfind sees is right and modified residues still occlude.
* Contacts are only *attributed* to residues backed by exactly one standard
  protein token. Modified residues, ligands and nucleotides stay "unknown".
"""

from __future__ import annotations

from typing import Any

# pyconfind geometry knobs. These MUST match MarinFold's
# ``contacts_v1.GenerationConfig`` defaults (verified against
# ``experiments/exp74_evals_protenix_pyconfind_contacts/pyconfind_contacts.py``)
# so contacts predicted by a MarinFold model mean the same thing as the ones we
# train on. ``assembly=None`` analyses the structure as-is rather than
# implicitly expanding biological assembly 1.
PYCONFIND_KWARGS = dict(
    native_only=True,
    contact_distance=3.0,
    dcut=25.0,
    clash_distance=2.0,
    assembly=None,
)

# pyconfind emits a long tail of near-zero degrees (down to ~1e-8); MarinFold's
# ``min_contact_degree`` keeps that noise out.
MIN_CONTACT_DEGREE = 0.001

# A same-chain pair counts as a contact only this far apart in the primary
# sequence. Applied in ``TokenizedStructure.to_features``, not here, because
# chain identity is needed to restrict it to intra-chain pairs — see the note
# in :func:`compute_contacts`.
MIN_SEQ_SEPARATION = 6

# MarinFold's operating point as of 2026-08: precision ~= recall ~= 0.6, output
# is a truncated top-k contact list. Training samples around and above this so
# the model covers today's predictor and the better one we expect later.
MARINFOLD_PRECISION = 0.6
MARINFOLD_RECALL = 0.6
# Lower bound on sampled precision. p=0.4 gives eps_fp=1.5 — more false contacts
# than true ones — which brackets the current 0.6 comfortably from below.
MIN_SAMPLED_PRECISION = 0.4

# helico token_types 0..20 are the standard protein residues (20 AAs + UNK).
MAX_PROTEIN_TOKEN_TYPE = 20


def conditioning_from_precision_recall(
    precision: float, recall: float
) -> tuple[float, float, float]:
    """``(reveal, eps_fp, eps_fn)`` reproducing a predictor at this operating point.

    MarinFold emits a *truncated top-k* list, so a true contact is missing
    because it did not make the cut — there is no separate "reported but wrong"
    channel. All recall loss therefore folds into ``reveal`` and ``eps_fn``
    stays 0.

    ``eps_fp`` counts false contacts relative to the *revealed* ones, so with
    all recall loss in ``reveal``:

        precision = revealed / (revealed + eps_fp * revealed)
                  = 1 / (1 + eps_fp)          =>  eps_fp = (1 - p) / p

    At p=0.6 that is eps_fp=0.667 — well outside the old U(0, 0.3) sampling
    range, so the model was never trained anywhere near MarinFold's precision.
    """
    if not 0.0 < precision <= 1.0:
        raise ValueError(f"precision must be in (0, 1], got {precision}")
    if not 0.0 <= recall <= 1.0:
        raise ValueError(f"recall must be in [0, 1], got {recall}")
    return recall, (1.0 - precision) / precision, 0.0


def sample_conditioning(
    contact_state: "Any",
    generator: "Any" = None,
    mode: str | None = None,
    reveal: float | None = None,
    eps_fp: float | None = None,
    eps_fn: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
) -> "Any":
    """Mask and corrupt a 3-state contact matrix for training.

    Takes a fully-specified ``(N, N)`` uint8 matrix and returns one conditioned
    at a randomly sampled level, so a single model spans everything from
    ab initio (nothing known) to fully specified.

    Two things beyond plain masking matter here.

    **Mode.** A contact predictor emits a *contact list*, not a uniform sample of
    matrix cells, and a model trained on one transfers poorly to the other. The
    modes are:

    ``none``          everything unknown — the MSA-free ab initio baseline
    ``full``          every known pair specified
    ``pair-subset``   reveal a fraction of *pairs*; the rest become unknown
    ``contact-list``  reveal a fraction of *contacts*; the remaining known pairs
                      become absent or unknown (coin flip, because MarinFold
                      truncates its output to a token budget — so an unlisted
                      pair sometimes means "not a contact" and sometimes
                      "didn't fit")

    **Corruption.** Masking alone teaches the model that every asserted contact
    is true, which is brittle exactly where it matters. False positives and
    negatives are injected at rates expressed relative to the number of revealed
    contacts, matching how contact predictors report precision and recall.

    All arguments except ``contact_state`` are sampled when left as ``None``;
    pass them explicitly to pin a fixed conditioning level for validation.
    """
    import torch

    from helico.data import CONTACT_ABSENT, CONTACT_PRESENT, CONTACT_UNKNOWN

    def _rand() -> float:
        return float(torch.rand((), generator=generator))

    if mode is None:
        r = _rand()
        mode = ("none" if r < 0.15 else "full" if r < 0.30
                else "pair-subset" if r < 0.65 else "contact-list")

    n = contact_state.shape[-1]
    out = torch.full_like(contact_state, CONTACT_UNKNOWN)
    if mode == "none":
        return out

    known = contact_state != CONTACT_UNKNOWN
    if not bool(known.any()):
        return out

    device = contact_state.device

    def _sym_mask(p: float) -> "Any":
        """Symmetric Bernoulli mask — sampled on the upper triangle only.

        Drawn on CPU then moved: `generator` is a CPU generator (this runs in
        DataLoader workers), and a CPU generator cannot seed a CUDA draw.
        Sampling on CPU also makes the result device-independent for a given
        seed, which keeps the tests reproducible anywhere.
        """
        m = torch.rand((n, n), generator=generator) < p
        m = torch.triu(m, diagonal=1)
        return (m | m.T).to(device)

    if reveal is None:
        if recall is not None:
            reveal = recall
        else:
            reveal = 1.0 if mode == "full" else _rand()

    if mode in ("full", "pair-subset"):
        keep = known if mode == "full" else (known & _sym_mask(reveal))
        out[keep] = contact_state[keep]
    else:  # contact-list
        is_contact = contact_state == CONTACT_PRESENT
        listed = is_contact & _sym_mask(reveal)
        # MarinFold emits a truncated top-k list, so an unlisted pair means
        # "did not make the cut" — which carries no information either way.
        # Marking those ABSENT (as an earlier version did on a coin flip) would
        # assert millions of true negatives the predictor never claimed.
        out[listed] = CONTACT_PRESENT

    # --- corruption -------------------------------------------------------
    # For contact-list (the MarinFold-shaped mode) the noise level is drawn as a
    # precision, so the sampled range is anchored to a real operating point
    # rather than to an arbitrary epsilon cap. The old U(0, 0.3) on eps_fp
    # corresponds to precision >= 0.77 and never reached MarinFold's 0.6.
    if precision is not None or recall is not None:
        p_ = MARINFOLD_PRECISION if precision is None else precision
        r_ = MARINFOLD_RECALL if recall is None else recall
        _, eps_fp_pr, eps_fn_pr = conditioning_from_precision_recall(p_, r_)
        eps_fp = eps_fp_pr if eps_fp is None else eps_fp
        eps_fn = eps_fn_pr if eps_fn is None else eps_fn
    elif mode == "contact-list":
        if eps_fp is None:
            p_ = MIN_SAMPLED_PRECISION + _rand() * (1.0 - MIN_SAMPLED_PRECISION)
            eps_fp = (1.0 - p_) / p_
        if eps_fn is None:
            eps_fn = 0.0  # recall loss is already modelled by `reveal`
    if eps_fn is None:
        eps_fn = _rand() * 0.3
    if eps_fp is None:
        eps_fp = _rand() * 0.3

    revealed_contact = out == CONTACT_PRESENT
    n_revealed = int(revealed_contact.sum()) // 2
    if n_revealed == 0:
        return out

    # False negatives: drop revealed contacts back to whatever the unlisted
    # state is in this mode, so the corruption is indistinguishable from the
    # contact simply not having been reported.
    if eps_fn > 0:
        drop = revealed_contact & _sym_mask(eps_fn)
        unlisted = CONTACT_ABSENT if mode != "pair-subset" else CONTACT_UNKNOWN
        out[drop] = unlisted

    # False positives: assert contacts that are not there. Expressed as a
    # fraction of the revealed contacts, not of all pairs — contacts are ~0.1%
    # of pairs, so a per-pair rate would swamp the true signal many times over.
    n_fp = int(round(eps_fp * n_revealed))
    if n_fp > 0:
        # Restricted to `known`: the eligible region. Outside it a real
        # predictor structurally cannot emit a contact — pyconfind reports only
        # protein side-chain contacts, and pairs closer than
        # MIN_SEQ_SEPARATION are filtered out. Sampling the full upper triangle
        # put ~40% of false positives on ligand/nucleic tokens or at
        # |i-j| < 6, giving the model a free "this one is fake" cue that will
        # not exist at deployment.
        candidates = known & (out != CONTACT_PRESENT) & torch.triu(
            torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1
        )
        idx = candidates.nonzero(as_tuple=False)
        if len(idx):
            pick = torch.randperm(len(idx), generator=generator)[:n_fp].to(idx.device)
            sel = idx[pick]
            out[sel[:, 0], sel[:, 1]] = CONTACT_PRESENT
            out[sel[:, 1], sel[:, 0]] = CONTACT_PRESENT

    return out


def load_rotamer_library() -> Any:
    """Load (and locally cache) the Dunbrack 2010 backbone-dependent library.

    Parsing costs ~3.4 s, so callers should do this once and pass the result
    into :func:`compute_contacts` — in particular once per preprocess worker,
    not once per structure.
    """
    from pyconfind import cached_rotamer_library, load_library

    return load_library(cached_rotamer_library())


def build_gemmi(tokens: list) -> tuple[Any, list[int]]:
    """Build a ``gemmi.Structure`` from Helico tokens.

    Returns ``(structure, slot_to_token)``. Slot ``k`` is the ``k``-th emitted
    residue, which is exactly pyconfind's ``Position`` index because
    ``positions_from_atoms`` preserves input order. ``slot_to_token[k]`` is the
    token index to attribute contacts to, or ``-1`` when the residue is not a
    single standard protein token.
    """
    import gemmi
    from pyconfind.pdb import LEGAL_RESIDUE_NAMES

    # Group tokens into residues. Tokens of one residue are contiguous, so a
    # key change starts a new group.
    groups: list[tuple[tuple[int, int], list[int]]] = []
    key_prev = None
    cur: list[int] = []
    for ti, tok in enumerate(tokens):
        key = (tok.chain_idx, tok.res_idx)
        if key != key_prev:
            cur = []
            groups.append((key, cur))
            key_prev = key
        cur.append(ti)

    st = gemmi.Structure()
    model = gemmi.Model("1")
    chains: dict[str, Any] = {}
    slot_to_token: list[int] = []

    for (chain_idx, _res_idx), tis in groups:
        resname = tokens[tis[0]].res_name
        if resname not in LEGAL_RESIDUE_NAMES:
            continue  # nucleotide / ligand / water — pyconfind would drop it anyway
        cname = str(chain_idx)
        if cname not in chains:
            chains[cname] = gemmi.Chain(cname)
        ch = chains[cname]

        res = gemmi.Residue()
        res.name = resname
        # Sequential numbering within the chain: unique and ordered by
        # construction, so no insertion-code or numbering-gap handling needed.
        res.seqid = gemmi.SeqId(len(ch) + 1, " ")
        n_atoms = 0
        for ti in tis:
            tok = tokens[ti]
            if tok.atom_coords is None:
                continue
            for name, elem, xyz in zip(tok.atom_names, tok.atom_elements, tok.atom_coords):
                atom = gemmi.Atom()
                atom.name = name
                atom.element = gemmi.Element(elem if elem else "C")
                atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                atom.occ = 1.0
                atom.altloc = "\x00"
                res.add_atom(atom)
                n_atoms += 1
        if n_atoms == 0:
            continue

        ch.add_residue(res)
        if len(tis) == 1 and tokens[tis[0]].token_type <= MAX_PROTEIN_TOKEN_TYPE:
            slot_to_token.append(tis[0])
        else:
            slot_to_token.append(-1)

    for cname in chains:
        model.add_chain(chains[cname])
    st.add_model(model)
    return st, slot_to_token


def compute_contacts(
    tokens: list,
    rotamer_library: Any,
    min_degree: float = MIN_CONTACT_DEGREE,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute contacts for one tokenized structure.

    Returns ``(edges, eligible)`` where ``edges`` is a sorted list of
    ``(ti, tj)`` token-index pairs with ``ti < tj``, and ``eligible`` is the
    sorted list of token indices pyconfind actually analysed. ``eligible`` is
    what makes "no contact" distinguishable from "unknown" downstream.

    The ``MIN_SEQ_SEPARATION`` rule is deliberately *not* applied here: it is
    only meaningful within a chain, and chain identity is more naturally
    available in ``to_features``. Inter-chain contacts are always kept.
    """
    from pyconfind import analyze

    st, slot_to_token = build_gemmi(tokens)
    if sum(1 for s in slot_to_token if s >= 0) < 2:
        return [], []

    analysis = analyze(st, rotamer_library=rotamer_library, **PYCONFIND_KWARGS)
    if len(analysis.positions) != len(slot_to_token):
        # Would silently misalign every contact; fail loudly instead.
        raise RuntimeError(
            f"pyconfind position/slot mismatch: {len(analysis.positions)} positions "
            f"vs {len(slot_to_token)} emitted residues"
        )

    edges: set[tuple[int, int]] = set()
    for contact in analysis.report.contacts:
        if contact.degree < min_degree:
            continue
        ti = slot_to_token[contact.pos_i]
        tj = slot_to_token[contact.pos_j]
        if ti < 0 or tj < 0 or ti == tj:
            continue
        edges.add((min(ti, tj), max(ti, tj)))

    eligible = [s for s in slot_to_token if s >= 0]
    return sorted(edges), eligible
