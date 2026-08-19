"""Step 2 -- re-seat MarinFold's prompt indices onto Helico token indices.

MarinFold emits residue pairs indexed into the *published prompt* sequence.
Helico's tokenizer indexes the *resolved* residues of the ground-truth
structure. On these 333 targets the two agree outright on only 52; feeding
MarinFold's indices straight in would shift contacts on the other 281 and look
exactly like "real contacts do not help". This repo has been bitten by that
before, which is why the map gets its own script and its own control.

Three cases, in order of preference, each verified before it is used:

* the Helico sequence equals the prompt (52 targets) -- identity map;
* the Helico sequence equals the prompt restricted to exp245's `resolved`
  positions (269) -- prompt position p maps to its rank within `resolved`;
* Helico resolves a residue exp245 did not (12, all by one residue except one
  by four) -- a greedy left-to-right walk recovers the correspondence, which is
  then verified strictly increasing and residue-compatible.

**The control.** A map error is invisible in every downstream number except
this one: exp245's own ground-truth contacts, pushed through the map, are
compared against Helico's `oracle_contact_state` on the same structure. Both
sides are pyconfind at the same thresholds (degree >= 0.001, separation >= 6),
so a correct map agrees essentially perfectly and a wrong one does not.

Writes `data/token_map.json` and `data/index_map_report.csv`.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/build_index_map.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

TOKEN_MAP = U.DATA / "token_map.json"
REPORT = U.DATA / "index_map_report.csv"

#: Below this the map is not trustworthy and the target is reported, not used.
#: Agreement is expected at ~1.0: both sides run pyconfind on the same
#: structure at the same thresholds, so the only source of disagreement is the
#: map itself (or a residue one pipeline resolved and the other did not).
MIN_JACCARD = 0.9


def compatible(a: str, b: str) -> bool:
    """Equal up to X, which the two pipelines use differently for modified residues."""
    return len(a) == len(b) and all(x == y or x == "X" or y == "X" for x, y in zip(a, b))


def _greedy_walk(mine: str, expect: str) -> list[int] | None:
    """Index of each `expect` position within `mine`, left to right."""
    out, mi = [], 0
    for e in expect:
        while mi < len(mine) and not (mine[mi] == e or mine[mi] == "X" or e == "X"):
            mi += 1
        if mi >= len(mine):
            return None
        out.append(mi)
        mi += 1
    if any(b <= a for a, b in zip(out, out[1:])):
        return None
    if not all(mine[t] == expect[k] or mine[t] == "X" or expect[k] == "X"
               for k, t in enumerate(out)):
        return None
    return out


def align_map(mine: str, expect: str) -> list[int] | None:
    """Index of each `expect` position within `mine`, or None if it is ambiguous.

    `mine` is `expect` with zero or more residues inserted -- Helico resolved a
    residue exp245 treated as missing. A greedy walk recovers the
    correspondence, and the walk is verified strictly increasing and
    residue-compatible. Taken from
    experiments/marinfold_contacts/byclass/add_foldbench_rest.py, which solved
    the same problem for exp226's top-up targets.

    **The walk is run in both directions and the two must agree.** Its
    docstring there warns it could mis-seat on a repeat, and on this target set
    it does: `7pv5_A` carries a modified cysteine that Helico writes as `X`, and
    because `X` matches anything the left-to-right walk placed the extra residue
    one position late. Every verification the single walk applies still passed,
    and the map was wrong by one residue for a tenth of that protein's contacts.
    A right-to-left walk seats the insertion elsewhere whenever the placement is
    not forced, so disagreement is exactly the signal that it is not, and the
    target is dropped rather than fuzzily aligned.
    """
    forward = _greedy_walk(mine, expect)
    if forward is None:
        return None
    n_mine, n_expect = len(mine) - 1, len(expect) - 1
    backward = _greedy_walk(mine[::-1], expect[::-1])
    if backward is None:
        return None
    backward = [n_mine - t for t in reversed(backward)]
    if forward != backward:
        return None
    return forward


def build_map(mine: str, prompt: str, resolved: list[int]) -> tuple[dict[int, int], str]:
    """prompt position -> Helico token index, plus which rule produced it."""
    resolved = [p for p in resolved if 0 <= p < len(prompt)]
    expect = "".join(prompt[p] for p in resolved)
    if compatible(mine, prompt):
        return {i: i for i in range(len(prompt))}, "identity"
    if compatible(mine, expect):
        return {p: k for k, p in enumerate(resolved)}, "resolved_rank"
    walk = align_map(mine, expect)
    if walk is None:
        return {}, "unmapped"
    return {p: walk[k] for k, p in enumerate(resolved)}, "aligned"


def oracle_pairs(gt_structure, chains, ccd, rotamer_library) -> set[tuple[int, int]]:
    """Helico's own ground-truth contacts, as upper-triangle token pairs."""
    import torch

    from helico.bench import oracle_contact_state
    from helico.data import CONTACT_PRESENT, tokenize_sequences

    tokenized = tokenize_sequences(chains, ccd)
    state = oracle_contact_state(gt_structure, tokenized, rotamer_library)
    if state is None:
        return set()
    idx = torch.nonzero(torch.triu(state == CONTACT_PRESENT, diagonal=1))
    return {(int(a), int(b)) for a, b in idx}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from helico.bench import structure_to_chains
    from helico.contacts import load_rotamer_library
    from helico.data import parse_ccd, parse_mmcif

    universe = U.load_gt_universe()
    with (U.CACHE / "upstream/eval_sets.csv").open() as handle:
        prompts = {row["stem"]: row["sequence"] for row in csv.DictReader(handle)}
    with (U.DATA / "targets.csv").open() as handle:
        targets = list(csv.DictReader(handle))
    if args.limit:
        targets = targets[:args.limit]

    ccd, rotamer_library = parse_ccd(), load_rotamer_library()

    token_map, rows = {}, []
    for n, target in enumerate(targets, 1):
        stem = target["target_id"]
        record = universe[stem]
        mapping, rule = build_map(target["input_seq"], prompts[stem],
                                  record["resolved"])

        # The control: exp245's own truth, mapped, against Helico's own truth.
        structure = parse_mmcif(U.GT_DIR / f"{stem}.cif.gz",
                                max_resolution=float("inf"))
        chains = structure_to_chains(structure)
        mine = oracle_pairs(structure, chains, ccd, rotamer_library)
        length = record["L"]
        theirs = set()
        for i, j, degree in record["contacts"]:
            i, j = int(i), int(j)
            if degree < U.MIN_DEG or (j - i) < U.MIN_SEP or not (i < j < length):
                continue
            a, b = mapping.get(i), mapping.get(j)
            if a is not None and b is not None:
                theirs.add((min(a, b), max(a, b)))

        union = len(mine | theirs)
        jaccard = len(mine & theirs) / union if union else float("nan")
        if mapping:
            token_map[stem] = {str(p): t for p, t in sorted(mapping.items())}
        rows.append({
            "target_id": stem, "eval_set": target["eval_set"], "rule": rule,
            "L_exp245": length, "L_helico": target["L_helico"],
            "n_mapped": len(mapping), "n_theirs": len(theirs), "n_mine": len(mine),
            "jaccard": round(jaccard, 6) if union else "",
            "ok": int(bool(mapping) and union > 0 and jaccard >= MIN_JACCARD),
        })
        if n % 50 == 0:
            print(f"  {n}/{len(targets)}", flush=True)

    with REPORT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    TOKEN_MAP.write_text(json.dumps(token_map))

    from collections import Counter
    print("\nrule:", dict(Counter(r["rule"] for r in rows)))
    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    jac = [r["jaccard"] for r in rows if r["jaccard"] != ""]
    print(f"agreement with Helico's own oracle contacts: "
          f"median {statistics.median(jac):.4f}, min {min(jac):.4f}")
    print(f"pass (>= {MIN_JACCARD}): {len(good)}/{len(rows)}")
    for r in bad:
        print(f"  FAIL {r['target_id']:8s} {r['rule']:13s} jaccard={r['jaccard']} "
              f"theirs={r['n_theirs']} mine={r['n_mine']} "
              f"L {r['L_exp245']}/{r['L_helico']}")
    print(f"\n{len(token_map)} maps -> {TOKEN_MAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
