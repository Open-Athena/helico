"""Export contacts from MarinFold's new default checkpoint for the by-class set.

MarinFold main promoted `contacts-v1-exp199-cooldown-1.5B` to default in
exp238 -- the same exp199 run continued to step 290,400 and annealed, 304.5B
tokens against 152.3B. This re-exports the contact arms from that checkpoint so
the folding comparison can be rerun against it.

Rollouts were regenerated locally with exp211's worker (100 rollouts, per-request
seeds, occurrence-frequency voting) rather than reusing exp211's, because those
are the *previous* checkpoint's. The index maps are unchanged: the same
prompt->token mapping this directory already verified per target, reused here
rather than recomputed, so the only thing that differs between the two arms is
the model that produced the contacts.

Writes `data/arms/cooldown_{L,L2,L5}.json` and reports precision/recall per
class next to the previous checkpoint's.

Run:
    uv run python experiments/marinfold_contacts/byclass/export_cooldown_contacts.py
"""

from __future__ import annotations

import csv
import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROLLOUTS = Path("/data/marinfold_cooldown/rollouts/cooldown_eval2/contacts")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARMS = DATA / "arms"


def main() -> None:
    from helico.bench import structure_to_chains
    from helico.data import parse_mmcif

    import sys
    sys.path.insert(0, str(HERE))
    from add_foldbench_rest import align_map, compatible

    EXP211 = Path("/home/bizon/git/MarinFold/.claude/worktrees/"
                  "contact-consistency-exp199-9d394e/experiments/"
                  "exp211_evals_contact_set_3d_self_consistency")
    uni = {}
    for line in (EXP211 / "_scratch/gt_universe.jsonl").read_text().splitlines():
        r = json.loads(line)
        uni[(r["dataset"], r["stem"])] = r
    fb_map = json.loads((DATA / "token_map_foldbench_rest.json").read_text())

    with (DATA / "targets.csv").open() as f:
        targets = [r for r in csv.DictReader(f) if r["in_eval2"] == "1"]

    votes = pd.concat([pd.read_parquet(p)
                       for p in sorted(glob.glob(str(ROLLOUTS / "*.parquet")))])
    gt_export = json.loads((ARMS / "gt_from_exp211.json").read_text())
    out = {"L": {}, "L2": {}, "L5": {}}
    stats, dropped = [], []

    for t in targets:
        tid, ds, stem = t["target_id"], t["dataset"], t["stem"]
        st = parse_mmcif(DATA / "gt" / f"{tid}.cif.gz", max_resolution=float("inf"))
        chains = [c for c in structure_to_chains(st) if c["type"] == "protein"] if st else []
        if not chains:
            dropped.append((tid, "no protein chain"))
            continue
        mine = max(chains, key=lambda c: len(c["sequence"]))["sequence"]
        L = len(mine)

        if ds == "foldbench_rest":
            token_of = {int(k): v for k, v in fb_map[tid].items()}
        else:
            rec = uni[(ds, stem)]
            prompt, resolved = t["input_seq"], rec["resolved"]
            expect = "".join(prompt[p] for p in resolved)
            if compatible(mine, prompt):
                token_of = {i: i for i in range(len(prompt))}
            elif compatible(mine, expect):
                token_of = {p: k for k, p in enumerate(resolved)}
            else:
                walk = align_map(mine, expect)
                if walk is None:
                    dropped.append((tid, "no verified map"))
                    continue
                token_of = {p: walk[k] for k, p in enumerate(resolved)}

        v = votes[(votes.dataset == ds) & (votes.stem == stem)]
        v = v[~v.duplicate]
        if v.empty:
            dropped.append((tid, "no rollout contacts"))
            continue
        ranked = [p for p, _ in Counter(
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(v.i, v.j)).most_common()]

        pairs = []
        for a, b in ranked:
            ta, tb = token_of.get(a), token_of.get(b)
            if ta is None or tb is None:
                continue
            pairs.append([min(ta, tb), max(ta, tb)])
        for label, n in (("L5", max(1, L // 5)), ("L2", max(1, L // 2)), ("L", L)):
            out[label][tid] = pairs[:n]
        stats.append({"target_id": tid, "dataset": ds,
                      "n_pred": len(out["L"][tid]), "n_gt": len(gt_export.get(tid, []))})

    for label, mapping in out.items():
        (ARMS / f"cooldown_{label}.json").write_text(json.dumps(mapping))
    print(f"exported {len(stats)} targets, dropped {len(dropped)}")
    for tid, why in dropped:
        print(f"  dropped {tid}: {why}")

    # Accuracy audit next to the previous checkpoint's arm.
    prev = json.loads((ARMS / "marinfold_L.json").read_text())
    byds: dict[str, list] = {}
    for s in stats:
        byds.setdefault(s["dataset"], []).append(s["target_id"])
    print(f"\n{'class':16s} {'n':>4s} {'prec new':>9s} {'rec new':>8s} "
          f"{'prec prev':>10s} {'rec prev':>9s}")
    for ds, ids in sorted(byds.items()):
        def acc(arm):
            ps, rs = [], []
            for tid in ids:
                pred = {tuple(p) for p in arm.get(tid, [])}
                true = {tuple(p) for p in gt_export.get(tid, [])}
                if pred and true:
                    ps.append(len(pred & true) / len(pred))
                    rs.append(len(pred & true) / len(true))
            return (sum(ps) / len(ps), sum(rs) / len(rs)) if ps else (float("nan"),) * 2
        pn, rn = acc(out["L"])
        pp, rp = acc(prev)
        print(f"{ds:16s} {len(ids):4d} {pn:9.3f} {rn:8.3f} {pp:10.3f} {rp:9.3f}")


if __name__ == "__main__":
    main()
