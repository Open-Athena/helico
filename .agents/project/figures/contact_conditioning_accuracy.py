"""Figure: how the contact pathway learns, measured with real predicted contacts.

Regenerate with:
    uv run python .agents/project/figures/contact_conditioning_accuracy.py

A checkpoint sweep of `contacts-msafree-01`, restricted to the FoldBench monomers
that survive MarinFold exp226's homology filter (< 40% identity to either
training arm). Only 15 of the original 100 clear it, and 14 of those are paired
across the whole sweep -- small, but the alternative is a training curve measured
on targets MarinFold has effectively memorised. Three conditioning arms are
benched at each checkpoint:

  real    -- MarinFold contacts-v1-exp199-1.5B, vote-aggregated, truncated at top-L
  oracle  -- the ground-truth contact map (the ceiling)
  off     -- contacts withheld (the no-information control)

An earlier version of this figure plotted a *synthetic* series: the oracle map
degraded with a uniform noise model to MarinFold's measured 60/60 operating
point. Real predictor errors are structured and cost far more than uniform ones
at matched precision and recall, so that series overstated the deployment
condition and has been replaced by the measured one.

Step 0 is the warm start itself -- Protenix v1 weights with `use_msa=False` and
the contact projection still at its zero initialisation. Conditioning is an exact
no-op there by construction, so all three arms should coincide; that they do is
the control, and the measured spread is annotated.

Every Helico arm is genuinely MSA-free (benched with HELICO_BENCH_SINGLE_SEQ=0
and HELICO_BENCH_NO_MSA=1, trained with the MSA loader disabled), so no
alignment-derived signal reaches the model by any route. The Protenix +MSA
reference lines do of course use alignments -- that is the comparison.

Validation-set numbers are deliberately absent. 38% of that set's structures
share a chain sequence verbatim with training, so its absolute values measure
memorisation as much as folding. FoldBench is the number of record.

Annotations are limited to what was measured -- counts, means, standard errors.
Interpretation belongs in the writeup, not on the axes.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "contact_conditioning_accuracy.png"
BYCLASS = ROOT / "experiments/marinfold_contacts/byclass/data/targets.csv"


def eval2_pdb_codes() -> set[str]:
    """PDB codes of the FoldBench monomers that clear the homology filter."""
    with BYCLASS.open() as f:
        return {r["stem"].split("_")[0].lower() for r in csv.DictReader(f)
                if r["dataset"] == "foldbench100" and r["in_eval2"] == "1"}

STEPS = [0, 1000, 2000, 3000, 5000, "final"]
# Contacts are withheld here, so there is nothing for training to improve; three
# points establish that it is flat rather than drifting.
OFF_STEPS = [0, 1000, "final"]

C_REAL, C_ORACLE, C_OFF = "#b8452f", "#1b5e9c", "#7a7a7a"
C_MSA, C_V2MSA, C_SS, C_V2SS = "#2e7d32", "#1a7f5a", "#9a9a9a", "#a0762b"


def bench_dir(step, arm):
    """Where a (checkpoint, arm) pair's per-target CSV lives.

    The `final` checkpoint was already benched for the real-contacts figure, so
    the sweep reuses those runs rather than repeating them.
    """
    if step == "final":
        return {"real": "bench_mf2_rollout_L", "oracle": "bench_mf2_oracle",
                "off": "bench_mf2_off"}[arm]
    return f"bench_curve_s{step}_{arm}"


def load(arm_or_dir):
    """{pdb_id: lddt} for the monomer category, NaNs dropped.

    Protenix v2 is scored by experiments/marinfold_contacts/score_protenix_v2.py
    rather than modal/bench.py -- it runs through ByteDance's own implementation,
    and the upstream DockQ-based scorer returns 0.000 on monomers.
    """
    if arm_or_dir.startswith("v2_"):
        f = ROOT / f"experiments/marinfold_contacts/upstream/{arm_or_dir}_scores.csv"
        if not f.exists():
            return {}
        keep = eval2_pdb_codes()
        return {r["pdb_id"]: float(r["lddt"]) for r in csv.DictReader(f.open())
                if r["pdb_id"].split("-")[0].lower() in keep}

    f = ROOT / arm_or_dir / "results" / "monomer_protein.csv"
    if not f.exists():
        return {}
    keep = eval2_pdb_codes()
    out = {}
    for row in csv.DictReader(f.open()):
        if row["pdb_id"].split("-")[0].lower() not in keep:
            continue
        try:
            v = float(row.get("lddt", ""))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out[row["pdb_id"]] = v
    return out


def paired_stats(a, b, keys):
    """Paired mean difference b - a, its standard error, t, and #improved."""
    d = [b[k] - a[k] for k in keys]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    se = sd / len(d) ** 0.5
    return m, se, m / se, sum(1 for x in d if x > 0)


def main():
    arms = {}
    for s in STEPS:
        for arm in ("real", "oracle"):
            arms[(s, arm)] = load(bench_dir(s, arm))
    for s in OFF_STEPS:
        arms[(s, "off")] = load(bench_dir(s, "off"))

    refs = {k: load(k) for k in ("v2_msa", "v2_singleseq")}
    refs["protenix_msa"] = load("bench_mf2_protenix_msa")
    refs["protenix_ss"] = load("bench_mf2_protenix_singleseq")

    missing = [k for k, v in {**arms, **refs}.items() if not v]
    if missing:
        raise SystemExit(f"no rows for: {missing}")

    keys = None
    for d in (*arms.values(), *refs.values()):
        keys = set(d) if keys is None else keys & set(d)
    keys = sorted(keys)
    n = len(keys)

    def mean(d):
        return sum(d[k] for k in keys) / n

    print(f"n = {n} homology-filtered FoldBench monomers\n")
    print(f"{'step':>7} {'off':>8} {'real MF':>9} {'oracle':>8}")
    for s in STEPS:
        off = mean(arms[(s, "off")]) if (s, "off") in arms else float("nan")
        print(f"{str(s):>7} {off:8.4f} {mean(arms[(s, 'real')]):9.4f} "
              f"{mean(arms[(s, 'oracle')]):8.4f}")
    print()
    for k, lab in [("protenix_ss", "Protenix v1, single sequence"),
                   ("v2_singleseq", "Protenix v2, single sequence"),
                   ("protenix_msa", "Protenix v1 + MSA"),
                   ("v2_msa", "Protenix v2 + MSA")]:
        print(f"  {lab:30s} {mean(refs[k]):.4f}")
    print()
    for a, b, lab in [
        (arms[(0, "real")], arms[(0, "oracle")], "step 0: oracle vs real (no-op)"),
        (arms[(0, "off")], arms[(0, "oracle")], "step 0: oracle vs off (no-op)"),
        (arms[("final", "off")], arms[("final", "real")], "final: real vs off"),
        (arms[("final", "real")], arms[("final", "oracle")], "final: oracle vs real"),
        (arms[(1000, "real")], arms[("final", "real")], "real: step 1000 -> final"),
    ]:
        m, se, t, up = paired_stats(a, b, keys)
        print(f"  {lab:34s} {m:+.4f} +/- {se:.4f}  t={t:+.2f}  {up}/{n}")
    step0 = [mean(arms[(0, a)]) for a in ("real", "oracle", "off")]
    print(f"  {'step 0 spread across arms':34s} {max(step0) - min(step0):.4f}")

    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    xs = list(range(len(STEPS)))
    for k, c, lab, xf, va in [
        ("v2_msa", C_V2MSA, "Protenix v2 + MSA", 0.985, "bottom"),
        ("protenix_msa", C_MSA, "Protenix v1 + MSA", 0.985, "top"),
        ("v2_singleseq", C_V2SS, "Protenix v2, single sequence", 0.985, "bottom"),
        ("protenix_ss", C_SS, "Protenix v1, single sequence", 0.545, "bottom"),
    ]:
        v = mean(refs[k])
        ax.axhline(v, color=c, ls="--", lw=1.5, zorder=1)
        ax.annotate(f"{lab}  ({v:.3f})", xy=(xf, v),
                    xycoords=("axes fraction", "data"), ha="right", va=va,
                    fontsize=8.5, color=c, fontweight="bold", zorder=6)

    ax.plot(xs, [mean(arms[(s, "oracle")]) for s in STEPS], "-o", color=C_ORACLE,
            lw=1.8, ms=7, zorder=4, label="oracle contacts")
    ax.plot(xs, [mean(arms[(s, "real")]) for s in STEPS], "-s", color=C_REAL,
            lw=1.8, ms=6.5, zorder=4,
            label="MarinFold contacts (exp199, top-L)")
    ax.plot([xs[STEPS.index(s)] for s in OFF_STEPS],
            [mean(arms[(s, "off")]) for s in OFF_STEPS], "-^", color=C_OFF,
            lw=1.8, ms=6.5, zorder=4, label="contacts withheld")

    for arm, col, dy in (("oracle", C_ORACLE, 10), ("real", C_REAL, -17)):
        v = mean(arms[("final", arm)])
        ax.annotate(f"{v:.3f}", (xs[-1], v), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8.5, color=col,
                    fontweight="bold")
    v = mean(arms[("final", "off")])
    ax.annotate(f"{v:.3f}", (xs[-1], v), textcoords="offset points",
                xytext=(0, -17), ha="center", fontsize=8.5, color=C_OFF,
                fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.set_xlim(-0.35, len(STEPS) - 0.4)
    ax.set_ylim(0.28, 0.95)
    ax.set_xlabel("training step (contacts-msafree-01)")
    ax.set_ylabel("FoldBench lDDT")
    ax.set_title(f"Learning to use contacts\n{n} homology-filtered FoldBench "
                 f"monomers, Helico arms MSA-free", fontsize=11, loc="left")
    ax.legend(loc=(0.045, 0.545), fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
