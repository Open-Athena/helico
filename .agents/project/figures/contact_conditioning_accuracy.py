"""Figure: MSA-free contact-conditioned folding vs Protenix.

Regenerate with:
    uv run python .agents/project/figures/contact_conditioning_accuracy.py

Reads the FoldBench per-target CSVs written by `modal/bench.py --output-dir
bench_*`. Every Helico arm here is genuinely MSA-free (benched with
HELICO_BENCH_SINGLE_SEQ=1, trained with the MSA loader disabled), so no
alignment-derived signal reaches the model by any route.

Restricted to the protein categories: the contact map is a protein side-chain
feature, so nucleic-acid-only targets carry no signal. They also served as the
empirical null -- both arms are identical by construction there.

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

PROTEIN_CATS = [
    "interface_antibody_antigen",
    "interface_protein_ligand",
    "interface_protein_peptide",
    "interface_protein_protein",
    "monomer_protein",
]
CAT_LABEL = {
    "interface_antibody_antigen": "antibody-antigen",
    "interface_protein_ligand": "protein-ligand",
    "interface_protein_peptide": "protein-peptide",
    "interface_protein_protein": "protein-protein",
    "monomer_protein": "protein monomer",
}
CAT_MARKER = {
    "interface_antibody_antigen": "o",
    "interface_protein_ligand": "s",
    "interface_protein_peptide": "^",
    "interface_protein_protein": "D",
    "monomer_protein": "v",
}

STEPS = [1000, 2000, 3000, 4000, 5000, "final"]

C_ON, C_MF, C_OFF = "#1b5e9c", "#7b52a1", "#b8452f"
C_MSA, C_NOMSA = "#2e7d32", "#7a7a7a"


def bench_dir(step, arm):
    """Checkpoint of contacts-msafree-01.

    arm "on" is the oracle contact map (the ceiling); "mf" degrades it to a
    truncated top-k list at MarinFold's measured operating point, 60% precision
    and 60% recall -- the deployment condition; "off" withholds contacts.
    """
    stem = "final" if step == "final" else f"s{step}"
    return f"bench_mf01_{stem}_{arm}"


def load_lddt(d):
    """{(category, pdb_id): lddt} over protein categories, NaNs dropped."""
    out = {}
    for cat in PROTEIN_CATS:
        f = ROOT / d / "results" / f"{cat}.csv"
        if not f.exists():
            continue
        for row in csv.DictReader(f.open()):
            try:
                v = float(row.get("lddt", ""))
            except (TypeError, ValueError):
                continue
            if not math.isnan(v):
                out[(cat, row["pdb_id"])] = v
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
        for arm in ("on", "mf"):
            arms[(s, arm)] = load_lddt(bench_dir(s, arm))
    arms[(1000, "off")] = load_lddt(bench_dir(1000, "off"))
    arms[("final", "off")] = load_lddt(bench_dir("final", "off"))
    ptx_msa = load_lddt("bench_protenix_msa")
    ptx_ss = load_lddt("bench_protenix_singleseq")

    keys = None
    for d in [*arms.values(), ptx_msa, ptx_ss]:
        keys = set(d) if keys is None else keys & set(d)
    keys = sorted(keys)
    n = len(keys)

    def mean(d):
        return sum(d[k] for k in keys) / n

    print(f"n = {n} paired protein targets\n")
    print(f"{'step':>7} {'off':>8} {'MarinFold':>10} {'oracle':>8}")
    for s in STEPS:
        off = mean(arms[(s, "off")]) if (s, "off") in arms else float("nan")
        print(f"{str(s):>7} {off:8.4f} {mean(arms[(s, 'mf')]):10.4f} "
              f"{mean(arms[(s, 'on')]):8.4f}")
    print(f"\nProtenix + MSA      {mean(ptx_msa):.4f}")
    print(f"Protenix single-seq {mean(ptx_ss):.4f}\n")
    for a, b, lab in [
        (arms[("final", "off")], arms[("final", "mf")], "MarinFold 60/60 vs contacts off"),
        (arms[("final", "mf")], arms[("final", "on")], "oracle vs MarinFold 60/60"),
        (ptx_msa, arms[("final", "mf")], "MarinFold 60/60 vs Protenix+MSA"),
        (ptx_msa, arms[("final", "on")], "oracle vs Protenix+MSA"),
    ]:
        m, se, t, up = paired_stats(a, b, keys)
        print(f"  {lab:34s} {m:+.4f} +/- {se:.4f}  t={t:+.2f}  {up}/{n}")

    fig, (ax, sx) = plt.subplots(1, 2, figsize=(13.4, 5.6))

    xs = list(range(len(STEPS)))
    ax.axhline(mean(ptx_msa), color=C_MSA, ls="--", lw=1.6, zorder=1)
    ax.axhline(mean(ptx_ss), color=C_NOMSA, ls="--", lw=1.6, zorder=1)
    ax.annotate(f"Protenix + MSA  ({mean(ptx_msa):.3f})",
                xy=(0.015, mean(ptx_msa)), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=9, color=C_MSA,
                fontweight="bold", zorder=6)
    ax.annotate(f"Protenix, single sequence  ({mean(ptx_ss):.3f})",
                xy=(0.015, mean(ptx_ss)), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=9, color=C_NOMSA,
                fontweight="bold", zorder=6)

    ax.plot(xs, [mean(arms[(s, "on")]) for s in STEPS], "-o", color=C_ON, lw=1.8,
            ms=7, zorder=3, label="oracle contacts (100%)")
    ax.plot(xs, [mean(arms[(s, "mf")]) for s in STEPS], "-s", color=C_MF, lw=1.8,
            ms=6.5, zorder=3,
            label="MarinFold operating point (60% prec, 60% recall)")
    ax.plot([0, len(STEPS) - 1],
            [mean(arms[(1000, "off")]), mean(arms[("final", "off")])],
            "-^", color=C_OFF, lw=1.8, ms=6.5, zorder=3, label="contacts withheld")

    for s, x in zip(STEPS, xs):
        if s != "final":
            continue
        for arm, col, dy in (("on", C_ON, 9), ("mf", C_MF, -16)):
            v = mean(arms[(s, arm)])
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8.5,
                        color=col, fontweight="bold")
    v = mean(arms[("final", "off")])
    ax.annotate(f"{v:.3f}", (len(STEPS) - 1, v), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=8.5, color=C_OFF,
                fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.set_xlim(-0.35, len(STEPS) - 0.35)
    ax.set_ylim(0.22, 0.93)
    ax.set_xlabel("training step (contacts-msafree-01)")
    ax.set_ylabel("FoldBench lDDT")
    ax.set_title(f"A. MSA-free folding from contacts\n{n} paired protein targets",
                 fontsize=11, loc="left")
    ax.legend(loc=(0.03, 0.40), fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")

    final_mf, final_off = arms[("final", "mf")], arms[("final", "off")]
    sx.plot([0, 1], [0, 1], "-", color="0.55", lw=1.3, zorder=1)
    sx.annotate("y = x", xy=(0.36, 0.385), fontsize=9, color="0.45",
                rotation=45, ha="center", va="center")
    for cat in PROTEIN_CATS:
        ks = [k for k in keys if k[0] == cat]
        if not ks:
            continue
        sx.scatter([ptx_msa[k] for k in ks], [final_mf[k] for k in ks],
                   marker=CAT_MARKER[cat], s=58, color=C_MF, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3,
                   label=CAT_LABEL[cat])
    sx.scatter([ptx_msa[k] for k in keys], [final_off[k] for k in keys],
               marker="x", s=34, color=C_OFF, alpha=0.55, zorder=2,
               label="same targets, contacts withheld")

    m, se, t, up = paired_stats(ptx_msa, final_mf, keys)
    sx.annotate(f"above y=x: {up}/{n} targets\n"
                f"mean d = {m:+.3f} +/- {se:.3f}  (t={t:.1f})",
                xy=(0.035, 0.965), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95))

    sx.set_xlim(0.25, 1.0)
    sx.set_ylim(0.25, 1.0)
    sx.set_aspect("equal")
    sx.set_xlabel("Protenix + MSA  lDDT")
    sx.set_ylabel("Helico + contacts @ 60/60, MSA-free  lDDT")
    sx.set_title(f"B. Per-target at the MarinFold operating point\n"
                 f"final checkpoint, {n} paired protein targets",
                 fontsize=11, loc="left")
    sx.legend(loc="lower left", fontsize=7.8, framealpha=0.95)
    sx.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
