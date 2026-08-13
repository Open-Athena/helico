"""Figure: contact-prediction accuracy by target class, MarinFold vs Protenix v2.

Regenerate with:
    uv run python .agents/project/figures/contact_accuracy_by_dataset.py

This exists to answer a specific objection. MarinFold's own evaluations report
its models as roughly on par with Protenix v2 single sequence at contact
recapitulation, while the folding results here rest on MarinFold supplying a map
about 1.9x more precise. Both are true: the aggregate is a mean over a target mix
that is 71% designed proteins, where Protenix is much the better contact
predictor. MarinFold's advantage turns out to be confined to foldbench100 -- it
also loses on CAMEO hard and ties on CASP free modelling, both natural sets, so
"natural vs designed" is not the axis that explains it.

Metric is R-precision -- precision among the top-R predicted contacts, where R is
the ground-truth contact count, so the budget matches the answer's size. Primary
separation >= 6, all ranges pooled.

**Homology-filtered.** Every target shown is in MarinFold exp226's `eval2`:
< 40% identity to either training arm (4.1M AFDB + 66.8M ESM-Atlas), mmseqs
-s 7.5, a hit counted only at evalue <= 1e-3 and qcov >= 0.50. 307 of the
expanded 776 survive. Numbers on the unfiltered set are not shown -- they are
dominated by targets MarinFold has effectively memorised.

Data (`experiments/marinfold_contacts/rprecision_eval2.csv.gz`) is exp226's
per-protein table verbatim, not recomputed here. Protenix contacts are read off
its predicted structure with pyconfind, which is the stronger of its two routes.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments/marinfold_contacts/rprecision_eval2.csv.gz"
OUT = Path(__file__).parent / "contact_accuracy_by_dataset.png"

# Same order as folding_by_dataset.py so the two figures read together.
GROUPS = [
    ("foldbench_rest", "FoldBench monomers\n(exp226's net-new)"),
    ("foldbench100", "FoldBench monomers\n(the original 100)"),
    ("cameo_hard", "CAMEO hard"),
    ("casp_fm", "CASP free\nmodelling"),
    ("denovo_pdb", "de novo designs"),
]
SERIES = [
    ("Protenix-v2 single-seq", "Protenix v2, single sequence", "#a0762b"),
    ("MarinFold #199 (1.5B, seq only)", "MarinFold exp199 (no MSA)", "#b8452f"),
    ("Protenix-v2 + MSA", "Protenix v2 + MSA", "#1a7f5a"),
]


def paired_stats(a, b):
    """Paired mean difference b - a, its standard error, and #improved."""
    d = [y - x for x, y in zip(a, b)]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
    return m, se, sum(1 for x in d if x > 0)


def main():
    import gzip

    with gzip.open(SRC, "rt") as f:
        rows = [r for r in csv.DictReader(f) if r["cut"] == "R" and r["range"] == "all"]
    by = {g: [r for r in rows if r["dataset"] == g] for g, _l in GROUPS}
    natural = [r for r in rows if r["designed_any"].lower() in ("false", "0", "")]

    def mean(rs, col):
        return sum(float(r[col]) for r in rs) / len(rs)

    print(f"R-precision on eval2 (< 40% training identity): {len(rows)} targets, "
          f"{len(natural)} natural\n")
    print(f"{'class':22s} {'n':>4s}" + "".join(f" {s[1][:20]:>22s}" for s in SERIES))
    for g, _lab in GROUPS:
        rs = by[g]
        print(f"{g:22s} {len(rs):4d}" + "".join(f" {mean(rs, c):22.3f}" for c, _l, _k in SERIES))
    print(f"{'natural (pooled)':22s} {len(natural):4d}"
          + "".join(f" {mean(natural, c):22.3f}" for c, _l, _k in SERIES))
    print()
    for lab, rs in [("natural", natural), *[(g, by[g]) for g, _l in GROUPS]]:
        m, se, up = paired_stats([float(r["Protenix-v2 single-seq"]) for r in rs],
                                 [float(r["MarinFold #199 (1.5B, seq only)"]) for r in rs])
        print(f"  MarinFold - Protenix v2 SS, {lab:22s} {m:+.3f} +/- {se:.3f}  "
              f"({up}/{len(rs)})")

    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    xs = list(range(len(GROUPS) + 1))          # +1 for the pooled natural bar
    w = 0.8 / len(SERIES)
    for i, (col, lab, colr) in enumerate(SERIES):
        vals = [mean(by[g], col) for g, _l in GROUPS] + [mean(natural, col)]
        off = (i - (len(SERIES) - 1) / 2) * w
        ax.bar([x + off for x in xs], vals, width=w, color=colr, alpha=0.9, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=8,
                    color=colr, fontweight="bold")

    ax.axvline(len(GROUPS) - 0.5, color="0.75", lw=1.2, ls="--")
    labels = [f"{lab}\nn = {len(by[g])}" for g, lab in GROUPS]
    labels[-1] += "\n(designed)"
    labels.append(f"ALL NATURAL\npooled, n = {len(natural)}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("R-precision  (precision among the top-R predicted contacts)")
    ax.set_title("Contact-prediction accuracy by target class, homology-filtered\n"
                 f"MarinFold exp226 eval2: {len(rows)} targets at < 40% identity to "
                 f"MarinFold's training data", fontsize=11.5, loc="left")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
