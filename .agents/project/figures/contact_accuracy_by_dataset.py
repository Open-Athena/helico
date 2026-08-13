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

Data (`experiments/marinfold_contacts/rprecision_by_dataset.csv`) is extracted
verbatim from MarinFold's own experiment outputs, not recomputed here:

  MarinFold exp199   MarinFold exp180, exp199_cw_p06_aug_step145199_rows.csv.gz
  Protenix v2        MarinFold exp74,  contact_precision_all.csv

Both are restricted to the 554 targets scored by every arm, so every comparison
is paired. Protenix contacts are read off its predicted structure with pyconfind
(`predictor=structure`); the distogram-derived variant is weaker everywhere and
is reported in the writeup rather than plotted.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments/marinfold_contacts/rprecision_by_dataset.csv"
OUT = Path(__file__).parent / "contact_accuracy_by_dataset.png"

# (key, label, sublabel). foldbench100 is the set the folding results here use.
GROUPS = [
    ("denovo_pdb", "de novo designs", "denovo_pdb"),
    ("foldbench100", "natural: FoldBench\nmonomers", "foldbench100"),
    ("cameo_hard", "natural: CAMEO\nhard", "cameo_hard"),
    ("casp_fm", "natural: CASP\nfree modelling", "casp_fm"),
]
SERIES = [
    ("marinfold_exp199", "MarinFold exp199 (no MSA)", "#b8452f"),
    ("px_ss_structure", "Protenix v2, single sequence", "#a0762b"),
    ("px_msa_structure", "Protenix v2 + MSA", "#1a7f5a"),
]


def paired_stats(a, b):
    """Paired mean difference b - a, its standard error, and #improved."""
    d = [y - x for x, y in zip(a, b)]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
    return m, se, sum(1 for x in d if x > 0)


def main():
    rows = list(csv.DictReader(SRC.open()))
    by = {g: [r for r in rows if r["dataset"] == g] for g, _l, _s in GROUPS}

    def mean(rs, col):
        return sum(float(r[col]) for r in rs) / len(rs)

    print(f"R-precision, {len(rows)} paired targets\n")
    hdr = f"{'target class':16s} {'n':>4s}" + "".join(f" {s[0][:13]:>15s}" for s in SERIES)
    print(hdr)
    for g, _lab, _sub in GROUPS:
        rs = by[g]
        print(f"{g:16s} {len(rs):4d}" + "".join(f" {mean(rs, c):15.3f}" for c, _l, _k in SERIES))
    print(f"{'ALL (weighted)':16s} {len(rows):4d}"
          + "".join(f" {mean(rows, c):15.3f}" for c, _l, _k in SERIES))
    print()
    for g, _lab, _sub in [*GROUPS, ("ALL", "", "")]:
        rs = rows if g == "ALL" else by[g]
        m, se, up = paired_stats([float(r["px_ss_structure"]) for r in rs],
                                 [float(r["marinfold_exp199"]) for r in rs])
        print(f"  MarinFold - Protenix v2 SS on {g:14s} {m:+.3f} +/- {se:.3f}  "
              f"MarinFold better on {up}/{len(rs)}")
    print()
    for c, lab, _k in [("px_ss_distogram", "Protenix v2 SS, distogram", 0),
                       ("px_msa_distogram", "Protenix v2 + MSA, distogram", 0)]:
        print(f"  {lab:32s} " + "  ".join(f"{g}={mean(by[g], c):.3f}"
                                          for g, _l, _s in GROUPS))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    w = 0.26
    xs = list(range(len(GROUPS)))
    for i, (col, lab, colr) in enumerate(SERIES):
        vals = [mean(by[g], col) for g, _l, _s in GROUPS]
        off = (i - 1) * w
        ax.bar([x + off for x in xs], vals, width=w, color=colr, alpha=0.9, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x + off, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5,
                    color=colr, fontweight="bold")

    # The mix that produces the aggregate: 71% of targets are designed.
    frac = len(by["denovo_pdb"]) / len(rows)
    ax.annotate(f"{frac:.0%} of the 554 targets are here,\n"
                f"so this class sets the aggregate",
                xy=(0, 0.90), xycoords=("data", "axes fraction"), ha="center",
                fontsize=8.5, color="0.35", linespacing=1.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\nn = {len(by[g])}" for g, lab, _s in GROUPS], fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("R-precision  (precision among the top-R predicted contacts)")
    ax.set_title("Contact-prediction accuracy by target class\n"
                 "554 paired targets, MarinFold's own evaluation sets",
                 fontsize=11.5, loc="left")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
