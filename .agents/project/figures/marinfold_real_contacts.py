"""Figure: MarinFold predicted contacts vs the Protenix baselines.

Reads bench_mf2_* on the paired FoldBench monomer set. Helico arms are MSA-free
(no alignment, no conservation profile); the Protenix +MSA arms of course use
alignments -- that is the comparison.

Protenix v2 runs through ByteDance's own implementation, at its recommended
inference settings (5 samples / 10 cycles) versus 3 samples / 6 cycles for the
Helico arms, which favours the baseline.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "marinfold_real_contacts.png"

# (top-n label, bench arm, measured precision, measured recall)
LEVELS = [("L/5", "rollout_L5", 0.795, 0.179),
          ("L/2", "rollout_L2", 0.676, 0.379),
          ("L",   "rollout_L",  0.505, 0.564)]
C_REAL, C_SYN, C_OFF = "#b8452f", "#7b52a1", "#7a7a7a"
C_ORACLE, C_MSA, C_SS = "#1b5e9c", "#2e7d32", "#8a6d3b"
C_V2MSA, C_V2SS = "#1a7f5a", "#a0762b"


def load_v2(tag):
    """Protenix v2 scores, produced by the official implementation.

    Scored through experiments/marinfold_contacts/score_protenix_v2.py, which
    uses the same compute_lddt and the same atom correspondence as the bench --
    the upstream DockQ-based scorer returns 0.000 on monomers.
    """
    f = ROOT / f"experiments/marinfold_contacts/upstream/{tag}_scores.csv"
    if not f.exists():
        return {}
    return {r["pdb_id"]: float(r["lddt"]) for r in csv.DictReader(f.open())}


def load(arm):
    if arm.startswith("v2_"):
        return load_v2(arm)
    f = ROOT / f"bench_mf2_{arm}" / "results" / "monomer_protein.csv"
    out = {}
    for row in csv.DictReader(f.open()):
        try:
            v = float(row.get("lddt", ""))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out[row["pdb_id"]] = v
    return out


def paired(a, b, keys):
    d = [b[k] - a[k] for k in keys]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    return m, sd / len(d) ** 0.5


def main():
    arms = {a: load(a) for a in
            ["off", "oracle", "protenix_msa", "protenix_singleseq",
             "v2_msa", "v2_singleseq", "single_L", "v2ss_derived",
             *[r for _l, r, *_ in LEVELS]]}
    keys = None
    for d in arms.values():
        keys = set(d) if keys is None else keys & set(d)
    keys = sorted(keys)
    n = len(keys)
    mean = lambda a: sum(arms[a][k] for k in keys) / n  # noqa: E731

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 5.6))

    # --- Panel A: accuracy vs how many contacts are supplied ---------------
    xs = range(len(LEVELS))
    real = [mean(r) for _l, r, *_ in LEVELS]
    ax.plot(list(xs), real, "-o", color=C_REAL, lw=1.9, ms=8, zorder=3,
            label="MarinFold contacts (exp199)")

    refs = [(mean("v2_msa"), C_V2MSA, "Protenix v2 + MSA", 0.30, "bottom"),
            (mean("oracle"), C_ORACLE, "oracle contacts", 0.63, "bottom"),
            (mean("protenix_msa"), C_MSA, "Protenix v1 + MSA", 0.955, "top"),
            (mean("v2_singleseq"), C_V2SS, "Protenix v2, single seq", 0.955, "bottom"),
            (mean("v2ss_derived"), C_SYN,
             "contacts read off v2 single-seq structure", 0.63, "top"),
            (mean("off"), C_OFF, "no contacts", 0.30, "top")]
    for val, c, lab, xf, va in refs:
        ax.axhline(val, color=c, ls="--", lw=1.5, zorder=1)
        ax.annotate(f"{lab}  ({val:.3f})", xy=(xf, val),
                    xycoords=("axes fraction", "data"), ha="right", va=va,
                    fontsize=8.5, color=c, fontweight="bold", zorder=6)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"top-{l}\np={p:.2f} r={rc:.2f}" for l, _r, p, rc in LEVELS],
                       fontsize=9)
    ax.set_ylim(0.28, 0.95)
    ax.set_ylabel("FoldBench lDDT")
    ax.set_xlabel("contact budget (and MarinFold's measured accuracy there)")
    ax.set_title(f"A. Folding from predicted contacts\n"
                 f"{n} paired FoldBench monomer targets", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")

    # --- Panel B: per-target vs the best single-sequence baseline ----------
    r, s = arms["rollout_L"], arms["v2_singleseq"]
    bx.plot([0, 1], [0, 1], "-", color="0.55", lw=1.3, zorder=1)
    bx.scatter([s[k] for k in keys], [r[k] for k in keys], s=42, color=C_REAL,
               alpha=0.75, edgecolors="white", linewidths=0.7, zorder=3)
    m, se = paired(s, r, keys)
    above = sum(1 for k in keys if r[k] > s[k])
    bx.annotate(f"contacts better on {above}/{n} targets\n"
                f"mean d = {m:+.3f} +/- {se:.3f}",
                xy=(0.035, 0.965), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95))
    bx.annotate("y = x", xy=(0.22, 0.245), fontsize=9, color="0.45", rotation=45,
                ha="center", va="center")
    bx.set_xlim(0.1, 1.0)
    bx.set_ylim(0.1, 1.0)
    bx.set_aspect("equal")
    bx.set_xlabel("Protenix v2, single sequence   lDDT")
    bx.set_ylabel("Helico + MarinFold contacts, top-L   lDDT")
    bx.set_title(f"B. Per-target vs the strongest single-sequence baseline\n"
                 f"{n} paired targets", fontsize=11, loc="left")
    bx.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"n={n}")
    for a in ("off", "protenix_singleseq", "v2_singleseq", "v2ss_derived",
              "single_L", *[r for _l, r, *_ in LEVELS], "oracle",
              "protenix_msa", "v2_msa"):
        print(f"  {a:14s} {mean(a):.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
