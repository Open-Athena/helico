"""Figure: real MarinFold contacts vs synthetic noise at matched precision/recall.

Reads bench_mf2_* (98 paired FoldBench monomer targets, all MSA-free). Each
synthetic arm was generated at the precision/recall measured for its real
counterpart, so the real-vs-synthetic gap isolates error *structure* from error
*rate* -- the question helico#11 exists to answer.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "marinfold_real_contacts.png"

# (top-n label, real arm, synthetic arm, measured precision, recall)
LEVELS = [("L/5", "rollout_L5", "synth_L5", 0.795, 0.179),
          ("L/2", "rollout_L2", "synth_L2", 0.676, 0.379),
          ("L",   "rollout_L",  "synth_L",  0.505, 0.564)]
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
             "v2_msa", "v2_singleseq", "single_L",
             *[x for _l, r, s, *_ in LEVELS for x in (r, s)]]}
    keys = None
    for d in arms.values():
        keys = set(d) if keys is None else keys & set(d)
    keys = sorted(keys)
    n = len(keys)
    mean = lambda a: sum(arms[a][k] for k in keys) / n  # noqa: E731

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 5.6))

    # --- Panel A: real vs matched synthetic across contact budgets ----------
    xs = range(len(LEVELS))
    real = [mean(r) for _l, r, _s, *_ in LEVELS]
    syn = [mean(s) for _l, _r, s, *_ in LEVELS]
    for y, c, lab, mk in ((syn, C_SYN, "synthetic noise at the same precision/recall", "s"),
                          (real, C_REAL, "real MarinFold contacts (exp199)", "o")):
        ax.plot(list(xs), y, f"-{mk}", color=c, lw=1.9, ms=8, zorder=3, label=lab)
    for x, (lab, r, s, p, rc) in enumerate(LEVELS):
        m, se = paired(arms[s], arms[r], keys)
        ax.annotate("", xy=(x, mean(r) + 0.008), xytext=(x, mean(s) - 0.008),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.3))
        ax.annotate(f"{m:+.3f}", xy=(x, (mean(r) + mean(s)) / 2), xytext=(7, 0),
                    textcoords="offset points", fontsize=9, color="0.2", va="center")

    # The three ceiling lines sit within 0.01 lDDT of each other, so their
    # labels are staggered horizontally rather than stacked on top of one
    # another. (xfrac, va) is per line.
    refs = [(mean("v2_msa"), C_V2MSA, "Protenix v2 + MSA", 0.30, "bottom"),
            (mean("oracle"), C_ORACLE, "oracle contacts", 0.63, "bottom"),
            (mean("protenix_msa"), C_MSA, "Protenix v1 + MSA", 0.955, "top"),
            (mean("v2_singleseq"), C_V2SS, "Protenix v2, single seq", 0.955, "bottom"),
            (mean("protenix_singleseq"), C_SS, "Protenix v1, single seq", 0.63, "top"),
            (mean("off"), C_OFF, "no contacts", 0.30, "top")]
    for val, c, lab, xf, va in refs:
        ax.axhline(val, color=c, ls="--", lw=1.5, zorder=1)
        ax.annotate(f"{lab}  ({val:.3f})", xy=(xf, val),
                    xycoords=("axes fraction", "data"), ha="right", va=va,
                    fontsize=8.5, color=c, fontweight="bold", zorder=6)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"top-{l}\np={p:.2f} r={rc:.2f}" for l, _r, _s, p, rc in LEVELS],
                       fontsize=9)
    ax.set_ylim(0.28, 0.95)
    ax.set_ylabel("FoldBench lDDT")
    ax.set_xlabel("contact budget (and MarinFold's measured accuracy there)")
    ax.set_title(f"A. Real predictor errors cost more than the same error rate\n"
                 f"{n} paired FoldBench monomer targets, MSA-free", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")

    # --- Panel B: per-target, real vs synthetic at top-L -------------------
    r, s = arms["rollout_L"], arms["synth_L"]
    bx.plot([0, 1], [0, 1], "-", color="0.55", lw=1.3, zorder=1)
    bx.scatter([s[k] for k in keys], [r[k] for k in keys], s=42, color=C_REAL,
               alpha=0.75, edgecolors="white", linewidths=0.7, zorder=3)
    m, se = paired(s, r, keys)
    below = sum(1 for k in keys if r[k] < s[k])
    bx.annotate(f"real below synthetic on {below}/{n} targets\n"
                f"mean d = {m:+.3f} +/- {se:.3f}",
                xy=(0.035, 0.965), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95))
    bx.annotate("y = x", xy=(0.30, 0.325), fontsize=9, color="0.45", rotation=45,
                ha="center", va="center")
    bx.set_xlim(0.15, 1.0)
    bx.set_ylim(0.15, 1.0)
    bx.set_aspect("equal")
    bx.set_xlabel("synthetic noise, p=0.505 r=0.564   lDDT")
    bx.set_ylabel("real MarinFold contacts, top-L   lDDT")
    bx.set_title(f"B. Per-target at MarinFold's operating point\n"
                 f"{n} paired targets", fontsize=11, loc="left")
    bx.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"n={n}")
    for a in ("off", "protenix_singleseq", "single_L", *[r for _l, r, _s, *_ in LEVELS],
              *[s for _l, _r, s, *_ in LEVELS], "oracle", "protenix_msa"):
        print(f"  {a:14s} {mean(a):.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
