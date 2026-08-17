"""Figure: folding accuracy by target class, on homology-filtered targets only.

Regenerate with:
    uv run python .agents/project/figures/folding_by_dataset.py

The companion to `contact_accuracy_by_dataset.py`: that one asks how good the
predicted contacts are per class, this one asks what they are worth after
folding, in lDDT.

**Every target shown survives two independent filters.**

  MarinFold homology   MarinFold exp226's eval2: < 40% identity to either
                       training arm (4.1M AFDB + 66.8M ESM-Atlas), mmseqs
                       -s 7.5, hit counted iff evalue <= 1e-3 and qcov >= 0.50
  Helico training      released on or after 2021-09-30, Helico's training cutoff

Neither filter alone is enough. The first leaves targets Helico itself trained
on; the second leaves targets whose fold MarinFold has effectively memorised.
238 of the 380 benched targets clear both; 67 of those are natural proteins.

Contacts come from `contacts-v1-exp199-cooldown-1.5B`, MarinFold's current
default since exp238 -- 100 rollouts, occurrence-frequency voting, truncated at
top-L. Helico arms come from `modal/bench_byclass.py`; Protenix v2 runs through
ByteDance's own implementation. The oracle arm conditions on contacts derived
from the answer, so it is a ceiling, not a prediction.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BYCLASS = ROOT / "experiments/marinfold_contacts/byclass"
OUT = Path(__file__).parent / "folding_by_dataset.png"

# Natural classes first, designs last: the designs are 72% of the filtered set
# and behave completely differently, so pooling them hides the result.
GROUPS = [
    ("foldbench_rest", "FoldBench monomers\n(exp226's net-new)"),
    ("foldbench100", "FoldBench monomers\n(the original 100)"),
    ("cameo_hard", "CAMEO hard"),
    ("casp_fm", "CASP free\nmodelling"),
    ("denovo_pdb", "de novo designs"),
]
SERIES = [
    ("off", "Helico, no contacts", "#7a7a7a"),
    ("v2_singleseq", "Protenix v2, single sequence", "#a0762b"),
    ("cool_L", "Helico + MarinFold contacts", "#b8452f"),
    ("v2_msa", "Protenix v2 + MSA", "#1a7f5a"),
    ("oracle", "Helico + oracle contacts", "#1b5e9c"),
]


def load(arm: str) -> dict[str, float]:
    """{target_id: lddt}, pooling the main run with the net-new top-up."""
    out = {}
    for tag in (arm, f"fbrest_{arm}"):
        f = BYCLASS / "results" / f"{tag}.csv"
        if not f.exists():
            continue
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "ok":
                    continue
                try:
                    v = float(r["lddt"])
                except (TypeError, ValueError):
                    continue
                if not math.isnan(v):
                    out[r["target_id"]] = v
    return out


def paired_stats(a: dict, b: dict, keys):
    """Paired mean difference b - a, standard error, and #improved."""
    d = [b[k] - a[k] for k in keys]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
    return m, (sd / len(d) ** 0.5 if len(d) > 1 else 0.0), sum(1 for x in d if x > 0)


def main():
    with (BYCLASS / "data/targets.csv").open() as f:
        meta = {r["target_id"]: r for r in csv.DictReader(f)}

    arms = {k: load(k) for k, _l, _c in SERIES}
    present = [s for s in SERIES if arms[s[0]]]
    missing = [s[0] for s in SERIES if not arms[s[0]]]
    if missing:
        print(f"NOTE: no results yet for {missing}; plotting without them\n")

    # Paired across every arm that has data, and homology-filtered.
    keys = set.intersection(*[set(arms[s[0]]) for s in present])
    keys = sorted(k for k in keys if meta[k]["in_eval2"] == "1")
    by = {g: [k for k in keys if meta[k]["dataset"] == g] for g, _l in GROUPS}
    natural = [k for k in keys if meta[k]["designed"] in ("0", "False", "false")]

    def mean(arm, ks):
        return sum(arms[arm][k] for k in ks) / len(ks)

    print(f"homology-filtered and paired: n = {len(keys)}  ({len(natural)} natural)\n")
    hdr = f"{'class':22s} {'n':>4s}" + "".join(f" {s[0]:>14s}" for s in present)
    print(hdr)
    for g, _lab in GROUPS:
        ks = by[g]
        if not ks:
            continue
        print(f"{g:22s} {len(ks):4d}" + "".join(f" {mean(s[0], ks):14.3f}" for s in present))
    print(f"{'natural (pooled)':22s} {len(natural):4d}"
          + "".join(f" {mean(s[0], natural):14.3f}" for s in present))
    print()
    for lab, ks in [("natural", natural), *[(g, by[g]) for g, _l in GROUPS if by[g]]]:
        m, se, up = paired_stats(arms["off"], arms["cool_L"], ks)
        line = f"  contacts vs none, {lab:22s} {m:+.3f} +/- {se:.3f}  ({up}/{len(ks)})"
        if arms.get("v2_singleseq"):
            m2, se2, up2 = paired_stats(arms["v2_singleseq"], arms["cool_L"], ks)
            line += f"   vs Protenix v2 SS {m2:+.3f} +/- {se2:.3f} ({up2}/{len(ks)})"
        print(line)

    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    groups = [(g, lab) for g, lab in GROUPS if by[g]]
    xs = list(range(len(groups) + 1))          # +1 for the pooled natural bar
    w = 0.8 / len(present)
    for i, (arm, lab, colr) in enumerate(present):
        vals = [mean(arm, by[g]) for g, _l in groups] + [mean(arm, natural)]
        off = (i - (len(present) - 1) / 2) * w
        ax.bar([x + off for x in xs], vals, width=w, color=colr, alpha=0.9, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x + off, v + 0.010, f"{v:.2f}", ha="center", fontsize=7.4,
                    color=colr, fontweight="bold", rotation=90)

    ax.axvline(len(groups) - 0.5, color="0.75", lw=1.2, ls="--")
    labels = [f"{lab}\nn = {len(by[g])}" for g, lab in groups]
    labels[-1] += "\n(designed)"
    labels.append(f"ALL NATURAL\npooled, n = {len(natural)}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("FoldBench lDDT")
    ax.set_title("Folding accuracy by target class, homology-filtered\n"
                 f"{len(keys)} targets: < 40% identity to MarinFold's training data "
                 f"AND outside Helico's training window",
                 fontsize=11.5, loc="left")
    ax.legend(loc="upper center", fontsize=8.5, framealpha=0.95, ncol=5,
              bbox_to_anchor=(0.5, 1.005), frameon=False)
    ax.grid(axis="y", alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
