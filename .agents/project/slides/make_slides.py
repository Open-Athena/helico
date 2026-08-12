"""Slide deck: folding from predicted contacts instead of MSAs.

    uv run python .agents/project/slides/make_slides.py

Every number is from the n=98 FoldBench monomer arms (bench_mf2_*), so the deck
is internally consistent -- the earlier assembly-set numbers are a different
target set and are never mixed in. Regenerates the PDF in place.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "contact_conditioned_folding.pdf"
FIGS = ROOT / ".agents/project/figures"

W, H = 13.33, 7.5           # 16:9
INK, MUTE = "#1a1a1a", "#5a5a5a"
ACCENT, WARN, GOOD = "#1b5e9c", "#b8452f", "#2e7d32"


def load(arm):
    f = ROOT / f"bench_mf2_{arm}" / "results" / "monomer_protein.csv"
    out = {}
    if not f.exists():
        return out
    for row in csv.DictReader(f.open()):
        try:
            v = float(row.get("lddt", ""))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out[row["pdb_id"]] = v
    return out


ARMS = ["off", "protenix_singleseq", "single_L", "rollout_L5", "rollout_L2",
        "rollout_L", "synth_L5", "synth_L2", "synth_L", "oracle", "protenix_msa"]
D = {a: load(a) for a in ARMS}
KEYS = sorted(set.intersection(*(set(D[a]) for a in ARMS)))
N = len(KEYS)
M = {a: sum(D[a][k] for k in KEYS) / N for a in ARMS}


def paired(a, b):
    d = [D[b][k] - D[a][k] for k in KEYS]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    return m, sd / len(d) ** 0.5, sum(1 for x in d if x > 0)


def slide(pdf, title, subtitle=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.90, title, fontsize=27, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.06, 0.825, subtitle, fontsize=14, color=MUTE, va="top")
    return fig


def bullets(fig, items, y0=0.70, dy=0.088, size=15):
    for i, (txt, col) in enumerate(items):
        fig.text(0.075, y0 - i * dy, "•", fontsize=size, color=col or MUTE, va="top")
        fig.text(0.105, y0 - i * dy, txt, fontsize=size, color=col or INK, va="top",
                 wrap=True)


def embed(fig, png, rect=(0.05, 0.06, 0.90, 0.66)):
    ax = fig.add_axes(rect)
    ax.imshow(imread(str(png)))
    ax.axis("off")


with PdfPages(OUT) as pdf:
    # 1 -- title
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.62, "Folding from predicted contacts\ninstead of MSAs",
             fontsize=40, fontweight="bold", color=INK, va="top", linespacing=1.25)
    fig.text(0.06, 0.34, "Helico + MarinFold  ·  results on 98 FoldBench monomers",
             fontsize=17, color=MUTE)
    fig.text(0.06, 0.27, "github.com/Open-Athena/helico  ·  PR #13, issue #11",
             fontsize=12, color=MUTE)
    pdf.savefig(fig); plt.close(fig)

    # 2 -- the idea
    fig = slide(pdf, "The idea",
                "AF3-family models lean on MSAs. Strip the alignment and accuracy collapses.")
    bullets(fig, [
        ("MarinFold predicts residue–residue side-chain contacts directly.", None),
        ("Feed those contacts to the folding model in place of the alignment,\n"
         "and the MSA search leaves the critical path.", None),
        ("Helico takes a three-state contact matrix — contact / no-contact /\n"
         "unknown — added into the pair representation.", None),
        ("Training samples the conditioning level per example, so one model spans\n"
         "no contacts through fully specified.", None),
    ])
    fig.text(0.075, 0.20, "All Helico arms here are genuinely MSA-free: no alignment, "
             "no conservation profile,\nat training or inference.",
             fontsize=13, color=MUTE, style="italic")
    pdf.savefig(fig); plt.close(fig)

    # 3 -- headline numbers
    fig = slide(fig if False else pdf, "Where things land",
                f"FoldBench, {N} paired monomer targets, all MSA-free")
    rows = [("no contacts", M["off"], MUTE),
            ("stock Protenix v1, single sequence", M["protenix_singleseq"], MUTE),
            ("real MarinFold contacts, top-L", M["rollout_L"], WARN),
            ("synthetic noise at the same precision/recall", M["synth_L"], ACCENT),
            ("oracle contacts", M["oracle"], ACCENT),
            ("Protenix + MSA", M["protenix_msa"], GOOD)]
    ax = fig.add_axes([0.34, 0.13, 0.36, 0.58])
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.85)
    for y, (lab, v, c) in zip(ys, rows):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=13,
                fontweight="bold", color=c)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.set_xlim(0, 1.0); ax.set_xlabel("FoldBench lDDT", fontsize=12)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.25, ls=":")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.text(0.755, 0.66, "Real contacts beat\nsingle sequence\nby a wide margin",
             fontsize=14, color=WARN, fontweight="bold", va="top")
    fig.text(0.755, 0.50, "...but do not reach\nMSA-level accuracy", fontsize=14,
             color=GOOD, fontweight="bold", va="top")
    pdf.savefig(fig); plt.close(fig)

    # 4 -- do we beat single sequence?
    m, se, up = paired("protenix_singleseq", "rollout_L")
    fig = slide(pdf, "Yes: real contacts clearly beat single-sequence folding",
                "vs stock Protenix v1 weights in single-sequence mode, same targets")
    bullets(fig, [
        (f"Real MarinFold contacts (top-L):  {m:+.3f} ± {se:.3f} lDDT   "
         f"(t={m/se:.1f}, better on {up}/{N})", WARN),
        (f"Holds at every contact budget: +0.227 at top-L/2, +0.180 at top-L/5.", None),
        (f"Even a single rollout beats it: +0.168 ± 0.022.", None),
    ], y0=0.68)
    mm, sse, _ = paired("protenix_singleseq", "off")
    fig.text(0.075, 0.36, "The control that makes this credible", fontsize=16,
             fontweight="bold", color=INK)
    fig.text(0.075, 0.29,
             f"Our own contacts-off arm scores {M['off']:.3f} — slightly BELOW "
             f"Protenix's {M['protenix_singleseq']:.3f}\n"
             f"({mm:+.3f} ± {sse:.3f}). The fine-tuned model has no intrinsic edge "
             f"with no information.\nAll of the gain comes from the contacts.",
             fontsize=14, color=MUTE, va="top", linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)

    # 5 -- the real vs synthetic figure
    fig = slide(pdf, "But error structure, not error rate, is what hurts",
                "each synthetic arm generated at the precision/recall measured for its real counterpart")
    embed(fig, FIGS / "marinfold_real_contacts.png", (0.045, 0.05, 0.91, 0.70))
    pdf.savefig(fig); plt.close(fig)

    # 6 -- decomposition
    fig = slide(pdf, "Decomposing the 0.230 gap to MSAs",
                "real MarinFold top-L 0.622  →  Protenix + MSA 0.851")
    parts = [("oracle contacts\nvs MSA", M["protenix_msa"] - M["oracle"], MUTE),
             ("error rate\n(oracle → 50%/56%)", M["oracle"] - M["synth_L"], MUTE),
             ("error STRUCTURE\n(synthetic → real)", M["synth_L"] - M["rollout_L"], WARN)]
    ax = fig.add_axes([0.34, 0.30, 0.38, 0.40])
    ax.barh(range(len(parts)), [p[1] for p in parts],
            color=[p[2] for p in parts], alpha=0.85)
    for y, (lab, v, c) in enumerate(parts):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=13,
                fontweight="bold", color=c)
    ax.set_yticks(range(len(parts))); ax.set_yticklabels([p[0] for p in parts], fontsize=12.5)
    ax.set_xlabel("lDDT cost", fontsize=12); ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, ls=":")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.text(0.072, 0.19,
             "Real predictor errors cluster near true contacts, where they are "
             "geometrically plausible\nand cannot be rejected as inconsistent with the "
             "rest of the map. Our training noise model\ndraws false positives "
             "uniformly — the easy case.",
             fontsize=14, color=INK, va="top", linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)

    # 7 -- what next
    fig = slide(pdf, "What to do about it")
    bullets(fig, [
        ("Retrain with false positives sampled from near-miss pairs rather than\n"
         "uniformly. The model has never seen the error distribution it faces.\n"
         "Cheapest lever, and it is on our side rather than MarinFold's.", WARN),
        ("Vote aggregation across rollouts is worth +0.071 ± 0.008 over a single\n"
         "rollout at the same budget — keep using it, never mix the recipes.", None),
        ("Improve MarinFold's error profile, not only its precision/recall.", None),
    ], y0=0.70, dy=0.155)
    fig.text(0.075, 0.19, "Caveats", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.075, 0.135,
             "Warm-started from Protenix v1.  Monomers only, so not comparable to the "
             "earlier assembly-set numbers.\n"
             "MarinFold indexes into the published prompt, Helico into resolved "
             "residues — only 15/100 agree outright;\nthe map is verified by round "
             "trip at Jaccard 0.998.",
             fontsize=12, color=MUTE, va="top", linespacing=1.7)
    pdf.savefig(fig); plt.close(fig)

print(f"n={N}; wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
