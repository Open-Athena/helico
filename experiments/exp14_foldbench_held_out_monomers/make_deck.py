"""Slide deck for exp14: Helico on exp245's held-out FoldBench monomers.

Figures are deliberately plain -- bars, dots, error bars, an identity line --
and carry no interpretation. What a slide means is written on the slide, not
drawn into the figure.

Every mean shown has a 95% percentile bootstrap interval over 10,000 resamples
of the proteins, and every arm on a given slide sees the same resamples, so the
intervals are comparable and differences between arms are paired. That is
stated on each slide rather than annotated in the axes.

All four metrics `score_monomer` computes are shown. Note that TM-score,
GDT-TS and RMSD are computed over one representative atom per residue (CA), so
the RMSD here is a CA RMSD -- see `helico.bench.extract_backbone_coords`.

Numbers come from `data/`; nothing is hardcoded.

    uv run python experiments/exp14_foldbench_held_out_monomers/make_deck.py
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

OUT = U.HERE / "exp14_deck.pdf"
W, H = 13.33, 7.5

INK, MUTE, GRID = "#1b1b1b", "#5c5c5c", "#d8d8d8"
SET_COLOR = {"eval-val": "#4269d0", "eval-test": "#b8452f", "eval-denovo": "#3ca951"}
SET_ORDER = ("eval-val", "eval-test", "eval-denovo")

#: Reporting order, top to bottom.
ARM_ORDER = ("off", "v2ss", "mf_L5", "mf_L2", "mf_L", "v2msa", "oracle",
             "protenix_v2_single_seq", "protenix_v2_msa")
SHORT = {
    "off": "Helico, no contacts",
    "v2ss": "Helico + Protenix-v2 SS contacts",
    "mf_L5": "Helico + MarinFold L/5",
    "mf_L2": "Helico + MarinFold L/2",
    "mf_L": "Helico + MarinFold top-L",
    "v2msa": "Helico + Protenix-v2 MSA contacts",
    "oracle": "Helico + oracle contacts",
    "protenix_v2_single_seq": "Protenix v2, single sequence",
    "protenix_v2_msa": "Protenix v2 + MSA",
}
METRIC_LABEL = {"lddt": "lDDT", "tm_score": "TM-score", "gdt_ts": "GDT-TS",
                "rmsd": "CA RMSD (Å)"}

#: A second pass over the same numbers, requested in a different reading order
#: and with natural and designed proteins separated rather than shown as three
#: eval sets. The arms conditioned on contacts read off a Protenix structure
#: are left out here: they answer "is the conditioning channel faithful", not
#: "how good are the predictors", which is what this ordering is for.
SPLIT_ORDER = ("protenix_v2_msa", "protenix_v2_single_seq", "off",
               "mf_L5", "mf_L2", "mf_L", "oracle")
#: eval-val + eval-test pooled, against the designed monomers.
SPLIT_PANELS = (("all-natural", "natural proteins (eval-val + eval-test)",
                 "#4269d0"),
                ("eval-denovo", "de novo designed proteins (eval-denovo)",
                 "#3ca951"))


def slide(pdf, title, subtitle=None):
    """Title and a wrapped subtitle. Matplotlib does not wrap `fig.text`, so a
    long subtitle runs off the page rather than reflowing -- wrap it here."""
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.055, 0.955, "\n".join(textwrap.wrap(title, 62)), fontsize=24,
             fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.055, 0.876, "\n".join(textwrap.wrap(subtitle, 118)),
                 fontsize=12, color=MUTE, va="top", linespacing=1.45)
    return fig


def finish(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def tidy(ax, *, xlabel=None, ylabel=None, xgrid=True, ygrid=False):
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=11)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.7)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTE, length=0, labelsize=10)


def metric_bars(fig, frame, metric, arms):
    """Horizontal bars, one group per eval set, with bootstrap intervals."""
    sub = frame[(frame.metric == metric) & frame.eval_set.isin(SET_ORDER)]
    ax = fig.add_axes((0.30, 0.10, 0.66, 0.70))
    height = 0.26
    positions = {arm: len(arms) - i for i, arm in enumerate(arms)}
    for k, eval_set in enumerate(SET_ORDER):
        rows = sub[sub.eval_set == eval_set].set_index("arm")
        offset = (1 - k) * height
        ys, means, los, his = [], [], [], []
        for arm in arms:
            if arm not in rows.index:
                continue
            ys.append(positions[arm] + offset)
            means.append(rows.loc[arm, "mean"])
            los.append(rows.loc[arm, "mean"] - rows.loc[arm, "ci_lo"])
            his.append(rows.loc[arm, "ci_hi"] - rows.loc[arm, "mean"])
        n = int(rows.n.iloc[0]) if len(rows) else 0
        ax.barh(ys, means, height=height * 0.9, color=SET_COLOR[eval_set],
                label=f"{eval_set} (n={n})",
                xerr=[los, his], error_kw={"ecolor": INK, "elinewidth": 1.1,
                                          "capsize": 2.5, "capthick": 1.1})
    ax.set_yticks([positions[a] for a in arms])
    ax.set_yticklabels([SHORT[a] for a in arms], fontsize=11)
    ax.set_ylim(0.4, len(arms) + 0.8)
    tidy(ax, xlabel=METRIC_LABEL[metric])
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.075), labelcolor=INK, fontsize=11)
    return ax


def metric_bars_split(fig, frame, metric, arms):
    """Two panels on one slide: natural proteins, then designed ones."""
    axes = []
    for k, (eval_set, panel_title, color) in enumerate(SPLIT_PANELS):
        rows = frame[(frame.metric == metric)
                     & (frame.eval_set == eval_set)].set_index("arm")
        left = 0.285 + k * 0.375
        ax = fig.add_axes((left, 0.135, 0.30, 0.595))
        present = [a for a in arms if a in rows.index]
        positions = [len(present) - i for i, a in enumerate(present)]
        means = [rows.loc[a, "mean"] for a in present]
        lo = [rows.loc[a, "mean"] - rows.loc[a, "ci_lo"] for a in present]
        hi = [rows.loc[a, "ci_hi"] - rows.loc[a, "mean"] for a in present]
        ax.barh(positions, means, height=0.62, color=color,
                xerr=[lo, hi], error_kw={"ecolor": INK, "elinewidth": 1.1,
                                        "capsize": 3, "capthick": 1.1})
        ax.set_yticks(positions)
        if k == 0:
            ax.set_yticklabels([SHORT[a] for a in present], fontsize=10.5)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(0.35, len(present) + 0.65)
        n = int(rows.n.iloc[0]) if len(rows) else 0
        ax.set_title(f"{panel_title}\nn = {n}", fontsize=11, color=INK, pad=8)
        tidy(ax, xlabel=METRIC_LABEL[metric])
        axes.append(ax)
    # One shared x range so the two panels are read against each other rather
    # than each auto-scaling to its own spread.
    lo = min(ax.get_xlim()[0] for ax in axes)
    hi = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(lo, hi)
    return axes


def scatter_panel(ax, frame, x_arm, y_arm, metric, lower_better):
    merged = frame.pivot_table(index="target_id", columns="arm", values=metric)
    meta = frame.drop_duplicates("target_id").set_index("target_id")
    merged = merged.dropna(subset=[x_arm, y_arm])
    sets = meta.loc[merged.index, "eval_set"]
    for eval_set in SET_ORDER:
        mask = (sets == eval_set).to_numpy()
        if not mask.any():
            continue
        ax.plot(merged[x_arm].to_numpy()[mask], merged[y_arm].to_numpy()[mask],
                "o", ms=3.4, alpha=0.55, color=SET_COLOR[eval_set], ls="none",
                label=eval_set)
    lo = float(min(merged[x_arm].min(), merged[y_arm].min()))
    hi = float(max(merged[x_arm].max(), merged[y_arm].max()))
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color=MUTE, lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    better = int((merged[y_arm] < merged[x_arm]).sum()) if lower_better \
        else int((merged[y_arm] > merged[x_arm]).sum())
    ax.set_title(f"{SHORT[y_arm]}\n{better}/{len(merged)} above the line",
                 fontsize=9.5, color=INK, pad=6)
    tidy(ax, ygrid=True)
    return len(merged)


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    metrics = pd.read_csv(U.DATA / "headline_metrics.csv")
    per_target = pd.read_csv(U.DATA / "per_target.csv")
    per_target = per_target[per_target.status == "ok"]
    deltas = pd.read_csv(U.DATA / "paired_deltas.csv")
    arms = [a for a in ARM_ORDER if a in set(metrics.arm)]
    n_units = int(metrics[(metrics.metric == "lddt")
                          & (metrics.eval_set == "all")].n.iloc[0])

    with PdfPages(OUT) as pdf:
        # --- 1. title -------------------------------------------------
        fig = plt.figure(figsize=(W, H))
        fig.patch.set_facecolor("white")
        fig.text(0.055, 0.70, "Folding from predicted contacts", fontsize=40,
                 fontweight="bold", color=INK)
        fig.text(0.055, 0.615, "Helico on MarinFold exp245's held-out FoldBench "
                 "monomer sets", fontsize=19, color=MUTE)
        for i, line in enumerate([
                f"{n_units} monomers scored by all 9 predictors  ·  "
                "eval-val 95 / eval-test 210 / eval-denovo 19",
                "Contacts from exp232 m2-p06, decontaminated against every "
                "protein scored here",
                "Helico is MSA-free throughout: one sequence and a contact map",
                "Means carry 95% percentile bootstrap intervals, 10,000 "
                "resamples of the proteins",
        ]):
            fig.text(0.055, 0.47 - i * 0.055, "— " + line, fontsize=13.5,
                     color=INK if i < 3 else MUTE)
        fig.text(0.055, 0.10, "Open-Athena/helico #14  ·  data and structures at "
                 "hf://buckets/timodonnell/helico-experiments/"
                 "exp14_foldbench_held_out_monomers/",
                 fontsize=10.5, color=MUTE)
        finish(pdf, fig)

        # --- 2-5. one slide per metric --------------------------------
        for metric, lower_better in (("lddt", False), ("tm_score", False),
                                     ("gdt_ts", False), ("rmsd", True)):
            direction = "lower is better" if lower_better else "higher is better"
            fig = slide(
                pdf, f"{METRIC_LABEL[metric]} by conditioning arm",
                f"Mean per eval set, {direction}. Error bars are 95% percentile "
                f"bootstrap intervals over 10,000 resamples of the proteins; "
                f"every arm sees the same resamples.")
            metric_bars(fig, metrics, metric, arms)
            finish(pdf, fig)

        # --- 6. paired deltas -----------------------------------------
        fig = slide(
            pdf, "Paired differences against MarinFold contacts",
            "eval-test only. Intervals are 95% percentile bootstrap on the "
            "per-target difference, which is narrower than differencing two "
            "per-arm intervals.")
        sub = deltas[(deltas.eval_set == "eval-test") & (deltas.a == "mf_L")]
        ax = fig.add_axes((0.34, 0.15, 0.60, 0.63))
        ys = range(len(sub))[::-1]
        ax.barh(list(ys), sub.delta, height=0.55, color="#b8452f",
                xerr=[sub.delta - sub.ci_lo, sub.ci_hi - sub.delta],
                error_kw={"ecolor": INK, "elinewidth": 1.1, "capsize": 3})
        ax.axvline(0, color=MUTE, lw=1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"vs {SHORT.get(b, b)}" for b in sub.b], fontsize=11)
        tidy(ax, xlabel="difference in mean lDDT (MarinFold top-L minus the other arm)")
        finish(pdf, fig)

        # --- 7-8. per-protein scatters --------------------------------
        for metric, lower_better in (("lddt", False), ("tm_score", False)):
            fig = slide(
                pdf, f"Per-protein {METRIC_LABEL[metric]}: MarinFold vs each "
                     f"predictor",
                "One point per protein. x is Helico + MarinFold top-L, y is the "
                "other predictor; the line is y = x. No bootstrap here — these "
                "are individual proteins, not means.")
            others = [a for a in ARM_ORDER if a != "mf_L" and a in set(per_target.arm)]
            grid = fig.subplots(2, 4, gridspec_kw={
                "left": 0.055, "right": 0.975, "top": 0.745, "bottom": 0.115,
                "hspace": 0.52, "wspace": 0.30})
            for ax, other in zip(grid.flat, others):
                scatter_panel(ax, per_target, "mf_L", other, metric, lower_better)
            for ax in list(grid.flat)[len(others):]:
                ax.axis("off")
            handles = [plt.Line2D([], [], marker="o", ls="none", ms=6,
                                  color=SET_COLOR[s], label=s) for s in SET_ORDER]
            fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
                       bbox_to_anchor=(0.5, 0.005), fontsize=11, labelcolor=INK)
            fig.text(0.055, 0.045, f"x axis: Helico + MarinFold top-L "
                     f"({METRIC_LABEL[metric]})", fontsize=10, color=MUTE)
            finish(pdf, fig)

        # --- 9. contact precision vs accuracy -------------------------
        fig = slide(
            pdf, "Accuracy against the precision of the contacts supplied",
            "One point per protein, for the three predicted-contact arms. No "
            "bootstrap here — these are individual proteins.")
        accuracy = []
        marinfold = pd.read_csv(U.DATA / "marinfold_arm_accuracy.csv")
        for cut in ("L", "L2", "L5"):
            accuracy.append(marinfold[["target_id", f"precision_{cut}"]]
                            .rename(columns={f"precision_{cut}": "precision"})
                            .assign(arm=f"mf_{cut}"))
        v2 = pd.read_csv(U.DATA / "v2_arm_accuracy.csv")
        accuracy.append(v2[["target_id", "arm", "precision"]])
        merged = per_target.merge(pd.concat(accuracy, ignore_index=True),
                                  on=["target_id", "arm"], how="inner")
        ax = fig.add_axes((0.09, 0.13, 0.60, 0.64))
        for arm, color in (("v2ss", "#7a4fbf"), ("mf_L", "#b8452f"),
                           ("v2msa", "#1a7f5a")):
            part = merged[merged.arm == arm]
            if part.empty:
                continue
            ax.plot(part.precision, part.lddt, "o", ms=3.6, alpha=0.5,
                    color=color, ls="none", label=SHORT[arm])
        tidy(ax, xlabel="precision of the contacts supplied", ylabel="lDDT",
             ygrid=True)
        ax.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=10)
        finish(pdf, fig)

        # --- 10. does the confidence head still work without an MSA? ---
        fig = slide(
            pdf, "Predicted confidence against measured accuracy",
            "One point per protein: the mean pLDDT Helico's confidence head "
            "reported, against the lDDT that was measured. This is the score "
            "used to pick one of the three diffusion samples. No bootstrap "
            "here — these are individual proteins.")
        ax = fig.add_axes((0.09, 0.13, 0.55, 0.64))
        conf_arms = (("off", "#5c5c5c"), ("v2ss", "#7a4fbf"),
                     ("mf_L", "#b8452f"), ("v2msa", "#1a7f5a"),
                     ("oracle", "#4269d0"))
        summary = []
        for arm, color in conf_arms:
            part = per_target[(per_target.arm == arm) & per_target.mean_plddt.notna()]
            if part.empty:
                continue
            ax.plot(part.mean_plddt, part.lddt, "o", ms=3.4, alpha=0.5,
                    color=color, ls="none", label=SHORT[arm])
            summary.append((arm, part.mean_plddt.mean(), part.lddt.mean(),
                            part.mean_plddt.corr(part.lddt, method="spearman")))
        tidy(ax, xlabel="mean pLDDT reported", ylabel="lDDT measured", ygrid=True)
        ax.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9.5)
        fig.text(0.70, 0.74, "mean pLDDT · mean lDDT · Spearman",
                 fontsize=10.5, color=INK, va="top", fontweight="bold")
        for i, (arm, plddt, lddt, rho) in enumerate(summary):
            fig.text(0.70, 0.685 - i * 0.052,
                     f"{SHORT[arm]}\n{plddt:.1f}  ·  {lddt:.3f}  ·  {rho:.2f}",
                     fontsize=9.5, color=MUTE, va="top", linespacing=1.5)
        finish(pdf, fig)

        # --- 11. how it was run ---------------------------------------
        fig = slide(pdf, "How each number was produced", None)
        lines = [
            ("Helico", "contacts-msafree-01 step 6000 · 6 trunk recycles · "
                       "1 trunk run · 3 diffusion samples · seed 42 · bfloat16 "
                       "· no MSA · best of the 3 by the confidence head's own "
                       "ranking score (pTM for a single chain)"),
            ("Protenix v2", "protenix 2.0.0, model protenix-v2 · 10 trunk "
                            "recycles · 1 seed · 5 diffusion samples · 200 "
                            "diffusion steps · top-1 of the 5 by the model's "
                            "own ranking score. More inference compute than "
                            "the Helico arms get, which is conservative here."),
            ("MarinFold contacts", "exp232 m2-p06 step 145199 · 100 rollouts per "
                                   "protein · occurrence-frequency voting · "
                                   "cut at top-L, L/2, L/5 of the prompt length"),
            ("Hardware", "one NVIDIA H100 80GB per worker, 8 workers · "
                         "~13 s per protein per Helico arm"),
            ("Metrics", "lDDT over all matched atoms; TM-score, GDT-TS and RMSD "
                        "over one atom per residue (CA), so RMSD is a CA RMSD"),
            ("Excluded", "9 of 333 units: 7 whose contact index map could not be "
                         "verified, 2 whose Protenix prediction covered under "
                         "90% of ground-truth atoms"),
        ]
        for i, (head, body) in enumerate(lines):
            y = 0.75 - i * 0.115
            fig.text(0.055, y, head, fontsize=14, fontweight="bold", color=INK,
                     va="top")
            fig.text(0.055, y - 0.036, body, fontsize=12, color=MUTE, va="top",
                     wrap=True)
        finish(pdf, fig)

        # --- 12-15. the same metrics, natural and designed separated ---
        split_arms = [a for a in SPLIT_ORDER if a in set(metrics.arm)]
        for metric, lower_better in (("lddt", False), ("tm_score", False),
                                     ("gdt_ts", False), ("rmsd", True)):
            direction = "lower is better" if lower_better else "higher is better"
            fig = slide(
                pdf, f"{METRIC_LABEL[metric]}: natural against designed proteins",
                f"Mean per group, {direction}. Error bars are 95% percentile "
                f"bootstrap intervals over 10,000 resamples of the proteins; "
                f"every arm sees the same resamples. Both panels share an "
                f"x range.")
            metric_bars_split(fig, metrics, metric, split_arms)
            finish(pdf, fig)

    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
