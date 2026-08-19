"""Step 7 -- the figures.

Three, each answering one of the issue's questions:

1. `scoreboard.png` -- mean lDDT per arm on each eval set, with 95% bootstrap
   intervals. The dot-and-whisker form rather than bars: these are means with
   uncertainty, and 27 bars would hide the intervals that decide every claim.
2. `val_vs_test.png` -- each arm's eval-val mean beside its eval-test mean.
   This is H2, and the whole reason the sets were split.
3. `contact_quality.png` -- per-target contact precision against per-target
   lDDT, for the three predicted-contact arms. This is the question the
   Protenix-v2 arms exist to answer: is Helico tracking contact quality, or
   adding something of its own on top?

Colours are validated for colour-vision deficiency (dataviz `validate_palette`,
all checks pass); identity is carried by row labels and direct labels as well as
hue, never by hue alone. The CSVs behind every figure are in `data/`.

Run:
    uv run python experiments/exp14_foldbench_held_out_monomers/plot_results.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream as U  # noqa: E402

PLOTS = U.HERE / "plots"

#: One colour per eval set. Validated as a categorical triple.
SET_COLOR = {"eval-val": "#4269d0", "eval-test": "#b8452f", "eval-denovo": "#3ca951"}
SET_ORDER = ("eval-val", "eval-test", "eval-denovo")

#: One colour per contact source, for the scatter. Validated as a categorical
#: quadruple together with the oracle's blue.
ARM_COLOR = {"mf_L": "#b8452f", "v2ss": "#7a4fbf", "v2msa": "#1a7f5a",
             "oracle": "#1b5e9c"}

INK, MUTED, GRID = "#1b1b1b", "#5c5c5c", "#d8d8d8"


def style(ax, *, xlabel: str) -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_yticklabels():
        label.set_color(INK)


def scoreboard(headline: pd.DataFrame) -> None:
    frame = headline[headline.eval_set.isin(SET_ORDER)]
    arms = list(dict.fromkeys(headline.arm))
    labels = dict(zip(headline.arm, headline.label))
    positions = {arm: len(arms) - i for i, arm in enumerate(arms)}
    offsets = {"eval-val": 0.24, "eval-test": 0.0, "eval-denovo": -0.24}

    fig, ax = plt.subplots(figsize=(9, 0.62 * len(arms) + 2.1))
    for eval_set in SET_ORDER:
        sub = frame[frame.eval_set == eval_set]
        if sub.empty:
            continue
        y = [positions[a] + offsets[eval_set] for a in sub.arm]
        ax.hlines(y, sub.ci_lo, sub.ci_hi, color=SET_COLOR[eval_set], lw=2,
                  alpha=0.85)
        ax.plot(sub.mean_lddt, y, "o", ms=8, color=SET_COLOR[eval_set],
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"{eval_set} (n={int(sub.n.iloc[0])})", ls="none")

    ax.set_yticks([positions[a] for a in arms])
    ax.set_yticklabels([labels[a] for a in arms])
    ax.set_ylim(0.4, len(arms) + 0.7)
    style(ax, xlabel="mean lDDT (95% bootstrap interval over proteins)")
    # Below the axes, not inside them: with nine rows there is no empty corner
    # the legend can sit in without landing on an interval.
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), labelcolor=INK)
    ax.set_title("Folding accuracy by conditioning arm\n"
                 "exp245's held-out FoldBench monomers, Helico MSA-free throughout",
                 color=INK, loc="left", pad=14, fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "scoreboard.png", dpi=200)
    plt.show()


def val_vs_test(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("no val-vs-test rows to plot")
        return
    frame = frame.iloc[::-1].reset_index(drop=True)
    y = range(len(frame))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(frame) + 2.1))
    ax.hlines(y, frame.eval_val, frame.eval_test, color=GRID, lw=2)
    ax.plot(frame.eval_val, y, "o", ms=8, color=SET_COLOR["eval-val"],
            markeredgecolor="white", markeredgewidth=1.2, ls="none",
            label="eval-val (97)")
    ax.plot(frame.eval_test, y, "o", ms=8, color=SET_COLOR["eval-test"],
            markeredgecolor="white", markeredgewidth=1.2, ls="none",
            label="eval-test (217)")
    for i, row in frame.iterrows():
        ax.annotate(f"{row.change:+.3f}",
                    (max(row.eval_val, row.eval_test), i),
                    textcoords="offset points", xytext=(10, 0), va="center",
                    fontsize=9, color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels(frame.label)
    style(ax, xlabel="mean lDDT")
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), labelcolor=INK)
    ax.set_title("The working set against the held-out set", color=INK,
                 loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "val_vs_test.png", dpi=200)
    plt.show()


def contact_quality(per_target: pd.DataFrame, precision: pd.DataFrame) -> None:
    merged = per_target.merge(precision, on=["target_id", "arm"], how="inner")
    arms = [a for a in ("mf_L", "v2ss", "v2msa") if a in set(merged.arm)]
    if not arms:
        print("no contact-quality rows to plot")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arm in arms:
        sub = merged[merged.arm == arm]
        ax.plot(sub.precision, sub.lddt, "o", ms=5, alpha=0.55,
                color=ARM_COLOR[arm], ls="none",
                label=f"{arm} (n={len(sub)}, r={sub.precision.corr(sub.lddt):.2f})")
    ax.set_ylabel("lDDT of the Helico prediction", color=INK)
    style(ax, xlabel="precision of the contacts it was given")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Does Helico track contact quality?", color=INK, loc="left",
                 pad=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "contact_quality.png", dpi=200)
    plt.show()


def msa_depth(depth: pd.DataFrame, arms) -> None:
    """The depth buckets, as a standalone figure for the notebook."""
    buckets = [b for b in ("<=10", "11-100", "101-1000", ">1000")
               if b in set(depth.bucket)]
    fig, axes = plt.subplots(1, len(buckets), figsize=(13, 4.6), sharey=True)
    shades = ("#b8452f", "#c98a3a", "#4a90c4", "#4269d0")
    for ax, bucket, color in zip(np.atleast_1d(axes), buckets, shades):
        rows = depth[depth.bucket == bucket].set_index("arm")
        present = [a for a in arms if a in rows.index]
        positions = [len(present) - i for i, _ in enumerate(present)]
        ax.barh(positions, [rows.loc[a, "mean_lddt"] for a in present],
                height=0.66, color=color,
                xerr=[[rows.loc[a, "mean_lddt"] - rows.loc[a, "ci_lo"] for a in present],
                      [rows.loc[a, "ci_hi"] - rows.loc[a, "mean_lddt"] for a in present]],
                error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 2})
        ax.set_yticks(positions)
        ax.set_yticklabels([rows.loc[a, "label"] for a in present], fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_title(f"depth {bucket}  (n={int(rows.n.iloc[0])})", fontsize=10,
                     color=INK)
        style(ax, xlabel="lDDT")
    fig.suptitle("Natural proteins by alignment depth", color=INK, x=0.02,
                 ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "msa_depth.png", dpi=200)
    plt.show()


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    PLOTS.mkdir(exist_ok=True)

    scoreboard(pd.read_csv(U.DATA / "headline.csv"))
    val_vs_test(pd.read_csv(U.DATA / "val_vs_test.csv")
                if (U.DATA / "val_vs_test.csv").stat().st_size > 1
                else pd.DataFrame())

    per_target = pd.read_csv(U.DATA / "per_target.csv")
    per_target = per_target[per_target.status == "ok"]
    marinfold = pd.read_csv(U.DATA / "marinfold_arm_accuracy.csv")
    frames = [marinfold[["target_id", f"precision_{cut}"]]
              .rename(columns={f"precision_{cut}": "precision"})
              .assign(arm=f"mf_{cut}")
              for cut in ("L", "L2", "L5")]
    v2_path = U.DATA / "v2_arm_accuracy.csv"
    if v2_path.exists():
        v2 = pd.read_csv(v2_path)
        frames.append(v2[["target_id", "arm", "precision"]])
    contact_quality(per_target, pd.concat(frames, ignore_index=True))

    depth_path = U.DATA / "depth_strata.csv"
    if depth_path.exists():
        order = ("protenix_v2_msa", "esmfold2", "esmfold",
                 "protenix_v2_single_seq", "off", "mf_L", "oracle")
        msa_depth(pd.read_csv(depth_path), order)

    print(f"-> {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
