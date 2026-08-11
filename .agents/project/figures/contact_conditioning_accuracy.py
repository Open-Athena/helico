"""Figure: structural accuracy of contact-conditioned Helico vs Protenix.

Regenerate with:
    uv run python .agents/project/figures/contact_conditioning_accuracy.py

Panel A reads the FoldBench per-target CSVs written by `modal/bench.py
--output-dir bench_*`; panel B pulls the in-training validation history from
W&B (project timodonnell/helico). Both panels are restricted to the protein
categories -- the contact map is a protein side-chain feature, so
nucleic-acid-only targets are not informative here (they also served as the
empirical null: both arms are identical by construction there).

The two panels measure different things on purpose. Panel A is the absolute
anchor against Protenix on a fixed 28-target paired set. Panel B is the only
*within-run* progress signal we have: the two benched Helico checkpoints come
from different training runs, so the step-to-step difference in panel A is not
a trajectory. See the design doc for the full argument.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "contact_conditioning_accuracy.png"

# Contacts are a protein side-chain feature (pyconfind rotamers), so only
# protein-containing categories carry signal.
PROTEIN_CATS = [
    "interface_antibody_antigen",
    "interface_protein_ligand",
    "interface_protein_peptide",
    "interface_protein_protein",
    "monomer_protein",
]

# Checkpoint provenance. These are two DIFFERENT training runs, not two points
# on one trajectory -- `contacts-lrmult1000` is a standalone 3k-step run and
# `contacts-m1000-long` is the long run. Kept explicit so the caveat in panel A
# cannot quietly drift away from the data.
ARMS = {
    "bench_m1000_off": ("contacts-lrmult1000/final.pt", 3000, "off"),
    "bench_m1000_on": ("contacts-lrmult1000/final.pt", 3000, "on"),
    "bench_s8000_off": ("contacts-m1000-long/step_8000.pt", 8000, "off"),
    "bench_s8000_on": ("contacts-m1000-long/step_8000.pt", 8000, "on"),
    "bench_protenix_msa": ("protenix-v1 (+MSA)", None, "msa"),
    "bench_protenix_nomsa": ("protenix-v1 (single seq)", None, "nomsa"),
}

C_ON, C_OFF = "#1b5e9c", "#b8452f"
C_MSA, C_NOMSA = "#2e7d32", "#7a7a7a"


def load_lddt(bench_dir):
    """{(category, pdb_id): lddt} over protein categories, NaNs dropped."""
    out = {}
    for cat in PROTEIN_CATS:
        f = ROOT / bench_dir / "results" / f"{cat}.csv"
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


def fetch_val_history():
    """Per-run in-training validation lDDT, keyed by run id."""
    import wandb

    api = wandb.Api()
    runs = [r for r in api.runs("timodonnell/helico") if "m1000-long" in r.name]
    hist = {}
    for r in sorted(runs, key=lambda x: x.created_at):
        rows = []
        for row in r.scan_history(page_size=10000):
            if row.get("val/lddt@contacts100") is None:
                continue
            rows.append(
                {
                    "step": row["_step"],
                    "c0": row.get("val/lddt@contacts0"),
                    "c50": row.get("val/lddt@contacts50"),
                    "c100": row["val/lddt@contacts100"],
                }
            )
        if rows:
            hist[r.id] = rows
    return hist


def main():
    arms = {d: load_lddt(d) for d in ARMS}

    # Restrict to targets every arm scored, so all six numbers are paired.
    keys = set(arms["bench_s8000_on"])
    for d in arms:
        keys &= set(arms[d])
    keys = sorted(keys)
    n = len(keys)
    mean = {d: sum(arms[d][k] for k in keys) / n for d in arms}

    print(f"n = {n} paired protein targets")
    for d in ARMS:
        print(f"  {d:24s} {mean[d]:.4f}")
    for a, b, lab in [
        ("bench_s8000_off", "bench_s8000_on", "contacts off -> on @ step_8000"),
        ("bench_protenix_msa", "bench_s8000_on", "helico contacts-on vs Protenix+MSA"),
        ("bench_protenix_nomsa", "bench_protenix_msa", "Protenix: single-seq -> +MSA"),
    ]:
        m, se, t, up = paired_stats(arms[a], arms[b], keys)
        print(f"  {lab:38s} {m:+.4f} +/- {se:.4f}  t={t:+.2f}  {up}/{n} up")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # ---- Panel A: FoldBench, absolute anchor vs Protenix -------------------
    steps = [3000, 8000]
    on = [mean["bench_m1000_on"], mean["bench_s8000_on"]]
    off = [mean["bench_m1000_off"], mean["bench_s8000_off"]]

    ax.axhline(mean["bench_protenix_msa"], color=C_MSA, ls="--", lw=1.6, zorder=1)
    ax.axhline(mean["bench_protenix_nomsa"], color=C_NOMSA, ls="--", lw=1.6, zorder=1)
    ax.annotate(
        f"Protenix + MSA  ({mean['bench_protenix_msa']:.3f})",
        xy=(0.52, mean["bench_protenix_msa"]), xycoords=("axes fraction", "data"),
        ha="center", va="top", fontsize=9.5, color=C_MSA, fontweight="bold")
    ax.annotate(
        f"Protenix, single sequence  ({mean['bench_protenix_nomsa']:.3f})",
        xy=(0.015, mean["bench_protenix_nomsa"]), xycoords=("axes fraction", "data"),
        ha="left", va="bottom", fontsize=9.5, color=C_NOMSA, fontweight="bold")

    # Dotted, not solid: these two checkpoints are different runs.
    ax.plot(steps, on, ":", color=C_ON, lw=1.5, zorder=2)
    ax.plot(steps, off, ":", color=C_OFF, lw=1.5, zorder=2)
    ax.plot(steps, on, "o", color=C_ON, ms=11, zorder=3,
            label="Helico, contacts given (100%)")
    ax.plot(steps, off, "s", color=C_OFF, ms=10, zorder=3,
            label="Helico, no contacts (all unknown)")

    for x, y in zip(steps, on):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9.5, color=C_ON, fontweight="bold")
    for x, y in zip(steps, off):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -17),
                    ha="center", fontsize=9.5, color=C_OFF, fontweight="bold")

    m, se, t, up = paired_stats(arms["bench_s8000_off"], arms["bench_s8000_on"], keys)
    ax.annotate(
        "", xy=(8000, on[1] - 0.012), xytext=(8000, off[1] + 0.012),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.4))
    ax.annotate(
        f"contacts add\n{m:+.3f} lDDT\n(t={t:.1f}, {up}/{n})",
        xy=(8000, (on[1] + off[1]) / 2), xytext=(-14, 0),
        textcoords="offset points", ha="right", va="center", fontsize=9.5, color="0.2")

    ax.set_xlim(1800, 9400)
    ax.set_ylim(0.13, 0.95)
    ax.set_xticks(steps)
    ax.set_xlabel("training step of the benchmarked checkpoint")
    ax.set_ylabel("FoldBench lDDT")
    ax.set_title(f"A. Oracle contacts reach MSA-level accuracy\n"
                 f"FoldBench, {n} paired protein targets", fontsize=11.5, loc="left")
    ax.legend(loc=(0.03, 0.30), fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")
    ax.text(0.5, 0.015,
            "dotted: the two checkpoints are different training runs, not one trajectory",
            transform=ax.transAxes, ha="center", fontsize=8.2, style="italic", color="0.4")

    # ---- Panel B: within-run training progress ----------------------------
    try:
        hist = fetch_val_history()
    except Exception as exc:  # noqa: BLE001 - figure is still useful without panel B
        hist = {}
        print(f"W&B history unavailable ({type(exc).__name__}: {exc})")

    if hist:
        for i, (rid, rows) in enumerate(hist.items()):
            xs = [r["step"] for r in rows]
            first = i == 0
            for key, colour, lab in [
                ("c100", C_ON, "100% of contacts given"),
                ("c50", "#7b52a1", "50% of contacts given"),
                ("c0", C_OFF, "no contacts (all unknown)"),
            ]:
                ys = [r[key] for r in rows]
                bx.plot(xs, ys, "-o", color=colour, ms=4.5, lw=1.4, alpha=0.75,
                        label=lab if first else None)
        bx.set_xlabel("training step")
        bx.set_ylabel("validation lDDT")
        bx.set_title("B. Within-run progress plateaus after ~step 5000\n"
                     "50 held-out structures, 4 restarts overlaid",
                     fontsize=11.5, loc="left")
        bx.legend(loc=(0.55, 0.45), fontsize=9.5, framealpha=0.95)
        bx.grid(alpha=0.25, ls=":")
        bx.text(0.985, 0.015,
                "separate lines are independent restarts of the same config;\n"
                "their spread is the run-to-run noise floor",
                transform=bx.transAxes, ha="right", fontsize=8.2, style="italic",
                color="0.4")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
