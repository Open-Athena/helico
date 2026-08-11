"""Figure: structural accuracy of contact-conditioned Helico vs Protenix.

Regenerate with:
    uv run python .agents/project/figures/contact_conditioning_accuracy.py

Panels A and B read the FoldBench per-target CSVs written by `modal/bench.py
--output-dir bench_*`; panel C pulls the in-training validation history from
W&B (project timodonnell/helico). Everything is restricted to the protein
categories -- the contact map is a protein side-chain feature, so
nucleic-acid-only targets carry no signal (they also served as the empirical
null: both arms are identical by construction there).

Panel A's connected trajectory is a single run (`contacts-lrmult1000`, steps
0-3000); the step-8000 point is from a different run (`contacts-m1000-long`)
and is drawn detached for that reason. Step 0 is the warm start: Protenix v1
weights with use_msa=False and a zero-init contact projection. Both arms are
benched at step 0 and the measured difference is annotated.

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

# Contacts are a protein side-chain feature (pyconfind rotamers), so only
# protein-containing categories carry signal.
PROTEIN_CATS = [
    "interface_antibody_antigen",
    "interface_protein_ligand",
    "interface_protein_peptide",
    "interface_protein_protein",
    "monomer_protein",
]
CAT_LABEL = {
    "interface_antibody_antigen": "antibody–antigen",
    "interface_protein_ligand": "protein–ligand",
    "interface_protein_peptide": "protein–peptide",
    "interface_protein_protein": "protein–protein",
    "monomer_protein": "protein monomer",
}
CAT_MARKER = {
    "interface_antibody_antigen": "o",
    "interface_protein_ligand": "s",
    "interface_protein_peptide": "^",
    "interface_protein_protein": "D",
    "monomer_protein": "v",
}

# Every arm, and where its weights came from. The step-0 arms are the warm
# start (Protenix v1 + use_msa=False); `bench_protenix_nomsa` doubles as the
# contacts-off arm there because at step 0 conditioning is a no-op.
REQUIRED = [
    "bench_m1000_off", "bench_m1000_on",
    "bench_s8000_off", "bench_s8000_on",
    "bench_protenix_msa", "bench_protenix_nomsa",
]
OPTIONAL = ["bench_t1000_on", "bench_t1000_off",
            "bench_t2000_on", "bench_t2000_off", "bench_step0_on"]

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
            rows.append({"step": row["_step"], "c0": row.get("val/lddt@contacts0"),
                         "c50": row.get("val/lddt@contacts50"),
                         "c100": row["val/lddt@contacts100"]})
        if rows:
            hist[r.id] = rows
    return hist


def panel_trajectory(ax, mean, arms, keys, n):
    """Accuracy vs training step, with the Protenix reference lines."""
    ax.axhline(mean["bench_protenix_msa"], color=C_MSA, ls="--", lw=1.6, zorder=1)
    ax.axhline(mean["bench_protenix_nomsa"], color=C_NOMSA, ls="--", lw=1.6, zorder=1)
    ax.annotate(f"Protenix + MSA  ({mean['bench_protenix_msa']:.3f})",
                xy=(0.5, mean["bench_protenix_msa"]), xycoords=("axes fraction", "data"),
                ha="center", va="top", fontsize=9, color=C_MSA, fontweight="bold")
    ax.annotate(f"Protenix, single sequence  ({mean['bench_protenix_nomsa']:.3f})",
                xy=(0.38, mean["bench_protenix_nomsa"]), xycoords=("axes fraction", "data"),
                ha="center", va="bottom", fontsize=9, color=C_NOMSA, fontweight="bold")

    # Within-run trajectory: contacts-lrmult1000, steps 0 -> 3000.
    traj_on = [(0, mean.get("bench_step0_on", mean["bench_protenix_nomsa"])),
               (1000, mean.get("bench_t1000_on")), (2000, mean.get("bench_t2000_on")),
               (3000, mean["bench_m1000_on"])]
    traj_off = [(0, mean["bench_protenix_nomsa"]),
                (1000, mean.get("bench_t1000_off")), (2000, mean.get("bench_t2000_off")),
                (3000, mean["bench_m1000_off"])]
    traj_on = [(s, v) for s, v in traj_on if v is not None]
    traj_off = [(s, v) for s, v in traj_off if v is not None]

    ax.plot([s for s, _ in traj_on], [v for _, v in traj_on], "-o", color=C_ON,
            lw=1.8, ms=8, zorder=3, label="contacts given (100%)")
    ax.plot([s for s, _ in traj_off], [v for _, v in traj_off], "-s", color=C_OFF,
            lw=1.8, ms=7, zorder=3, label="no contacts (all unknown)")

    # Different run -> drawn detached, with a break in the x axis implied by the gap.
    ax.plot([8000], [mean["bench_s8000_on"]], "o", color=C_ON, ms=9,
            mfc="white", mew=2.2, zorder=3)
    ax.plot([8000], [mean["bench_s8000_off"]], "s", color=C_OFF, ms=8,
            mfc="white", mew=2.2, zorder=3)
    ax.annotate("step 8000\n(different run)", xy=(8000, mean["bench_s8000_off"]),
                xytext=(0, -30), textcoords="offset points", ha="center",
                fontsize=8.5, color="0.35")

    for s, v in traj_on:
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=C_ON, fontweight="bold")
    for s, v in traj_off:
        if s == 0:
            continue  # coincides with the contacts-on label at step 0
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, color=C_OFF, fontweight="bold")

    if "bench_step0_on" in mean:
        gap = abs(mean["bench_step0_on"] - mean["bench_protenix_nomsa"])
        ax.annotate(f"step 0: Δ = {gap:.4f}",
                    xy=(0, mean["bench_protenix_nomsa"]), xytext=(14, 34),
                    textcoords="offset points", ha="left", fontsize=8.5, color="0.3",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=1))

    ax.set_xlim(-600, 9000)
    ax.set_ylim(0.13, 0.95)
    ax.set_xticks([0, 1000, 2000, 3000, 8000])
    ax.set_xlabel("training step")
    ax.set_ylabel("FoldBench lDDT")
    ax.set_title(f"A. FoldBench lDDT vs training step\n"
                 f"{n} paired protein targets", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")


def panel_scatter(ax, arms, keys, n):
    """Per-target contacts vs MSA, the paired view behind panel A's means."""
    ax.plot([0, 1], [0, 1], "-", color="0.55", lw=1.3, zorder=1)
    ax.annotate("y = x", xy=(0.42, 0.445), fontsize=9, color="0.45",
                rotation=45, ha="center", va="center")

    for cat in PROTEIN_CATS:
        ks = [k for k in keys if k[0] == cat]
        if not ks:
            continue
        ax.scatter([arms["bench_protenix_msa"][k] for k in ks],
                   [arms["bench_s8000_on"][k] for k in ks],
                   marker=CAT_MARKER[cat], s=58, color=C_ON, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3,
                   label=CAT_LABEL[cat])
    # Same targets with contacts withheld, to show where the model falls back to.
    ax.scatter([arms["bench_protenix_msa"][k] for k in keys],
               [arms["bench_s8000_off"][k] for k in keys],
               marker="x", s=34, color=C_OFF, alpha=0.55, zorder=2,
               label="same targets, contacts withheld")

    m, se, t, up = paired_stats(arms["bench_protenix_msa"], arms["bench_s8000_on"], keys)
    ax.annotate(f"above y=x: {up}/{n} targets\nmean Δ = {m:+.3f} ± {se:.3f}  (t={t:.1f})",
                xy=(0.035, 0.965), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95))

    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Protenix + MSA  lDDT")
    ax.set_ylabel("Helico + contacts  lDDT")
    ax.set_title(f"B. Per-target lDDT: contacts vs MSA\n"
                 f"step_8000 checkpoint, {n} paired protein targets",
                 fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")


def panel_validation(bx, hist):
    for i, (_rid, rows) in enumerate(hist.items()):
        xs = [r["step"] for r in rows]
        for key, colour, lab in [("c100", C_ON, "100% of contacts given"),
                                 ("c50", "#7b52a1", "50% of contacts given"),
                                 ("c0", C_OFF, "no contacts (all unknown)")]:
            bx.plot(xs, [r[key] for r in rows], "-o", color=colour, ms=4, lw=1.3,
                    alpha=0.75, label=lab if i == 0 else None)
    bx.set_xlabel("training step")
    bx.set_ylabel("validation lDDT")
    bx.set_title("C. Validation lDDT vs training step\n"
                 "50 held-out structures, 4 independent restarts", fontsize=11, loc="left")
    bx.legend(loc=(0.03, 0.49), fontsize=8.5, framealpha=0.95)
    bx.grid(alpha=0.25, ls=":")
    bx.text(0.985, 0.03, "separate lines are independent restarts\nof the same config",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=8, style="italic",
            color="0.4")


def main():
    # `results/` is created empty when a bench starts and only filled when it
    # finishes, so presence of the directory means nothing -- gate on rows.
    arms = {d: load_lddt(d) for d in REQUIRED + OPTIONAL}
    missing = [d for d in OPTIONAL if not arms[d]]
    for d in missing:
        del arms[d]
    if missing:
        print(f"not yet benched, omitted from the figure: {', '.join(missing)}")
    for d in REQUIRED:
        if not arms[d]:
            raise SystemExit(f"required bench arm {d} has no scored targets")
    present = list(arms)

    # Restrict to targets every arm scored, so all numbers are paired.
    keys = set(arms["bench_s8000_on"])
    for d in arms:
        keys &= set(arms[d])
    keys = sorted(keys)
    n = len(keys)
    mean = {d: sum(arms[d][k] for k in keys) / n for d in arms}

    print(f"n = {n} paired protein targets")
    for d in present:
        print(f"  {d:24s} {mean[d]:.4f}")
    for a, b, lab in [
        ("bench_s8000_off", "bench_s8000_on", "contacts off -> on @ step_8000"),
        ("bench_protenix_msa", "bench_s8000_on", "helico contacts-on vs Protenix+MSA"),
        ("bench_protenix_nomsa", "bench_protenix_msa", "Protenix: single-seq -> +MSA"),
    ]:
        m, se, t, up = paired_stats(arms[a], arms[b], keys)
        print(f"  {lab:38s} {m:+.4f} +/- {se:.4f}  t={t:+.2f}  {up}/{n} up")
    if "bench_step0_on" in arms:
        m, se, t, up = paired_stats(arms["bench_protenix_nomsa"], arms["bench_step0_on"], keys)
        print(f"  {'step-0 no-op control (on - off)':38s} {m:+.4f} +/- {se:.4f}  t={t:+.2f}")

    fig, (ax, sx, bx) = plt.subplots(1, 3, figsize=(18.6, 5.6))
    panel_trajectory(ax, mean, arms, keys, n)
    panel_scatter(sx, arms, keys, n)
    try:
        panel_validation(bx, fetch_val_history())
    except Exception as exc:  # noqa: BLE001 - figure is still useful without panel C
        print(f"W&B history unavailable ({type(exc).__name__}: {exc})")

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
