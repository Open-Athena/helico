"""Figure: exactly where contact conditioning enters Helico, and what it replaces.

Regenerate with:
    uv run python .agents/project/figures/architecture_contact_conditioning.py

Drawn from the code, not from memory -- every shape and every module name here
is `src/helico/model/helico.py`:

  linear_contact = linear_no_bias(NUM_CONTACT_STATES, d_pair, zeros_init=True)

  z_init  = linear_zinit1(s_init)[:, :, None] + linear_zinit2(s_init)[:, None, :]
  z_init += trunk_relpe(...)
  z_init += linear_token_bond(token_bonds)
  z_init += linear_contact(contact_onehot)        # <- the whole modification

  for cycle in range(n_cycles):
      z = z_init + linear_z_cycle(layernorm_z_cycle(z))   # z_init re-added here
      z = z + template_embedder(batch, z)
      if use_msa: z = msa_module(...)                     # <- skipped
      s, z = pairformer(s, z, ...)

Two things the diagram has to get right because both are easy to state wrongly:

1. The contacts enter the *pair* representation at its initialisation, not the
   MSA slot. `z_init` is re-added at the top of every recycling iteration, so
   the contact term is re-injected each cycle and reaches the template embedder,
   the Pairformer, the distogram head, diffusion and confidence.
2. Turning the MSA off means closing *two* doors, not one. Skipping the MSA
   module leaves `msa_profile` and `deletion_mean` -- a PSSM and an insertion
   rate -- inside `s_inputs`, which is a separate path into the trunk. Gating
   only the module was worth +0.311 lDDT of leaked alignment signal.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "architecture_contact_conditioning.png"

INK, MUTE = "#1a1a1a", "#6b6b6b"
NEW = "#c0392b"      # the added path
GONE = "#c4c4c4"     # the removed path
KEEP = "#2f6f9f"     # unchanged trunk


def box(ax, cx, cy, w, h, title, sub=None, ec=KEEP, fc="#eef4fa", tc=INK,
        lw=1.6, ls="-", fs=12):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.025",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=3))
    ax.text(cx, cy + (0.022 if sub else 0), title, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.030, sub, ha="center", va="center", fontsize=9,
                color=tc, alpha=0.75, zorder=4)


def arrow(ax, p0, p1, color=KEEP, lw=1.8, ls="-", z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=z, shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.93, "Contacts replace the MSA with one zero-initialised "
            "projection", ha="center", fontsize=23, fontweight="bold", color=INK)

    # ---- the trunk, left to right ----------------------------------------
    y = 0.615
    xs = [0.105, 0.295, 0.495, 0.695, 0.880]
    box(ax, xs[0], y, 0.155, 0.135, "sequence")
    box(ax, xs[1], y, 0.155, 0.135, "input", "embedder")
    box(ax, xs[2], y, 0.185, 0.135, "pair", "representation", ec=INK,
        fc="white", lw=2.2, fs=13)
    box(ax, xs[3], y, 0.155, 0.135, "Pairformer", "48 blocks")
    box(ax, xs[4], y, 0.135, 0.135, "diffusion")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 0.080, y), (b - 0.080, y))
    arrow(ax, (xs[4], y - 0.070), (xs[4], y - 0.150))
    ax.text(xs[4], y - 0.190, "structure", ha="center", fontsize=11.5, color=MUTE)

    # Recycling: the pair representation is rebuilt every cycle, so whatever
    # is added into it is re-injected each time round.
    ax.add_patch(FancyArrowPatch((xs[3], y + 0.075), (xs[2], y + 0.075),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color=KEEP, connectionstyle="arc3,rad=0.55",
                                 zorder=1))
    ax.text((xs[2] + xs[3]) / 2, y + 0.185, "recycling", ha="center",
            fontsize=10, color=KEEP, style="italic")

    # ---- what goes in: contacts (new) and the MSA (gone) ------------------
    box(ax, 0.495, 0.285, 0.235, 0.135, "predicted contacts",
        "N x N,  contact / no / unknown", ec=NEW, fc="#fdf0ee", tc=NEW, fs=13)
    arrow(ax, (0.495, 0.357), (0.495, 0.542), color=NEW, lw=3.0, z=4)
    ax.text(0.520, 0.462, "Linear(3 → 128)", fontsize=11.5, color=NEW,
            fontweight="bold", va="center")
    ax.text(0.520, 0.428, "zero-initialised", fontsize=10, color=NEW, va="center")

    box(ax, 0.845, 0.285, 0.195, 0.135, "MSA", "512 x N alignment",
        ec=GONE, fc="#f5f5f5", tc=GONE, ls="--", lw=1.6, fs=13)
    arrow(ax, (0.790, 0.330), (0.580, 0.535), color=GONE, lw=1.8, ls=":", z=1)
    ax.plot([0.655, 0.715], [0.400, 0.460], color=GONE, lw=3.2, zorder=5)
    ax.plot([0.655, 0.715], [0.460, 0.400], color=GONE, lw=3.2, zorder=5)

    # ---- one line of takeaway --------------------------------------------
    ax.text(0.5, 0.115, "384 new weights. Zero-init makes the warm start from "
            "Protenix exact, and the pair representation is rebuilt\n"
            "every recycling iteration — so the contacts re-enter the trunk on "
            "every cycle.", ha="center", fontsize=12.5, color=INK, linespacing=1.9)

    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
