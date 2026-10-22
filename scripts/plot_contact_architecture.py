"""Render the planned contact-diffusion architecture as SVG, PDF and PNG.

AF3 baseline: https://www.nature.com/articles/s41586-024-07487-w (Fig. 2).
Helico paths: model/helico.py and contact_diffusion.py. Search is planned;
the pilot implemented only one contact-intervention round, not the full tree.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/images"


def main():
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("#fafbfd")
    ax.set(xlim=(0, 18), ylim=(0, 12))
    ax.axis("off")
    ink = "#172b4d"
    muted = "#52657c"
    base = "#e8eef6"
    new = "#d9f2e8"
    edge = "#178366"
    search = "#fff0d6"
    orange = "#b66c14"

    def text(x, y, s, size=11, color=ink, **kwargs):
        ax.text(x, y, s, fontsize=size, color=color, va="center", **kwargs)

    def box(x, y, w, h, title, body="", face=base, border="#b2c1d3", title_size=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                   linewidth=1.3, edgecolor=border, facecolor=face, zorder=3))
        if body:
            text(x + w/2, y + h*.71, title, title_size, ha="center", weight="bold", zorder=4)
            text(x + w/2, y + h*.32, body, 10.3, ha="center", linespacing=1.45, zorder=4)
        else:
            text(x + w/2, y + h/2, title, title_size, ha="center", weight="bold", zorder=4)

    def arrow(points, color=muted, dashed=False, width=1.6):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width,
                    linestyle="--" if dashed else "-", zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14,
                                    lw=width, color=color, linestyle="--" if dashed else "-", zorder=2))

    text(.4, 11.55, "Helico · contact-conditioned structure generation", 23, weight="bold")
    text(.4, 11.12, "Planned full-model fine-tuning from Protenix v1  |  AF3 backbone with a discrete contact process", 12, muted)
    for x, label, color in [(.4, "Retained from AF3", base), (5.0, "New / changed in Helico", new), (11.0, "Planned inference search", search)]:
        ax.add_patch(FancyBboxPatch((x, 10.58), .25, .25, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#a9b9c9"))
        text(x+.4, 10.7, label, 11)

    # Main network, left to right. Contacts enter pair initialization and remain
    # available through recycling; alignment evidence is a parallel input.
    box(.45, 8.45, 3.0, 1.2, "Molecular inputs", "Sequences · ligand chemistry\nBonds · atom reference features")
    box(.45, 6.95, 3.0, 1.05, "MSAs + templates", "Paired / unpaired alignments\nMSA dropout during training")
    box(.45, 5.35, 3.0, 1.15, "Contact input  Cₜ", "n × n × 3 states\ncontact / no contact / masked", new, edge)
    box(4.3, 8.45, 3.1, 1.2, "Input embedding", "Single + pair initialization\nAdd contact embedding to z", new, edge)
    box(4.3, 5.35, 3.1, 1.15, "Contact embedding", "Learned 3 → pair channels\nZero-initialized projection", new, edge)
    box(8.25, 7.05, 3.0, 2.6, "Recycled trunk", "Template module\nMSA module\n48-block Pairformer\nSingle s + pair z")
    box(12.2, 8.65, 2.5, 1.0, "Contact head", "Symmetric probabilities\nn × n", new, edge, 11.5)
    box(12.2, 7.35, 2.5, .9, "Distogram head", "Distance-bin probabilities", title_size=11.5)
    box(12.2, 5.65, 2.5, 1.1, "Coordinate diffusion", "Continuous atom denoising\nConditioned on s and z", title_size=11.0)
    box(15.35, 5.65, 2.15, 1.1, "3D samples", "All-atom coordinates", title_size=11.5)
    box(15.35, 7.35, 2.15, .9, "Confidence", "pLDDT · PAE · ranking", title_size=11.5)
    arrow([(3.5,9.05),(4.23,9.05)])
    arrow([(7.46,9.05),(8.18,9.05)])
    arrow([(3.5,7.47),(8.18,7.47)])
    arrow([(3.5,5.92),(4.23,5.92)],edge)
    arrow([(5.85,6.57),(5.85,8.38)],edge)
    arrow([(11.32,9.2),(12.13,9.2)],edge)
    arrow([(11.32,7.8),(12.13,7.8)])
    arrow([(10.2,6.98),(10.2,6.2),(12.13,6.2)])
    arrow([(14.77,6.2),(15.28,6.2)])
    arrow([(16.43,6.82),(16.43,7.28)])
    arrow([(11.0,9.72),(11.0,10.05),(16.43,10.05),(16.43,8.32)],muted,dashed=True)
    text(14.1,10.23,"Trunk features also feed confidence",9.2,muted,ha="center")
    # Trunk recycling loop is distinct from the contact-search loop below.
    arrow([(8.18,8.7),(7.87,8.7),(7.87,9.95),(9.0,9.95),(9.0,9.72)],muted)
    text(7.83,10.13,"recycle",9.0,muted,ha="center")

    ax.plot([.4,17.55],[4.88,4.88],color="#d3dce8",lw=1)
    text(.45,4.57,"NEW TRAINING SIGNAL",11,edge,weight="bold")
    box(.45,2.92,3.8,1.15,"Ground-truth contacts  C₀", "Closest heavy-atom distance < 5 Å\nMissing geometry is not a negative",new,edge)
    box(4.85,2.92,4.1,1.15,"Absorbing-mask process", "Sample t; mask unordered pairs together\nRevelation independent of contact value",new,edge)
    box(9.55,2.92,3.6,1.15,"Contact denoising loss", "BCE on valid masked pairs\nVisible-pair loss tracked separately",new,edge)
    box(13.75,2.92,3.75,1.15,"Structure objectives retained", "Coordinate diffusion · distogram\nConfidence supervision")
    arrow([(4.31,3.5),(4.78,3.5)],edge)
    arrow([(6.85,4.14),(6.85,4.35),(.18,4.35),(.18,4.75),(1.95,4.75),(1.95,5.28)],edge,dashed=True)
    text(8.4,2.52,"Masking mixture covers all-masked, fully revealed and very sparse inputs; MSAs can be present or absent.",10.5,muted,ha="center")
    text(11.35,4.3,"C₀ target + contact-head prediction",9.6,edge,ha="center")

    # Search is intentionally labelled planned: a finite tree of clean contact
    # hypotheses is not automatically a calibrated reverse-diffusion sampler.
    text(.45,2.05,"PLANNED INFERENCE-TIME SEARCH",11,orange,weight="bold")
    box(.45,.55,4.45,1.02,"Start with masked contacts", "Run trunk + coordinate diffusion",search,orange,11.5)
    box(5.6,.55,5.1,1.02,"Propose contact hypotheses", "High contact probability, absent in 3D sample\nBranch on a small set of pair constraints",search,orange,11.5)
    box(11.4,.55,6.1,1.02,"Rerun · score · retain diverse branches", "Check constraint satisfaction, geometry and confidence\nFeed each branch's contact state back into Cₜ",search,orange,11.5)
    arrow([(4.96,1.06),(5.53,1.06)],orange)
    arrow([(10.76,1.06),(11.33,1.06)],orange)
    text(.45,.08,"Two distinct processes: discrete masking of contacts around an AF3-style continuous coordinate diffusion model. Diagram shows the planned architecture.",9.5,muted)

    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ["svg","pdf","png"]:
        path = OUT/f"masked_contact_architecture.{ext}"
        fig.savefig(path,dpi=180,bbox_inches="tight",facecolor=fig.get_facecolor())
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
