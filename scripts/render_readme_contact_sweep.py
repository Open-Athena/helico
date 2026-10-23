"""Render the README GIF from saved predictions (no model or GPU required).

    uv run --no-project --with numpy --with scipy --with pillow \
        python scripts/render_readme_contact_sweep.py

Only rigid alignment, camera rotation, and spline drawing are used. Coordinates
are never interpolated between predictions. Requires DejaVu Sans fonts (set
HELICO_DEMO_FONT_DIR if they are not in the usual Linux location).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/images/contact_sweep"
WIDTH, HEIGHT, AA = 1200, 700, 2
BG, PANEL, BORDER = "#0b1220", "#101d2e", "#243347"
WHITE, MUTED, TEAL = "#edf4fb", "#a0b3c9", "#5ce3c1"
FONT_DIR = Path(os.environ.get("HELICO_DEMO_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))


def font(size, bold=False):
    return ImageFont.truetype(
        str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size * AA
    )


def text(draw, xy, label, size=16, color=WHITE, bold=False, anchor=None):
    draw.text(tuple(int(x * AA) for x in xy), label, font=font(size, bold),
              fill=color, anchor=anchor)


def line(draw, points, color=BORDER, width=1):
    draw.line([(int(x * AA), int(y * AA)) for x, y in points], fill=color, width=width * AA)


def box(draw, bounds, fill=PANEL, outline=BORDER, radius=16):
    draw.rounded_rectangle(tuple(int(x * AA) for x in bounds), radius=radius * AA,
                           fill=fill, outline=outline, width=AA)


def aligned(pred, truth):
    p, q = pred - pred.mean(0), truth - truth.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return p @ rotation


def tube(coords, reference=False):
    """A narrow tube following the CA trace, colored from N to C terminus."""
    t = np.linspace(0, len(coords) - 1, (len(coords) - 1) * 8 + 1)
    spline = CubicSpline(np.arange(len(coords)), coords)
    points, tangent = spline(t), spline(t, 1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    # Parallel transport avoids sudden twists where the tangent turns vertical.
    normals = [np.cross(tangent[0], np.eye(3)[np.argmin(abs(tangent[0]))])]
    normals[0] /= np.linalg.norm(normals[0])
    for direction in tangent[1:]:
        normal = normals[-1] - direction * np.dot(normals[-1], direction)
        normals.append(normal / np.linalg.norm(normal))
    normals = np.array(normals)
    binormals = np.cross(tangent, normals)
    theta = np.linspace(0, 2 * np.pi, 11)[:-1]
    radial = (normals[:, None] * np.cos(theta)[None, :, None]
              + binormals[:, None] * np.sin(theta)[None, :, None])
    vertices = points[:, None] + 0.40 * radial
    faces = np.stack([vertices[:-1], vertices[1:], np.roll(vertices[1:], -1, axis=1),
                      np.roll(vertices[:-1], -1, axis=1)], axis=2).reshape(-1, 4, 3)
    face_normals = (radial[:-1] + radial[1:] + np.roll(radial[:-1], -1, axis=1)
                    + np.roll(radial[1:], -1, axis=1)).reshape(-1, 3)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    if reference:
        colors = np.tile([168, 185, 209], (len(faces), 1))
    else:
        stops = np.array([[71, 225, 193], [93, 179, 244], [175, 139, 245]])
        positions = np.repeat(t[:-1] / t[-1], len(theta))
        colors = np.column_stack([np.interp(positions, [0, .5, 1], stops[:, c])
                                  for c in range(3)])
    return faces, face_normals, colors


def draw_tube(draw, mesh, rotation, center, scale):
    faces, normals, colors = mesh
    rotated = faces @ rotation
    lighting = (normals @ rotation) @ np.array([-.35, .45, .82])
    shades = np.clip(.45 + .55 * np.maximum(lighting, 0), 0, 1)
    rgb = (colors * shades[:, None]).astype(int)
    xy = rotated[:, :, :2] * [scale, -scale] + center
    for i in np.argsort(rotated[:, :, 2].mean(1)):
        draw.polygon([tuple(p) for p in (xy[i] * AA).astype(int)], fill=tuple(rgb[i]))


def render(data, meshes, ref_mesh, camera, scale, stage, phase):
    image = Image.new("RGB", (WIDTH * AA, HEIGHT * AA), BG)
    draw = ImageDraw.Draw(image)
    row = data["conditions"][stage]
    count, total = row["n_contacts"], len(data["ordered_contact_pairs_zero_indexed"])
    text(draw, (38, 27), "HELICO", 17, TEAL, True)
    text(draw, (136, 29), "/  CONTACT-CONDITIONED FOLDING", 13, MUTED)
    text(draw, (36, 62), "From contacts to coordinates", 38, WHITE, True)
    text(draw, (38, 119), "8B2G  ·  SH3-like domain  ·  74 residues  ·  no MSA", 17, MUTED)
    for bounds in [(36, 166, 350, 538), (368, 166, 758, 538), (776, 166, 1164, 538)]:
        box(draw, bounds)
    text(draw, (56, 187), "01   INPUT CONTACT MAP", 13, MUTED, True)
    text(draw, (390, 187), "02   HELICO PREDICTION", 13, TEAL, True)
    text(draw, (798, 187), "03   EXPERIMENTAL REFERENCE", 13, MUTED, True)

    length = len(data["sequence"])
    cell, left, top = 2.85, 85, 231
    edge = cell * length
    draw.rectangle((left * AA, top * AA, (left + edge) * AA, (top + edge) * AA), fill=BG)
    for pos in range(0, length + 1, 10):
        line(draw, [(left, top + pos * cell), (left + edge, top + pos * cell)], "#182638")
        line(draw, [(left + pos * cell, top), (left + pos * cell, top + edge)], "#182638")
    line(draw, [(left, top), (left + edge, top + edge)], "#33465b")
    pairs = data["ordered_contact_pairs_zero_indexed"][:count]
    for i, j in pairs:
        for a, b in [(i, j), (j, i)]:
            x, y = left + a * cell, top + b * cell
            draw.rectangle((x * AA, y * AA, (x + cell) * AA, (y + cell) * AA), fill=TEAL)
    text(draw, (left, top + edge + 8), "1", 11, MUTED)
    text(draw, (left + edge, top + edge + 8), "74", 11, MUTED, anchor="ra")
    text(draw, (left + edge / 2, top + edge + 8), "residue position", 11, MUTED, anchor="ma")
    text(draw, (70, 482), f"{count:02d}", 29, TEAL, True)
    text(draw, (123, 494), f"/ {total} supplied pairs", 15, MUTED)

    # Identical camera and scale for all conditions and both structure panels.
    angle = .20 * np.sin(2 * np.pi * phase)
    c, s = np.cos(angle), np.sin(angle)
    rotation = camera @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    draw_tube(draw, meshes[stage], rotation, np.array([563, 354]), scale)
    draw_tube(draw, ref_mesh, rotation, np.array([970, 354]), scale)
    text(draw, (390, 490), f"Cα lDDT  {row['ca_lddt']:.3f}", 18, TEAL, True)
    text(draw, (390, 516), f"Mean pLDDT  {row['mean_plddt']:.1f} / 100", 12, MUTED)
    text(draw, (798, 490), "Same reference at every step", 15, MUTED)
    text(draw, (798, 516), "Cα backbone traces · shared scale", 12, MUTED)

    text(draw, (38, 562), "CONTACTS SUPPLIED", 12, MUTED, True)
    xs = np.linspace(290, 1110, len(data["conditions"]))
    line(draw, [(xs[0], 586), (xs[-1], 586)], BORDER, 3)
    if stage:
        line(draw, [(xs[0], 586), (xs[stage], 586)], TEAL, 3)
    for i, (x, condition) in enumerate(zip(xs, data["conditions"])):
        color = TEAL if i <= stage else "#475a70"
        radius = 7 if i == stage else 4
        draw.ellipse(((x-radius)*AA, (586-radius)*AA, (x+radius)*AA, (586+radius)*AA), fill=color)
        text(draw, (x, 604), str(condition["n_contacts"]), 15, color, i == stage, anchor="ma")
    text(draw, (38, 588), "More constraints, same sequence", 12, MUTED)
    line(draw, [(36, 644), (1164, 644)])
    text(draw, (38, 661), "Known-structure contacts · independent predictions · fixed seed and sampling", 13, MUTED)
    text(draw, (1162, 661), f"{stage+1:02d} / {len(meshes):02d}", 13, TEAL, anchor="ra")
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main():
    data = json.loads((DATA / "data.json").read_text())
    truth = np.array(data["reference_ca"])
    centered = truth - truth.mean(0)
    predictions = [aligned(np.array(row["predicted_ca"]), truth) for row in data["conditions"]]
    meshes = [tube(p) for p in predictions]
    ref_mesh = tube(centered, reference=True)
    # Stable principal-axis view; preserve handedness.
    _, _, vt = np.linalg.svd(centered)
    camera = vt.T[:, [1, 0, 2]]
    camera[:, 2] *= np.linalg.det(camera)
    all_points = np.concatenate(predictions + [centered])
    radius = np.linalg.norm(all_points, axis=1).max() + 1
    scale = min(150 / radius, 9)
    frames = []
    snapshots = []
    for stage in range(len(meshes)):
        for frame in range(24):
            image = render(data, meshes, ref_mesh, camera, scale, stage, frame / 24)
            if frame == 0:
                snapshots.append(image)
            frames.append(image)
    # A shared palette keeps text, backgrounds and molecule colors stable.
    palette_source = Image.new("RGB", (WIDTH, HEIGHT * len(snapshots)))
    for i, snapshot in enumerate(snapshots):
        palette_source.paste(snapshot, (0, HEIGHT * i))
    palette = palette_source.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [80] * len(frames)
    for i in range(len(meshes)):
        durations[i * 24 + 23] = 600
    frames[0].save(DATA.parent / "contact_sweep.gif", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0, optimize=True)
    snapshots[-1].save(DATA / "poster.png")
    # A compact static comparison is also useful for reduced-motion readers.
    comparison = Image.new("RGB", (WIDTH, HEIGHT * 2))
    comparison.paste(snapshots[0], (0, 0))
    comparison.paste(snapshots[-1], (0, HEIGHT))
    comparison.save(DATA / "comparison.png")
    print(f"Rendered {len(frames)} frames; GIF {(DATA.parent / 'contact_sweep.gif').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
