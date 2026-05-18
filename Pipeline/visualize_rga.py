"""
Rhetorical Geometry Analysis — Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone script. Reads a completed schema v0.4 result JSON and produces:

  Figure 1  — Poincaré Disk  (DOC-XXX_poincare_disk.png)
              Nodes at adaptive calibrated positions. Zone rings drawn from
              adaptive_zone_boundaries in graph_structure. Labels placed with
              adjustText to prevent collisions.

  Figure 2  — T_μν Stress-Energy Field  (DOC-XXX_T_munu_field.png)
              Stress contribution of each proposition rendered as a Gaussian
              kernel on the disk plane, summed into a 2-D heatmap. Intensity
              at the origin (r=0) reflects the summed weight of all edges
              terminating at the governance zone anchors, not independent
              stress at that location. This is noted in the figure caption.

Usage:
    python visualize_rga.py <result_json_path> [output_dir]

    output_dir defaults to the directory containing result_json_path.
"""

from __future__ import annotations

import sys
import json
import math
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text


# ── Color scheme ──────────────────────────────────────────────────────────────

CLASSIFICATION_COLOR: Dict[str, str] = {
    "anchored_true":           "#2E7D32",   # deep green
    "anchored_false":          "#C62828",   # deep red
    "bridge_narrative":        "#E65100",   # amber-orange
    "contextually_misleading": "#F57C00",   # orange
    "inferentially_true":      "#66BB6A",   # light green
    "inferentially_false":     "#EF9A9A",   # light red
    "ambiguous":               "#9E9E9E",   # mid grey
    "out_of_scope":            "#BDBDBD",   # light grey
}

ANCHOR_COLOR        = "#1565C0"   # deep blue
ANCHOR_EDGE_COLOR   = "#0D47A1"
PROP_EDGE_COLOR_POS = "#4CAF50"   # green edges (supports)
PROP_EDGE_COLOR_NEG = "#E53935"   # red edges (contradicts)
ZONE_RING_COLOR     = "#B0BEC5"   # light blue-grey for zone boundaries
BG_COLOR            = "#FAFAFA"


# ── Polar-to-Cartesian ─────────────────────────────────────────────────────────

def _xy(r: float, theta: float) -> Tuple[float, float]:
    return r * math.cos(theta), r * math.sin(theta)


# ── Zone ring labels ───────────────────────────────────────────────────────────

ZONE_RING_LABELS = {
    "governance":      "governance",
    "false_attractor": "false attractor",
}


# ── Figure 1: Poincaré Disk ────────────────────────────────────────────────────

def draw_poincare_disk(result: Dict, output_path: Path) -> None:
    graph = result["graph_structure"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    zone_boundaries: Dict[str, float] = graph.get("adaptive_zone_boundaries", {})
    node_radii: Dict[str, float] = graph.get("node_radii", {})

    # Build lookup: node id → node dict
    node_map = {n["id"]: n for n in nodes}

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 8),
        facecolor=BG_COLOR,
        gridspec_kw={"width_ratios": [3, 1.1]},
    )
    fig.patch.set_facecolor(BG_COLOR)

    ax   = axes[0]   # main disk
    axin = axes[1]   # governance zone inset

    # ── Main disk ─────────────────────────────────────────────────────────────

    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.axis("off")

    # Disk boundary
    boundary = mpatches.Circle((0, 0), 1.0, fill=False,
                           edgecolor="#CFD8DC", linewidth=1.5, zorder=1)
    ax.add_patch(boundary)

    # Disk background fill
    disk_bg = mpatches.Circle((0, 0), 1.0, facecolor="#ECEFF1", alpha=0.35,
                          zorder=0)
    ax.add_patch(disk_bg)

    # Adaptive zone rings
    ring_order = [
        ("governance", "governance boundary"),
        ("false_attractor", "false attractor boundary"),
    ]
    for key, label in ring_order:
        r_ring = zone_boundaries.get(key)
        if r_ring and 0 < r_ring < 1:
            ring = mpatches.Circle(
                (0, 0), r_ring,
                fill=False,
                edgecolor=ZONE_RING_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
                zorder=2,
            )
            ax.add_patch(ring)
            ax.text(
                0, r_ring + 0.025,
                label,
                ha="center", va="bottom",
                fontsize=7, color=ZONE_RING_COLOR,
                style="italic",
                zorder=6,
            )

    # Edges
    node_id_set = {n["id"] for n in nodes}
    for edge in edges:
        src_id = edge["source"]
        tgt_id = edge["target"]
        if src_id not in node_map or tgt_id not in node_map:
            continue
        src = node_map[src_id]
        tgt = node_map[tgt_id]
        x0, y0 = _xy(src["r"], src["theta"])
        x1, y1 = _xy(tgt["r"], tgt["theta"])
        color = PROP_EDGE_COLOR_NEG if edge.get("sign") == "negative" else PROP_EDGE_COLOR_POS
        ax.plot(
            [x0, x1], [y0, y1],
            color=color, linewidth=0.6, alpha=0.35, zorder=3,
        )

    # Nodes
    label_objects = []
    label_xy      = []

    for node in nodes:
        nid   = node["id"]
        r     = node.get("r", 0.02)
        theta = node.get("theta", 0.0)
        x, y  = _xy(r, theta)

        if node["type"] == "anchor":
            ax.scatter(x, y, s=80, color=ANCHOR_COLOR,
                       edgecolors=ANCHOR_EDGE_COLOR, linewidths=0.8,
                       marker="D", zorder=5)
        else:
            cls   = node.get("classification", "ambiguous")
            color = CLASSIFICATION_COLOR.get(cls, "#9E9E9E")
            size  = 55 + node.get("stress", 0.3) * 30
            ax.scatter(x, y, s=size, color=color,
                       edgecolors="white", linewidths=0.5,
                       alpha=0.88, zorder=5)

        # Collect label
        txt = ax.text(
            x, y, nid,
            fontsize=6.5, ha="center", va="center",
            fontweight="bold" if node["type"] == "anchor" else "normal",
            color="white" if node["type"] == "anchor" else "#212121",
            zorder=6,
            path_effects=[
                pe.withStroke(linewidth=1.5, foreground="white")
            ],
        )
        label_objects.append(txt)
        label_xy.append((x, y))

    # Adjust label positions to prevent collisions
    adjust_text(
        label_objects,
        x=[xy[0] for xy in label_xy],
        y=[xy[1] for xy in label_xy],
        ax=ax,
        expand_text=(1.3, 1.3),
        force_text=(0.4, 0.4),
        force_points=(0.2, 0.2),
        only_move={"text": "xy", "points": "xy"},
        avoid_points=False,
        lim=300,
    )

    # Legend
    legend_patches = []
    seen_cls = set()
    for node in nodes:
        if node["type"] == "anchor":
            if "anchor" not in seen_cls:
                legend_patches.append(
                    mpatches.Patch(color=ANCHOR_COLOR, label="anchor (governance)")
                )
                seen_cls.add("anchor")
        else:
            cls = node.get("classification", "ambiguous")
            if cls not in seen_cls:
                color = CLASSIFICATION_COLOR.get(cls, "#9E9E9E")
                legend_patches.append(mpatches.Patch(color=color, label=cls))
                seen_cls.add(cls)

    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=7,
        framealpha=0.7,
        title="classification",
        title_fontsize=7,
    )

    doc_id = result.get("document_metadata", {}).get("document_id", "")
    doc_name = result.get("document_metadata", {}).get("common_name", "")
    ax.set_title(
        f"{doc_id}  |  {doc_name}\nPoincaré Disk — Rhetorical Geometry Analysis",
        fontsize=10, pad=10, color="#263238",
    )

    lambda2_note = ""
    for entry in result.get("audit_trail", []):
        if entry.get("module") == "HSHGeometry" and entry.get("status") == "complete":
            lambda2_note = entry.get("note", "")
            break
    if lambda2_note:
        ax.annotate(
            lambda2_note,
            xy=(0, -1.08), fontsize=7, ha="center", color="#546E7A",
        )

    # ── Governance inset ─────────────────────────────────────────────────────

    axin.set_facecolor(BG_COLOR)
    axin.set_aspect("equal")
    gov_r = zone_boundaries.get("governance", 0.30)
    pad   = gov_r * 0.25
    axin.set_xlim(-(gov_r + pad), (gov_r + pad))
    axin.set_ylim(-(gov_r + pad), (gov_r + pad))
    axin.axis("off")
    axin.set_title("governance zone (inset)", fontsize=8, color="#263238", pad=6)

    gov_ring = mpatches.Circle((0, 0), gov_r, fill=False,
                           edgecolor=ZONE_RING_COLOR, linewidth=1.0,
                           linestyle="--", alpha=0.7)
    axin.add_patch(gov_ring)

    # Edges within the governance zone
    gov_ids = {n["id"] for n in nodes
               if n.get("zone") == "governance" or n["type"] == "anchor"}
    for edge in edges:
        if edge["source"] in gov_ids and edge["target"] in gov_ids:
            src = node_map.get(edge["source"])
            tgt = node_map.get(edge["target"])
            if src and tgt:
                x0, y0 = _xy(src["r"], src["theta"])
                x1, y1 = _xy(tgt["r"], tgt["theta"])
                color = (PROP_EDGE_COLOR_NEG
                         if edge.get("sign") == "negative"
                         else PROP_EDGE_COLOR_POS)
                axin.plot([x0, x1], [y0, y1],
                          color=color, linewidth=0.7, alpha=0.4)

    # Nodes in governance zone
    inset_labels = []
    inset_label_xy = []

    for node in nodes:
        if node.get("zone") != "governance" and node["type"] != "anchor":
            continue
        r     = node.get("r", 0.02)
        theta = node.get("theta", 0.0)
        x, y  = _xy(r, theta)

        if node["type"] == "anchor":
            axin.scatter(x, y, s=60, color=ANCHOR_COLOR,
                         edgecolors=ANCHOR_EDGE_COLOR, linewidths=0.7,
                         marker="D", zorder=5)
        else:
            cls   = node.get("classification", "ambiguous")
            color = CLASSIFICATION_COLOR.get(cls, "#9E9E9E")
            axin.scatter(x, y, s=45, color=color,
                         edgecolors="white", linewidths=0.4,
                         alpha=0.88, zorder=5)

        txt = axin.text(
            x, y, node["id"],
            fontsize=6, ha="center", va="center",
            fontweight="bold" if node["type"] == "anchor" else "normal",
            color="white" if node["type"] == "anchor" else "#212121",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )
        inset_labels.append(txt)
        inset_label_xy.append((x, y))

    if inset_labels:
        adjust_text(
            inset_labels,
            x=[xy[0] for xy in inset_label_xy],
            y=[xy[1] for xy in inset_label_xy],
            ax=axin,
            expand_text=(1.5, 1.5),
            force_text=(0.6, 0.6),
            force_points=(0.3, 0.3),
            only_move={"text": "xy", "points": "xy"},
            avoid_points=False,
            lim=400,
        )

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[visualize_rga] Figure 1 saved: {output_path}")


# ── Figure 2: T_μν Stress-Energy Field ────────────────────────────────────────

def draw_stress_field(result: Dict, output_path: Path) -> None:
    """Render the T_μν stress-energy field as a 2-D heatmap on the Poincaré disk.

    Each proposition contributes a Gaussian kernel centered at its disk
    position, scaled by its stress_energy_contribution. Anchor nodes are
    excluded from the kernel sum because they define the reference frame
    rather than contributing stress to it.

    The intensity concentration at the disk center (r≈0) reflects the
    convergence of proposition-to-anchor edge weights pulling toward
    the governance zone. It is not an independent stress value at that
    location but the summed geodesic tension of all edges terminating
    at the anchors.
    """
    graph    = result["graph_structure"]
    nodes    = graph["nodes"]
    doc_id   = result.get("document_metadata", {}).get("document_id", "")
    doc_name = result.get("document_metadata", {}).get("common_name", "")

    GRID = 400
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, GRID),
        np.linspace(-1, 1, GRID),
    )
    disk_mask = (xx**2 + yy**2) < 0.9998
    field = np.zeros((GRID, GRID))

    # Gaussian bandwidth relative to disk radius
    sigma = 0.14

    for node in nodes:
        if node["type"] == "anchor":
            continue
        r     = node.get("r", 0.45)
        theta = node.get("theta", 0.0)
        stress = node.get("stress", 0.3)
        cx, cy = _xy(r, theta)
        kernel = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        field += stress * kernel

    # Mask outside disk
    field[~disk_mask] = np.nan

    cmap = LinearSegmentedColormap.from_list(
        "rga_stress",
        ["#E3F2FD", "#90CAF9", "#EF9A9A", "#E53935", "#B71C1C"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("#ECEFF1")
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")

    im = ax.imshow(
        field,
        extent=(-1.0, 1.0, -1.0, 1.0),
        origin="lower",
        cmap=cmap,
        alpha=0.88,
        zorder=2,
    )

    # Disk boundary
    boundary = mpatches.Circle((0, 0), 1.0, fill=False,
                           edgecolor="#90A4AE", linewidth=1.5, zorder=5)
    ax.add_patch(boundary)

    # Node markers (light, for spatial reference)
    for node in nodes:
        r     = node.get("r", 0.02)
        theta = node.get("theta", 0.0)
        x, y  = _xy(r, theta)
        if node["type"] == "anchor":
            ax.scatter(x, y, s=40, color=ANCHOR_COLOR, marker="D",
                       edgecolors="white", linewidths=0.5, zorder=6, alpha=0.7)
        else:
            cls   = node.get("classification", "ambiguous")
            color = CLASSIFICATION_COLOR.get(cls, "#9E9E9E")
            ax.scatter(x, y, s=25, color=color,
                       edgecolors="white", linewidths=0.3, zorder=6, alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                        label="stress-energy contribution")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"{doc_id}  |  {doc_name}\n"
        r"$T_{\mu\nu}$ Stress-Energy Field — Rhetorical Geometry Analysis",
        fontsize=10, pad=10, color="#263238",
    )

    caption = (
        "Gaussian kernels centered at each proposition's Poincaré disk position, "
        "scaled by stress_energy_contribution.\n"
        "Intensity at center (r≈0) reflects convergent edge weight toward "
        "governance-zone anchors, not independent stress at that location."
    )
    fig.text(
        0.5, 0.01, caption,
        ha="center", va="bottom", fontsize=7, color="#546E7A",
        style="italic", wrap=True,
    )

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[visualize_rga] Figure 2 saved: {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Poincaré disk and T_μν field figures from RGA result JSON."
    )
    parser.add_argument("result_json", help="Path to completed schema v0.4 result JSON.")
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help="Directory for output PNGs. Defaults to same directory as result JSON.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_json)
    if not result_path.exists():
        sys.exit(f"[visualize_rga] ERROR: result JSON not found: {result_path}")

    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    out_dir = Path(args.output_dir) if args.output_dir else result_path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_id = result.get("document_metadata", {}).get("document_id", "DOC-unknown")
    # Derive a short file prefix from document_id, e.g. DOC-18300528-001 → DOC-001
    parts = doc_id.split("-")
    short_id = f"{parts[0]}-{parts[-1]}" if len(parts) >= 2 else doc_id

    disk_path   = out_dir / f"{short_id}_poincare_disk.png"
    stress_path = out_dir / f"{short_id}_T_munu_field.png"

    draw_poincare_disk(result, disk_path)
    draw_stress_field(result, stress_path)


if __name__ == "__main__":
    main()
