#!/usr/bin/env python3
"""Plot v26 condition results with C0/C1 represented once each."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l4_action_only_v26"
OUT = EXP / "results/notion_visualizations"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
FAMILY_LABELS = ["FC clear", "FC blocked", "Translate", "Rotate", "Lift & relocate"]
CONDITION_LABELS = ["C0\nImage", "C1\nR numeric", "C2\nR qualitative", "C3\nF numeric",
                    "C4\nF qualitative", "C5\nM numeric", "C6\nM qualitative"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    runs = [json.loads(line) for line in (EXP / "logs/runs.jsonl").read_text().splitlines() if line]
    assert len(runs) == 1750
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 16})

    def group(condition: str, resolution: str, family: str | None = None) -> list[dict]:
        return [r for r in runs if r["geometry_condition"] == condition
                and r["direction_resolution"] == resolution
                and (family is None or r["scene_family"] == family)]

    def rate(condition: str, resolution: str, family: str | None = None) -> float:
        rows = group(condition, resolution, family)
        assert len(rows) == (25 if family is not None else 125)
        return 100 * sum(r["scoring"]["action_matches_advisory"] for r in rows) / len(rows)

    # C0/C1 have no direction-dependent input or output in action-only v26.
    # Keep the D4-labelled run as the single, explicitly documented representative.
    fig, ax = plt.subplots(figsize=(14, 6.2))
    x = np.arange(7)
    w = 0.31
    bars = []
    for ci in range(2):
        bars.extend(ax.bar(x[ci], rate(f"C{ci}", "D4"), width=0.53,
                           color="#748899", label="C0/C1: one representative" if ci == 0 else None))
    for ci in range(2, 7):
        bars.extend(ax.bar(x[ci] - w/2, rate(f"C{ci}", "D4"), width=w,
                           color="#4E79A7", label="D4" if ci == 2 else None))
        bars.extend(ax.bar(x[ci] + w/2, rate(f"C{ci}", "D8"), width=w,
                           color="#F28E2B", label="D8" if ci == 2 else None))
    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_ylim(0, 72)
    ax.set_ylabel("Advisory action-label agreement (%)")
    ax.set_title("v26 Action Agreement: C0/C1 Once, C2-C6 by D4/D8")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1.1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#243447")
    fig.text(0.5, 0.015,
             "C0/C1: one D4-labelled repeat retained only as representative (n=125 each); C2-C6: D4/D8 distinct inputs (n=125 each).",
             ha="center", fontsize=10, color="#34495E")
    save(fig, "16_condition_agreement_c0c1_single.png")

    # Baselines are shown only once, in the first heatmap, clearly marked as non-directional.
    configurations = [
        ([("C0", "D4"), ("C1", "D4")] + [(f"C{i}", "D4") for i in range(2, 7)],
         ["C0*", "C1*"] + [f"C{i} D4" for i in range(2, 7)],
         "v26 Baselines C0/C1 + C2-C6 D4", "05_family_condition_baseline_plus_d4.png"),
        ([(f"C{i}", "D8") for i in range(2, 7)],
         [f"C{i} D8" for i in range(2, 7)],
         "v26 C2-C6 D8 (No Duplicate Baselines)", "05_family_condition_d8_c2c6.png"),
    ]
    for columns, labels, title, filename in configurations:
        matrix = np.array([[rate(c, d, family) for c, d in columns] for family in FAMILIES])
        fig, ax = plt.subplots(figsize=(11.7 if len(columns) == 7 else 9.3, 6.3))
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(columns)), labels)
        ax.set_yticks(range(5), FAMILY_LABELS)
        ax.set_title(title + "\nEach cell = 5 scenes × 5 seeds = 25 calls")
        for i in range(5):
            for j in range(len(columns)):
                ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center",
                        color="white" if matrix[i, j] >= 60 else "#243447", fontweight="bold")
        fig.colorbar(image, ax=ax, label="Advisory action-label agreement (%)", shrink=0.82)
        if len(columns) == 7:
            fig.text(0.5, 0.015, "* C0/C1 are non-directional; D4-labelled repeats are single representatives.",
                     ha="center", fontsize=10, color="#34495E")
        save(fig, filename)

    print("Created three C0/C1-deduplicated condition figures in", OUT)


if __name__ == "__main__":
    main()
