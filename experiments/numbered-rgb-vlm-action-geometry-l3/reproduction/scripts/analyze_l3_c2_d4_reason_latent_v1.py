#!/usr/bin/env python3
"""Compare C2/D4 scene cosine structure at block 24 and final RMSNorm."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/l3_v19_c2_d4_reason_latent_20260918"
FAMILIES = ["FC_BLOCKED", "FC_CLEAR", "LIFT_AND_RELOCATE", "ROTATE", "TRANSLATE"]
FAMILY_LABELS = ["FC blocked", "FC clear", "Lift", "Rotate", "Translate"]
HEATMAP_ORDER = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
FAMILY_COLORS = {
    "FC_BLOCKED": "#247fa0",
    "FC_CLEAR": "#188038",
    "LIFT_AND_RELOCATE": "#b85983",
    "ROTATE": "#8a73bd",
    "TRANSLATE": "#d5a03e",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0).all()
    return x / norms


def cosine(x: np.ndarray) -> np.ndarray:
    x = unit_rows(x)
    result = x @ x.T
    np.fill_diagonal(result, 1.0)
    return result


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n_pairs": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def within_values(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return np.array([matrix[i, j] for pos, i in enumerate(ids) for j in ids[pos + 1:]])


def between_values(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(np.flatnonzero(left), np.flatnonzero(right))].ravel()


def hypothesis(matrix: np.ndarray, mask: np.ndarray) -> dict:
    within = describe(within_values(matrix, mask))
    between = describe(between_values(matrix, mask, ~mask))
    return {
        "within": within,
        "clear_vs_other": between,
        "gap_within_minus_between": within["mean"] - between["mean"],
    }


def family_matrix(matrix: np.ndarray, records: list[dict]) -> np.ndarray:
    out = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.float64)
    labels = np.array([r["scene_family"] for r in records])
    for i, left in enumerate(FAMILIES):
        for j, right in enumerate(FAMILIES):
            lm, rm = labels == left, labels == right
            if i == j:
                values = within_values(matrix, lm)
            else:
                values = between_values(matrix, lm, rm)
            out[i, j] = values.mean()
    return out


def nearest_neighbors(matrix: np.ndarray, records: list[dict]) -> tuple[list[dict], float]:
    rows = []
    matches = 0
    for i, record in enumerate(records):
        candidates = matrix[i].copy()
        candidates[i] = -np.inf
        j = int(np.argmax(candidates))
        same = record["scene_family"] == records[j]["scene_family"]
        matches += same
        rows.append({
            "scene_id": record["scene_id"],
            "short_id": record["short_id"],
            "neighbor_scene_id": records[j]["scene_id"],
            "neighbor_short_id": records[j]["short_id"],
            "cosine": float(matrix[i, j]),
            "same_family": bool(same),
        })
    return rows, matches / len(records)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelcolor": "#465367",
        "axes.titlecolor": "#213044",
        "text.color": "#213044",
        "xtick.color": "#627287",
        "ytick.color": "#627287",
        "svg.fonttype": "none",
    })


def save(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=210, facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def add_family_boundaries(ax) -> None:
    for boundary in (4.5, 9.5, 14.5, 19.5):
        ax.axhline(boundary, color="white", lw=2.2)
        ax.axvline(boundary, color="white", lw=2.2)


def heatmap_pair(figures: Path, records: list[dict], mid: np.ndarray, final: np.ndarray) -> None:
    order = [i for family in HEATMAP_ORDER for i, record in enumerate(records) if record["scene_family"] == family]
    records = [records[i] for i in order]
    mid = mid[np.ix_(order, order)]
    final = final[np.ix_(order, order)]
    offdiag = np.concatenate([mid[~np.eye(25, dtype=bool)], final[~np.eye(25, dtype=bool)]])
    vmin, vmax = np.quantile(offdiag, [0.01, 0.99])
    labels = [r["short_id"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.1), constrained_layout=True)
    for ax, matrix, title in zip(axes, (mid, final), ("Block 24", "Final RMSNorm")):
        im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, loc="left", fontsize=15, weight="bold")
        ax.set_xticks(range(25), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(25), labels, fontsize=7)
        add_family_boundaries(ax)
        ax.set_xlabel("Scene")
        ax.set_ylabel("Scene")
    fig.colorbar(im, ax=axes, shrink=0.82, label="Cosine similarity")
    fig.suptitle("C2 D4 pre-reason scene similarity", x=0.03, ha="left", fontsize=20, weight="bold")
    fig.text(0.03, -0.005, "One deterministic vector per scene · fixed JSON prefix · no answer sampling", fontsize=10, color="#657387")
    save(fig, figures / "01_cosine_block24_vs_final")


def difference_heatmap(figures: Path, records: list[dict], mid: np.ndarray, final: np.ndarray) -> None:
    diff = final - mid
    limit = np.quantile(np.abs(diff[~np.eye(25, dtype=bool)]), 0.99)
    labels = [r["short_id"] for r in records]
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title("Final − Block 24 cosine similarity", loc="left", fontsize=18, weight="bold")
    ax.set_xticks(range(25), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(25), labels, fontsize=7)
    add_family_boundaries(ax)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Change in cosine")
    fig.text(0.12, 0.02, "Red = pair became more similar · Blue = pair became less similar", fontsize=10, color="#657387")
    save(fig, figures / "02_cosine_final_minus_block24")


def hypothesis_bars(figures: Path, summaries: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharey=True)
    for ax, key, title in zip(axes, ("fc_clear_family", "semantic_none"),
                              ("FC_CLEAR family: 5 scenes", "GT NONE: 4 scenes")):
        vals = [
            summaries[key]["block24"]["within"]["mean"],
            summaries[key]["block24"]["clear_vs_other"]["mean"],
            summaries[key]["final_norm"]["within"]["mean"],
            summaries[key]["final_norm"]["clear_vs_other"]["mean"],
        ]
        colors = ["#188038", "#a8c7fa", "#0b8043", "#5b8def"]
        bars = ax.bar([0, 1, 3, 4], vals, color=colors, width=0.78)
        ax.set_xticks([0, 1, 3, 4], ["Within\nBlock 24", "vs others\nBlock 24", "Within\nFinal", "vs others\nFinal"])
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
        ax.grid(axis="y", color="#e3e8ed", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Mean pairwise cosine")
    all_vals = [bar.get_height() for ax in axes for bar in ax.patches]
    lo, hi = min(all_vals), max(all_vals)
    axes[0].set_ylim(lo - max(0.002, (hi-lo)*0.35), hi + max(0.002, (hi-lo)*0.35))
    fig.suptitle("Clear-scene similarity hypothesis", x=0.045, ha="left", fontsize=19, weight="bold")
    fig.text(0.045, 0.01, "Primary grouping is scene family; semantic NONE is shown as a robustness view.", fontsize=9.5, color="#657387")
    fig.subplots_adjust(top=0.81, bottom=0.2, wspace=0.18)
    save(fig, figures / "03_clear_hypothesis")


def family_heatmaps(figures: Path, mid: np.ndarray, final: np.ndarray) -> None:
    vmin = min(mid.min(), final.min())
    vmax = max(mid.max(), final.max())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), constrained_layout=True)
    for ax, matrix, title in zip(axes, (mid, final), ("Block 24", "Final RMSNorm")):
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(5), FAMILY_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(5), FAMILY_LABELS)
        ax.set_title(title, loc="left", fontsize=14, weight="bold")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i,j] > (vmin+vmax)/2 else "#213044")
    fig.colorbar(im, ax=axes, shrink=0.82, label="Mean pairwise cosine")
    fig.suptitle("Mean similarity by scene family", x=0.03, ha="left", fontsize=19, weight="bold")
    save(fig, figures / "04_family_similarity_block24_vs_final")


def layerwise_plot(figures: Path, layer_metrics: dict) -> None:
    x = np.arange(1, 50)
    labels = [str(i) for i in range(1, 49)] + ["Final"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    for ax, key, title in zip(axes, ("fc_clear_family", "semantic_none"),
                              ("FC_CLEAR family", "GT NONE scenes")):
        within = np.array(layer_metrics[key]["within"])
        between = np.array(layer_metrics[key]["between"])
        gap = within - between
        ax.plot(x, within, color="#188038", lw=2.1, label="Within clear")
        ax.plot(x, between, color="#1a73e8", lw=2.1, label="Clear vs others")
        ax2 = ax.twinx()
        ax2.plot(x, gap, color="#9334e6", lw=1.6, ls="--", label="Gap")
        ax.axvline(24, color="#5f6368", lw=1, ls=":")
        ax.axvline(49, color="#5f6368", lw=1, ls=":")
        ax.set_ylabel("Mean cosine")
        ax2.set_ylabel("Within − between", color="#9334e6")
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
        ax.grid(color="#e3e8ed", linewidth=0.65)
        lines = ax.get_lines()[:2] + ax2.get_lines()[:1]
        ax.legend(lines, [line.get_label() for line in lines], loc="best", frameon=False, ncol=3)
    axes[-1].set_xticks([1, 8, 16, 24, 32, 40, 49], ["1", "8", "16", "24", "32", "40", "Final"])
    axes[-1].set_xlabel("Readout")
    fig.suptitle("How clear-scene separation changes through the network", x=0.05, ha="left", fontsize=18, weight="bold")
    fig.subplots_adjust(top=0.9, bottom=0.09, left=0.09, right=0.91, hspace=0.3)
    save(fig, figures / "05_layerwise_clear_separation")


def clear_profiles(figures: Path, records: list[dict], mid: np.ndarray, final: np.ndarray) -> None:
    clear = np.array([r["family_clear"] for r in records])
    ids = np.flatnonzero(clear)
    rows = []
    for i in ids:
        other_clear = clear.copy(); other_clear[i] = False
        rows.append((
            records[i]["short_id"],
            mid[i, other_clear].mean(), mid[i, ~clear].mean(),
            final[i, other_clear].mean(), final[i, ~clear].mean(),
        ))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
    x = np.arange(len(rows)); width = 0.36
    for ax, cols, title in zip(axes, ((1,2), (3,4)), ("Block 24", "Final RMSNorm")):
        a = [row[cols[0]] for row in rows]; b = [row[cols[1]] for row in rows]
        ax.bar(x-width/2, a, width, label="To other FC_CLEAR", color="#188038")
        ax.bar(x+width/2, b, width, label="To non-clear families", color="#1a73e8")
        ax.set_xticks(x, [row[0] for row in rows])
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
        ax.grid(axis="y", color="#e3e8ed", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean cosine per clear scene")
    all_vals = [p.get_height() for ax in axes for p in ax.patches]
    lo, hi = min(all_vals), max(all_vals)
    axes[0].set_ylim(lo - max(0.002, (hi-lo)*0.35), hi + max(0.002, (hi-lo)*0.35))
    fig.suptitle("Each FC_CLEAR scene: similarity to clear peers vs other scenes", x=0.05, ha="left", fontsize=18, weight="bold")
    fig.subplots_adjust(top=0.85, bottom=0.14, wspace=0.18)
    save(fig, figures / "06_clear_scene_profiles")


def clear_rows_heatmap(figures: Path, records: list[dict], mid: np.ndarray, final: np.ndarray) -> None:
    clear = np.array([r["family_clear"] for r in records])
    labels = [r["short_id"] for r in records]
    row_labels = [records[i]["short_id"] for i in np.flatnonzero(clear)]
    data = np.concatenate([mid[clear], final[clear]], axis=0)
    vmin, vmax = np.quantile(data, [0.01, 0.99])
    fig, axes = plt.subplots(2, 1, figsize=(14.2, 5.6), constrained_layout=True)
    for ax, matrix, title in zip(axes, (mid[clear], final[clear]), ("Block 24", "Final RMSNorm")):
        im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(25), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(5), row_labels)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        for boundary in (4.5, 9.5, 14.5, 19.5):
            ax.axvline(boundary, color="white", lw=2)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Cosine similarity")
    fig.suptitle("FC_CLEAR rows against all 25 scenes", x=0.03, ha="left", fontsize=18, weight="bold")
    save(fig, figures / "07_clear_to_all_scenes")


def pca_pair(figures: Path, records: list[dict], mid_raw: np.ndarray, final_raw: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    for ax, raw, title in zip(axes, (mid_raw, final_raw), ("Block 24", "Final RMSNorm")):
        x = unit_rows(raw)
        model = PCA(n_components=2, svd_solver="full")
        xy = model.fit_transform(x)
        for family in FAMILIES:
            ids = [i for i,r in enumerate(records) if r["scene_family"] == family]
            ax.scatter(xy[ids,0], xy[ids,1], s=75, color=FAMILY_COLORS[family], label=family, edgecolor="white")
            for i in ids:
                ax.text(xy[i,0], xy[i,1], records[i]["short_id"], fontsize=8)
        ax.set_title(f"{title} · PC1+PC2 {model.explained_variance_ratio_.sum():.1%}", loc="left", fontsize=13, weight="bold")
        ax.grid(color="#e3e8ed", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [x.replace("_", " ") for x in labels], ncol=5, frameon=False, loc="lower center")
    fig.suptitle("Supplementary PCA of the same cosine-normalized vectors", x=0.04, ha="left", fontsize=18, weight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.2, wspace=0.23)
    save(fig, figures / "08_pca_block24_vs_final")


def scene_gallery(figures: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(18, 10.5))
    for ax, record in zip(axes.flat, records):
        with Image.open(ROOT / record["image_path"]) as image:
            ax.imshow(image)
        ax.axis("off")
        reason = record["ground_truth"]["blocking_reason"]
        ax.set_title(f"{record['short_id']} · target {record['target_object_id']} · {reason}\nreason {record['reason_correct_count']}/5 · blocker {record['blocker_correct_count']}/5",
                     fontsize=8, color=FAMILY_COLORS[record["scene_family"]], pad=5)
    fig.suptitle("C2 D4 scene key", x=0.02, ha="left", fontsize=19, weight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.015, left=0.01, right=0.995, hspace=0.33, wspace=0.06)
    fig.savefig(figures / "09_scene_gallery.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    out = args.experiment.resolve()
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    style()

    records = json.loads((out / "records.json").read_text())
    manifest = json.loads((out / "extraction_manifest.json").read_text())
    assert sha(out / "hidden_states.npz") == manifest["hidden_states_sha256"]
    assert sha(out / "records.json") == manifest["records_sha256"]
    states = np.load(out / "hidden_states.npz", allow_pickle=False)
    assert states["scene_ids"].tolist() == [r["scene_id"] for r in records]
    assert states["block_states"].shape == (25, 48, 2048)

    all_matrices = {}
    layer_metrics = {
        "fc_clear_family": {"within": [], "between": []},
        "semantic_none": {"within": [], "between": []},
    }
    family_clear = np.array([r["family_clear"] for r in records], dtype=bool)
    semantic_clear = np.array([r["semantic_clear"] for r in records], dtype=bool)
    for i in range(49):
        key = f"block_{i+1}" if i < 48 else "final_norm"
        raw = states["block_states"][:, i] if i < 48 else states["final_norm_states"]
        matrix = cosine(raw)
        all_matrices[key] = matrix
        for name, mask in (("fc_clear_family", family_clear), ("semantic_none", semantic_clear)):
            layer_metrics[name]["within"].append(float(within_values(matrix, mask).mean()))
            layer_metrics[name]["between"].append(float(between_values(matrix, mask, ~mask).mean()))
    np.savez_compressed(out / "cosine_similarity_matrices.npz", **all_matrices,
                        scene_ids=np.array([r["scene_id"] for r in records]))

    mid = all_matrices["block_24"]
    final = all_matrices["final_norm"]
    summaries = {
        "fc_clear_family": {"block24": hypothesis(mid, family_clear), "final_norm": hypothesis(final, family_clear)},
        "semantic_none": {"block24": hypothesis(mid, semantic_clear), "final_norm": hypothesis(final, semantic_clear)},
    }
    for value in summaries.values():
        value["gap_change_final_minus_block24"] = (
            value["final_norm"]["gap_within_minus_between"] - value["block24"]["gap_within_minus_between"]
        )
    family_mid = family_matrix(mid, records)
    family_final = family_matrix(final, records)
    nn_mid, nn_mid_accuracy = nearest_neighbors(mid, records)
    nn_final, nn_final_accuracy = nearest_neighbors(final, records)

    with (out / "pairwise_cosine_block24_final.csv").open("w", newline="") as handle:
        fields = ["scene_a", "scene_b", "short_a", "short_b", "family_a", "family_b", "a_fc_clear", "b_fc_clear",
                  "a_semantic_clear", "b_semantic_clear", "block24_cosine", "final_cosine", "final_minus_block24"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(25):
            for j in range(i+1, 25):
                writer.writerow({
                    "scene_a": records[i]["scene_id"], "scene_b": records[j]["scene_id"],
                    "short_a": records[i]["short_id"], "short_b": records[j]["short_id"],
                    "family_a": records[i]["scene_family"], "family_b": records[j]["scene_family"],
                    "a_fc_clear": records[i]["family_clear"], "b_fc_clear": records[j]["family_clear"],
                    "a_semantic_clear": records[i]["semantic_clear"], "b_semantic_clear": records[j]["semantic_clear"],
                    "block24_cosine": mid[i,j], "final_cosine": final[i,j], "final_minus_block24": final[i,j]-mid[i,j],
                })

    summary = {
        "experiment": "l3_v19_c2_d4_reason_latent_20260918",
        "source": "l3_primary_blocker_v19 / C2 D4",
        "n_scenes": 25,
        "independent_point_unit": "scene, not sampling seed",
        "readout": manifest["readout"],
        "assistant_prefix": manifest["assistant_prefix"],
        "primary_comparison": ["block_24", "final_norm"],
        "hypotheses": summaries,
        "family_similarity": {"families": FAMILIES, "block24": family_mid.tolist(), "final_norm": family_final.tolist()},
        "nearest_neighbor_same_family_accuracy": {"block24": nn_mid_accuracy, "final_norm": nn_final_accuracy},
        "nearest_neighbors": {"block24": nn_mid, "final_norm": nn_final},
        "layerwise": layer_metrics,
        "gt_reason_scene_counts": dict(Counter(r["ground_truth"]["blocking_reason"] for r in records)),
        "historical_v19": {
            "reason_correct": sum(r["reason_correct_count"] for r in records),
            "blocker_correct": sum(r["blocker_correct_count"] for r in records),
            "total": 125,
        },
        "interpretation_limits": [
            "Cosine structure is descriptive and does not prove causal use of a representation.",
            "FC_CLEAR is a scene-family label; one member (C5) has GT CLEARANCE_OVERLAP, so GT NONE is reported separately.",
            "Pairwise cosine values are not independent observations because each scene participates in many pairs.",
            "A fixed teacher-forced JSON prefix moves the readout to the reason decision point but is not an actual sampled generation trajectory.",
            "Transformers BF16 extraction and historical vLLM constrained decoding are different backends.",
        ],
        "hidden_states_sha256": manifest["hidden_states_sha256"],
        "analysis_code_sha256": sha(Path(__file__)),
    }
    write_json(out / "analysis_summary.json", summary)

    heatmap_pair(figures, records, mid, final)
    difference_heatmap(figures, records, mid, final)
    hypothesis_bars(figures, summaries)
    family_heatmaps(figures, family_mid, family_final)
    layerwise_plot(figures, layer_metrics)
    clear_profiles(figures, records, mid, final)
    clear_rows_heatmap(figures, records, mid, final)
    pca_pair(figures, records, states["block_states"][:, 23], states["final_norm_states"])
    scene_gallery(figures, records)

    fc = summaries["fc_clear_family"]
    sem = summaries["semantic_none"]
    report = f"""# C2 D4 pre-reason latent cosine analysis

## Design

- 25 scenes, one deterministic point per scene; no answer generation and no sampling seed.
- Fixed assistant prefix: `{{\"blocking_reason\": \"`
- Readout: immediately before the first reason value token.
- Primary comparison: decoder block 24 vs final RMSNorm.

## Main result

### FC_CLEAR family (5 scenes)

- Block 24: within clear {fc['block24']['within']['mean']:.6f}; clear vs other {fc['block24']['clear_vs_other']['mean']:.6f}; gap {fc['block24']['gap_within_minus_between']:.6f}.
- Final: within clear {fc['final_norm']['within']['mean']:.6f}; clear vs other {fc['final_norm']['clear_vs_other']['mean']:.6f}; gap {fc['final_norm']['gap_within_minus_between']:.6f}.
- Gap change, final minus block 24: {fc['gap_change_final_minus_block24']:.6f}.

### Semantic clear / GT NONE (4 scenes)

- Block 24: within clear {sem['block24']['within']['mean']:.6f}; clear vs other {sem['block24']['clear_vs_other']['mean']:.6f}; gap {sem['block24']['gap_within_minus_between']:.6f}.
- Final: within clear {sem['final_norm']['within']['mean']:.6f}; clear vs other {sem['final_norm']['clear_vs_other']['mean']:.6f}; gap {sem['final_norm']['gap_within_minus_between']:.6f}.
- Gap change, final minus block 24: {sem['gap_change_final_minus_block24']:.6f}.

## Reproducibility and limits

- Hidden state archive SHA-256: `{manifest['hidden_states_sha256']}`
- The five historical v19 seeds are metadata only; they do not create five latent points.
- This is descriptive representation analysis, not proof of the model's causal reasoning.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    artifacts = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", artifacts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
