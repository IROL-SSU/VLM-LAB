#!/usr/bin/env python3
"""Analyze and plot C2 decision-boundary cosine similarities."""
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
from matplotlib.font_manager import FontProperties
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_c2d8_decision_latent_v22"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT = FontProperties(fname=FONT_PATH)
FONT_BOLD = FontProperties(fname=FONT_BOLD_PATH)
FAMILY_ORDER = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
FAMILY_PREFIX = {
    "FC_CLEAR": "C",
    "FC_BLOCKED": "B",
    "TRANSLATE": "T",
    "ROTATE": "R",
    "LIFT_AND_RELOCATE": "L",
}
FAMILY_COLORS = {
    "FC_CLEAR": "#2FA36B",
    "FC_BLOCKED": "#E45858",
    "TRANSLATE": "#3B82F6",
    "ROTATE": "#7C5CE7",
    "LIFT_AND_RELOCATE": "#F59E0B",
}
RETRIEVE = "RETRIEVE_NOW"
REARRANGE = "REARRANGE_FIRST"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    assert np.all(norms > 0)
    return values / norms


def cosine(values: np.ndarray, center: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    normalized = unit_rows(values)
    result = normalized @ normalized.T
    np.fill_diagonal(result, 1.0)
    return np.clip(result, -1.0, 1.0)


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def summarize_matrix(matrix: np.ndarray) -> dict:
    values = off_diagonal(matrix)
    return {
        "pair_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def split_pair_summary(matrix: np.ndarray, labels: list[str]) -> dict:
    same, different = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (same if labels[i] == labels[j] else different).append(float(matrix[i, j]))
    return {
        "same": {"n": len(same), "mean": float(np.mean(same)), "std": float(np.std(same))},
        "different": {"n": len(different), "mean": float(np.mean(different)), "std": float(np.std(different))},
        "mean_gap_same_minus_different": float(np.mean(same) - np.mean(different)),
    }


def nearest_neighbor_summary(matrix: np.ndarray, records: list[dict]) -> dict:
    family_matches = 0
    expected_matches = 0
    majority_matches = 0
    neighbors = []
    for i, record in enumerate(records):
        row = matrix[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        other = records[j]
        family_matches += record["scene_family"] == other["scene_family"]
        expected_matches += record["expected_decision"] == other["expected_decision"]
        majority_matches += record["majority_decision"] == other["majority_decision"]
        neighbors.append({
            "scene_id": record["scene_id"],
            "nearest_scene_id": other["scene_id"],
            "cosine": float(matrix[i, j]),
            "same_family": record["scene_family"] == other["scene_family"],
            "same_expected_decision": record["expected_decision"] == other["expected_decision"],
            "same_majority_decision": record["majority_decision"] == other["majority_decision"],
        })
    n = len(records)
    return {
        "same_family": family_matches,
        "same_family_rate": family_matches / n,
        "same_expected_decision": expected_matches,
        "same_expected_decision_rate": expected_matches / n,
        "same_majority_decision": majority_matches,
        "same_majority_decision_rate": majority_matches / n,
        "neighbors": neighbors,
    }


def short_id(record: dict) -> str:
    suffix = int(record["scene_id"].rsplit("v", 1)[1])
    return f"{FAMILY_PREFIX[record['scene_family']]}{suffix}"


def style_axis(ax) -> None:
    ax.tick_params(colors="#667085", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#D7DDE5")


def heatmap_pair(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    records: list[dict],
    *,
    centered: bool,
    condition_label: str,
) -> None:
    labels = [record["short_id"] for record in records]
    if centered:
        vmin, vmax = -1.0, 1.0
        cmap = "coolwarm"
        subtitle = "전체 25개 장면 평균 벡터를 뺀 뒤 cosine 계산"
        filename_title = "평균 중심화 cosine"
    else:
        values = np.concatenate([off_diagonal(left), off_diagonal(right)])
        vmin = max(-1.0, float(np.quantile(values, 0.01)) - 0.01)
        vmax = 1.0
        cmap = "viridis"
        subtitle = "각 셀은 두 장면의 decision-boundary hidden state cosine 유사도"
        filename_title = "Raw cosine"
    # Give the colorbar its own GridSpec column. Passing both heatmap axes
    # directly to fig.colorbar can place the bar over the right-hand panel
    # after subplots_adjust is applied.
    fig = plt.figure(figsize=(16, 7.6), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        left=0.07,
        right=0.95,
        bottom=0.12,
        top=0.82,
        wspace=0.20,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    colorbar_ax = fig.add_subplot(grid[0, 2])
    images = []
    for ax, matrix, title in zip(axes, (left, right), ("중간 · Decoder block 24", "끝단 · Final RMSNorm")):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        images.append(image)
        ax.set_title(title, loc="left", fontproperties=FONT_BOLD, fontsize=14, pad=12, color="#172033")
        ax.set_xticks(range(25), labels=labels, rotation=90, fontproperties=FONT, fontsize=7.5)
        ax.set_yticks(range(25), labels=labels, fontproperties=FONT, fontsize=7.5)
        for boundary in (4.5, 9.5, 14.5, 19.5):
            ax.axhline(boundary, color="white", linewidth=1.4)
            ax.axvline(boundary, color="white", linewidth=1.4)
        style_axis(ax)
    fig.suptitle(f"{condition_label} 결정 분기 latent · {filename_title}", x=0.06, ha="left", y=0.965,
                 fontproperties=FONT_BOLD, fontsize=20, color="#172033")
    fig.text(0.06, 0.91, subtitle, fontproperties=FONT, fontsize=10.5, color="#667085")
    colorbar = fig.colorbar(images[-1], cax=colorbar_ax)
    colorbar.set_label("cosine similarity", fontproperties=FONT, fontsize=9, labelpad=10)
    colorbar.ax.tick_params(labelsize=8, colors="#667085")
    fig.text(0.06, 0.025, "C=FC_CLEAR · B=FC_BLOCKED · T=TRANSLATE · R=ROTATE · L=LIFT_AND_RELOCATE",
             fontproperties=FONT, fontsize=9, color="#667085")
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def plot_delta(path: Path, delta: np.ndarray, records: list[dict]) -> None:
    labels = [record["short_id"] for record in records]
    off = off_diagonal(delta)
    limit = max(0.001, float(np.quantile(np.abs(off), 0.99)))
    fig = plt.figure(figsize=(9.8, 8.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.05),
        left=0.14,
        right=0.89,
        bottom=0.12,
        top=0.9,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(range(25), labels=labels, rotation=90, fontproperties=FONT, fontsize=8)
    ax.set_yticks(range(25), labels=labels, fontproperties=FONT, fontsize=8)
    for boundary in (4.5, 9.5, 14.5, 19.5):
        ax.axhline(boundary, color="white", linewidth=1.4)
        ax.axvline(boundary, color="white", linewidth=1.4)
    ax.set_title("끝단에서 더 비슷해졌는가?", loc="left", fontproperties=FONT_BOLD,
                 fontsize=18, pad=34, color="#172033")
    ax.text(0, -1.25, "Final cosine − block 24 cosine · 빨강은 증가, 파랑은 감소",
            fontproperties=FONT, fontsize=10, color="#667085")
    style_axis(ax)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("cosine change", fontproperties=FONT, fontsize=9, labelpad=10)
    colorbar.ax.tick_params(labelsize=8, colors="#667085")
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def plot_branch_margin(path: Path, records: list[dict]) -> None:
    margins = np.array([record["branch_logit_margin_retrieve_minus_rearrange"] for record in records])
    y = np.arange(len(records))
    colors = ["#3B82F6" if value >= 0 else "#F59E0B" for value in margins]
    fig, ax = plt.subplots(figsize=(12.5, 10.2))
    ax.barh(y, margins, color=colors, alpha=0.9)
    ax.axvline(0, color="#172033", linewidth=1.2)
    ax.set_yticks(y, labels=[record["short_id"] for record in records], fontproperties=FONT, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E1E6ED", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("TR logit − ARR logit  (양수: RETRIEVE 쪽)", fontproperties=FONT, fontsize=10)
    ax.set_title("결정 분기에서의 두 토큰 선호", loc="left", fontproperties=FONT_BOLD,
                 fontsize=19, pad=38, color="#172033")
    ax.text(0.0, 1.015, "막대는 Transformers 재추출 logits · 우측 표기는 기존 vLLM 5회 응답",
            transform=ax.transAxes, fontproperties=FONT, fontsize=10, color="#667085")
    span = max(abs(margins.min()), abs(margins.max()))
    ax.set_xlim(-span * 1.35, span * 1.35)
    for i, (value, record) in enumerate(zip(margins, records)):
        label = f"{record['retrieve_count']}R/{record['rearrange_count']}A · 기대 {'R' if record['expected_decision'] == RETRIEVE else 'A'}"
        ax.text(span * 1.31, i, label, va="center", ha="right", fontproperties=FONT,
                fontsize=8.2, color="#475467")
    for boundary in (4.5, 9.5, 14.5, 19.5):
        ax.axhline(boundary, color="#C9D1DC", linewidth=1.2)
    style_axis(ax)
    fig.text(0.08, 0.02, "TR은 RETRIEVE_NOW 경로, ARR은 REARRANGE_FIRST 경로의 첫 분기 토큰",
             fontproperties=FONT, fontsize=9, color="#667085")
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.08, top=0.9)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def plot_group_summary(path: Path, summaries: dict) -> None:
    categories = [
        ("같은 family", "family", "same"),
        ("다른 family", "family", "different"),
        ("같은 기대 결정", "expected", "same"),
        ("다른 기대 결정", "expected", "different"),
        ("같은 다수결", "majority", "same"),
        ("다른 다수결", "majority", "different"),
    ]
    mid = [summaries["mid"][group][side]["mean"] for _, group, side in categories]
    final = [summaries["final"][group][side]["mean"] for _, group, side in categories]
    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.bar(x - width / 2, mid, width, label="Block 24", color="#7C5CE7")
    ax.bar(x + width / 2, final, width, label="Final", color="#3B82F6")
    ax.set_xticks(x, [item[0] for item in categories], fontproperties=FONT, fontsize=9)
    ax.set_ylabel("평균 raw cosine", fontproperties=FONT, fontsize=10)
    ax.set_title("집단별 평균 cosine", loc="left", fontproperties=FONT_BOLD,
                 fontsize=19, pad=28, color="#172033")
    ax.text(0, 1.01, "쌍(pair)을 독립 표본으로 간주한 검정은 하지 않은 기술 통계",
            transform=ax.transAxes, fontproperties=FONT, fontsize=10, color="#667085")
    ax.grid(axis="y", color="#E1E6ED", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, prop=FONT)
    low = min(mid + final)
    high = max(mid + final)
    pad = max(0.002, (high - low) * 0.35)
    ax.set_ylim(low - pad, min(1.0, high + pad))
    style_axis(ax)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.86)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.10f}" for value in row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    output = args.experiment.resolve()
    records = json.loads((output / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "extraction_manifest.json").read_text(encoding="utf-8"))
    condition_key = manifest["source_condition"]
    assert condition_key in {"C2_D4", "C2_D8"}, condition_key
    condition_label = condition_key.replace("_", " ")
    assert sha(output / "latent_states.npz") == manifest["latent_states_sha256"]
    assert sha(output / "records.json") == manifest["records_sha256"]
    states = np.load(output / "latent_states.npz", allow_pickle=False)
    assert states["scene_ids"].tolist() == [record["scene_id"] for record in records]
    mid_states = states["mid_layer24_states"]
    final_states = states["final_norm_states"]
    assert mid_states.shape == final_states.shape == (25, 2048)
    for record in records:
        record["short_id"] = short_id(record)

    mid_raw = cosine(mid_states)
    final_raw = cosine(final_states)
    mid_centered = cosine(mid_states, center=True)
    final_centered = cosine(final_states, center=True)
    delta = final_raw - mid_raw
    labels = [record["short_id"] for record in records]
    families = [record["scene_family"] for record in records]
    expected = [record["expected_decision"] for record in records]
    majority = [record["majority_decision"] for record in records]
    outcomes = [record["outcome"] for record in records]

    summary = {
        "n_scenes": 25,
        "condition": condition_key,
        "readout": manifest["decision_boundary"]["readout_definition"],
        "independent_unit": "scene, not seed",
        "historical_responses": 125,
        "historical_decision_counts": {
            RETRIEVE: sum(record["retrieve_count"] for record in records),
            REARRANGE: sum(record["rearrange_count"] for record in records),
        },
        "historical_outcomes": dict(Counter(outcomes)),
        "raw_cosine": {
            "mid_layer24": summarize_matrix(mid_raw),
            "final_norm": summarize_matrix(final_raw),
            "pairwise_correlation_mid_vs_final": float(np.corrcoef(off_diagonal(mid_raw), off_diagonal(final_raw))[0, 1]),
            "final_minus_mid": summarize_matrix(delta),
            "mean_absolute_pair_change": float(np.mean(np.abs(off_diagonal(delta)))),
        },
        "mean_centered_cosine": {
            "mid_layer24": summarize_matrix(mid_centered),
            "final_norm": summarize_matrix(final_centered),
            "pairwise_correlation_mid_vs_final": float(np.corrcoef(off_diagonal(mid_centered), off_diagonal(final_centered))[0, 1]),
        },
        "group_pair_cosine": {
            "mid": {
                "family": split_pair_summary(mid_raw, families),
                "expected": split_pair_summary(mid_raw, expected),
                "majority": split_pair_summary(mid_raw, majority),
            },
            "final": {
                "family": split_pair_summary(final_raw, families),
                "expected": split_pair_summary(final_raw, expected),
                "majority": split_pair_summary(final_raw, majority),
            },
        },
        "nearest_neighbor": {
            "mid_layer24": nearest_neighbor_summary(mid_raw, records),
            "final_norm": nearest_neighbor_summary(final_raw, records),
        },
        "branch_preference": {},
        "limitations": [
            "25 scenes only; pairwise cosine values are not independent samples.",
            "Raw decoder hidden states can be anisotropic, so mean-centered cosine is shown as a sensitivity view.",
            "Scene family, object identity, image layout, prompt length, and geometry payload all vary together.",
            manifest["historical_overlay"],
            "A latent similarity pattern is descriptive and does not establish a causal reasoning mechanism.",
        ],
        "source_code_sha256": sha(Path(__file__)),
        "latent_states_sha256": manifest["latent_states_sha256"],
    }
    branch_matches_majority = 0
    branch_matches_expected = 0
    for record in records:
        branch = RETRIEVE if record["branch_logit_margin_retrieve_minus_rearrange"] >= 0 else REARRANGE
        branch_matches_majority += branch == record["majority_decision"]
        branch_matches_expected += branch == record["expected_decision"]
    summary["branch_preference"] = {
        "matches_historical_majority": branch_matches_majority,
        "matches_historical_majority_rate": branch_matches_majority / 25,
        "matches_scene_design": branch_matches_expected,
        "matches_scene_design_rate": branch_matches_expected / 25,
        "caution": "Transformers BF16 raw logits versus historical vLLM constrained sampling; exact backend equivalence is not assumed.",
    }

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    heatmap_pair(
        figures / "01_raw_cosine_mid_vs_final.png",
        mid_raw,
        final_raw,
        records,
        centered=False,
        condition_label=condition_label,
    )
    heatmap_pair(
        figures / "02_centered_cosine_mid_vs_final.png",
        mid_centered,
        final_centered,
        records,
        centered=True,
        condition_label=condition_label,
    )
    plot_delta(figures / "03_final_minus_mid_cosine.png", delta, records)
    plot_branch_margin(figures / "04_decision_branch_logits.png", records)
    plot_group_summary(figures / "05_group_mean_cosine.png", summary["group_pair_cosine"])

    write_matrix_csv(output / "cosine_mid_layer24_raw.csv", mid_raw, labels)
    write_matrix_csv(output / "cosine_final_norm_raw.csv", final_raw, labels)
    write_matrix_csv(output / "cosine_mid_layer24_centered.csv", mid_centered, labels)
    write_matrix_csv(output / "cosine_final_norm_centered.csv", final_centered, labels)
    with (output / "pairwise_cosine.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "scene_a", "scene_b", "same_family", "same_expected", "same_majority",
            "mid_raw", "final_raw", "final_minus_mid", "mid_centered", "final_centered",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(25):
            for j in range(i + 1, 25):
                writer.writerow({
                    "scene_a": records[i]["scene_id"],
                    "scene_b": records[j]["scene_id"],
                    "same_family": families[i] == families[j],
                    "same_expected": expected[i] == expected[j],
                    "same_majority": majority[i] == majority[j],
                    "mid_raw": f"{mid_raw[i, j]:.10f}",
                    "final_raw": f"{final_raw[i, j]:.10f}",
                    "final_minus_mid": f"{delta[i, j]:.10f}",
                    "mid_centered": f"{mid_centered[i, j]:.10f}",
                    "final_centered": f"{final_centered[i, j]:.10f}",
                })
    with (output / "scene_latent_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "short_id", "scene_id", "family", "target_id", "expected_decision", "retrieve_count",
            "rearrange_count", "correct_count", "outcome", "majority_decision",
            "branch_logit_retrieve_TR", "branch_logit_rearrange_ARR",
            "branch_margin_retrieve_minus_rearrange", "branch_two_way_probability_retrieve",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "short_id": record["short_id"],
                "scene_id": record["scene_id"],
                "family": record["scene_family"],
                "target_id": record["target_object_id"],
                "expected_decision": record["expected_decision"],
                "retrieve_count": record["retrieve_count"],
                "rearrange_count": record["rearrange_count"],
                "correct_count": record["correct_count"],
                "outcome": record["outcome"],
                "majority_decision": record["majority_decision"],
                "branch_logit_retrieve_TR": record["branch_logit_retrieve_TR"],
                "branch_logit_rearrange_ARR": record["branch_logit_rearrange_ARR"],
                "branch_margin_retrieve_minus_rearrange": record["branch_logit_margin_retrieve_minus_rearrange"],
                "branch_two_way_probability_retrieve": record["branch_two_way_probability_retrieve"],
            })
    write_json(output / "analysis_summary.json", summary)

    mid_stats = summary["raw_cosine"]["mid_layer24"]
    final_stats = summary["raw_cosine"]["final_norm"]
    family_mid = summary["group_pair_cosine"]["mid"]["family"]
    family_final = summary["group_pair_cosine"]["final"]["family"]
    expected_mid = summary["group_pair_cosine"]["mid"]["expected"]
    expected_final = summary["group_pair_cosine"]["final"]["expected"]
    nearest_mid = summary["nearest_neighbor"]["mid_layer24"]
    nearest_final = summary["nearest_neighbor"]["final_norm"]
    report = f"""# {condition_label} 결정 분기 latent cosine — 중간층 vs 끝단

## 무엇을 측정했나

- 대상: L2 v16의 {condition_label} 25개 장면. 기존 5개 seed 응답 125개는 결과 라벨로만 연결했다.
- 독립 latent 점은 장면당 하나, 총 25개다. 같은 장면의 5개 seed는 결정 전 입력이 같으므로 복제하지 않았다.
- 실제 출력은 `{{\"decision\": \"RE...`까지 공통이다. 토크나이저에서 그 다음 토큰이 `TR`(RETRIEVE_NOW)과 `ARR`(REARRANGE_FIRST)로 처음 갈라진다.
- 공통 토큰 `RE` 위치에서 decoder block 24 출력과 block 48 뒤 final RMSNorm 출력을 각각 2,048차원 벡터로 추출했다.
- 모델 답변을 다시 생성하지 않았다. 기존 vLLM 결과와 같은 checkpoint·이미지·prompt를 Transformers BF16 forward로 재입력했다.

## 핵심 결과

- Raw cosine 300쌍 평균: block 24 **{mid_stats['mean']:.6f}**, final **{final_stats['mean']:.6f}**.
- 중간–끝단 pairwise cosine 패턴 상관: **{summary['raw_cosine']['pairwise_correlation_mid_vs_final']:.4f}**.
- 끝단에서 pairwise cosine이 바뀐 절댓값 평균: **{summary['raw_cosine']['mean_absolute_pair_change']:.6f}**.
- 같은 family와 다른 family의 평균 차이(같음−다름): block 24 **{family_mid['mean_gap_same_minus_different']:.6f}**, final **{family_final['mean_gap_same_minus_different']:.6f}**.
- 같은 기대 결정과 다른 기대 결정의 평균 차이: block 24 **{expected_mid['mean_gap_same_minus_different']:.6f}**, final **{expected_final['mean_gap_same_minus_different']:.6f}**.
- 최근접 장면이 같은 family인 경우: block 24 **{nearest_mid['same_family']}/25**, final **{nearest_final['same_family']}/25**.
- 재추출 branch logit 방향과 기존 5회 다수결의 일치: **{branch_matches_majority}/25**. 장면 설계 기대와의 일치: **{branch_matches_expected}/25**.

## 그림 읽는 법

1. `01_raw_cosine_mid_vs_final.png`: 원래 hidden state의 cosine. 모델 표현의 공통 방향 때문에 값이 전반적으로 높을 수 있다.
2. `02_centered_cosine_mid_vs_final.png`: 25개 장면 평균 벡터를 뺀 민감도 분석. 절대값보다 장면 관계가 유지되는지를 본다.
3. `03_final_minus_mid_cosine.png`: 끝단 cosine에서 중간층 cosine을 뺀 값. 빨강은 두 장면이 더 비슷해졌고 파랑은 덜 비슷해졌음을 뜻한다.
4. `04_decision_branch_logits.png`: 분기점에서 `TR`과 `ARR`의 raw logit 차이. 기존 5회 응답은 오른쪽에 함께 적었다.
5. `05_group_mean_cosine.png`: 같은/다른 family·기대 결정·기존 다수결별 평균 cosine 비교다.

## 해석 제한

- cosine이 높다고 모델이 같은 이유로 판단했다는 뜻은 아니다.
- 25개 장면뿐이고 family, 물체, 이미지 배치, geometry payload 길이가 함께 달라진다.
- 300개 장면쌍은 서로 독립인 300개 표본이 아니므로 유의성 검정을 하지 않았다.
- Raw hidden state는 anisotropy 때문에 cosine이 높게 몰릴 수 있어 mean-centered 결과를 함께 제시했다.
- hidden state는 Transformers에서 재추출했고 기존 응답은 vLLM constrained decoding 결과다. backend 수치 동일성이나 인과성을 주장하지 않는다.

## 산출물

- `latent_states.npz`: block 24와 final state `[25, 2048]`, branch logits `[25, 2]`.
- `records.json`, `extraction_manifest.json`: 입력·token 경계·hash·실행 환경 감사 정보.
- `cosine_*csv`: 네 종류 cosine matrix.
- `pairwise_cosine.csv`: 300개 장면쌍의 중간/끝단 cosine.
- `scene_latent_summary.csv`: 장면별 기존 응답과 분기 logits.
- `analysis_summary.json`: 기술 통계와 한계.
- `figures/`: PNG 5개.

## 재현

```bash
.qwen3-vl/venv/bin/python scripts/extract_l2_c2d8_decision_latent_v22.py --condition-key {condition_key} --output {output}
.l1-latent/plot-venv/bin/python scripts/analyze_l2_c2d8_decision_latent_v22.py --experiment {output}
```
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): sha(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", artifacts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
