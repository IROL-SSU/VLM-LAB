#!/usr/bin/env python3
"""Summarize v18 approach decisions without assigning scene-family ground truth.

The analysis itself uses only the standard library. Matplotlib is optional and
creates static PNG/PDF figures; use --no-plots to skip it. The v16 comparison
describes a semantic change in task and labels, not a change in accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_approach_access_geometry_v18"
BASELINE = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16"
FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")
DECISIONS = ("ACCESSIBLE", "BLOCKED")
V16_DECISIONS = ("RETRIEVE_NOW", "REARRANGE_FIRST")
V16_MAPPING = dict(zip(V16_DECISIONS, DECISIONS))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def decision(row, labels=DECISIONS):
    parsed = row.get("parsed_response")
    value = parsed.get("decision") if isinstance(parsed, dict) else row.get("decision")
    return value if row.get("schema_valid") is True and value in labels else "INVALID"


def key(row):
    return row["scene_id"], row["condition_key"], row["seed"]


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def distribution(rows, labels=DECISIONS):
    tally = Counter(decision(row, labels) for row in rows)
    return {
        "n": len(rows), labels[0]: tally[labels[0]], labels[1]: tally[labels[1]],
        "INVALID": tally["INVALID"],
        "positive_rate": tally[labels[0]] / len(rows) if rows else None,
    }


def transition_counts(pairs, left="c0_decision", right="condition_decision"):
    tally = Counter((row[left], row[right]) for row in pairs)
    valid = sum(tally[(a, b)] for a in DECISIONS for b in DECISIONS)
    to_accessible = tally[("BLOCKED", "ACCESSIBLE")]
    to_blocked = tally[("ACCESSIBLE", "BLOCKED")]
    return {
        "n": len(pairs), "valid_pairs": valid, "invalid_pairs": len(pairs) - valid,
        "unchanged_accessible": tally[("ACCESSIBLE", "ACCESSIBLE")],
        "unchanged_blocked": tally[("BLOCKED", "BLOCKED")],
        "blocked_to_accessible": to_accessible, "accessible_to_blocked": to_blocked,
        "changed_valid_pairs": to_accessible + to_blocked,
        "net_accessible_change": to_accessible - to_blocked,
    }


def validate(config, schedule, records, baseline_records):
    expected = config["scheduled_calls"]
    assert expected == 1500, f"Expected a 1,500-call experiment, got {expected}"
    assert len(schedule) == len(records) == expected, (len(schedule), len(records), expected)
    scheduled = {row["run_id"]: row for row in schedule}
    observed = {row["run_id"]: row for row in records}
    assert len(scheduled) == len(observed) == expected, "Duplicate run IDs"
    assert set(scheduled) == set(observed), "Missing or unexpected run IDs"
    assert len({key(row) for row in records}) == expected, "Duplicate scene-condition-seed keys"
    baseline_map = {key(row): row for row in baseline_records}
    assert len(baseline_map) == len(baseline_records) == expected
    assert set(baseline_map) == {key(row) for row in records}, "v16/v18 pairing differs"
    seeds = set(config["seeds"])
    assert len(seeds) == 5 and config["scene_count"] == 25
    scene_groups = defaultdict(list)
    checked_fields = ("scene_id", "scene_family", "condition_key", "seed", "image_sha256", "geometry_sha256", "target_object_id")
    for row in records:
        planned = scheduled[row["run_id"]]
        previous = baseline_map[key(row)]
        for field in checked_fields:
            assert row[field] == planned[field] == previous[field], (row["run_id"], field)
        for field in ("prompt_sha256", "user_prompt"):
            assert row[field] == planned[field], (row["run_id"], field)
        for field in ("model_id", "generation", "system_prompt", "schema", "schema_sha256"):
            assert row[field] == config[field], (row["run_id"], field)
        assert row["model_id"] == previous["model_id"]
        assert row["generation"] == previous["generation"]
        if row["schema_valid"]:
            assert decision(row) in DECISIONS, (row["run_id"], "Invalid recorded decision")
        scene_groups[(row["scene_id"], row["condition_key"])].append(row)
    assert len(scene_groups) == 300
    assert len({row["condition_key"] for row in records}) == 12
    assert len({row["scene_id"] for row in records}) == 25
    assert set(row["scene_family"] for row in records) == set(FAMILIES)
    assert set(Counter(row["condition_key"] for row in records).values()) == {125}
    assert set(Counter((row["condition_key"], row["scene_family"]) for row in records).values()) == {25}
    for rows in scene_groups.values():
        assert len(rows) == 5 and {row["seed"] for row in rows} == seeds
    return baseline_map


def create_plots(results_dir, condition_results, scene_results, comparisons, keys):
    try:
        import matplotlib
    except ModuleNotFoundError:
        return [], "matplotlib not installed; core JSON/CSV/report files were generated."
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    condition_map = {row["condition_key"]: row for row in condition_results}
    comparison_map = {(row["condition_key"], row["scene_family"]): row for row in comparisons}
    family_labels = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT / RELOCATE"]

    def save(fig, stem, caption):
        fig.savefig(plot_dir / (stem + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(plot_dir / (stem + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        plots.append({"png": "plots/" + stem + ".png", "pdf": "plots/" + stem + ".pdf", "caption": caption})

    def heatmap(data, xlabels, ylabels, title, subtitle, cmap, vmin, vmax, formatter, colorbar_label, stem, caption):
        fig, ax = plt.subplots(figsize=(12.0, 6.3), layout="constrained")
        mesh = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_title(title + "\n" + subtitle, loc="left", pad=16)
        for i, line in enumerate(data):
            for j, value in enumerate(line):
                dark = value > 65 if vmin == 0 else abs(value) > 65
                ax.text(j, i, formatter(value), ha="center", va="center", fontsize=9,
                        color="white" if dark else "#172033")
        fig.colorbar(mesh, ax=ax, shrink=0.8, label=colorbar_label)
        save(fig, stem, caption)

    matrix = [[100 * condition_map[c]["by_family"][f]["positive_rate"] for c in keys] for f in FAMILIES]
    heatmap(matrix, keys, family_labels, "v18 approach accessibility: responses by family and condition",
            "ACCESSIBLE response rate (%); each cell = 5 scenes x 5 seeds. Not execution accuracy.",
            "Blues", 0, 100, lambda value: f"{value:.0f}%", "ACCESSIBLE (%)",
            "01_accessible_by_family_condition", "조건·family별 ACCESSIBLE 비율. 각 칸은 25회이며 정확도를 뜻하지 않습니다.")

    fc_scenes = sorted({row["scene_id"] for row in scene_results if row["scene_family"] == "FC_BLOCKED"})
    scene_map = {(row["scene_id"], row["condition_key"]): row for row in scene_results}
    fc_matrix = [[20 * scene_map[(scene, c)]["ACCESSIBLE"] for c in keys] for scene in fc_scenes]
    heatmap(fc_matrix, keys, [s.removeprefix("scene_") for s in fc_scenes],
            "FC_BLOCKED: approach decisions for each scene", "ACCESSIBLE count / 5 seeds; columns retain the same condition order.",
            "Blues", 0, 100, lambda value: f"{round(value / 20)}/5", "ACCESSIBLE (%)",
            "02_fc_blocked_per_scene", "FC_BLOCKED 장면별 ACCESSIBLE 횟수(/5). 한 장면 안에서 seed 간 응답이 달라지는지도 확인할 수 있습니다.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout="constrained")
    for ax, condition in zip(axes, ("C0", "C2_D8")):
        rows = [comparison_map[(condition, f)] for f in FAMILIES]
        for offset, field, color, label in ((-0.2, "v16_retrieve_now", "#94a3b8", "v16 RETRIEVE_NOW"),
                                            (0.2, "v18_accessible", "#2563eb", "v18 ACCESSIBLE")):
            values = [percentage(row[field], row["n"]) for row in rows]
            bars = ax.bar([i + offset for i in range(len(rows))], values, width=0.38, color=color, label=label)
            ax.bar_label(bars, labels=[f"{value:.0f}" for value in values], padding=3, fontsize=9)
        ax.set_xticks(range(len(FAMILIES)), family_labels, rotation=30, ha="right")
        ax.set_title(condition)
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Response rate (%)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Retrieval question (v16) vs approach question (v18)\nDifferent tasks: changes in response distribution, not accuracy", fontsize=13)
    save(fig, "03_v16_v18_c0_c2d8", "C0 및 C2 D8의 v16 RETRIEVE_NOW ↔ v18 ACCESSIBLE 비율 비교. 질문과 라벨의 의미가 달라 정확도 향상으로 해석하지 않습니다.")

    deltas = [[comparison_map[(c, f)]["positive_rate_delta_pp"] for c in keys] for f in FAMILIES]
    heatmap(deltas, keys, family_labels, "v16 to v18: response-rate change across all conditions",
            "v18 ACCESSIBLE minus v16 RETRIEVE_NOW (percentage points). Same scenes and seeds; different tasks.",
            "RdBu", -100, 100, lambda value: f"{value:+.0f}", "Change (percentage points)",
            "04_v16_v18_family_delta", "모든 조건에서 v16→v18 응답 비율 변화(%p). FC_BLOCKED 외 family로 변화가 확산됐는지 확인합니다.")
    return plots, None


def write_report(results_dir, config, summary, scenes, comparisons, plots):
    conditions = summary["condition_results"]
    keys = [row["condition_key"] for row in conditions]
    comparison_map = {(row["condition_key"], row["scene_family"]): row for row in comparisons}
    fc_rows = [row for row in comparisons if row["scene_family"] == "FC_BLOCKED"]
    old_fc = sum(row["v16_retrieve_now"] for row in fc_rows)
    new_fc = sum(row["v18_accessible"] for row in fc_rows)
    fc_n = sum(row["n"] for row in fc_rows)
    biggest = sorted((row for row in comparisons if row["scene_family"] != "FC_BLOCKED"),
                     key=lambda row: abs(row["positive_rate_delta_pp"]), reverse=True)[:5]
    lines = [
        "# L2 접근 경로 판단 × Geometry 단일정보 — v18 결과", "",
        "25장면 × 12조건 × 5 seed = 1,500회. 직전 합의한 짧은 프롬프트로 선반 입구에서 타깃까지의 접근 경로만 질문했습니다. v16과 동일한 이미지·Geometry·모델·샘플링 설정·seed 조합을 사용하고 system/user 프롬프트 및 출력 라벨을 변경했습니다.", "",
        "접근 가능성 정답(GT)이나 실제 로봇 실행 결과는 없습니다. `FC_BLOCKED` 등은 기존 장면 family 이름이며 접근 차단 정답으로 사용하지 않았습니다. 아래 수치는 모델의 응답 비율입니다. 특히 v16의 `RETRIEVE_NOW`와 v18의 `ACCESSIBLE`은 질문의 의미가 달라, 대응시킨 비율 차이는 정확도·회수 성공률 개선을 뜻하지 않습니다.", "",
        "## 주요 관찰", "",
        f"- FC_BLOCKED 전체 응답: v16 RETRIEVE_NOW {old_fc}/{fc_n} ({percentage(old_fc, fc_n):.1f}%) → v18 ACCESSIBLE {new_fc}/{fc_n} ({percentage(new_fc, fc_n):.1f}%), {percentage(new_fc-old_fc, fc_n):+.1f}%p. 12조건을 합산한 기술 통계이며 독립 장면 수가 늘어난 것은 아닙니다.",
    ]
    for condition in ("C0", "C2_D8"):
        row = comparison_map[(condition, "FC_BLOCKED")]
        lines.append(f"- FC_BLOCKED / {condition}: {row['v16_retrieve_now']}/{row['n']} → {row['v18_accessible']}/{row['n']} ({row['positive_rate_delta_pp']:+.1f}%p). 같은 장면·seed의 v16 REARRANGE_FIRST→v18 ACCESSIBLE {row['blocked_to_accessible']}회, v16 RETRIEVE_NOW→v18 BLOCKED {row['accessible_to_blocked']}회.")
    lines.append("- FC_BLOCKED 외 변화가 큰 조합: " + "; ".join(f"{row['condition_key']} / {row['scene_family']} {row['v16_retrieve_now']}/25→{row['v18_accessible']}/25 ({row['positive_rate_delta_pp']:+.1f}%p)" for row in biggest) + ".")
    lines += ["", "## 실험 및 무결성", "",
              f"- 모델: `{config['model_id']}`; seed: {', '.join(map(str, config['seeds']))}.",
              f"- 완료 {summary['completed_calls']}/1,500; JSON schema 유효 {summary['schema_valid_calls']}/1,500; 인프라 오류 {summary['infrastructure_errors']}회.",
              f"- 생성 시간 합계: {summary['total_generation_seconds']:.1f}초. 모델 적재 및 분석 시간은 제외합니다.",
              "- 1,500개의 run ID 및 장면·조건·seed 조합이 누락·중복 없이 일치하며, 모든 조건은 125회, family×조건은 25회입니다.",
              "- 이미지/Geometry SHA256, target ID, model ID, sampling 설정을 v16 로그와 대조했습니다. 응답 비율 분모에는 전체 실행 수를 사용하고 잘못된 응답은 별도 집계합니다.", "",
              "## 프롬프트", "", "System:", "", "```text", config["system_prompt"], "```", "", "User:", "", "```text", config["user_prompt_template"], "```", "",
              "C0의 Geometry는 `null`이며, 나머지 조건은 각 조건의 기존 단일정보 Geometry JSON입니다.", "",
              "## 조건별 응답", "",
              "| 조건 | 정보 | ACCESSIBLE /125 | 비율 | BLOCKED | INVALID | 5 seed 전원 일치 /25 | C0 대비 B→A | C0 대비 A→B |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in conditions:
        lines.append(f"| {row['condition_key']} | {row['condition_name']} | {row['ACCESSIBLE']} | {100*row['positive_rate']:.1f}% | {row['BLOCKED']} | {row['INVALID']} | {row['unanimous_scenes']} | {row['blocked_to_accessible']} | {row['accessible_to_blocked']} |")
    lines += ["", "B→A는 같은 v18의 C0 BLOCKED→해당 조건 ACCESSIBLE, A→B는 반대 방향입니다. seed 전원 일치는 응답의 안정성을 나타내며 정답 여부를 나타내지 않습니다.", "",
              "## Family별 ACCESSIBLE 횟수 /25", "", "| 조건 | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
    for row in conditions:
        lines.append("| " + row["condition_key"] + " | " + " | ".join(str(row["by_family"][f]["ACCESSIBLE"]) for f in FAMILIES) + " |")
    lines += ["", "## v16→v18 응답 비율 비교", "", "각 칸은 v16 RETRIEVE_NOW → v18 ACCESSIBLE 횟수(/25), 괄호는 비율 차이(%p)입니다. 서로 다른 질문의 응답 분포를 비교합니다.", "",
              "| 조건 | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
    for condition in keys:
        cells = [comparison_map[(condition, f)] for f in FAMILIES]
        lines.append("| " + condition + " | " + " | ".join(f"{r['v16_retrieve_now']}→{r['v18_accessible']} ({r['positive_rate_delta_pp']:+.0f})" for r in cells) + " |")
    lines += ["", "## 장면별 ACCESSIBLE 횟수 /5", "", "| 장면 | target | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
    scene_map = {(row["scene_id"], row["condition_key"]): row for row in scenes}
    for scene in sorted({row["scene_id"] for row in scenes}):
        rows = [scene_map[(scene, condition)] for condition in keys]
        lines.append(f"| {scene} | {rows[0]['target_object_id']} | " + " | ".join(str(row["ACCESSIBLE"]) for row in rows) + " |")
    if plots:
        lines += ["", "## 시각화", ""]
        for plot in plots:
            lines += [f"![{plot['caption']}]({plot['png']})", "", plot["caption"], ""]
    elif summary.get("plot_note"):
        lines += ["", "시각화 생성 상태: " + summary["plot_note"], ""]
    lines += ["", "## 해석 범위", "",
              "이 프롬프트에는 접근 주체의 부피, 접근 방향/궤적, 그리퍼 크기가 지정돼 있지 않습니다. 모델의 ACCESSIBLE 응답은 해당 이미지와 Geometry를 바탕으로 한 접근 경로 판단이며, 파지·타깃 제거 가능성이나 실제 무충돌 경로의 검증은 아닙니다.", "",
              "v16과 비교할 때 질문을 회수에서 접근으로 바꾸면서 문장 길이, 판단 보조 문구, 출력 라벨도 함께 바뀌었습니다. 관찰된 차이를 특정 문구 하나의 효과로 분리할 수 없습니다. 5회 seed 반복은 같은 장면의 반복 측정이므로 독립적인 1,500장면 실험으로 해석하지 않습니다.", "",
              "원자료: `raw_scene_seed_decisions.csv`, `scene_results.json`, `paired_vs_c0.csv`, `paired_vs_v16.csv`. 조건·family 요약: `condition_results.csv`, `family_results.csv`, `v16_v18_family_comparison.csv`. 각 행의 실제 seed와 결정을 원자료에서 확인할 수 있습니다.", ""]
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(dest=DEST, baseline=BASELINE, make_plots=True):
    dest, baseline = Path(dest), Path(baseline)
    config = read_json(dest / "config/experiment_config.json")
    schedule = read_jsonl(dest / "config/run_table.jsonl")
    records = read_jsonl(dest / "logs/runs.jsonl")
    previous = read_jsonl(baseline / "logs/runs.jsonl")
    previous_map = validate(config, schedule, records, previous)
    keys = [item["condition"] + ("_" + item["resolution"] if item.get("resolution") else "") for item in config["conditions"]]
    names = {key: item["name"] for key, item in zip(keys, config["conditions"])}
    assert len(keys) == len(set(keys)) == 12 and keys[0] == "C0"
    by_condition, by_scene = defaultdict(list), defaultdict(list)
    for row in records:
        by_condition[row["condition_key"]].append(row)
        by_scene[(row["scene_id"], row["condition_key"])].append(row)
    c0 = {(row["scene_id"], row["seed"]): row for row in by_condition["C0"]}
    condition_results, scene_results, family_results = [], [], []
    paired_c0, paired_previous, raw_decisions, comparisons, seed_results = [], [], [], [], []
    for condition in keys:
        rows = sorted(by_condition[condition], key=lambda row: (row["scene_id"], row["seed"]))
        unanimous = 0
        for scene_id in sorted({row["scene_id"] for row in rows}):
            scene_rows = sorted(by_scene[(scene_id, condition)], key=lambda row: row["seed"])
            tally = distribution(scene_rows)
            is_unanimous = max(tally["ACCESSIBLE"], tally["BLOCKED"]) == len(config["seeds"])
            unanimous += is_unanimous
            scene_results.append({"scene_id": scene_id, "scene_family": scene_rows[0]["scene_family"],
                                  "condition_key": condition, "target_object_id": scene_rows[0]["target_object_id"],
                                  **tally, "unanimous": is_unanimous,
                                  "responses": [{"seed": row["seed"], "decision": decision(row)} for row in scene_rows]})
        condition_pairs = []
        condition_previous_pairs = []
        for row in rows:
            base = c0[(row["scene_id"], row["seed"])]
            old = previous_map[key(row)]
            core = {field: row[field] for field in ("scene_id", "scene_family", "condition_key", "seed")}
            pair = {**core, "c0_decision": decision(base), "condition_decision": decision(row)}
            paired_c0.append(pair)
            condition_pairs.append(pair)
            old_decision = decision(old, V16_DECISIONS)
            previous_pair = {**core, "v16_decision": old_decision,
                             "v16_mapped_decision": V16_MAPPING.get(old_decision, "INVALID"),
                             "v18_decision": decision(row)}
            paired_previous.append(previous_pair)
            condition_previous_pairs.append(previous_pair)
            raw_decisions.append({**core, "run_id": row["run_id"], "target_object_id": row["target_object_id"],
                                  "decision": decision(row), "schema_valid": row["schema_valid"],
                                  "raw_response": row.get("raw_response"), "v16_decision": old_decision})
        families = {}
        for family in FAMILIES:
            family_rows = [row for row in rows if row["scene_family"] == family]
            families[family] = distribution(family_rows)
            family_results.append({"condition_key": condition, "condition_name": names[condition], "scene_family": family, **families[family]})
            prior_rows = [previous_map[key(row)] for row in family_rows]
            prior_dist = distribution(prior_rows, V16_DECISIONS)
            pairs = [row for row in condition_previous_pairs if row["scene_family"] == family]
            comparisons.append({"condition_key": condition, "scene_family": family, "n": len(family_rows),
                                "v16_retrieve_now": prior_dist["RETRIEVE_NOW"], "v16_rearrange_first": prior_dist["REARRANGE_FIRST"],
                                "v16_invalid": prior_dist["INVALID"], "v18_accessible": families[family]["ACCESSIBLE"],
                                "v18_blocked": families[family]["BLOCKED"], "v18_invalid": families[family]["INVALID"],
                                "positive_rate_delta_pp": 100 * (families[family]["positive_rate"] - prior_dist["positive_rate"]),
                                **transition_counts(pairs, "v16_mapped_decision", "v18_decision")})
        prior_dist = distribution([previous_map[key(row)] for row in rows], V16_DECISIONS)
        tally = distribution(rows)
        condition_results.append({"condition_key": condition, "condition_name": names[condition], **tally,
                                  "unanimous_scenes": unanimous, "by_family": families, **transition_counts(condition_pairs),
                                  "v16_RETRIEVE_NOW": prior_dist["RETRIEVE_NOW"],
                                  "v16_to_v18_positive_delta_pp": 100 * (tally["positive_rate"] - prior_dist["positive_rate"]),
                                  "paired_vs_v16": transition_counts(condition_previous_pairs, "v16_mapped_decision", "v18_decision")})
        for seed in config["seeds"]:
            seed_results.append({"condition_key": condition, "seed": seed, **distribution([row for row in rows if row["seed"] == seed])})
    result_dir = dest / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = dest / "logs/attempts.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    summary = {
        "experiment_version": config["experiment_version"], "model_id": config["model_id"],
        "completed_calls": len(records), "scene_count": 25, "condition_count": len(keys), "seed_count": len(config["seeds"]),
        "schema_valid_calls": sum(decision(row) in DECISIONS for row in records),
        "infrastructure_errors": sum(row.get("status") == "infrastructure_error" for row in attempts),
        "access_ground_truth_available": False, "retrieval_execution_gt_available": False, "accuracy": None,
        "evaluation": "response_distributions_and_paired_semantic_task_change_no_family_ground_truth",
        "baseline": str(baseline), "v16_label_correspondence_for_description_only": V16_MAPPING,
        "condition_results": condition_results,
        "total_generation_seconds": round(sum(row.get("latency_seconds", 0) for row in records), 3),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    integrity = {"status": "pass", "scheduled_calls": len(schedule), "completed_calls": len(records),
                 "unique_run_ids": len({row["run_id"] for row in records}), "all_scene_condition_seed_sets_complete": True,
                 "same_images_geometry_targets_model_generation_as_v16": True,
                 "condition_counts": dict(Counter(row["condition_key"] for row in records)),
                 "finish_reasons": dict(Counter(row.get("finish_reason", "unknown") for row in records)),
                 "logs_sha256": hashlib.sha256((dest / "logs/runs.jsonl").read_bytes()).hexdigest(),
                 "v16_logs_sha256": hashlib.sha256((baseline / "logs/runs.jsonl").read_bytes()).hexdigest(),
                 "run_table_sha256": hashlib.sha256((dest / "config/run_table.jsonl").read_bytes()).hexdigest()}
    for name, value in (("scene_results.json", scene_results), ("paired_vs_c0.json", paired_c0),
                        ("paired_vs_v16.json", paired_previous), ("integrity_report.json", integrity)):
        write_json(result_dir / name, value)
    fields = ["condition_key", "condition_name", "n", "ACCESSIBLE", "BLOCKED", "INVALID", "positive_rate", "unanimous_scenes",
              "blocked_to_accessible", "accessible_to_blocked", "changed_valid_pairs", "net_accessible_change", "valid_pairs", "invalid_pairs",
              "v16_RETRIEVE_NOW", "v16_to_v18_positive_delta_pp"]
    write_csv(result_dir / "condition_results.csv", condition_results, fields)
    write_csv(result_dir / "family_results.csv", family_results, ["condition_key", "condition_name", "scene_family", "n", "ACCESSIBLE", "BLOCKED", "INVALID", "positive_rate"])
    write_csv(result_dir / "seed_results.csv", seed_results, ["condition_key", "seed", "n", "ACCESSIBLE", "BLOCKED", "INVALID", "positive_rate"])
    for filename, rows in (("raw_scene_seed_decisions.csv", raw_decisions), ("paired_vs_c0.csv", paired_c0),
                           ("paired_vs_v16.csv", paired_previous), ("v16_v18_family_comparison.csv", comparisons)):
        write_csv(result_dir / filename, rows, list(rows[0]))
    scene_map = {(row["scene_id"], row["condition_key"]): row for row in scene_results}
    wide_scenes = [{"scene_id": scene, "scene_family": scene_map[(scene, "C0")]["scene_family"],
                    "target_object_id": scene_map[(scene, "C0")]["target_object_id"],
                    **{condition: scene_map[(scene, condition)]["ACCESSIBLE"] for condition in keys}}
                   for scene in sorted({row["scene_id"] for row in scene_results})]
    write_csv(result_dir / "accessible_counts_by_scene.csv", wide_scenes, ["scene_id", "scene_family", "target_object_id"] + keys)
    plots, plot_note = create_plots(result_dir, condition_results, scene_results, comparisons, keys) if make_plots else ([], "Plots disabled with --no-plots.")
    summary.update({"plots": plots, "plot_note": plot_note})
    write_json(result_dir / "summary.json", summary)
    write_report(result_dir, config, summary, scene_results, comparisons, plots)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = analyze(args.dest, args.baseline, make_plots=not args.no_plots)
    print(json.dumps({"report": str(args.dest / "results/report.md"),
                      "completed_calls": summary["completed_calls"], "schema_valid_calls": summary["schema_valid_calls"],
                      "plots": len(summary["plots"]), "plot_note": summary["plot_note"],
                      "conditions": [{key: row[key] for key in ("condition_key", "ACCESSIBLE", "BLOCKED", "unanimous_scenes")} for row in summary["condition_results"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
