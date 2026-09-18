#!/usr/bin/env python3
"""Summarize straight-approach v19 and paired changes from v18, without GT."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import analyze_l2_approach_access_geometry_v18 as stats
import run_l2_straight_access_geometry_v19 as run

DEST, SOURCE = run.DEST, run.SOURCE
FAMILIES = stats.FAMILIES


def plots(out, keys, families, scenes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "font.size": 10, "axes.unicode_minus": False})
    directory = out / "plots"
    directory.mkdir(exist_ok=True)
    labels = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT"]
    lookup = {(r["condition_key"], r["scene_family"]): r for r in families}
    files = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(f"plots/{name}.png")

    for name, title, delta in (("01_accessible_heatmap", "v19 정면 직접 접근: ACCESSIBLE 응답", False),
                               ("02_change_vs_v18", "v18 → v19: 접근 가능 응답의 변화", True)):
        data = [[lookup[(k, f)]["delta_accessible"] if delta else lookup[(k, f)]["ACCESSIBLE"] for k in keys] for f in FAMILIES]
        fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
        mesh = ax.imshow(data, cmap="RdBu" if delta else "Blues", vmin=-25 if delta else 0, vmax=25, aspect="auto")
        ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right")
        ax.set_yticks(range(5), labels)
        ax.set_title(title + ("\n각 칸: ACCESSIBLE 횟수 차이 /25 (정확도 변화 아님)" if delta else "\n각 칸: 5장면 × 5개 seed 중 접근 가능 횟수 /25"), loc="left", pad=16)
        for i, line in enumerate(data):
            for j, value in enumerate(line):
                text = f"{value:+d}" if delta else f"{value}/25"
                ax.text(j, i, text, ha="center", va="center", color="white" if abs(value) >= 17 else "#172033")
        fig.colorbar(mesh, ax=ax, shrink=.85, label="응답 횟수 차이" if delta else "ACCESSIBLE 횟수")
        save(fig, name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True, layout="constrained")
    for ax, key, label in zip(axes, ("C0", "C2_D8", "C4_D8"), ("RGB only", "R 정성형 D8", "F 정성형 D8")):
        group = [lookup[(key, f)] for f in FAMILIES]
        for offset, field, color, version in ((-.2, "v18_accessible", "#9AA8B9", "v18 기존 접근 질문"),
                                              (.2, "ACCESSIBLE", "#168C82", "v19 정면 직접 접근")):
            values = [r[field] for r in group]
            bars = ax.bar([i + offset for i in range(5)], values, .38, color=color, label=version)
            ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xticks(range(5), labels, rotation=32, ha="right")
        ax.set_title(f"{key} · {label}", weight="bold")
        ax.set_ylim(0, 30)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ACCESSIBLE 횟수 /25")
    axes[1].legend(loc="upper center", fontsize=9)
    fig.suptitle("접근 질문을 바꿨을 때 같은 장면들의 응답 변화\n숫자는 접근 가능 응답 횟수이며 정확도가 아닙니다", fontsize=16)
    save(fig, "03_v18_v19_focused_comparison")

    scene_ids = sorted({r["scene_id"] for r in scenes if r["scene_family"] == "FC_BLOCKED"})
    lookup_scene = {(r["scene_id"], r["condition_key"]): r for r in scenes}
    values = [[lookup_scene[(scene, k)]["ACCESSIBLE"] for k in keys] for scene in scene_ids]
    fig, ax = plt.subplots(figsize=(13, 5.6), layout="constrained")
    mesh = ax.imshow(values, cmap="Blues", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right")
    ax.set_yticks(range(5), [s.replace("scene_", "") for s in scene_ids])
    ax.set_title("FC_BLOCKED 장면별 비교\n각 칸: v18 → v19 ACCESSIBLE 횟수 /5 · 색은 v19 횟수", loc="left", pad=16)
    for i, scene in enumerate(scene_ids):
        for j, k in enumerate(keys):
            r = lookup_scene[(scene, k)]
            ax.text(j, i, f"{r['v18_accessible']}→{r['ACCESSIBLE']}", ha="center", va="center", color="white" if r["ACCESSIBLE"] >= 3 else "#172033")
    fig.colorbar(mesh, ax=ax, shrink=.8, label="v19 ACCESSIBLE /5")
    save(fig, "04_fc_blocked_scene_comparison")
    return files


def main():
    config, schedule = run.prepare()
    records = run.engine.rows(DEST / "logs/runs.jsonl")
    prior = run.engine.rows(SOURCE / "logs/runs.jsonl")
    old = stats.validate(config, schedule, records, prior)
    assert len(records) == 1500 and all(r["schema_valid"] for r in records)
    assert all(json.loads(r["raw_response"]) == r["parsed_response"] and set(r["parsed_response"]) == {"decision"} for r in records)
    assert [stats.key(r) for r in records] == [stats.key(r) for r in schedule] == [stats.key(r) for r in prior]
    keys = [c["condition"] + ("_" + c["resolution"] if c["resolution"] else "") for c in config["conditions"]]
    names = {k: c["name"] for k, c in zip(keys, config["conditions"])}
    by_cond, by_scene = defaultdict(list), defaultdict(list)
    for r in records:
        by_cond[r["condition_key"]].append(r)
        by_scene[(r["scene_id"], r["condition_key"])].append(r)
    paired = [{"scene_id": r["scene_id"], "scene_family": r["scene_family"], "condition_key": r["condition_key"],
               "seed": r["seed"], "v18_decision": stats.decision(old[stats.key(r)]), "v19_decision": stats.decision(r)} for r in records]
    scenes, families, conditions = [], [], []
    for key in keys:
        group = by_cond[key]
        unanimous = 0
        for scene in sorted({r["scene_id"] for r in group}):
            local = sorted(by_scene[(scene, key)], key=lambda r: r["seed"])
            tally = stats.distribution(local)
            previous = stats.distribution([old[stats.key(r)] for r in local])
            same = max(tally["ACCESSIBLE"], tally["BLOCKED"]) == 5
            unanimous += same
            scenes.append({"scene_id": scene, "scene_family": local[0]["scene_family"], "condition_key": key,
                           "target_object_id": local[0]["target_object_id"], **tally, "unanimous": same,
                           "v18_accessible": previous["ACCESSIBLE"],
                           "responses": [{"seed": r["seed"], "decision": stats.decision(r)} for r in local]})
        family_map = {}
        for family in FAMILIES:
            local = [r for r in group if r["scene_family"] == family]
            tally = stats.distribution(local)
            previous = stats.distribution([old[stats.key(r)] for r in local])
            changes = stats.transition_counts([r for r in paired if r["condition_key"] == key and r["scene_family"] == family], "v18_decision", "v19_decision")
            value = {"condition_key": key, "condition_name": names[key], "scene_family": family,
                     **tally, "v18_accessible": previous["ACCESSIBLE"],
                     "delta_accessible": tally["ACCESSIBLE"] - previous["ACCESSIBLE"], **changes}
            families.append(value)
            family_map[family] = value
        changes = stats.transition_counts([r for r in paired if r["condition_key"] == key], "v18_decision", "v19_decision")
        conditions.append({"condition_key": key, "condition_name": names[key], **stats.distribution(group),
                           "unanimous_scenes": unanimous, "v18_accessible": sum(stats.decision(old[stats.key(r)]) == "ACCESSIBLE" for r in group),
                           **changes, "by_family": family_map})
    out = DEST / "results"
    out.mkdir(exist_ok=True)
    attempts = run.engine.rows(DEST / "logs/attempts.jsonl")
    summary = {"experiment_version": config["experiment_version"], "model_id": config["model_id"],
               "completed_calls": len(records), "schema_valid_calls": len(records),
               "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in attempts),
               "all_first_attempt": all(r["attempt"] == 1 for r in records),
               "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
               "total_generation_seconds": round(sum(r["latency_seconds"] for r in records), 3),
               "accuracy": None, "approach_execution_gt_available": False,
               "paired_vs_v18": stats.transition_counts(paired, "v18_decision", "v19_decision"),
               "condition_results": conditions, "finished_at": run.engine.now()}
    for filename, data in (("scene_results.json", scenes), ("family_results.json", families), ("paired_vs_v18.json", paired)):
        stats.write_json(out / filename, data)
    stats.write_csv(out / "paired_vs_v18.csv", paired, list(paired[0]))
    stats.write_csv(out / "family_results.csv", families, list(families[0]))
    stats.write_csv(out / "condition_results.csv", conditions, [k for k in conditions[0] if k != "by_family"])
    stats.write_csv(out / "scene_results.csv", scenes, [k for k in scenes[0] if k != "responses"])
    summary["plots"] = plots(out, keys, families, scenes)
    stats.write_json(out / "summary.json", summary)
    integrity = {"status": "pass", "same_v18_inputs_model_sampling_schema_order": True,
                 "unique_complete_scene_condition_seed_combinations": 1500,
                 "schema_valid_calls": 1500,
                 "logs_sha256": run.engine.digest(DEST / "logs/runs.jsonl"),
                 "v18_logs_sha256": run.engine.digest(SOURCE / "logs/runs.jsonl")}
    stats.write_json(out / "integrity_report.json", integrity)
    lines = ["# L2 정면 직접 접근 × Geometry 단일정보 — v19 결과", "",
             "25장면 × 12조건 × 5 seed = 1,500회. v18과 이미지·Geometry·모델·생성 설정·출력 schema·seed·실행 순서를 동일하게 유지하고 합의한 system/user 프롬프트를 변경했다.", "",
             "숫자는 ACCESSIBLE/BLOCKED 응답 분포다. 실제 정면 접근 GT가 없으므로 family 이름을 정답으로 사용하거나 정확도로 환산하지 않는다. system 문구, straight 조건, proximity 설명, 출력 지시가 함께 바뀌었으므로 straight 한 단어의 효과를 분리한 실험은 아니다.", "",
             f"완료 {len(records)}/1500, JSON 유효 1500/1500, 인프라 오류 {summary['infrastructure_errors']}, 모든 첫 attempt: {summary['all_first_attempt']}. 생성 시간 합계 {summary['total_generation_seconds']}초.", "",
             f"v18 대비 전체 전환: ACCESSIBLE→BLOCKED {summary['paired_vs_v18']['accessible_to_blocked']}회, BLOCKED→ACCESSIBLE {summary['paired_vs_v18']['blocked_to_accessible']}회.", "",
             "## 실제 사용 프롬프트", "", "System:", "```text", config["system_prompt"], "```", "",
             "User:", "```text", config["user_prompt_template"], "```", "",
             "## 조건·family별 ACCESSIBLE 횟수: v18 → v19 (/25)", "",
             "| 조건 | " + " | ".join(FAMILIES) + " | 5회 만장일치 장면 /25 |",
             "|---|" + "---:|" * 6]
    for row in conditions:
        lines.append("| " + row["condition_key"] + " | " + " | ".join(f"{row['by_family'][f]['v18_accessible']} → {row['by_family'][f]['ACCESSIBLE']}" for f in FAMILIES) + f" | {row['unanimous_scenes']} |")
    lines += ["", "## 조건별 전체 응답 및 v18 대응 변화", "", "| 조건 | ACCESSIBLE /125 | BLOCKED /125 | A→B | B→A |", "|---|---:|---:|---:|---:|"]
    for row in conditions:
        lines.append(f"| {row['condition_key']} {row['condition_name']} | {row['ACCESSIBLE']} | {row['BLOCKED']} | {row['accessible_to_blocked']} | {row['blocked_to_accessible']} |")
    lines += ["", "## 장면별 ACCESSIBLE 횟수: v18 → v19 (/5)", "", "| 장면 | 타깃 | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
    lookup = {(r["scene_id"], r["condition_key"]): r for r in scenes}
    for scene in sorted({r["scene_id"] for r in scenes}):
        lines.append(f"| {scene} | {lookup[(scene, 'C0')]['target_object_id']} | " + " | ".join(f"{lookup[(scene, k)]['v18_accessible']}→{lookup[(scene, k)]['ACCESSIBLE']}" for k in keys) + " |")
    lines += ["", "## 그림", ""]
    for file in summary["plots"]:
        lines += [f"![v19 결과]({file})", ""]
    lines += ["## 재현", "", "```bash", ".qwen3-vl/venv/bin/python scripts/run_l2_straight_access_geometry_v19.py",
              "uv run --no-project --python 3.13 --with matplotlib python scripts/analyze_l2_straight_access_geometry_v19.py", "```", "",
              "추론은 완료 run을 건너뛰며 원시 응답은 logs/runs.jsonl에 보존된다. paired_vs_v18.csv에서 모든 scene·condition·seed별 변경을 확인할 수 있다.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed": len(records), "paired": summary["paired_vs_v18"], "report": str(out / "report.md"),
                      "conditions": [{"key": r["condition_key"], "accessible": r["ACCESSIBLE"], "fc_blocked": r["by_family"]["FC_BLOCKED"]["ACCESSIBLE"]} for r in conditions]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
