#!/usr/bin/env python3
"""Image-backed comparison of all 25 scenes, four conditions, and 500 calls."""
import base64
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l2_v16_prompt_replay_v21"
OUT = EXP / "results/four_condition_comparison"
KEYS = ("C0", "C1", "C2_D4", "C2_D8")
NAMES = ("이미지 단독", "R 수치형", "R 정성형 4방향", "R 정성형 8방향")
FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")
SHORT = {"RETRIEVE_NOW": "회수", "REARRANGE_FIRST": "재배치"}
# Descriptions of visible layouts, not model reasoning or execution ground truth.
NOTES = {
    "scene_fc_clear_v01": "청록색 타깃 53 양옆에 병 62와 캔 98이 떨어져 보이며, 타깃 전면은 드러나 있습니다.",
    "scene_fc_clear_v02": "청록색 타깃 29 왼쪽의 병 10과 오른쪽의 컵 66 사이에 영상상 간격이 보입니다.",
    "scene_fc_clear_v03": "청록색 타깃 95 오른쪽에 파란 컵 20이 있으며, 타깃을 가리는 물체는 보이지 않습니다.",
    "scene_fc_clear_v04": "청록색 타깃 61 왼쪽의 검은 컵 38, 오른쪽의 보라색 컵 13과 영상상 간격이 보입니다.",
    "scene_fc_clear_v05": "타깃 62 오른쪽에 머그 82가 있습니다. 손잡이는 타깃 쪽을 향하지만 영상상 간격이 보입니다.",
    "scene_fc_blocked_v01": "타깃 64 오른쪽에 캔 40이 가깝게 놓여 있습니다. 타깃 전면은 대부분 드러나 있습니다.",
    "scene_fc_blocked_v02": "타깃 53 왼쪽에 병 49가 가깝게 놓여 있습니다. 단순 가림보다는 옆쪽 간격을 살펴볼 장면입니다.",
    "scene_fc_blocked_v03": "타깃 12 오른쪽에 파란 컵 33이 가까이 있습니다. 타깃 전면이 드러나 있어 가림 여부만으로 판단하기 어렵습니다.",
    "scene_fc_blocked_v04": "타깃 89 왼쪽에 검은 컵 19가 가까이 놓여 있습니다. 타깃 윗부분과 전면은 보입니다.",
    "scene_fc_blocked_v05": "타깃 59 바로 오른쪽의 머그 26 손잡이가 타깃 쪽으로 뻗어 있습니다. 네 조건 모두 재배치로 응답한 FC_BLOCKED 장면입니다.",
    "scene_translate_v01": "큰 상자 35 뒤로 타깃 67의 오른쪽 일부가 보입니다.",
    "scene_translate_v02": "상자 95 왼쪽으로 타깃 45가 보입니다. 상자는 타깃의 오른쪽 일부를 영상에서 가리고 있습니다.",
    "scene_translate_v03": "상자 45 오른쪽으로 타깃 18의 일부가 보입니다. 머그 39와 캔 41은 더 오른쪽에 있습니다.",
    "scene_translate_v04": "상자 84 왼쪽으로 타깃 13의 좁은 부분이 보입니다.",
    "scene_translate_v05": "상자 79 왼쪽으로 타깃 54의 일부가 보입니다. 나머지 물체들은 화면 오른쪽에 있습니다.",
    "scene_rotate_v01": "선반 오른쪽 상자 95의 왼쪽으로 타깃 12가 보이고, 더 왼쪽에는 병과 머그가 모여 있습니다.",
    "scene_rotate_v02": "선반 왼쪽의 상자 50 오른쪽으로 타깃 38이 보입니다. 더 오른쪽에는 병 29와 컵들이 있습니다.",
    "scene_rotate_v03": "선반 오른쪽의 비스듬한 상자 19 왼쪽으로 타깃 83이 보입니다. 타깃 왼쪽에는 병 23 등이 있습니다.",
    "scene_rotate_v04": "선반 왼쪽의 비스듬한 상자 95 오른쪽으로 타깃 47의 일부가 보입니다.",
    "scene_rotate_v05": "상자 82 왼쪽으로 타깃 44가 좁게 보입니다. 주변에는 병 두 개와 컵·머그가 놓여 있습니다.",
    "scene_lift_and_relocate_v01": "상자 64 오른쪽으로 타깃 59가 보입니다. 상자 왼쪽에는 병과 머그가 모여 있습니다.",
    "scene_lift_and_relocate_v02": "상자 14 왼쪽으로 타깃 70이 보이며, 양옆에 컵 77과 머그·병들이 놓여 있습니다.",
    "scene_lift_and_relocate_v03": "상자 35 오른쪽으로 타깃 12가 보입니다. 왼쪽에는 머그들이, 오른쪽에는 병 92와 캔 50이 있습니다.",
    "scene_lift_and_relocate_v04": "상자 73 왼쪽으로 타깃 68이 좁게 보이고, 타깃 왼쪽에 병 30이 있습니다. 완전 가림 물체 ID 16의 record는 기존 입력 정책에 따라 제외돼 있습니다.",
    "scene_lift_and_relocate_v05": "상자 15 오른쪽으로 타깃 43이 보입니다. 그 오른쪽에는 컵 76과 머그 24가 겹쳐 보입니다.",
}
INSIGHTS = {
    "scene_fc_clear_v01": "C1만 재배치 3/5로 다수 판단까지 바뀝니다. C0·D4·D8은 모두 회수 5/5. 수치 정보를 추가한 뒤 원본 기대와의 일치가 낮아진 사례입니다.",
    "scene_fc_clear_v04": "C1에서만 재배치가 1/5 나옵니다. 나머지 세 조건은 회수 5/5입니다.",
    "scene_fc_clear_v05": "C1과 D8에서 각각 재배치 1/5가 나옵니다. C0·D4는 회수 5/5. D8이 모든 FC_CLEAR 응답을 보존한 것은 아닙니다.",
    "scene_fc_blocked_v01": "C0·D4는 회수 5/5, C1은 회수 4/5지만 D8은 재배치 5/5입니다. 원본 설계 기준에서는 D8이 일치하고, 회수로 유도하려는 목표에서는 반대입니다.",
    "scene_fc_blocked_v02": "C0·C1·D4는 회수 4/5, D8은 회수 2/5입니다. D8도 재배치 5/5에는 도달하지 않았습니다.",
    "scene_fc_blocked_v03": "C0·D4·D8 모두 회수 5/5이고 C1만 재배치 2/5입니다. 원본 기준에서 D8이 끝까지 구분하지 못한 대표 장면이지만, 회수 유도 목표에서는 해석이 반대입니다.",
    "scene_fc_blocked_v04": "재배치 응답이 C0 0/5 → C1·D4 3/5 → D8 5/5로 증가합니다. 원본 기준의 개선과 회수 유도 목표를 혼동하지 않아야 합니다.",
    "scene_fc_blocked_v05": "네 조건 모두 재배치 5/5입니다. 원본 기준에서는 전부 일치하지만, FC_BLOCKED도 회수하도록 유도하려는 목표에서는 전부 반대 응답입니다.",
    "scene_translate_v02": "C0은 회수 2/재배치 3, C1은 회수 5/재배치 0, D4·D8은 회수 0/재배치 5입니다. TRANSLATE에서 조건 차이를 만든 유일한 장면입니다.",
    "scene_rotate_v03": "C1에서만 바로 회수가 1/5 나옵니다. C0·D4·D8은 재배치 5/5입니다.",
    "scene_rotate_v04": "C0에서 바로 회수가 1/5 나오지만, C1·D4·D8에서는 재배치 5/5입니다.",
}

CSS = """
*{box-sizing:border-box}body{margin:0;background:#edf1f6;color:#18253a;font:17px/1.55 'Noto Sans CJK KR',sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:32px;margin:0 0 12px}h2{font-size:27px;margin:0 0 10px}h3{font-size:21px;margin:0 0 8px}
p{margin:8px 0}.intro,.summary,.family{background:white;border-radius:16px;padding:24px;margin:0 0 22px}a{color:#1958ad}nav{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}
.warning{background:#fff3d9;border-left:5px solid #d49d31;padding:12px 16px;border-radius:6px}.muted{color:#55647a}.small{font-size:14px}
.card{padding:18px 0;border-top:1px solid #dce2eb;break-inside:avoid}.layout{display:grid;grid-template-columns:480px 1fr;gap:24px;align-items:start}
figure{margin:0}figure img{width:100%;height:auto;display:block;border-radius:10px}figcaption{font-size:14px;color:#57677d;margin-top:7px}
table{width:100%;border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:center;border:1px solid #dbe3ed}th{background:#eef3f9;font-size:15px}
td .counts{font-size:21px;font-weight:700;white-space:nowrap}.good{background:#e8f6ee;color:#155d36}.mixed{background:#fff3d9;color:#855400}.bad{background:#fdebea;color:#9a272b}
.tag{display:inline-block;padding:1px 9px;border-radius:5px;font-size:14px}.seeds{font-size:12px;color:#52637b;letter-spacing:1px}.explain{font-size:16px}.geometry{font-size:14px;background:#f5f7fb;padding:9px 12px;border-radius:7px;margin:10px 0}
details{margin:12px 0}summary{cursor:pointer;color:#225ba3}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f4f6fa;padding:12px}.seed-table{font-size:13px}.seed-table td{padding:5px}
@media(max-width:1000px){main{padding:12px}.layout{grid-template-columns:1fr}figure{max-width:650px}h1{font-size:26px}td .counts{font-size:18px}}
@media print{body{background:white}main{padding:0}.family{break-before:page}.card{break-inside:avoid}details,nav{display:none}}
"""


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = read(EXP / "config/frozen_scenes.json")
    all_records = [json.loads(line) for line in (EXP / "logs/runs.jsonl").read_text().splitlines() if line]
    rows = [r for r in all_records if r["condition_key"] in KEYS]
    assert len(rows) == 500 and len({(r['scene_id'],r['condition_key'],r['seed']) for r in rows}) == 500
    lookup = {}
    for scene in scenes:
        for k in KEYS:
            group = sorted([r for r in rows if r["scene_id"] == scene["scene_id"] and r["condition_key"] == k], key=lambda r:r["seed"])
            assert len(group) == 5 and {r['seed'] for r in group} == {28101,28102,28103,28104,28105}
            assert all(r['schema_valid'] and json.loads(r['raw_response']) == r['parsed_response'] for r in group)
            lookup[(scene["scene_id"],k)] = group
    stats, flat = [], []
    mismatch_union, always_agree = [], []
    for k in KEYS:
        match_counts = []
        for scene in scenes:
            expected = "RETRIEVE_NOW" if scene["scene_family"] == "FC_CLEAR" else "REARRANGE_FIRST"
            group = lookup[(scene["scene_id"],k)]
            matches = sum(r["parsed_response"]["decision"] == expected for r in group)
            match_counts.append(matches)
        stats.append({"condition":k,"matches":sum(match_counts),"all_five_match_scenes":match_counts.count(5),
                      "mixed_scenes":sum(0<n<5 for n in match_counts),"all_five_mismatch_scenes":match_counts.count(0)})
    overview_rows = ''.join(f"<tr><td>{k}</td><td>{n}</td><td>{s['matches']}/125</td><td>{s['all_five_match_scenes']}</td><td>{s['mixed_scenes']}</td><td>{s['all_five_mismatch_scenes']}</td></tr>" for k,n,s in zip(KEYS,NAMES,stats))
    intro = f"""<section class="intro"><h1>같은 이미지, 네 가지 입력 — 25장면 전체 비교</h1>
    <p>v16 원문 프롬프트 재실행 v21 · C0 / C1 / C2_D4 / C2_D8 · 총 500개 응답</p>
    <div class="warning"><b>숫자는 회수 / 재배치 응답 횟수입니다.</b> 각 칸 합계 5회. 색은 원본 노션 설계 기준과의 일치 정도입니다.<br>
    FC_CLEAR → 회수, 나머지 → 재배치가 원본 기준입니다. 실제 로봇 회수 성공·실패가 아닙니다.<br>
    FC_BLOCKED를 회수로 유도하려는 현재 목표는 원본 기준과 반대이므로 해당 카드에 두 기준을 함께 표시합니다.</div>
    <p class="muted small">초록: 5/5 일치 · 노랑: 1~4/5 일치 · 빨강: 0/5 일치. 모든 조건에서 동일한 원본 이미지를 사용했습니다. 이미지를 클릭하면 원본 크기로 볼 수 있습니다.</p>
    <nav>{''.join(f'<a href="#{f}">{f}</a>' for f in FAMILIES)}</nav></section>
    <section class="summary"><h2>조건별 전체 요약</h2><table><tr><th>조건</th><th>정보</th><th>원본 설계 일치</th><th>5회 모두 일치 장면</th><th>응답 혼재 장면</th><th>5회 모두 불일치 장면</th></tr>{overview_rows}</table>
    <p class="muted">C1은 TRANSLATE v02·FC_CLEAR 일부에서 C0보다 일치가 낮아집니다. D4는 FC_CLEAR와 TRANSLATE·ROTATE·LIFT를 모두 5/5 일치시키지만 FC_BLOCKED 네 장면에 불일치가 남습니다. D8의 잔여 불일치는 세 장면입니다.</p></section>"""
    sections, md, image_checks = [], ["# 25장면 × 네 조건 이미지 비교", "", "각 칸은 **회수 / 재배치** 횟수(합계 5회). 원본 노션 기준: FC_CLEAR는 회수, 나머지는 재배치. 실제 회수 성공률이 아니다. FC_BLOCKED 회수 유도 목표는 별도로 본다.", ""], []
    for family in FAMILIES:
        family_scenes = sorted([s for s in scenes if s["scene_family"] == family], key=lambda s:s["scene_id"])
        section = [f'<section class="family" id="{family}"><h2>{family} — 5장면 전체</h2>', '<p class="muted">각 칸: 회수 / 재배치 횟수. 색은 원본 설계 기준과의 일치 정도이며 물리적 정확도가 아닙니다.</p>']
        md += [f"## {family}", ""]
        for scene in family_scenes:
            sid, target = scene["scene_id"], scene["target_object_id"]
            path = Path(scene["image_path"])
            image_bytes = path.read_bytes()
            assert hashlib.sha256(image_bytes).hexdigest() == scene["image_sha256"]
            image_checks.append(sid)
            data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
            expected = "RETRIEVE_NOW" if family == "FC_CLEAR" else "REARRANGE_FIRST"
            cells, counts_text, match_by_cond, records_by_cond = [], [], [], {}
            for k in KEYS:
                group = lookup[(sid,k)]
                records_by_cond[k] = group
                decisions = [r['parsed_response']['decision'] for r in group]
                retrieve, rearrange = decisions.count('RETRIEVE_NOW'), decisions.count('REARRANGE_FIRST')
                matches = decisions.count(expected)
                match_by_cond.append(matches)
                cls = 'good' if matches == 5 else 'bad' if matches == 0 else 'mixed'
                seeds_text = ' '.join('회' if d == 'RETRIEVE_NOW' else '재' for d in decisions)
                cells.append(f'<td class="{cls}"><div class="counts">{retrieve} / {rearrange}</div><div class="small">설계 일치 {matches}/5</div><div class="seeds">{seeds_text}</div></td>')
                counts_text.append(f'{retrieve} / {rearrange}')
                flat.append({"scene_id":sid,"family":family,"target_id":target,"condition":k,"retrieve":retrieve,"rearrange":rearrange,"design_matches":matches,
                             "mismatch_seeds":','.join(str(r['seed']) for r in group if r['parsed_response']['decision'] != expected)})
            if min(match_by_cond) == 5:
                always_agree.append(sid)
            else:
                mismatch_union.append(sid)
            insight = INSIGHTS.get(sid, f"네 조건 모두 {SHORT[expected]} 5/5입니다. 이 장면에서는 Geometry 추가에 따른 응답 차이가 없습니다.")
            assert sid in NOTES
            numeric = records_by_cond['C1'][0]['geometry_json_raw']
            nearest = min(numeric['relations'], key=lambda r:r['surface_gap_cm'])
            neighbor = nearest['object_id']
            d4 = next(r for r in records_by_cond['C2_D4'][0]['geometry_json_raw']['relations'] if r['object_id']==neighbor)
            d8 = next(r for r in records_by_cond['C2_D8'][0]['geometry_json_raw']['relations'] if r['object_id']==neighbor)
            geo_text = f"가장 가까운 물체 ID {neighbor}: 표면 간격 {nearest['surface_gap_cm']} cm, 중심 방위 {nearest['center_bearing_deg']}°. D4 점유 {', '.join(d4['occupied_target_sides'])}; D8 점유 {', '.join(d8['occupied_target_sides'])}. clearance={d8['clearance_zone']}."
            alternative = ''
            if family == 'FC_BLOCKED':
                alternative = '<p class="small"><b>회수 유도 목표로 보면:</b> ' + ' · '.join(f'{k} 회수 {counts_text[i].split(" / ")[0]}/5' for i,k in enumerate(KEYS)) + '. 이 수치는 실제 회수 가능 정답이 아닙니다.</p>'
            seed_rows = []
            for i, seed in enumerate((28101,28102,28103,28104,28105)):
                seed_rows.append(f'<tr><td>{seed}</td>' + ''.join(f'<td>{SHORT[records_by_cond[k][i]["parsed_response"]["decision"]]}</td>' for k in KEYS) + '</tr>')
            details = '<details><summary>seed별 응답과 실제 Geometry JSON 보기</summary><table class="seed-table"><tr><th>seed</th>' + ''.join(f'<th>{k}</th>' for k in KEYS) + '</tr>' + ''.join(seed_rows) + '</table>'
            for k in KEYS[1:]:
                details += f'<h4>{k}</h4><pre>{esc(json.dumps(records_by_cond[k][0]["geometry_json_raw"],ensure_ascii=False,indent=2))}</pre>'
            details += '</details>'
            section.append(f'''<article class="card" id="{sid}"><h3>{sid.replace('scene_','')} · 타깃 ID {target}</h3><div class="layout">
            <figure><a href="{path.as_uri()}" target="_blank"><img src="{data_url}" alt="{sid} 타깃 {target}" width="1280" height="960"></a><figcaption>실제 추론 원본 · 1280×960 · 모든 조건 동일 이미지</figcaption></figure>
            <div><p class="small"><b>원본 설계 기대: {SHORT[expected]}</b> · seed 순서 28101→28105</p>
            <table><tr>{''.join(f'<th>{k}<br>{n}</th>' for k,n in zip(KEYS,NAMES))}</tr><tr>{''.join(cells)}</tr></table>
            <p class="explain"><b>이미지 관찰:</b> {esc(NOTES[sid])}</p><p class="explain"><b>결과 비교:</b> {esc(insight)}</p>
            {alternative}<div class="geometry">{esc(geo_text)}</div></div></div>{details}</article>''')
            md += [f"### {sid} — 타깃 {target}", "", f"![{sid}]({path})", "", f"이미지 관찰: {NOTES[sid]}", "",
                   "| C0 | C1 | C2_D4 | C2_D8 |", "|---:|---:|---:|---:|", "| " + " | ".join(counts_text) + " |", "",
                   f"원본 설계 기대: {SHORT[expected]}. 조건별 일치 횟수: " + ' / '.join(str(n) for n in match_by_cond) + ' (각각 /5).', "", insight, "", geo_text, ""]
            if alternative:
                md += ['FC_BLOCKED의 회수 유도 목표는 위 설계 기준과 반대이며, 각 칸 앞의 회수 횟수를 따로 확인해야 한다.', '']
        section.append('</section>')
        section_html = ''.join(section)
        sections.append(section_html)
        family_doc = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{family} 네 조건 비교</title><style>{CSS}</style></head><body><main>{section_html}</main></body></html>'
        (OUT / f'{family.lower()}.html').write_text(family_doc, encoding='utf-8')
    footer = '<section class="intro"><h2>해석 범위</h2><p>입력 이미지와 측정값은 관찰·기록이고, 모델이 그 정보를 실제로 어떻게 사용했는지는 설명 응답을 수집하지 않아 알 수 없습니다. OVERLAP은 표면 간격 ≤5 cm의 범주이며 물체 충돌이나 그리퍼 간섭의 검증 결과가 아닙니다. 단일 RGB와 이 Geometry만으로 실제 회수 성공을 확정하지 않습니다.</p></section>'
    document = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v21 네 조건·25장면 전체 비교</title><style>{CSS}</style></head><body><main>{intro}{"".join(sections)}{footer}</main></body></html>'
    (OUT / 'index.html').write_text(document, encoding='utf-8')
    (OUT / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    with (OUT / 'scene_condition_comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (OUT / 'seed_responses.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['scene_id','condition','seed','decision'])
        writer.writerows([r['scene_id'],r['condition_key'],r['seed'],r['parsed_response']['decision']] for r in rows)
    assert len(image_checks) == len(set(image_checks)) == 25
    assert len(flat) == 100 and len(mismatch_union) + len(always_agree) == 25
    report = {"condition_summary":stats,"scenes_all_20_responses_match_original_design":always_agree,
              "scenes_with_any_original_design_mismatch":mismatch_union,"verified_original_images":25,"verified_calls":500,
              "image_bytes_unchanged":True,"model_rerun_performed":False}
    (OUT / 'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
