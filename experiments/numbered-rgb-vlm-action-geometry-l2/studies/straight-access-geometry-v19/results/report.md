# L2 정면 직접 접근 × Geometry 단일정보 — v19 결과

25장면 × 12조건 × 5 seed = 1,500회. v18과 이미지·Geometry·모델·생성 설정·출력 schema·seed·실행 순서를 동일하게 유지하고 합의한 system/user 프롬프트를 변경했다.

숫자는 ACCESSIBLE/BLOCKED 응답 분포다. 실제 정면 접근 GT가 없으므로 family 이름을 정답으로 사용하거나 정확도로 환산하지 않는다. system 문구, straight 조건, proximity 설명, 출력 지시가 함께 바뀌었으므로 straight 한 단어의 효과를 분리한 실험은 아니다.

완료 1500/1500, JSON 유효 1500/1500, 인프라 오류 0, 모든 첫 attempt: True. 생성 시간 합계 111.431초.

v18 대비 전체 전환: ACCESSIBLE→BLOCKED 0회, BLOCKED→ACCESSIBLE 581회.

## 실제 사용 프롬프트

System:
```text
Assess direct access to the target from the shelf opening.
Numbered badges identify objects.
Return only a JSON object.
```

User:
```text
Target object ID: {target_object_id}
Geometry information:
{geometry_json}

Can the robot approach the target straight from the shelf opening
without moving or contacting other objects?

A nearby object blocks access only if it obstructs this approach.

Return {"decision": "ACCESSIBLE"} or {"decision": "BLOCKED"}.
```

## 조건·family별 ACCESSIBLE 횟수: v18 → v19 (/25)

| 조건 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | 5회 만장일치 장면 /25 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 25 → 25 | 25 → 25 | 0 → 20 | 0 → 20 | 0 → 11 | 24 |
| C1 | 25 → 25 | 20 → 25 | 1 → 20 | 6 → 25 | 4 → 25 | 25 |
| C2_D4 | 25 → 25 | 16 → 25 | 0 → 20 | 12 → 25 | 0 → 18 | 23 |
| C2_D8 | 25 → 25 | 13 → 25 | 1 → 20 | 6 → 25 | 0 → 25 | 25 |
| C3_D4 | 25 → 25 | 25 → 25 | 12 → 25 | 11 → 25 | 7 → 25 | 25 |
| C3_D8 | 25 → 25 | 25 → 25 | 12 → 25 | 5 → 25 | 14 → 25 | 25 |
| C4_D4 | 25 → 25 | 25 → 25 | 12 → 25 | 6 → 20 | 3 → 20 | 23 |
| C4_D8 | 25 → 25 | 25 → 25 | 20 → 25 | 16 → 25 | 11 → 21 | 24 |
| C5_D4 | 25 → 25 | 25 → 25 | 15 → 21 | 5 → 25 | 2 → 25 | 24 |
| C5_D8 | 25 → 25 | 25 → 25 | 11 → 25 | 17 → 25 | 10 → 25 | 25 |
| C6_D4 | 25 → 25 | 25 → 25 | 11 → 25 | 15 → 25 | 5 → 25 | 25 |
| C6_D8 | 25 → 25 | 25 → 25 | 19 → 25 | 9 → 25 | 3 → 25 | 25 |

## 조건별 전체 응답 및 v18 대응 변화

| 조건 | ACCESSIBLE /125 | BLOCKED /125 | A→B | B→A |
|---|---:|---:|---:|---:|
| C0 RGB only | 101 | 24 | 0 | 51 |
| C1 R Numeric | 120 | 5 | 0 | 64 |
| C2_D4 R Qualitative D4 | 113 | 12 | 0 | 60 |
| C2_D8 R Qualitative D8 | 120 | 5 | 0 | 75 |
| C3_D4 F Numeric D4 | 125 | 0 | 0 | 45 |
| C3_D8 F Numeric D8 | 125 | 0 | 0 | 44 |
| C4_D4 F Qualitative D4 | 115 | 10 | 0 | 44 |
| C4_D8 F Qualitative D8 | 121 | 4 | 0 | 24 |
| C5_D4 M Numeric D4 | 121 | 4 | 0 | 49 |
| C5_D8 M Numeric D8 | 125 | 0 | 0 | 37 |
| C6_D4 M Qualitative D4 | 125 | 0 | 0 | 44 |
| C6_D8 M Qualitative D8 | 125 | 0 | 0 | 44 |

## 장면별 ACCESSIBLE 횟수: v18 → v19 (/5)

| 장면 | 타깃 | C0 | C1 | C2_D4 | C2_D8 | C3_D4 | C3_D8 | C4_D4 | C4_D8 | C5_D4 | C5_D8 | C6_D4 | C6_D8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_fc_blocked_v01 | 64 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_blocked_v02 | 53 | 5→5 | 5→5 | 1→5 | 2→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_blocked_v03 | 12 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_blocked_v04 | 89 | 5→5 | 4→5 | 5→5 | 1→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_blocked_v05 | 59 | 5→5 | 1→5 | 0→5 | 0→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_clear_v01 | 53 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_clear_v02 | 29 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_clear_v03 | 95 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_clear_v04 | 61 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_fc_clear_v05 | 62 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_lift_and_relocate_v01 | 59 | 0→1 | 0→5 | 0→5 | 0→5 | 1→5 | 3→5 | 0→5 | 1→5 | 1→5 | 4→5 | 1→5 | 2→5 |
| scene_lift_and_relocate_v02 | 70 | 0→5 | 4→5 | 0→5 | 0→5 | 5→5 | 5→5 | 1→5 | 5→5 | 1→5 | 5→5 | 4→5 | 1→5 |
| scene_lift_and_relocate_v03 | 12 | 0→5 | 0→5 | 0→5 | 0→5 | 1→5 | 5→5 | 2→5 | 5→5 | 0→5 | 0→5 | 0→5 | 0→5 |
| scene_lift_and_relocate_v04 | 68 | 0→0 | 0→5 | 0→2 | 0→5 | 0→5 | 0→5 | 0→0 | 0→1 | 0→5 | 0→5 | 0→5 | 0→5 |
| scene_lift_and_relocate_v05 | 43 | 0→0 | 0→5 | 0→1 | 0→5 | 0→5 | 1→5 | 0→5 | 0→5 | 0→5 | 1→5 | 0→5 | 0→5 |
| scene_rotate_v01 | 12 | 0→5 | 0→5 | 0→5 | 0→5 | 2→5 | 0→5 | 0→5 | 4→5 | 1→5 | 4→5 | 4→5 | 0→5 |
| scene_rotate_v02 | 38 | 0→5 | 4→5 | 5→5 | 0→5 | 0→5 | 0→5 | 0→1 | 0→5 | 0→5 | 4→5 | 2→5 | 0→5 |
| scene_rotate_v03 | 83 | 0→5 | 1→5 | 5→5 | 1→5 | 5→5 | 1→5 | 0→4 | 5→5 | 2→5 | 5→5 | 5→5 | 5→5 |
| scene_rotate_v04 | 47 | 0→5 | 1→5 | 2→5 | 5→5 | 2→5 | 2→5 | 3→5 | 5→5 | 2→5 | 4→5 | 4→5 | 4→5 |
| scene_rotate_v05 | 44 | 0→0 | 0→5 | 0→5 | 0→5 | 2→5 | 2→5 | 3→5 | 2→5 | 0→5 | 0→5 | 0→5 | 0→5 |
| scene_translate_v01 | 67 | 0→5 | 0→0 | 0→0 | 0→0 | 0→5 | 0→5 | 0→5 | 0→5 | 0→1 | 0→5 | 0→5 | 4→5 |
| scene_translate_v02 | 45 | 0→5 | 0→5 | 0→5 | 0→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 4→5 | 5→5 | 5→5 |
| scene_translate_v03 | 18 | 0→5 | 1→5 | 0→5 | 0→5 | 2→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| scene_translate_v04 | 13 | 0→0 | 0→5 | 0→5 | 0→5 | 0→5 | 0→5 | 0→5 | 5→5 | 0→5 | 0→5 | 0→5 | 0→5 |
| scene_translate_v05 | 54 | 0→5 | 0→5 | 0→5 | 1→5 | 5→5 | 2→5 | 2→5 | 5→5 | 5→5 | 2→5 | 1→5 | 5→5 |

## 그림

![v19 결과](plots/01_accessible_heatmap.png)

![v19 결과](plots/02_change_vs_v18.png)

![v19 결과](plots/03_v18_v19_focused_comparison.png)

![v19 결과](plots/04_fc_blocked_scene_comparison.png)

## 재현

```bash
.qwen3-vl/venv/bin/python scripts/run_l2_straight_access_geometry_v19.py
uv run --no-project --python 3.13 --with matplotlib python scripts/analyze_l2_straight_access_geometry_v19.py
```

추론은 완료 run을 건너뛰며 원시 응답은 logs/runs.jsonl에 보존된다. paired_vs_v18.csv에서 모든 scene·condition·seed별 변경을 확인할 수 있다.
