# L2 앞쪽 물체의 회수 방해 × Geometry 단일정보 — v20 결과

25장면 × 12조건 × 5 seed = 1,500회. v19와 이미지·Geometry·모델·생성 설정·실행 환경·seed·순서를 유지했다. 질문/system/출력 label을 변경했다.

BLOCKED = 앞쪽 물체가 타깃을 꺼내는 것을 조금이라도 방해한다고 응답. CLEAR = 그런 방해가 없다고 응답.

표의 숫자는 BLOCKED 응답 횟수이며 정답 개수가 아니다. 앞쪽 회수 방해 GT는 별도로 없으며, family 이름을 정답으로 취급하지 않는다. CLEAR도 옆쪽 간섭·파지 공간 등을 포함한 전체 회수 가능성을 보증하지 않는다.

완료/JSON 유효: 1500/1500. 인프라 오류 0, 모두 첫 시도: True. 생성 시간 합계 93.332초.

전체 응답: BLOCKED 194회, CLEAR 1306회.

## 실제 프롬프트

System:
```text
Assess whether objects in front of the target obstruct its removal.
Numbered badges identify objects.
Return only a JSON object.
```

User:
```text
Target object ID: {target_object_id}
Geometry information:
{geometry_json}

Does any object in front of target {target_object_id}
obstruct pulling the target out of the shelf, even slightly?

Return {"decision":"BLOCKED"} or {"decision":"CLEAR"}.
```

## 전체 조건: BLOCKED 응답 횟수

각 family는 5장면 × 5 seed = 25회, 조건별 전체는 125회. LIFT는 LIFT_AND_RELOCATE의 줄임말.

| 조건 | 정보 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT | BLOCKED /125 | CLEAR /125 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | RGB only | 0 | 1 | 10 | 5 | 0 | 16 | 109 |
| C1 | R Numeric | 0 | 5 | 2 | 0 | 0 | 7 | 118 |
| C2_D4 | R Qualitative D4 | 0 | 15 | 17 | 3 | 17 | 52 | 73 |
| C2_D8 | R Qualitative D8 | 1 | 19 | 16 | 5 | 11 | 52 | 73 |
| C3_D4 | F Numeric D4 | 0 | 2 | 1 | 0 | 0 | 3 | 122 |
| C3_D8 | F Numeric D8 | 0 | 1 | 3 | 1 | 0 | 5 | 120 |
| C4_D4 | F Qualitative D4 | 0 | 0 | 0 | 3 | 0 | 3 | 122 |
| C4_D8 | F Qualitative D8 | 0 | 5 | 3 | 1 | 0 | 9 | 116 |
| C5_D4 | M Numeric D4 | 0 | 5 | 1 | 3 | 0 | 9 | 116 |
| C5_D8 | M Numeric D8 | 0 | 6 | 5 | 2 | 0 | 13 | 112 |
| C6_D4 | M Qualitative D4 | 0 | 6 | 4 | 1 | 0 | 11 | 114 |
| C6_D8 | M Qualitative D8 | 0 | 3 | 6 | 5 | 0 | 14 | 111 |

## v19 → v20 BLOCKED 응답 횟수 (/25)

v19는 접근 방해, v20는 앞쪽 물체의 회수 방해를 질문한다. 이름이 같은 BLOCKED라도 의미가 달라 정확도 향상 비교가 아니다.

| 조건 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT_AND_RELOCATE |
|---|---:|---:|---:|---:|---:|
| C0 | 0 → 0 | 0 → 1 | 5 → 10 | 5 → 5 | 14 → 0 |
| C1 | 0 → 0 | 0 → 5 | 5 → 2 | 0 → 0 | 0 → 0 |
| C2_D4 | 0 → 0 | 0 → 15 | 5 → 17 | 0 → 3 | 7 → 17 |
| C2_D8 | 0 → 1 | 0 → 19 | 5 → 16 | 0 → 5 | 0 → 11 |
| C3_D4 | 0 → 0 | 0 → 2 | 0 → 1 | 0 → 0 | 0 → 0 |
| C3_D8 | 0 → 0 | 0 → 1 | 0 → 3 | 0 → 1 | 0 → 0 |
| C4_D4 | 0 → 0 | 0 → 0 | 0 → 0 | 5 → 3 | 5 → 0 |
| C4_D8 | 0 → 0 | 0 → 5 | 0 → 3 | 0 → 1 | 4 → 0 |
| C5_D4 | 0 → 0 | 0 → 5 | 4 → 1 | 0 → 3 | 0 → 0 |
| C5_D8 | 0 → 0 | 0 → 6 | 0 → 5 | 0 → 2 | 0 → 0 |
| C6_D4 | 0 → 0 | 0 → 6 | 0 → 4 | 0 → 1 | 0 → 0 |
| C6_D8 | 0 → 0 | 0 → 3 | 0 → 6 | 0 → 5 | 0 → 0 |

## 장면별 BLOCKED 응답 (/5)

| 장면 | 타깃 | C0 | C1 | C2_D4 | C2_D8 | C3_D4 | C3_D8 | C4_D4 | C4_D8 | C5_D4 | C5_D8 | C6_D4 | C6_D8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_fc_blocked_v01 | 64 | 0 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| scene_fc_blocked_v02 | 53 | 0 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_fc_blocked_v03 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| scene_fc_blocked_v04 | 89 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| scene_fc_blocked_v05 | 59 | 1 | 5 | 5 | 5 | 2 | 1 | 0 | 5 | 5 | 5 | 5 | 2 |
| scene_fc_clear_v01 | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_fc_clear_v02 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_fc_clear_v03 | 95 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_fc_clear_v04 | 61 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_fc_clear_v05 | 62 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v01 | 59 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v02 | 70 | 0 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v03 | 12 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v04 | 68 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v05 | 43 | 0 | 0 | 5 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_rotate_v01 | 12 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| scene_rotate_v02 | 38 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_rotate_v03 | 83 | 0 | 0 | 2 | 5 | 0 | 1 | 1 | 1 | 2 | 2 | 1 | 5 |
| scene_rotate_v04 | 47 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| scene_rotate_v05 | 44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_translate_v01 | 67 | 5 | 2 | 5 | 5 | 1 | 3 | 0 | 3 | 1 | 5 | 4 | 5 |
| scene_translate_v02 | 45 | 0 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_translate_v03 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| scene_translate_v04 | 13 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_translate_v05 | 54 | 5 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 그림

![v20 결과](plots/01_blocked_by_condition_family.png)

![v20 결과](plots/02_v19_v20_blocked_comparison.png)

![v20 결과](plots/03_scene_blocked_matrix.png)

## 재현

```bash
.qwen3-vl/venv/bin/python scripts/run_l2_front_obstruction_geometry_v20.py
uv run --no-project --python 3.13 --with matplotlib python scripts/analyze_l2_front_obstruction_geometry_v20.py
```
