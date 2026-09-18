# 노션 v16 직접 회수 프롬프트 재실행 — v21

원문: [L2 직접 회수 판단 × Geometry 단일정보 — v16 결과 (25장면·1,500회)](https://app.notion.com/p/3de952c9e27381e5969cdef5d80d54d2?pvs=204)

노션 system/user/schema가 원본 v16 config와 정확히 일치함을 확인했다. 이미지·Geometry·모델·생성 설정·실행 환경·seed·순서·프롬프트·schema를 동일하게 유지하여 새 추론 1,500회를 실행했다. 이전 응답은 재사용하지 않았다.

완료/JSON 유효 1500/1500, 오류 0, 모두 첫 시도: True. 생성 시간 합계 135.694초.

전체: RETRIEVE_NOW 505, REARRANGE_FIRST 995. 원본 v16과 개별 응답이 다른 호출 0/1500.

## 전체 조건 응답: 바로 회수 / 먼저 재배치

각 칸 합계는 25회. 정답 개수가 아니라 두 응답의 횟수다.

| 조건 | 정보 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT | 전체 회수 / 재배치 |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | RGB only | 25 / 0 | 19 / 6 | 2 / 23 | 1 / 24 | 0 / 25 | 47 / 78 |
| C1 | R Numeric | 20 / 5 | 13 / 12 | 5 / 20 | 1 / 24 | 0 / 25 | 39 / 86 |
| C2_D4 | R Qualitative D4 | 25 / 0 | 16 / 9 | 0 / 25 | 0 / 25 | 0 / 25 | 41 / 84 |
| C2_D8 | R Qualitative D8 | 24 / 1 | 7 / 18 | 0 / 25 | 0 / 25 | 0 / 25 | 31 / 94 |
| C3_D4 | F Numeric D4 | 22 / 3 | 22 / 3 | 4 / 21 | 4 / 21 | 5 / 20 | 57 / 68 |
| C3_D8 | F Numeric D8 | 25 / 0 | 21 / 4 | 5 / 20 | 4 / 21 | 1 / 24 | 56 / 69 |
| C4_D4 | F Qualitative D4 | 16 / 9 | 13 / 12 | 2 / 23 | 2 / 23 | 0 / 25 | 33 / 92 |
| C4_D8 | F Qualitative D8 | 17 / 8 | 14 / 11 | 2 / 23 | 0 / 25 | 0 / 25 | 33 / 92 |
| C5_D4 | M Numeric D4 | 18 / 7 | 8 / 17 | 4 / 21 | 6 / 19 | 2 / 23 | 38 / 87 |
| C5_D8 | M Numeric D8 | 23 / 2 | 19 / 6 | 3 / 22 | 4 / 21 | 0 / 25 | 49 / 76 |
| C6_D4 | M Qualitative D4 | 19 / 6 | 13 / 12 | 4 / 21 | 0 / 25 | 0 / 25 | 36 / 89 |
| C6_D8 | M Qualitative D8 | 22 / 3 | 21 / 4 | 2 / 23 | 0 / 25 | 0 / 25 | 45 / 80 |

## 원본 노션 기준 장면 설계 일치율

FC_CLEAR는 RETRIEVE_NOW, 나머지 네 family는 REARRANGE_FIRST를 기대하는 원본 노션의 참고 지표다. 실제 회수 성공률·물리적 정확도가 아니며, 이후 논의의 FC_BLOCKED를 회수 쪽으로 유도하려는 목표와도 다르다.

| 조건 | 원본 v16 | 재실행 v21 | 개별 응답 변경 /125 | 5회 만장일치 장면 /25 |
|---|---:|---:|---:|---:|
| C0 | 103/125 (82.4%) | 103/125 (82.4%) | 0 | 22 |
| C1 | 101/125 (80.8%) | 101/125 (80.8%) | 0 | 17 |
| C2_D4 | 109/125 (87.2%) | 109/125 (87.2%) | 0 | 23 |
| C2_D8 | 117/125 (93.6%) | 117/125 (93.6%) | 0 | 23 |
| C3_D4 | 87/125 (69.6%) | 87/125 (69.6%) | 0 | 14 |
| C3_D8 | 94/125 (75.2%) | 94/125 (75.2%) | 0 | 17 |
| C4_D4 | 99/125 (79.2%) | 99/125 (79.2%) | 0 | 16 |
| C4_D8 | 101/125 (80.8%) | 101/125 (80.8%) | 0 | 18 |
| C5_D4 | 98/125 (78.4%) | 98/125 (78.4%) | 0 | 16 |
| C5_D8 | 97/125 (77.6%) | 97/125 (77.6%) | 0 | 17 |
| C6_D4 | 102/125 (81.6%) | 102/125 (81.6%) | 0 | 18 |
| C6_D8 | 99/125 (79.2%) | 99/125 (79.2%) | 0 | 21 |

## 실제 사용 프롬프트

System:
```text
You assess whether a robot can retrieve a target object from a shelf.

Each numbered badge identifies the physical object underneath it.
Badge positions, sizes, and numbers are identity markers only.

Judge the current scene.
Return exactly one JSON object. Do not include explanations.
```

User:
```text
Target object ID: {target_object_id}

Geometry information:
{geometry_json}

Assume the robot can grasp and retrieve the target when it is
unobstructed.

In the current scene, can the robot pick up the target and take it
out through the shelf opening while leaving all other objects
in place and avoiding contact with them?

Consider whether surrounding objects obstruct access to the target
or its removal.

Partial visibility alone does not mean retrieval is blocked.
Being near another object alone does not mean retrieval is blocked.

Choose one:
- RETRIEVE_NOW: the target can be retrieved without rearranging
  other objects.
- REARRANGE_FIRST: another object must be moved before the target
  can be retrieved.

Return exactly one JSON object with the field "decision".
Do not identify the blocker or propose an action.
```

## 장면별 회수 / 재배치 횟수

| 장면 | 타깃 | C0 | C1 | C2_D4 | C2_D8 | C3_D4 | C3_D8 | C4_D4 | C4_D8 | C5_D4 | C5_D8 | C6_D4 | C6_D8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_fc_blocked_v01 | 64 | 5 / 0 | 4 / 1 | 5 / 0 | 0 / 5 | 3 / 2 | 3 / 2 | 2 / 3 | 4 / 1 | 2 / 3 | 5 / 0 | 0 / 5 | 5 / 0 |
| scene_fc_blocked_v02 | 53 | 4 / 1 | 4 / 1 | 4 / 1 | 2 / 3 | 4 / 1 | 3 / 2 | 3 / 2 | 3 / 2 | 1 / 4 | 5 / 0 | 2 / 3 | 2 / 3 |
| scene_fc_blocked_v03 | 12 | 5 / 0 | 3 / 2 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 3 / 2 | 4 / 1 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |
| scene_fc_blocked_v04 | 89 | 5 / 0 | 2 / 3 | 2 / 3 | 0 / 5 | 5 / 0 | 5 / 0 | 5 / 0 | 0 / 5 | 0 / 5 | 2 / 3 | 3 / 2 | 5 / 0 |
| scene_fc_blocked_v05 | 59 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 5 / 0 | 5 / 0 | 0 / 5 | 3 / 2 | 0 / 5 | 2 / 3 | 3 / 2 | 4 / 1 |
| scene_fc_clear_v01 | 53 | 5 / 0 | 2 / 3 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 4 / 1 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |
| scene_fc_clear_v02 | 29 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 4 / 1 | 5 / 0 | 3 / 2 | 0 / 5 | 3 / 2 | 5 / 0 | 5 / 0 | 5 / 0 |
| scene_fc_clear_v03 | 95 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 4 / 1 | 5 / 0 | 5 / 0 | 5 / 0 |
| scene_fc_clear_v04 | 61 | 5 / 0 | 4 / 1 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 2 / 3 | 3 / 2 | 5 / 0 | 5 / 0 | 2 / 3 | 5 / 0 |
| scene_fc_clear_v05 | 62 | 5 / 0 | 4 / 1 | 5 / 0 | 4 / 1 | 3 / 2 | 5 / 0 | 2 / 3 | 4 / 1 | 1 / 4 | 3 / 2 | 2 / 3 | 2 / 3 |
| scene_lift_and_relocate_v01 | 59 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 2 / 3 | 0 / 5 | 0 / 5 | 0 / 5 | 2 / 3 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_lift_and_relocate_v02 | 70 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 1 / 4 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_lift_and_relocate_v03 | 12 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 2 / 3 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_lift_and_relocate_v04 | 68 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_lift_and_relocate_v05 | 43 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 1 / 4 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_rotate_v01 | 12 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_rotate_v02 | 38 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 2 / 3 | 2 / 3 | 0 / 5 | 0 / 5 | 4 / 1 | 2 / 3 | 0 / 5 | 0 / 5 |
| scene_rotate_v03 | 83 | 0 / 5 | 1 / 4 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_rotate_v04 | 47 | 1 / 4 | 0 / 5 | 0 / 5 | 0 / 5 | 2 / 3 | 2 / 3 | 2 / 3 | 0 / 5 | 2 / 3 | 2 / 3 | 0 / 5 | 0 / 5 |
| scene_rotate_v05 | 44 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_translate_v01 | 67 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_translate_v02 | 45 | 2 / 3 | 5 / 0 | 0 / 5 | 0 / 5 | 1 / 4 | 2 / 3 | 2 / 3 | 2 / 3 | 4 / 1 | 1 / 4 | 2 / 3 | 0 / 5 |
| scene_translate_v03 | 18 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 3 / 2 | 2 / 3 | 0 / 5 | 0 / 5 | 0 / 5 | 1 / 4 | 0 / 5 | 2 / 3 |
| scene_translate_v04 | 13 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 1 / 4 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| scene_translate_v05 | 54 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 | 1 / 4 | 2 / 3 | 0 / 5 |

## 그림

![v21 결과](plots/01_rearrange_by_condition_family.png)

![v21 결과](plots/02_delta_vs_original_v16.png)

## 재현

```bash
.qwen3-vl/venv/bin/python scripts/run_l2_v16_prompt_replay_v21.py
uv run --no-project --python 3.13 --with matplotlib python scripts/analyze_l2_v16_prompt_replay_v21.py
```
