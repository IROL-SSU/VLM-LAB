# L2 single primary grasp blocker v10

## 실행 정의

System prompt:

```text
You are a robot manipulation planner.
Return only valid JSON.
```

User task prompt:

```text
The robot must grasp the target object.

Select the single numbered object that most obstructs grasping the target.
If no object obstructs the grasp, return null.
```

출력은 `object_to_remove_id` 하나이며 정답 방해물이 없으면 `null`이다.
GT는 Isaac Sim instance mask로 target-only silhouette의 5% 이상을 가리는 물체 중 가림 비율이 가장 큰 ID다. 이 장면 집합에는 양성 장면마다 후보가 정확히 하나 있다.

## 무결성

- 완료: 1500/1500
- 파싱 오류 / 후검증 실패 / 인프라 오류: 0 / 0 / 0

## 전체 결과

| 지표 | 결과 |
|---|---:|
| 정확한 ID 또는 null 완전일치 | 655/1500 (43.7%) |
| 방해물 존재 여부 정확도 | 60.0% |
| 존재 여부 balanced accuracy | 50.0% |
| 방해물 존재 recall | 100.0% |
| null specificity | 0.0% |
| 양성 장면 정확한 ID | 655/900 (72.8%) |
| 출력 null 횟수 | 0/1500 |

모델은 1,500회 모두 non-null ID를 출력했다. 따라서 양성 존재 recall은 100%지만 null specificity는 0%다.

## 조건별

| 조건 | 정확한 ID/null | 양성 ID 정확도 | null specificity |
|---|---:|---:|---:|
| C0  Numbered RGB only | 33.6% | 56.0% | 0.0% |
| C1  R Numeric | 41.6% | 69.3% | 0.0% |
| C2 D4 R Qualitative D4 | 46.4% | 77.3% | 0.0% |
| C2 D8 R Qualitative D8 | 46.4% | 77.3% | 0.0% |
| C3 D4 F Numeric D4 | 42.4% | 70.7% | 0.0% |
| C3 D8 F Numeric D8 | 43.2% | 72.0% | 0.0% |
| C4 D4 F Qualitative D4 | 44.8% | 74.7% | 0.0% |
| C4 D8 F Qualitative D8 | 48.0% | 80.0% | 0.0% |
| C5 D4 M Numeric D4 | 43.2% | 72.0% | 0.0% |
| C5 D8 M Numeric D8 | 49.6% | 82.7% | 0.0% |
| C6 D4 M Qualitative D4 | 45.6% | 76.0% | 0.0% |
| C6 D8 M Qualitative D8 | 39.2% | 65.3% | 0.0% |

## 장면 종류별

| 장면 종류 | 정확한 ID/null | 양성 ID 정확도 | null specificity |
|---|---:|---:|---:|
| TRANSLATE | 99.7% | 99.7% | N/A |
| ROTATE | 66.3% | 66.3% | N/A |
| LIFT_AND_RELOCATE | 52.3% | 52.3% | N/A |
| FC_CLEAR | 0.0% | N/A | 0.0% |
| FC_BLOCKED | 0.0% | N/A | 0.0% |

## v9 대비

| 방식 | 전체 exact | 양성 정답 blocker | null 정답 |
|---|---:|---:|---:|
| v9 instance별 occlusion 배열 | 49.1% | 15.8% | 99.2% |
| v10 단일 blocker ID | 43.7% | 72.8% | 0.0% |

v10은 양성 blocker ID 선택은 크게 좋아졌지만, `null`을 한 번도 선택하지 않아 전체 exact는 v9보다 낮다.

## 해석

짧은 문구의 `Select the single numbered object`가 선택 행동을 강하게 유도해, 뒤의 `If no object ... return null`보다 우세하게 작동했다. 따라서 이 결과는 target 앞의 물체를 하나 고르는 능력은 높지만, 먼저 제거할 물체가 실제로 필요한지 판단하는 능력은 전혀 분리되지 않았음을 보여준다.

주의: 이번 GT는 Isaac Sim의 instance-mask 가림 oracle이다. 실제 로봇 팔의 IK·충돌·gripper 폐쇄를 재생한 grasp rollout 평가는 아니다.
