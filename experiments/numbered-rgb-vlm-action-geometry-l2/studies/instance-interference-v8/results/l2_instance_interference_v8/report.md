# L2 all-instance interference v8

## 실험 정의

한 장면을 한 번 호출하고, 보이는 모든 번호 물체를 오름차순으로 한 번씩 출력한다. Target은 `TARGET`, 나머지는 `INTERFERES` 또는 `DOES_NOT_INTERFERE`로 분류한다.

프롬프트에는 수치 임계값을 노출하지 않았다. GT는 각 후보가 target-only silhouette의 5% 이상을 가린 경우 `INTERFERES`로 정의한다.

## 무결성

- 완료: 1500/1500
- 파싱 오류 / 후검증 실패 / 출력 불변식 실패: 0 / 0 / 0
- 인프라 오류: 0

## 전체 결과

| 평가 단위 | Accuracy | Balanced accuracy | Interference recall | Non-interference specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| 후보 instance | 61.8% | 63.3% | 65.6% | 61.0% | 590/310/2709/1731 |
| 장면 내 하나라도 방해 | 55.2% | 52.2% | 67.3% | 37.0% | 606/294/222/378 |

- 모든 번호 관계 완전일치: 389/1500 (25.9%)
- 방해 물체 ID 집합 완전일치: 389/1500 (25.9%)
- Target 역할 정확도: 1500/1500 (100.0%)

## 조건별 결과

| 조건 | 후보 정확도 | BA | 방해 recall | 비방해 specificity | 전체 관계 완전일치 | 장면 Boolean 정확도 |
|---|---:|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 65.4% | 65.4% | 65.3% | 65.4% | 32.0% | 55.2% |
| C1  R Numeric | 72.4% | 70.6% | 68.0% | 73.2% | 27.2% | 52.0% |
| C2 D4 R Qualitative D4 | 49.9% | 62.4% | 81.3% | 43.5% | 18.4% | 53.6% |
| C2 D8 R Qualitative D8 | 49.9% | 63.0% | 82.7% | 43.2% | 12.0% | 50.4% |
| C3 D4 F Numeric D4 | 67.9% | 59.4% | 46.7% | 72.2% | 27.2% | 50.4% |
| C3 D8 F Numeric D8 | 65.6% | 61.3% | 54.7% | 67.8% | 33.6% | 59.2% |
| C4 D4 F Qualitative D4 | 62.0% | 58.6% | 53.3% | 63.8% | 22.4% | 52.8% |
| C4 D8 F Qualitative D8 | 59.8% | 63.6% | 69.3% | 57.8% | 32.0% | 68.8% |
| C5 D4 M Numeric D4 | 62.7% | 60.6% | 57.3% | 63.8% | 24.0% | 48.8% |
| C5 D8 M Numeric D8 | 63.1% | 65.6% | 69.3% | 61.9% | 28.0% | 58.4% |
| C6 D4 M Qualitative D4 | 58.7% | 61.8% | 66.7% | 57.0% | 24.8% | 53.6% |
| C6 D8 M Qualitative D8 | 64.0% | 67.2% | 72.0% | 62.4% | 29.6% | 59.2% |

## 장면 종류별 결과

| 장면 종류 | 후보 정확도 | BA | 방해 recall | 비방해 specificity | 전체 관계 완전일치 | 장면 Boolean 정확도 |
|---|---:|---:|---:|---:|---:|---:|
| TRANSLATE | 71.2% | 70.5% | 68.0% | 73.0% | 33.0% | 68.0% |
| ROTATE | 59.1% | 55.6% | 49.0% | 62.3% | 10.3% | 54.3% |
| LIFT_AND_RELOCATE | 49.5% | 60.6% | 79.7% | 41.6% | 12.3% | 79.7% |
| FC_CLEAR | 75.1% | N/A | N/A | 75.1% | 66.3% | 66.3% |
| FC_BLOCKED | 63.0% | N/A | N/A | 63.0% | 7.7% | 7.7% |

## v7 단일 Boolean과 비교

| 방식 | Accuracy | BA | Recall | Specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| v8 instance별 출력의 OR | 55.2% | 52.2% | 67.3% | 37.0% | 606/294/222/378 |
| v7 단일 Boolean | 52.9% | 60.5% | 22.3% | 98.7% | 201/699/592/8 |

오답 후보 instance 행은 총 2041개이며 `candidate_instance_errors.csv`에 모두 기록했다.

## 대표 오류

- False negative: `scene_rotate_v05`에서 target 44를 가리는 object 82의 GT overlap ratio는 0.710이지만, 60회 중 0회만 `INTERFERES`로 출력했다.
- False positive: `scene_fc_blocked_v01`에서 object 40의 GT overlap ratio는 0.000이지만, 60회 중 60회 `INTERFERES`로 출력했다. 이 물체는 target 64 바로 옆에 있어, 프롬프트의 gripper-contact 표현이 시각적 가림 GT보다 넓게 해석된 사례다.
