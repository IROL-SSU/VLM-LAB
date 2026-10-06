# TRANSLATE RGB-D 모델 실험 로그

[중간 목표 설계](../) · [공통 데이터](dataset/)

이 폴더는 `9. 현재까지의 검증 결과`에 보고한 다섯 모델의 실험 진행 과정과 결과를 저장소만으로 감사(audit)할 수 있도록 정리한 snapshot이다. 모델이 바뀐 순서대로 번호를 붙였으며, 각 run 폴더에는 실행 당시 설정·전체 epoch history·checkpoint 선택 기록·테스트 예측·평가 결과·source snapshot이 들어 있다.

## 실험 진행 순서

| 순서 | 모델 | 원본 run ID | 핵심 변경 | 방향 | 방향+안전 | 정적 안전 | 최소거리 MAE |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | [Baseline](01-baseline/) | `translate_train_1000_20261001_v1` | 방향 분류와 방향별 거리 회귀 | 74% | 40% | 40% | 25.60 mm |
| 2 | [Signed scalar](02-signed-scalar/) | `translate_signed_safe_1000_20261002_v1` | 방향과 거리를 부호 있는 단일 변위로 결합 | 68% | 41% | 43% | 35.24 mm |
| 3 | [Joint + mirror](03-joint-mirror/) | `translate_joint_bins_flip_1000_20261002_v1` | 방향×거리 joint class, 복수 안전 정답, 좌우 반전 | 72% | **54%** | 61% | 40.53 mm |
| 4 | [Joint + short-safe](04-joint-short-safe/) | `translate_joint_bins_flip_preferred_1000_20261002_v1` | 안전 action 중 짧고 여유 있는 action 선호 | 71% | **54%** | **64%** | 38.72 mm |
| 5 | [Direction-first](05-direction-first/) | `translate_joint_bins_flip_direction_1000_20261002_v1` | 방향 marginal을 먼저 선택하고 해당 방향에서 거리 선택 | **79%** | 53% | 55% | 30.67 mm |

다섯 모델은 같은 100개 test 장면을 사용한다. Baseline 이후 이 test split을 반복 분석했으므로 위 표는 **동일 장면에서의 통제 비교**이며 untouched final test 성능이 아니다.

## 공통 실험 조건

- 합성 RGB-D shelf 장면 1,000개
- Train 800 / Validation 100 / Test 100
- LEFT 500 / RIGHT 500, 각 split 내 균형
- 동일한 15개 asset과 50개 target–blocker pair
- Split 사이에는 물체 배치만 분리되며 unseen-object/unseen-pair 평가는 아님
- 입력은 RGB, optical-Z depth, target/blocker visible mask와 calibration
- Frozen ImageNet ResNet-18 visual encoder
- Trainable PointNet-style geometry encoder와 fusion/head
- 정확한 target/blocker grounding을 가정
- 안전 평가는 target 전면 통로 확보와 shelf 경계/swept AABB 충돌 여부를 결합한 정적 proxy

공통 데이터 생성 설정과 split manifest는 [dataset/](dataset/)에 보존한다.

## 각 run 폴더 읽는 순서

1. `README.md` — 모델 변경 이유, 선택 checkpoint, 핵심 결과와 해석
2. `RUN_NOTES.md` — 원본 실험 폴더에 있던 실행 직후 기록을 그대로 보존한 문서
3. `config.json` — split ID, seed, optimizer, loss, 환경 버전과 경로
4. `history.csv` / `history.json` — 전체 epoch의 train/validation 추이
5. `selection_complete.json` — test 평가 전에 확정한 checkpoint와 선택 기준
6. `completion.json` — 최종 test 결과와 checkpoint SHA-256
7. `results/metrics.json` — 100개 test prediction과 상세 집계
8. `results/predictions.csv` — 장면별 정답·예측·오차·정적 유효성
9. `results/raw_predictions_before_gt.json` — GT 결합 전 raw prediction snapshot
10. `gradient_audit.json` / `input_fingerprints.json` — freeze/gradient와 입력 무결성 확인
11. `source/` — 해당 run 실행 당시 코드 snapshot

## 포함하지 않은 대용량 artifact

| Artifact | 제외 이유 | 대신 남긴 정보 |
|---|---|---|
| `best.pt`, `last.pt` | 모델별 약 46–49MB | `completion.json`과 README의 checkpoint SHA-256 |
| Frozen feature cache | 모델별 수십 MB | cache 설정·encoder hash·input fingerprint |
| RGB-D sample 1,000개 | 약 6.1GB | split manifest, dataset validation, overview/contact sheet |
| Test 개별 시각화 전체 | 모델별 약 20장 | 동일 순서의 `first20_predictions.jpg` contact sheet |

따라서 이 snapshot만으로 **실험 설계·학습 추이·checkpoint 선택·장면별 결과·코드 차이**는 추적할 수 있지만, 원본 RGB-D와 checkpoint가 필요한 완전한 재학습 또는 inference replay는 할 수 없다. 대용량 artifact가 추가로 필요하면 별도 release, object storage 또는 Git LFS에 checkpoint SHA와 연결해야 한다.

## 무결성

`SHA256SUMS`에는 이 폴더에 포함된 모든 snapshot 파일의 상대 경로와 SHA-256이 기록된다. 다른 컴퓨터에서는 이 디렉터리에서 다음 명령으로 복사 무결성을 확인할 수 있다.

```bash
sha256sum -c SHA256SUMS
```
