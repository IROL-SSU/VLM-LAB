# Diverse TRANSLATE RGB-D — 1,000-scene dataset

## 생성·검증 완료

**1,000개 장면 생성 및 1,000개 전수 검증 완료.** 디스크 크기 약 **6.1GB**. 데이터 생성 작업 당시에는 모델 재학습을 하지 않았다.

후속 학습 완료: [1,000개 데이터 학습 결과](../translate_train_1000_20261001_v1/README.md). 고정된 800/100/100 분할을 사용했으며 원본 데이터와 분할은 변경하지 않았다. 이 폴더의 `completion.json`은 데이터 생성 완료 시점의 기록이다.

| 항목 | 실제 결과 |
|---|---|
| 방향 | LEFT 500 / RIGHT 500 |
| 분할 | train 800 / validation 100 / test 100 |
| 각 split의 좌우 | train 400/400, validation 50/50, test 50/50 |
| 물체 / 역할 조합 | 15종 / ordered target–blocker 50조합 |
| 이동 거리 | **2.1–22.0cm**, 중앙값 6.6cm |
| 거리 구간 | 2~5cm 300개 / 5~9cm 400개 / 9~25cm 300개 |
| 거리 조건 완화 | **0개** — 계획한 거리 구간 수량 모두 충족 |
| 주변 물체 수 | 2개 158장 / 3개 163장 / 4개 170장 / 5개 179장 / 6개 154장 / 7개 176장 |
| 선반 색 | light oak 192 / walnut 194 / ivory 221 / slate 194 / sage 199 |
| 중복 검사 | 동일 RGB 0 / 동일 배치 0 / split 간 정의된 coarse 배치 중복 0 |
| 코드 검사 | 관련 unit test **23개 통과** |

모든 role 조합에 좌우 각 10개가 있으며, 15개 asset 모두 target 역할과 blocker 역할에 등장한다. 같은 물체만 보고 방향을 정할 수 없도록 구성했다. 중간 렌더러 오류는 완료 장면을 보존한 채 해당 구간을 재개했고, 최종 수량·누락·품질은 별도 검사로 확인했다.

## 범위

LEFT/RIGHT 물체 translation의 방향·거리 예측을 위한 합성 RGB-D 데이터다. 모델 학습은 이 작업에 포함하지 않았다. 기존 5개 학습 장면, 새 배치 검증 2개, 기존 checkpoint는 보존했다.

한 가지 선반 구조 안에서 장면 구성을 다양화했다. 완전히 다른 방/선반 구조나 새로운 object mesh를 추가한 것은 아니다.

## 다양성 및 분할 방식

- 실제 USD object asset **15종**, 서로 다른 ordered target–blocker 조합 **50개**.
- 조합당 **20개 독립 배치**: LEFT 10개, RIGHT 10개. 전체 LEFT **500**, RIGHT **500**.
- 각 조합에서 학습 16개, 검증 2개, 테스트 2개. 전체 **train 800 / validation 100 / test 100**.
- Target·blocker·주변 물체 위치와 yaw(-180~180°) 랜덤화. 주변 물체 **2~7개**, 장면당 전체 **4~9개**. 같은 장면 내 asset 중복 없음.
- 선반 색 5종(light oak / walnut / ivory / slate / sage), 벽 색 5종, key/fill/dome 광량 랜덤화.
- 카메라 위치: world X 22~34cm, Y -100~-90cm, Z 62~66cm, pitch 76~80°. 해상도 960×720. 각 장면의 실제 카메라 보정값 저장.
- Target 가시 비율 **40~80%**, blocker 가림 비율 **20~60%**, 다른 물체의 target 가림 ≤10%. 모든 선택 물체 최소 50 visible pixels.
- 거리 구간은 short 2~5cm, medium 5~9cm, long 9~25cm. 300/400/300개를 목표로 하되 특정 조합의 물리적/가림 제약 때문에 불가능한 구간은 다른 가능한 구간을 허용하고 `bin_fallback`으로 명시한다. **실제 분포는 `dataset_summary.json`을 기준**으로 한다.

분할은 생성 전에 `schedule.json`에 고정했다. 동일한 배치의 멀티뷰/색 변경 복제본을 나눠 담지 않고, 모든 샘플의 물체 배치를 새로 생성한다. 동일 물체/조합은 split 간 공유하므로 **새 배치 일반화용 분할**이며, unseen-object/unseen-pair 평가용은 아니다.

## label과 품질 검사

이전 파일럿과 같은 정책을 유지했다: target 앞 통로를 완전히 비우며 다른 물체 AABB/선반 경계와 연속 이동 중 충돌하지 않는, **8방향 후보 중 최소 1mm-grid 이동량**을 계산한다. 이번 데이터에는 그 최적 방향이 지정된 LEFT 또는 RIGHT인 장면만 채택했다.

Direction은 선반 좌표계, distance와 depth/point cloud 단위는 **m**이다. 회전·grasp·push/pull 분류·executor는 포함하지 않는다.

독립 검증은 모든 샘플의 RGB-D/mask 정렬, optical-Z 깊이, 보정 후 point cloud의 물체 bounds 일치, point-to-pixel 대응, 초기 물체 겹침 없음, label 재계산, 정답 목표의 통로 확보 및 이동 경로 충돌을 검사한다. 추가로 split 분리, 방향 균형, role 조합별 수량, RGB/정확한 배치 중복 및 coarse 배치 중복을 검사한다.

Coarse 중복 기준: 같은 role-labeled 전체 물체 집합에서 XY를 2cm, yaw를 15°로 양자화한 값이 동일한 경우. 카메라·재질은 무시하여 촬영 조건만 바꾼 복제본이 split을 넘는 경우를 검사한다. 이 기준이 모든 시각적 유사 장면을 제거한다는 뜻은 아니다.

Label은 **정적 보수적 AABB 기반 proxy**이며 실제 접촉 역학, 로봇 가동범위, grasp 성공을 검증하지 않았다. RGB-D에 보이지 않는 형상은 정답 생성/검증에만 사용하며 모델 입력으로 사용하면 안 된다.

## 데이터 사용

- `manifest.json`: 전체 1,000개.
- `manifest_train.json`, `manifest_validation.json`, `manifest_test.json`: 각각 800/100/100개. **이 manifest들의 `sample_dir`은 모두 이 dataset 루트 기준**이다.
- `splits.json`: 장면 ID 분할 목록.
- `schedule.json`: seed·role pair·방향·분할의 생성 전 고정 계획.
- `generation_config.json`: 범위·분포·허용 조건.
- `dataset_summary.json`: 최종 실제 분포와 중복 검사.
- `validation.json`: 샘플별 독립 품질 검사.
- `completion.json`: 1,000개 생성·검증 완료 여부.
- `overview.jpg`: 다양한 장면 20개 미리보기.
- `gallery.html`, `contact_sheets/`: 전체 장면을 50개씩 볼 수 있는 갤러리.
- `source/`: 생성·검증 코드 snapshot.

각 `samples/translate_diverse_XXXX/` 폴더:

- 모델 입력: `rgb.png`, `depth_z_m.npy`, `target_mask.png`, `blocker_mask.png`, `calibration.json`.
- 추가 관측 데이터: `instance_ids.png`, `depth_valid.png`, `pointcloud.npz`.
- 정답: `label.json`.
- 검증 전용: `objects.json`, `geometry_audit.json`, `target_reference_mask.png`, `visibility_audit.json`.
- 재현/시각화: `scene.usda`, `environment.json`, `sampling_audit.json`, `record.json`, `preview.png`, `topdown.png`, `goal_rgb.png`.

`objects.json`의 simulator GT bounds/중심, target-only reference mask, goal 이미지, label, scene ID는 encoder 입력에 넣지 않는다. `scene.usda`에는 재사용 목적으로 로드했으나 해당 장면에 등장하지 않는 hidden asset도 포함된다. 실제 선택 물체 목록은 `objects.json`이며, hidden asset은 모델 관측 데이터에 들어가지 않는다.

기존 5샘플 전용 `train_translate_pilot.py`는 정확히 5개를 요구한다. 후속 구현된 `scripts/train_translate_dataset.py`는 이 데이터의 split manifest를 읽어 mini-batch 학습하고 validation loss로 checkpoint를 선택한다. Test split은 모델 선택에 사용하지 않는다.

## 재현 / 재개

저장소 루트에서 실행하며 새 output 경로를 지정한다. Generator는 출력 폴더를 자동 덮어쓰지 않는다. 각 장면은 임시 폴더에 완전히 저장한 뒤 atomic rename하므로 완료된 샘플은 resume에서 재생성하지 않는다.

```bash
/home/ssu/isaacsim/python.sh scripts/generate_translate_1000.py --output experiments/translate_diverse_rerun

# 중단한 생성 재개
/home/ssu/isaacsim/python.sh scripts/generate_translate_1000.py --output experiments/translate_diverse_rerun --resume

# 전체 기록에서 manifest 재구성 (renderer 불필요)
python scripts/generate_translate_1000.py --output experiments/translate_diverse_rerun --resume --assemble-only

# 전체 1,000개 독립 검사와 갤러리
python scripts/audit_translate_1000.py experiments/translate_diverse_rerun

/home/ssu/miniforge3/envs/sam3/bin/python -m unittest discover -s scripts -p 'test_translate*.py' -v
```

GPU 메모리가 충분하면 서로 겹치지 않는 `[start-index, end-index)` 구간을 `--resume --worker-tag a` 등으로 나눠 렌더링할 수 있다. Worker별 progress/manifest는 분리되어 기록되며, 모든 worker 종료 뒤 반드시 `--assemble-only`와 전체 audit을 실행한다. 시드/배치 결과는 구간 분할과 무관하게 scene별 계획에 의해 결정된다.
