# TRANSLATE RGB-D — 1,000-scene 학습 결과

학습과 최종 테스트를 완료했다. **테스트 방향 정확도 74/100, 거리 MAE 25.60mm, 정적 목표 유효성 40/100**이다. 새로운 배치에서 일정한 예측 능력은 확인되지만, 아직 안정적인 sub-goal 생성 모델이라고 보기에는 부족하다.

## 학습 조건

- 데이터: `../translate_diverse_1000_20261001_v1`, 생성 전에 고정한 train 800 / validation 100 / test 100.
- 각 split은 LEFT/RIGHT 균형. 같은 15개 asset과 50개 target–blocker 조합을 공유하며, **물체 배치만 held-out**이다.
- 기존 파일럿과 동일한 frozen ImageNet ResNet-18 + trainable PointNet-style geometry encoder + fusion MLP + 방향/거리 head.
- RGB encoder 11,176,512개 파라미터 고정. Geometry/fusion/heads 368,744개 학습.
- 기존 5개 장면 checkpoint를 이어 학습하지 않았다. RGB만 ImageNet 사전학습 가중치를 사용하고 나머지는 새로 초기화했다.
- 입력: RGB, optical-Z depth, target/blocker visible mask, calibration. Simulator GT bounds, asset identity, goal image, label, scene ID는 encoder 입력이 아니다.
- Frozen RGB 특징을 캐시. Geometry는 관측 RGB-D에서 샘플링한 2,048개 point를 사용한다. RGB와 point sampling 모두 augmentation 없이 고정.
- Adam, 초기 learning rate 0.001, batch 32, 최대 200 epoch, cosine schedule 최저 0.00001, gradient clipping 5.
- Loss: 방향 cross entropy + 5 × SmoothL1(거리 / 0.25m, beta=0.02). 거리 회귀는 정답 방향 head만 감독한다.
- 검증 loss 최저 checkpoint를 선택. 30 epoch 연속 개선이 없어 **44 epoch에서 종료**, **14 epoch의 best.pt 선택**.
- Test는 선택 완료 후 best.pt를 새로 로드하여 **한 번만 평가**했다. 테스트 결과를 보고 재학습하거나 checkpoint를 바꾸지 않았다.

출력 구조는 기존 8방향 logits + 방향별 거리 head를 유지했다. 이번 감독 신호는 LEFT/RIGHT뿐이며 다른 6방향, rotation, grasp, executor는 학습·검증 대상이 아니다.

## 선택된 모델 결과

| 지표 | Train 800 | Validation 100 | Test 100 |
|---|---:|---:|---:|
| 방향 정확도 | 88.88% | 74.00% | 74.00% |
| 실제 예측 방향의 거리 MAE | 13.88mm | 21.21mm | 25.60mm |
| 정답 방향 head의 거리 MAE¹ | 7.30mm | 10.75mm | 10.98mm |
| 학습과 동일한 joint loss | 0.3724 | 0.6895 | 0.7563 |

¹ 정답 방향 head를 골라 계산한 진단용 수치다. 실제 추론에서는 방향도 모델이 선택하므로 이 값을 실제 거리 성능으로 사용하면 안 된다. 실제 출력의 거리 MAE는 25.60mm다.

최종 44 epoch 모델은 train 방향 100%, 거리 MAE 2.16mm까지 내려갔지만 validation loss는 악화됐다. 따라서 최종 모델 대신 사전에 정한 검증 loss 기준의 14 epoch 모델을 선택했다. 방향 정확도가 가장 높은 epoch를 선택한 것이 아니다.

### 테스트 세부 결과

- LEFT: 30/50 정답, 거리 MAE 34.12mm. RIGHT: 44/50 정답, 거리 MAE 17.08mm.
- 전체 예측: LEFT 36 / RIGHT 64. 다른 6방향 예측 없음.
- 거리 오차 중앙값 12.23mm, 90백분위 67.26mm, 최댓값 127.92mm.
- 방향 반전까지 반영한 **이동 벡터 오차 평균 57.40mm**. 거리의 크기 오차 25.60mm와 구분해야 한다.
- 방향이 맞은 74개에서 거리 MAE 10.81mm. 틀린 26개에서는 67.70mm.
- 방향 정답 + 거리 오차 5mm 이하: **25/100**.
- Train만으로 맞춘 상수 baseline: 항상 LEFT, 거리 75.13mm → test 방향 50%, 거리 MAE 29.44mm. 모델은 이 baseline보다 낫지만 목표 생성 품질은 여전히 제한적이다.

| 정답 거리 구간 | Test 수 | 방향 정답 | 거리 MAE |
|---|---:|---:|---:|
| 5cm 미만 | 25 | 25/25 | 8.48mm |
| 5cm 이상, 9cm 미만 | 46 | 38/46 | 17.76mm |
| 9cm 이상 | 29 | 11/29 | 52.79mm |

현재 결과에서는 긴 이동이 필요한 장면에서 오류가 특히 크다. 위 분석은 저장된 100개 예측의 사후 집계이며 추가 테스트 추론이나 모델 튜닝은 하지 않았다. 이 결과만으로 오류의 원인을 특정하지는 않는다.

### 목표 자체의 정적 기하 검사

예측한 방향·거리를 보정, snapping, 정답 대체 없이 검사했다.

- Target 앞 통로 확보: 54/100.
- 선반 내부이며 이동 경로가 다른 물체 AABB와 충돌하지 않음: 75/100.
- **두 조건 모두 충족: 40/100**.
- 실패 60개: 통로만 미확보 35, 충돌/경계 조건만 실패 14, 둘 다 실패 11.

최소 이동 label을 회귀하므로 방향을 맞혀도 약간 덜 움직이면 통로가 남을 수 있다. 반대로 더 움직여도 주변 물체/경계에 걸리면 유효하지 않다. 이 검사는 정적 보수적 AABB 기준이며 로봇 실행, 접촉 역학, grasp 성공률이 아니다.

## 검증 및 재현

- 관련 unit test **28개 통과**.
- 전체 input/manifest fingerprint가 학습 후에도 동일함을 확인.
- 저장된 history 전체에서 validation loss 최저 epoch가 checkpoint epoch와 일치함을 독립 확인.
- RGB encoder gradient 0 및 weights/BatchNorm buffer hash 불변 확인. Geometry/fusion/두 head에 gradient가 존재함을 확인.
- best.pt를 CPU로 재로드하여 train 전체 metric 재계산: 저장된 선택 epoch 수치와 일치.
- Train 1개에서 원본 RGB-D full forward와 cache 기반 추론 비교: 최대 절대 출력 차이 0.0000023842. Test는 재추론하지 않았다.
- Test IDs 100개가 train/validation과 분리되어 있고, 정답 수와 정적 유효성 집계가 보고서와 일치함을 확인.
- Checkpoint SHA256: `d64e87452d400eb5c0cb5e3cba85750db7cb6f8b710b8239d55cf7a8a3d3602f`.

단일 seed의 초기 baseline 실험이며 unseen-object, 실세계 센서, VLM grounding 오류에 대한 평가는 아니다. Target/blocker 선택 및 visible mask는 주어진다.

```bash
# 저장소 루트에서. 별도 output을 지정하여 기존 결과 보존.
/home/ssu/miniforge3/envs/sam3/bin/python scripts/train_translate_dataset.py \
  --output experiments/translate_train_1000_rerun

/home/ssu/miniforge3/envs/sam3/bin/python -m unittest discover -s scripts -p 'test_translate*.py' -v
```

## 파일

- [best.pt](best.pt): 선택된 전체 모델 가중치와 config.
- [last.pt](last.pt): 마지막 모델 및 optimizer/scheduler 상태. 선택된 모델이 아님.
- [config.json](config.json): 분할 IDs, 학습 설정, 환경 버전.
- [history.csv](history.csv), [learning_curves.png](learning_curves.png): train/validation 추이.
- [test_evaluation/metrics.json](test_evaluation/metrics.json): 테스트 집계와 100개 예측의 정답·기하 검사.
- [test_evaluation/predictions.csv](test_evaluation/predictions.csv): 분석용 예측 표.
- [test_evaluation/first20_predictions.jpg](test_evaluation/first20_predictions.jpg): manifest 순서 첫 20개. 성공 사례 선별이 아님. 주황 정답, 하늘색 예측.
- `cache_train.pt`, `cache_validation.pt`, `test_evaluation/cache_test.pt`: frozen RGB 특징 및 관측 point cache, scene IDs, sampling seed, encoder hash.
- `input_fingerprints.json`, `gradient_audit.json`, `source/`: 입력 hash, gradient/freeze 검사, 실행 코드 snapshot.
- `selection_complete.json`: **테스트 이전 시점**의 선택 확정 기록. 그 안의 `test_evaluated: false`는 당시 상태이며, 최종 완료 상태는 `completion.json`에 기록했다.
