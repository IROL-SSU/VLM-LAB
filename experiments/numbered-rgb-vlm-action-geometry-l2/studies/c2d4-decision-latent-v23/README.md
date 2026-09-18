# C2 D4 결정 분기 latent cosine — 중간층 vs 끝단

## 무엇을 측정했나

- 대상: L2 v16의 C2 D4 25개 장면. 기존 5개 seed 응답 125개는 결과 라벨로만 연결했다.
- 독립 latent 점은 장면당 하나, 총 25개다. 같은 장면의 5개 seed는 결정 전 입력이 같으므로 복제하지 않았다.
- 실제 출력은 `{"decision": "RE...`까지 공통이다. 토크나이저에서 그 다음 토큰이 `TR`(RETRIEVE_NOW)과 `ARR`(REARRANGE_FIRST)로 처음 갈라진다.
- 공통 토큰 `RE` 위치에서 decoder block 24 출력과 block 48 뒤 final RMSNorm 출력을 각각 2,048차원 벡터로 추출했다.
- 모델 답변을 다시 생성하지 않았다. 기존 vLLM 결과와 같은 checkpoint·이미지·prompt를 Transformers BF16 forward로 재입력했다.

## 핵심 결과

- Raw cosine 300쌍 평균: block 24 **0.992142**, final **0.983769**.
- 중간–끝단 pairwise cosine 패턴 상관: **0.2753**.
- 끝단에서 pairwise cosine이 바뀐 절댓값 평균: **0.009667**.
- 같은 family와 다른 family의 평균 차이(같음−다름): block 24 **0.002839**, final **0.008395**.
- 같은 기대 결정과 다른 기대 결정의 평균 차이: block 24 **0.001396**, final **0.008416**.
- 최근접 장면이 같은 family인 경우: block 24 **7/25**, final **18/25**.
- 재추출 branch logit 방향과 기존 5회 다수결의 일치: **19/25**. 장면 설계 기대와의 일치: **16/25**.

## 그림 읽는 법

1. `01_raw_cosine_mid_vs_final.png`: 원래 hidden state의 cosine. 모델 표현의 공통 방향 때문에 값이 전반적으로 높을 수 있다.
2. `02_centered_cosine_mid_vs_final.png`: 25개 장면 평균 벡터를 뺀 민감도 분석. 절대값보다 장면 관계가 유지되는지를 본다.
3. `03_final_minus_mid_cosine.png`: 끝단 cosine에서 중간층 cosine을 뺀 값. 빨강은 두 장면이 더 비슷해졌고 파랑은 덜 비슷해졌음을 뜻한다.
4. `04_decision_branch_logits.png`: 분기점에서 `TR`과 `ARR`의 raw logit 차이. 기존 5회 응답은 오른쪽에 함께 적었다.
5. `05_group_mean_cosine.png`: 같은/다른 family·기대 결정·기존 다수결별 평균 cosine 비교다.

## 해석 제한

- cosine이 높다고 모델이 같은 이유로 판단했다는 뜻은 아니다.
- 25개 장면뿐이고 family, 물체, 이미지 배치, geometry payload 길이가 함께 달라진다.
- 300개 장면쌍은 서로 독립인 300개 표본이 아니므로 유의성 검정을 하지 않았다.
- Raw hidden state는 anisotropy 때문에 cosine이 높게 몰릴 수 있어 mean-centered 결과를 함께 제시했다.
- hidden state는 Transformers에서 재추출했고 기존 응답은 vLLM constrained decoding 결과다. backend 수치 동일성이나 인과성을 주장하지 않는다.

## 산출물

- `latent_states.npz`: block 24와 final state `[25, 2048]`, branch logits `[25, 2]`.
- `records.json`, `extraction_manifest.json`: 입력·token 경계·hash·실행 환경 감사 정보.
- `cosine_*csv`: 네 종류 cosine matrix.
- `pairwise_cosine.csv`: 300개 장면쌍의 중간/끝단 cosine.
- `scene_latent_summary.csv`: 장면별 기존 응답과 분기 logits.
- `analysis_summary.json`: 기술 통계와 한계.
- `figures/`: PNG 5개.

## 재현

```bash
.qwen3-vl/venv/bin/python scripts/extract_l2_c2d8_decision_latent_v22.py --condition-key C2_D4 --output /home/ssu/ShelfScene/experiments/vlm_action_geometry_single_info_l2_c2d4_decision_latent_v23
.l1-latent/plot-venv/bin/python scripts/analyze_l2_c2d8_decision_latent_v22.py --experiment /home/ssu/ShelfScene/experiments/vlm_action_geometry_single_info_l2_c2d4_decision_latent_v23
```
