# Reproduction code snapshot

`scripts/`에는 L3 generation, 결과 집계와 C2-D4 reason latent 분석에 사용한 Python script를 보존합니다.

| Script | 역할 |
|---|---|
| `run_l3_maskfix_v17.py` | v17 single-causal-blocker generation 및 평가 |
| `run_l3_all_blockers_v18.py` | v18 all-visible-blockers generation 및 평가 |
| `run_l3_primary_blocker_v19.py` | v19 primary-blocker generation 및 평가 |
| `analyze_l3_primary_blocker_v19_paired.py` | v19 paired comparison 집계 |
| `analyze_l3_primary_blocker_v19_detailed.py` | v19 confusion·seed stability 상세 분석 |
| `analyze_l3_v19_c2_d4_reason.py` | v19 C2-D4 reason 결과 분석 |
| `extract_l3_c2_d4_reason_latent_v1.py` | 25 scenes의 block별 reason readout hidden state 추출 |
| `analyze_l3_c2_d4_reason_latent_v1.py` | cosine, nearest-neighbor, layerwise separation 및 figure 생성 |

이 폴더는 source project의 code snapshot이며 standalone package가 아닙니다. Generation 재실행에는 `/home/ssu/ShelfScene`의 공통 helper, source image/geometry artifact, Python environment, Qwen checkpoint와 vLLM runtime이 필요합니다. Latent extraction에는 같은 checkpoint를 로드할 수 있는 Transformers/PyTorch 환경과 source inputs가 필요합니다.

이미 보존된 결과는 각 study의 raw logs와 result files로 검산할 수 있습니다. Latent 분석은 보존된 `hidden_states.npz`와 `records.json`을 입력으로 다시 실행할 수 있으며, source artifact hash는 `artifact_hashes.json`에서 확인합니다.

`build_archive_manifest.py`는 이 Git archive 전용 도구입니다. 실험 폴더 root에서 실행하면 `audit/archive_manifest.csv`를 deterministic하게 다시 생성합니다.

```bash
python reproduction/build_archive_manifest.py
```
