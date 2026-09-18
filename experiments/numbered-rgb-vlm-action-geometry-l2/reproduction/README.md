# Reproduction code snapshot

`scripts/`에는 L2 artifact를 생성·실행·분석할 때 사용한 Python script를 보존합니다.

- `prepare_vlm_action_geometry_single_info_v1.py`, `vlm_action_geometry_v1_common.py`: 공통 scene/condition 준비와 helper
- `run_vlm_action_geometry_single_info_v1.py`, `run_l2_*.py`: generation 실행
- `setup_l2_*.py`: v11–v14 diagnostic setup
- `analyze_l2_*.py`, `plot_l2_*.py`, `compare_l2_*.py`: 집계와 비교
- `extract_l2_c2d8_decision_latent_v22.py`, `analyze_l2_c2d8_decision_latent_v22.py`: v22·v23 latent extraction/analysis
- `test_vlm_action_geometry_v1_common.py`: 공통 helper test

이 폴더는 source project의 code snapshot이며 standalone package가 아닙니다. 실제 generation 재실행에는 `/home/ssu/ShelfScene`의 source image/geometry artifact, Python environment, Qwen checkpoint와 vLLM runtime이 필요합니다. 이미 보존된 로그에 대한 집계 검증은 각 script의 CLI help와 study config의 source path를 기준으로 실행합니다.

`build_archive_manifest.py`는 이 Git archive 전용 도구입니다. 실험 폴더 root에서 실행하면 `audit/archive_manifest.csv`를 deterministic하게 다시 생성합니다.

```bash
python reproduction/build_archive_manifest.py
```
