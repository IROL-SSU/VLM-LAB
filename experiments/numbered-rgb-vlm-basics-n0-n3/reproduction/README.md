# Reproduction code snapshot

`scripts/`는 N0–N3 실험 당시 `/home/ssu/ShelfScene/scripts`에서 사용한 setup, inference, analysis 코드의 snapshot입니다.

주의사항:

- 스크립트의 기본 경로는 원본 `ShelfScene` repository layout을 가정합니다.
- 이 archive에는 전체 장면 이미지, USD와 model checkpoint가 없으므로 그대로 실행하기 전에 경로를 조정해야 합니다.
- config, prompt, schema와 raw logs는 각 단계 폴더에 보존되어 있습니다.
- 일부 N2/N3 실행은 당시 공통 runner를 재사용했습니다. 이 폴더는 실행 환경 전체를 vendoring한 독립 패키지가 아니라 실험 provenance용 코드 snapshot입니다.
- 분석을 다시 수행할 때 원시 로그를 덮어쓰지 말고 새 `results/` 또는 별도 experiment version에 출력하십시오.

주요 대응 관계:

| 단계 | Setup | Inference | Analysis |
|---|---|---|---|
| N0 | `setup_b0_n0_sparse_ids_text_vs_json.py` | 당시 공통 vLLM 실행 흐름 | `analyze_b0_n0_sparse_ids_text_vs_json.py` |
| N1 | `setup_b0_n1_explicit_visible_ids_ablation.py` | `run_b0_n1_explicit_visible_ids_ablation.py` | `analyze_b0_n1_explicit_visible_ids_ablation.py` |
| N2 | `setup_b0_n2_*` | 당시 공통 structured-output 실행 흐름 | `analyze_b0_n2_*` |
| N3 | `setup_n3_*` | `run_n3_d0_ltr_d1_depth_75scenes.py`, `run_n3_rgbd_prompt_methods.py` | `analyze_n3_*` |
