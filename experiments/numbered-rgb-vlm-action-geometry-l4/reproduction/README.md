# Reproduction code snapshot

`scripts/`에는 L4 v24–v26 준비·추론·replay·분석에 사용한 Python code를 보존합니다.

| Script | 역할 |
|---|---|
| `prepare_l4_action_full_from_v19_v24.py` | L3 v19 scene 입력으로 v24 matrix 준비 |
| `prepare_l4_blocker_conditioned_from_l3_v19_v25.py` | Frozen L3 v19 출력을 v25 입력에 결합 |
| `prepare_l4_action_only_from_v25_v26.py` | v25 입력으로 v26 action-only matrix 준비 |
| `run_vlm_action_geometry_single_info_v1.py` | v24–v26 generation 실행 공통 runner |
| `replay_vlm_action_geometry_single_info_v1.py` | v24·v25 Isaac Sim replay |
| `analyze_vlm_action_geometry_single_info_v1.py` | v24·v25 result 집계 |
| `analyze_l4_v25_c2_d8_failures.py` | v25 C2-D8 실패 taxonomy |
| `analyze_l4_action_only_v26.py` | v26 advisory action-label 집계와 paired v25 비교 |
| `plot_l4_*.py` | result figure 생성 |
| `vlm_action_geometry_v1_common.py` | 공통 scene/condition·hash·scoring helper |

이 코드는 source project의 snapshot이며 standalone package가 아닙니다. 원본 경로와 관련 Python dependency, Qwen checkpoint, 이미지·geometry 입력이 있어야 generation을 재실행할 수 있습니다. v24·v25 replay 재실행에는 Isaac Sim과 source scene asset도 필요합니다. v26은 출력 파라미터가 없어 replay 대상이 아닙니다.

`build_archive_manifest.py`는 Git archive 전용 도구입니다. 실험 폴더에서 다음 명령으로 `audit/archive_manifest.csv`를 deterministic하게 다시 생성할 수 있습니다.

```bash
python reproduction/build_archive_manifest.py
```
