# N3 24° downward / 90 cm — 120-scene input ablation

- Three scene families × 40 scenes
- Four full-order prompts × five seeds
- Three input conditions; 7,200 calls total
- Cell format: Full-order Exact / All-pairs Accuracy

## Full-order matrix

| Scene family | Prompt | Numbered RGB | RGB + Gray | RGB + Numbered Gray |
|---|---|---:|---:|---:|
| 분리 비가림 | Minimal direct | 30.000 / 76.667% | 37.000 / 74.778% | 37.500 / 76.111% |
| 분리 비가림 | Detailed direct | 46.000 / 85.667% | 45.000 / 80.111% | 43.000 / 78.889% |
| 분리 비가림 | Contact point | 40.500 / 82.778% | 39.500 / 77.889% | 42.500 / 78.000% |
| 분리 비가림 | Explicit all pairs | 24.000 / 68.556% | 18.000 / 63.556% | 18.000 / 61.444% |
| 밀집 비가림 | Minimal direct | 56.500 / 85.667% | 60.000 / 89.000% | 59.500 / 87.778% |
| 밀집 비가림 | Detailed direct | 74.500 / 94.111% | 65.000 / 91.000% | 59.000 / 89.444% |
| 밀집 비가림 | Contact point | 79.500 / 95.000% | 63.000 / 90.556% | 59.000 / 88.556% |
| 밀집 비가림 | Explicit all pairs | 41.500 / 75.667% | 27.000 / 68.556% | 28.500 / 66.667% |
| 구조적 가림 | Minimal direct | 85.500 / 94.889% | 84.000 / 94.556% | 86.000 / 95.778% |
| 구조적 가림 | Detailed direct | 84.500 / 96.556% | 83.000 / 95.222% | 86.500 / 96.333% |
| 구조적 가림 | Contact point | 87.500 / 97.111% | 83.000 / 95.333% | 88.500 / 97.222% |
| 구조적 가림 | Explicit all pairs | 50.500 / 79.000% | 42.000 / 72.333% | 38.500 / 66.778% |

## Pooled

| Input | Full-order Exact | All-pairs | Calls |
|---|---:|---:|---:|
| Numbered RGB | 58.375% | 85.972% | 2400 |
| RGB + Gray | 53.875% | 82.741% | 2400 |
| RGB + Numbered Gray | 53.875% | 81.917% | 2400 |

## Audit

- Every cell: 40 scenes × 5 seeds = 200 calls
- Scene, object geometry, IDs, GT, and RGB hashes match across inputs
- Camera-only change from the 16° source geometry
- Format failures and non-stop completions are recorded in audit/final_audit.json
