# N3 depth-input ablation — complete 40-scene comparison

The prior 20-scene occlusion restriction was historical: the first RGB-only prompt comparison stopped at 20 scenes, while both depth-input experiments already contained all 40. The missing 20 RGB-only scenes have now been run.

- Three scene families × 40 scenes each
- Three input conditions
- Four full-order prompts; five fixed-target comparison methods
- Five seeds per scene/method/input

## Pooled 40-scene results

| Input | Full-order exact | All-pairs | Target exact | Target relations |
|---|---:|---:|---:|---:|
| Numbered RGB | 44.292% | 78.583% | 58.633% | 78.907% |
| RGB + Gray | 45.208% | 78.204% | 59.133% | 78.253% |
| RGB + Numbered Gray | 45.542% | 77.852% | 58.467% | 77.427% |

## Full-order: Exact / All-pairs

| Scene family | Prompt | Numbered RGB | RGB + Gray | RGB + Numbered Gray |
|---|---|---:|---:|---:|
| Separated non-occlusion | Minimal direct | 22.000 / 70.333% | 24.000 / 68.222% | 26.000 / 66.778% |
| Separated non-occlusion | Detailed direct | 30.000 / 75.889% | 28.000 / 70.444% | 32.500 / 71.444% |
| Separated non-occlusion | Contact point | 26.000 / 73.667% | 25.500 / 69.333% | 27.500 / 68.889% |
| Separated non-occlusion | Explicit all pairs | 20.000 / 64.667% | 16.000 / 57.222% | 16.000 / 58.667% |
| Packed non-occlusion | Minimal direct | 46.500 / 78.111% | 47.000 / 82.111% | 49.000 / 83.222% |
| Packed non-occlusion | Detailed direct | 49.500 / 86.000% | 56.500 / 88.444% | 49.500 / 85.222% |
| Packed non-occlusion | Contact point | 54.000 / 84.667% | 46.000 / 83.333% | 52.000 / 83.778% |
| Packed non-occlusion | Explicit all pairs | 37.000 / 72.556% | 24.000 / 67.000% | 28.000 / 69.000% |
| Structured occlusion | Minimal direct | 76.500 / 94.111% | 79.000 / 94.111% | 77.000 / 93.000% |
| Structured occlusion | Detailed direct | 80.000 / 95.333% | 79.000 / 94.222% | 77.000 / 94.222% |
| Structured occlusion | Contact point | 76.000 / 93.556% | 79.500 / 94.000% | 79.000 / 94.667% |
| Structured occlusion | Explicit all pairs | 14.000 / 54.111% | 38.000 / 70.000% | 33.000 / 65.333% |

## Same fixed target: Partition exact / Relation accuracy

| Scene family | Prompt | Numbered RGB | RGB + Gray | RGB + Numbered Gray |
|---|---|---:|---:|---:|
| Separated non-occlusion | Minimal direct | 46.000 / 72.200% | 41.000 / 66.400% | 45.000 / 65.600% |
| Separated non-occlusion | Detailed direct | 56.500 / 77.000% | 47.500 / 70.000% | 50.500 / 71.200% |
| Separated non-occlusion | Contact point | 50.500 / 74.600% | 50.500 / 71.000% | 46.500 / 68.200% |
| Separated non-occlusion | Explicit all pairs | 33.000 / 63.400% | 25.500 / 54.600% | 25.500 / 57.200% |
| Separated non-occlusion | Single target | 62.500 / 83.200% | 60.500 / 77.200% | 57.000 / 72.800% |
| Packed non-occlusion | Minimal direct | 58.000 / 77.000% | 58.000 / 80.600% | 56.000 / 80.000% |
| Packed non-occlusion | Detailed direct | 58.000 / 82.000% | 67.000 / 86.400% | 57.000 / 81.000% |
| Packed non-occlusion | Contact point | 64.000 / 82.800% | 59.500 / 81.400% | 57.500 / 80.000% |
| Packed non-occlusion | Explicit all pairs | 45.000 / 73.000% | 34.500 / 67.800% | 38.500 / 70.200% |
| Packed non-occlusion | Single target | 49.000 / 73.400% | 60.500 / 79.600% | 57.500 / 77.400% |
| Structured occlusion | Minimal direct | 90.000 / 96.000% | 88.500 / 94.600% | 87.500 / 93.800% |
| Structured occlusion | Detailed direct | 90.000 / 96.000% | 86.000 / 93.600% | 88.000 / 95.200% |
| Structured occlusion | Contact point | 85.500 / 94.200% | 87.500 / 94.200% | 89.500 / 95.600% |
| Structured occlusion | Explicit all pairs | 34.000 / 57.800% | 58.500 / 72.600% | 52.000 / 67.200% |
| Structured occlusion | Single target | 57.500 / 81.000% | 62.000 / 83.800% | 69.000 / 86.000% |

## Audit

- Missing RGB-only calls: 500/500
- Every table cell has 40 scenes × 5 seeds: True
- Scene/GT/target signatures match: True
- Format failures across source records: 0
