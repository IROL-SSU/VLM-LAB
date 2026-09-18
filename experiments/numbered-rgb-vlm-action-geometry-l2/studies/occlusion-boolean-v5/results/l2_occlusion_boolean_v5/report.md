# L2 direct-occlusion Boolean v5

## Definition

`occluded_by_other_object=true` iff another physical object covers any part of the target-only projected silhouette. Proximity and grasp constraints are ignored.

## Prompt controls

- System prompt is supplied as a separate system-role message.
- No examples are present in the task prompt.
- `direct_graspable` is absent from both the task prompt and output schema.

## Integrity

- Completed: 1500/1500
- Parse errors: 0
- Post-validation failures: 0
- Infrastructure errors: 0

## Overall on the same v5 GT

| Output formulation | Accuracy | Balanced accuracy | Occlusion recall | No-occlusion specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| Direct occlusion Boolean v5 | 59.7% | 68.5% | 37.1% | 100.0% | 356/604/540/0 |
| Visibility-only L2 v4, inverted | 44.0% | 56.2% | 12.5% | 100.0% | 120/840/540/0 |
| Grasp-zone L2 v3, inverted | 67.3% | 74.4% | 49.3% | 99.4% | 473/487/537/3 |
| Original L1, non-FULL→true | 66.9% | 74.1% | 48.2% | 100.0% | 463/497/540/0 |

## Direct occlusion v5 by condition

| Condition | Accuracy | Balanced accuracy | Occlusion recall | No-occlusion specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 78.4% | 83.1% | 66.2% | 100.0% | 53/27/45/0 |
| C1  R Numeric | 58.4% | 67.5% | 35.0% | 100.0% | 28/52/45/0 |
| C2 D4 R Qualitative D4 | 72.8% | 78.8% | 57.5% | 100.0% | 46/34/45/0 |
| C2 D8 R Qualitative D8 | 74.4% | 80.0% | 60.0% | 100.0% | 48/32/45/0 |
| C3 D4 F Numeric D4 | 47.2% | 58.8% | 17.5% | 100.0% | 14/66/45/0 |
| C3 D8 F Numeric D8 | 56.0% | 65.6% | 31.2% | 100.0% | 25/55/45/0 |
| C4 D4 F Qualitative D4 | 53.6% | 63.7% | 27.5% | 100.0% | 22/58/45/0 |
| C4 D8 F Qualitative D8 | 50.4% | 61.3% | 22.5% | 100.0% | 18/62/45/0 |
| C5 D4 M Numeric D4 | 55.2% | 65.0% | 30.0% | 100.0% | 24/56/45/0 |
| C5 D8 M Numeric D8 | 54.4% | 64.4% | 28.7% | 100.0% | 23/57/45/0 |
| C6 D4 M Qualitative D4 | 59.2% | 68.1% | 36.2% | 100.0% | 29/51/45/0 |
| C6 D8 M Qualitative D8 | 56.8% | 66.2% | 32.5% | 100.0% | 26/54/45/0 |

## Accuracy by scene family

| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 84.0% | 56.0% | 72.0% | 100.0% | 80.0% |
| C1  R Numeric | 60.0% | 24.0% | 28.0% | 100.0% | 80.0% |
| C2 D4 R Qualitative D4 | 84.0% | 40.0% | 60.0% | 100.0% | 80.0% |
| C2 D8 R Qualitative D8 | 76.0% | 72.0% | 44.0% | 100.0% | 80.0% |
| C3 D4 F Numeric D4 | 40.0% | 8.0% | 8.0% | 100.0% | 80.0% |
| C3 D8 F Numeric D8 | 68.0% | 16.0% | 16.0% | 100.0% | 80.0% |
| C4 D4 F Qualitative D4 | 52.0% | 8.0% | 28.0% | 100.0% | 80.0% |
| C4 D8 F Qualitative D8 | 44.0% | 0.0% | 28.0% | 100.0% | 80.0% |
| C5 D4 M Numeric D4 | 52.0% | 16.0% | 28.0% | 100.0% | 80.0% |
| C5 D8 M Numeric D8 | 52.0% | 16.0% | 24.0% | 100.0% | 80.0% |
| C6 D4 M Qualitative D4 | 72.0% | 16.0% | 28.0% | 100.0% | 80.0% |
| C6 D8 M Qualitative D8 | 56.0% | 20.0% | 28.0% | 100.0% | 80.0% |

## Interpretation

- Changing the task/output formulation (direct occlusion field and no examples) changes balanced accuracy by +12.3 percentage points versus v4 on the same observable inputs and GT.
- V5 makes 604 false-negative occlusion decisions and 0 false-positive occlusion decisions.
- All 1500 v4/v5 calls have identical numbered RGB, geometry payload, target ID, condition, and seed; the intended experimental change is prompt/output formulation.
