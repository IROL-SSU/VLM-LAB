# L2 target-access wording v6

## Experimental scope

V6 uses the same pixel-exact Isaac occlusion GT as v5. It is a prompt/output wording ablation, not a new 3D robot approach-corridor oracle.

## Prompt controls

- System prompt is a separate system-role message.
- The task prompt contains no examples and none of the removed exclusion sentences.
- Output is only `target_access_blocked`.

## Integrity

- Completed: 1500/1500
- Parse errors: 0
- Post-validation failures: 0
- Infrastructure errors: 0

## Overall on the same GT

| Formulation | Accuracy | Balanced accuracy | Blocked recall | Unblocked specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| Target access v6 | 44.0% | 56.2% | 12.5% | 100.0% | 120/840/540/0 |
| Direct occlusion v5 | 59.7% | 68.5% | 37.1% | 100.0% | 356/604/540/0 |
| Visibility-only v4, inverted | 44.0% | 56.2% | 12.5% | 100.0% | 120/840/540/0 |
| Original L1, non-FULL→true | 66.9% | 74.1% | 48.2% | 100.0% | 463/497/540/0 |

## Target access v6 by condition

| Condition | Accuracy | Balanced accuracy | Blocked recall | Unblocked specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 44.0% | 56.2% | 12.5% | 100.0% | 10/70/45/0 |
| C1  R Numeric | 41.6% | 54.4% | 8.8% | 100.0% | 7/73/45/0 |
| C2 D4 R Qualitative D4 | 56.8% | 66.2% | 32.5% | 100.0% | 26/54/45/0 |
| C2 D8 R Qualitative D8 | 59.2% | 68.1% | 36.2% | 100.0% | 29/51/45/0 |
| C3 D4 F Numeric D4 | 39.2% | 52.5% | 5.0% | 100.0% | 4/76/45/0 |
| C3 D8 F Numeric D8 | 36.0% | 50.0% | 0.0% | 100.0% | 0/80/45/0 |
| C4 D4 F Qualitative D4 | 45.6% | 57.5% | 15.0% | 100.0% | 12/68/45/0 |
| C4 D8 F Qualitative D8 | 43.2% | 55.6% | 11.2% | 100.0% | 9/71/45/0 |
| C5 D4 M Numeric D4 | 40.0% | 53.1% | 6.2% | 100.0% | 5/75/45/0 |
| C5 D8 M Numeric D8 | 38.4% | 51.9% | 3.8% | 100.0% | 3/77/45/0 |
| C6 D4 M Qualitative D4 | 44.8% | 56.9% | 13.8% | 100.0% | 11/69/45/0 |
| C6 D8 M Qualitative D8 | 39.2% | 52.5% | 5.0% | 100.0% | 4/76/45/0 |

## Accuracy by scene family

| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 24.0% | 12.0% | 4.0% | 100.0% | 80.0% |
| C1  R Numeric | 12.0% | 16.0% | 0.0% | 100.0% | 80.0% |
| C2 D4 R Qualitative D4 | 36.0% | 52.0% | 16.0% | 100.0% | 80.0% |
| C2 D8 R Qualitative D8 | 32.0% | 48.0% | 36.0% | 100.0% | 80.0% |
| C3 D4 F Numeric D4 | 12.0% | 4.0% | 0.0% | 100.0% | 80.0% |
| C3 D8 F Numeric D8 | 0.0% | 0.0% | 0.0% | 100.0% | 80.0% |
| C4 D4 F Qualitative D4 | 16.0% | 16.0% | 16.0% | 100.0% | 80.0% |
| C4 D8 F Qualitative D8 | 12.0% | 12.0% | 12.0% | 100.0% | 80.0% |
| C5 D4 M Numeric D4 | 12.0% | 8.0% | 0.0% | 100.0% | 80.0% |
| C5 D8 M Numeric D8 | 8.0% | 0.0% | 4.0% | 100.0% | 80.0% |
| C6 D4 M Qualitative D4 | 12.0% | 16.0% | 16.0% | 100.0% | 80.0% |
| C6 D8 M Qualitative D8 | 4.0% | 12.0% | 0.0% | 100.0% | 80.0% |

## Interpretation

- Relative to v5, accuracy changes by -15.7 points and balanced accuracy by -12.3 points.
- V6 detects 120/960 blocked runs and produces 0 false blocked decisions.
- The wording asks whether an object prevents access until it is moved. Partial visual occlusion does not necessarily imply that stronger condition, so the wording and frozen pixel-occlusion GT are semantically misaligned.
- All 1500 v5/v6 calls have identical numbered RGB, geometry, target ID, condition, and seed.
