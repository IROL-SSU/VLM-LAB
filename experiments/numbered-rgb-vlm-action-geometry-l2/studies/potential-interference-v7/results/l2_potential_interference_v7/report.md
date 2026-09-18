# L2 potential-grasp-interference v7

## Experimental definition

The task prompt contains no numeric threshold. Ground truth treats `occlusion_ratio >= 0.05` as meaningful interference and ignores smaller render-level overlaps.

## Integrity

- Completed: 1500/1500
- Parse errors: 0
- Post-validation failures: 0
- Infrastructure errors: 0

## Overall on the same relaxed GT

| Formulation | Accuracy | Balanced accuracy | Interference recall | No-interference specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| Potential grasp interference v7 | 52.9% | 60.5% | 22.3% | 98.7% | 201/699/592/8 |
| Direct occlusion v5, rescored | 63.7% | 69.8% | 39.6% | 100.0% | 356/544/600/0 |
| Target access v6, rescored | 48.0% | 56.7% | 13.3% | 100.0% | 120/780/600/0 |
| Original L1 non-FULL, rescored | 70.9% | 75.7% | 51.4% | 100.0% | 463/437/600/0 |

## V7 by condition

| Condition | Accuracy | Balanced accuracy | Interference recall | No-interference specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 58.4% | 65.3% | 30.7% | 100.0% | 23/52/50/0 |
| C1  R Numeric | 44.0% | 52.7% | 9.3% | 96.0% | 7/68/48/2 |
| C2 D4 R Qualitative D4 | 65.6% | 71.3% | 42.7% | 100.0% | 32/43/50/0 |
| C2 D8 R Qualitative D8 | 64.8% | 69.0% | 48.0% | 90.0% | 36/39/45/5 |
| C3 D4 F Numeric D4 | 48.8% | 57.3% | 14.7% | 100.0% | 11/64/50/0 |
| C3 D8 F Numeric D8 | 48.0% | 56.7% | 13.3% | 100.0% | 10/65/50/0 |
| C4 D4 F Qualitative D4 | 46.4% | 55.3% | 10.7% | 100.0% | 8/67/50/0 |
| C4 D8 F Qualitative D8 | 44.8% | 54.0% | 8.0% | 100.0% | 6/69/50/0 |
| C5 D4 M Numeric D4 | 52.0% | 60.0% | 20.0% | 100.0% | 15/60/50/0 |
| C5 D8 M Numeric D8 | 49.6% | 57.7% | 17.3% | 98.0% | 13/62/49/1 |
| C6 D4 M Qualitative D4 | 56.8% | 64.0% | 28.0% | 100.0% | 21/54/50/0 |
| C6 D8 M Qualitative D8 | 55.2% | 62.7% | 25.3% | 100.0% | 19/56/50/0 |

## Accuracy by scene family

| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 48.0% | 20.0% | 24.0% | 100.0% | 100.0% |
| C1  R Numeric | 16.0% | 0.0% | 12.0% | 100.0% | 92.0% |
| C2 D4 R Qualitative D4 | 36.0% | 44.0% | 48.0% | 100.0% | 100.0% |
| C2 D8 R Qualitative D8 | 40.0% | 52.0% | 52.0% | 100.0% | 80.0% |
| C3 D4 F Numeric D4 | 36.0% | 0.0% | 8.0% | 100.0% | 100.0% |
| C3 D8 F Numeric D8 | 32.0% | 0.0% | 8.0% | 100.0% | 100.0% |
| C4 D4 F Qualitative D4 | 8.0% | 8.0% | 16.0% | 100.0% | 100.0% |
| C4 D8 F Qualitative D8 | 16.0% | 0.0% | 8.0% | 100.0% | 100.0% |
| C5 D4 M Numeric D4 | 28.0% | 8.0% | 24.0% | 100.0% | 100.0% |
| C5 D8 M Numeric D8 | 32.0% | 4.0% | 16.0% | 100.0% | 96.0% |
| C6 D4 M Qualitative D4 | 48.0% | 16.0% | 20.0% | 100.0% | 100.0% |
| C6 D8 M Qualitative D8 | 36.0% | 16.0% | 24.0% | 100.0% | 100.0% |

## Interpretation

- V7 improves balanced accuracy by +3.8 points over v6, but remains -9.3 points below v5 on the relaxed GT.
- V7 detects 201/900 interference runs and makes 8 false-interference decisions.
- Relaxing the GT correctly removes the 0.417% edge overlap, but the grasp-interference wording still asks for stronger physical evidence than direct visual occlusion and therefore lowers recall.
- V7 has identical observable inputs to v5 in 1500/1500 calls and to v6 in 1500/1500 calls.
