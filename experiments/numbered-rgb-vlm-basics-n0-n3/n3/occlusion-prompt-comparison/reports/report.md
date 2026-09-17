# N3 prompt comparison — numbered_rgb

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Design: 20 scenes × 5 prompts × 5 seeds = 500 calls
- Input: one numbered RGB image
- Forced JSON; temperature 0.3; batch size 1

## Result — complete order

| Scene type | Prompt | Full-order exact | All-pairs accuracy | 5/5 scenes | 0/5 scenes |
| --- | --- | --- | --- | --- | --- |
| Structured occlusion | Minimal direct | 80.000% | 95.111% | 16 | 4 |
| Structured occlusion | Detailed direct | 78.000% | 94.889% | 14 | 3 |
| Structured occlusion | Contact point | 77.000% | 93.778% | 13 | 4 |
| Structured occlusion | Explicit all pairs | 10.000% | 54.889% | 2 | 18 |

## Output compliance — complete-order prompts

| Scene type | Prompt | visible_ids exact | Response structure valid | JSON valid |
| --- | --- | --- | --- | --- |
| Structured occlusion | Minimal direct | 95.000% | 100.000% | 100.000% |
| Structured occlusion | Detailed direct | 100.000% | 100.000% | 100.000% |
| Structured occlusion | Contact point | 100.000% | 100.000% | 100.000% |
| Structured occlusion | Explicit all pairs | 100.000% | 38.000% | 100.000% |

## Result — same fixed target

| Scene type | Prompt | Relation accuracy | All relations exact |
| --- | --- | --- | --- |
| Structured occlusion | Minimal direct | 96.000% | 90.000% |
| Structured occlusion | Detailed direct | 94.400% | 86.000% |
| Structured occlusion | Contact point | 92.800% | 82.000% |
| Structured occlusion | Explicit all pairs | 54.800% | 23.000% |
| Structured occlusion | Single target | 78.000% | 50.000% |

## Metric meaning

- Full-order exact: the entire closest-to-farthest ID array must match; one inversion makes the call fail.
- All-pairs accuracy: among every unordered object pair, the predicted closer object is correct.
- Target relation accuracy: for the same fixed target, each other object is correctly labeled front/behind.
- All relations exact: every relation to that target must be correct in the call.

## Audit

- Calls/run IDs: 500/500
- Batch size 1: True
- Images per prompt / fixed order: 1 / True
- All finish reason `stop`: True
- Format failures: 0
- Maximum input/output tokens: 1850/226
- Maximum context with full output cap: 2874 / 8192
