# L2 simple obstruction score v14

## Setup

- Scene: `scene_lift_and_relocate_v05`
- Input: numbered RGB only
- Target: object 43
- Candidates: 52, 48, 15, 76, 24
- Model: Qwen3-VL-30B-A3B-Instruct
- Five seeds per candidate, 25 calls total
- No geometry payload and no score anchors or examples
- No action-level scalar GT; scores are analyzed descriptively

## Prompt

```text
Target object to grasp: object 43.
Candidate obstacle to evaluate: object {candidate_id}.

Score how much removing the candidate obstacle would improve the robot's
ability to grasp the target object.

Return an integer obstruction score from 0 to 10.
```

The JSON schema constrained only the output field and integer range.

## Results

Scores are listed in seed order 28101, 28102, 28103, 28104, 28105.

| Rank by mean | Object | Scores | Mean | Median | Range | Population SD |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 15 | 5, 5, 3, 2, 2 | 3.4 | 3 | 2–5 | 1.356 |
| 2= | 48 | 5, 0, 1, 2, 2 | 2.0 | 2 | 0–5 | 1.673 |
| 2= | 24 | 5, 0, 1, 2, 2 | 2.0 | 2 | 0–5 | 1.673 |
| 4 | 76 | 3, 1, 1, 2, 2 | 1.8 | 2 | 1–3 | 0.748 |
| 5 | 52 | 0, 0, 1, 0, 0 | 0.2 | 0 | 0–1 | 0.400 |

All 25 calls passed parsing and schema validation.

## Comparison with v13

| Object | v13 mean with 0/5/10 anchors | v14 mean without anchors |
|---:|---:|---:|
| 52 | 0.0 | 0.2 |
| 48 | 0.0 | 2.0 |
| 15 | 5.0 | 3.4 |
| 76 | 5.0 | 1.8 |
| 24 | 5.0 | 2.0 |

Removing the anchor definitions eliminated the previous 0-or-5 collapse. The model used 0, 1, 2, 3, and 5 and ranked object 15 highest on mean and median. However, the absolute scores remain seed-sensitive, especially for objects 48 and 24. A single call is therefore not reliable enough for removal selection; aggregation across seeds or a joint relative-ranking prompt is needed.
