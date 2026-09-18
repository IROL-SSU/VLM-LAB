# L2 single-image obstruction score v13

## Setup

- Scene: `scene_lift_and_relocate_v05`
- Input: numbered RGB only
- Target: object 43
- Candidates: 52, 48, 15, 76, 24
- Model: Qwen3-VL-30B-A3B-Instruct
- Five seeds per candidate, 25 calls total
- No geometry payload
- No action-level scalar GT; this experiment compares model score distributions and ranking

## Prompt

```text
The robot must grasp target object 43.

Score how much removing object {candidate_id} would improve the robot's ability
to grasp the target.

Use an integer score from 0 to 10:
- 0: Removing it would not help.
- 5: Removing it would moderately improve grasp access.
- 10: It must be removed before the target can be grasped.

Return only:
{"obstruction_score": <integer>}
```

## Results

| Object | Seed scores | Mean | Median | Range | Population SD |
|---:|---|---:|---:|---:|---:|
| 52 | 0, 0, 0, 0, 0 | 0.0 | 0 | 0–0 | 0.0 |
| 48 | 0, 0, 0, 0, 0 | 0.0 | 0 | 0–0 | 0.0 |
| 15 | 5, 5, 5, 5, 5 | 5.0 | 5 | 5–5 | 0.0 |
| 76 | 5, 5, 5, 5, 5 | 5.0 | 5 | 5–5 | 0.0 |
| 24 | 5, 5, 5, 5, 5 | 5.0 | 5 | 5–5 | 0.0 |

All 25 calls passed parsing and schema validation.

## Interpretation

The score formulation separated the two left-side objects (52 and 48) from the center/right objects (15, 76, and 24). It did not rank the three predicted obstructions: all received exactly 5. The model copied the explicitly defined anchor values 0 and 5 and never used an intermediate value or 10. Consequently, this prompt behaves more like a coarse three-level classifier than a calibrated 0–10 scorer.

Object 15 is the clear visual blocker of target 43, but it received the same score as objects 76 and 24. The current output therefore cannot yet identify which object should be removed first.
