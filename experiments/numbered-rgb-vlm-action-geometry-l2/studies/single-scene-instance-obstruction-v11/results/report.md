# L2 single-scene instance obstruction v11

## Input

- Scene: `scene_lift_and_relocate_v05`
- Target ID: `43`
- Input condition: C0 Numbered RGB only
- Model: Qwen3-VL-30B-A3B-Instruct
- Seeds: 28101, 28102, 28103, 28104, 28105

## Prompt

System:

```text
You are a robot manipulation planner.
Return only valid JSON.
```

Task:

```text
The robot must grasp the target object.

For each numbered object other than the target, determine whether that object
obstructs grasping the target.

Return one assessment for every non-target numbered object, from left to right.
```

## Result

All five seeds returned the same assessments.

| Left-to-right order | Object ID | `obstructs_grasp=true` |
|---:|---:|---:|
| 1 | 52 | 5/5 |
| 2 | 48 | 5/5 |
| 3 | 15 | 5/5 |
| 4 | 76 | 5/5 |
| 5 | 24 | 5/5 |

All five responses passed JSON-schema and semantic validation.

## Interpretation

The model did not distinguish the direct blocker from other visible objects. Object 15 is the clear visual blocker of target 43, while object 52 is far to the left and is not a plausible direct blocker. Marking both as obstructing shows that the broad phrase `obstructs grasping` induces an over-inclusive judgment.

The saved GT in this focused experiment is only a visual-occlusion reference, not a robot-arm grasp-rollout label; it marks object 15 true and the others false.
