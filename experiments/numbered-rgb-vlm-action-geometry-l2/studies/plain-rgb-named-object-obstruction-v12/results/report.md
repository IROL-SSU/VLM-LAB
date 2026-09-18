# L2 plain-RGB named-object obstruction v12

## Setup

- Image: plain RGB without numbered badges
- Target: the light blue cup partially visible immediately to the right of the Cheez-It box
- Model: Qwen3-VL-30B-A3B-Instruct
- Five independent seeds per candidate
- No object IDs or geometry information were included in the model prompt

## Prompt template

```text
The robot must grasp the light blue cup partially visible immediately to the right of the Cheez-It box.

Does {candidate_description} obstruct grasping the light blue cup?
Return the answer as obstructs_grasp=true or obstructs_grasp=false.
```

## Results

| Candidate | True | False | Majority judgment |
|---|---:|---:|---|
| Black mug on the left | 0/5 | 5/5 | Does not obstruct |
| Dark blue bottle partly behind the left side of the box | 4/5 | 1/5 | Obstructs |
| Cheez-It box in the center | 5/5 | 0/5 | Obstructs |
| Purple cup on the right | 5/5 | 0/5 | Obstructs |
| Brown mug behind the purple cup | 5/5 | 0/5 | Obstructs |

All 25 responses passed schema validation. The dark blue bottle was the only seed-sensitive candidate: seed 28103 returned false and the other four returned true.

## Interpretation

Asking about one named object at a time eliminated the previous all-true result for the distant black mug. The model consistently recognized the central Cheez-It box as an obstruction. It still treated both right-side vessels as grasp obstructions, suggesting that `obstructs grasping` includes perceived gripper-clearance or approach interference, not only visual occlusion.

The saved labels are visual-occlusion references only. They are not treated as definitive action-level correctness because no robot-arm grasp rollout was performed for these five named candidates.
