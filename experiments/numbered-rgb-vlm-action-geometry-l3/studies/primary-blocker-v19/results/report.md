# L3 strongest-primary-blocker rerun v19

25 scenes × 12 conditions × 5 seeds = 1,500 independent calls. The scalar blocker_id must be the visible causal object whose removal would most improve direct graspability.

Dataset limitation: every blocked scene has exactly one visible valid blocker ID, so ranking among multiple valid blockers is not directly evaluated.

Completion: 1500/1500; parse errors: 0; semantic validation failures: 0; infrastructure errors: 0

Overall reason: 514/1500 (34.3%); blocker: 1123/1500 (74.9%); joint: 512/1500 (34.1%).

| Condition | Reason /125 | Blocker /125 | Joint /125 | Joint accuracy |
|---|---:|---:|---:|---:|
| C0 | 50 | 95 | 48 | 38.4% |
| C1 | 30 | 85 | 30 | 24.0% |
| C2_D4 | 60 | 116 | 60 | 48.0% |
| C2_D8 | 72 | 116 | 72 | 57.6% |
| C3_D4 | 34 | 80 | 34 | 27.2% |
| C3_D8 | 22 | 74 | 22 | 17.6% |
| C4_D4 | 48 | 90 | 48 | 38.4% |
| C4_D8 | 45 | 91 | 45 | 36.0% |
| C5_D4 | 36 | 92 | 36 | 28.8% |
| C5_D8 | 38 | 96 | 38 | 30.4% |
| C6_D4 | 40 | 94 | 40 | 32.0% |
| C6_D8 | 39 | 94 | 39 | 31.2% |
