# L2 direct retrieval v16 — geometry conditions

25 identical numbered images × 12 conditions × 5 seeds = 1,500 independent calls. The v15 system prompt, task wording, model, sampling settings and output schema are unchanged; only the Geometry information block changes. No gripper data is supplied.

Geometry uses the original R/F/M definitions from maskfix_validation, matched by image hash and simulator/display-ID correspondence. Only visible-object records are supplied. R-Qualitative includes the original 5 cm clearance-zone categories. The single fully hidden object (lift_v04, ID 16) is excluded from payload records; measurements of visible objects are otherwise reused unchanged.

These are predictions. Scene-design agreement is descriptive agreement with FC_CLEAR→RETRIEVE_NOW and other scene families→REARRANGE_FIRST, NOT physical retrieval accuracy. In particular, FC_BLOCKED scenes were designed with finger-clearance assumptions that are not included in this prompt.

- Schema-valid calls: 1500/1500
- Infrastructure errors: 0
- C0 responses changed versus previous v15: 4/125

| Condition | Retrieve | Rearrange | FC_CLEAR retrieve /25 | FC_BLOCKED rearrange /25 | Other families rearrange /75 | Unanimous scenes /25 | Changed vs C0 /125 | Scene-design agreement /125 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 RGB only | 47 | 78 | 25 | 6 | 72 | 22 | 0 | 103 |
| C1 R Numeric | 39 | 86 | 20 | 12 | 69 | 17 | 16 | 101 |
| C2_D4 R Qualitative D4 | 41 | 84 | 25 | 9 | 75 | 23 | 6 | 109 |
| C2_D8 R Qualitative D8 | 31 | 94 | 24 | 18 | 75 | 23 | 16 | 117 |
| C3_D4 F Numeric D4 | 57 | 68 | 22 | 3 | 62 | 14 | 22 | 87 |
| C3_D8 F Numeric D8 | 56 | 69 | 25 | 4 | 65 | 17 | 15 | 94 |
| C4_D4 F Qualitative D4 | 33 | 92 | 16 | 12 | 71 | 16 | 16 | 99 |
| C4_D8 F Qualitative D8 | 33 | 92 | 17 | 11 | 73 | 18 | 20 | 101 |
| C5_D4 M Numeric D4 | 38 | 87 | 18 | 17 | 63 | 16 | 27 | 98 |
| C5_D8 M Numeric D8 | 49 | 76 | 23 | 6 | 68 | 17 | 14 | 97 |
| C6_D4 M Qualitative D4 | 36 | 89 | 19 | 12 | 71 | 18 | 21 | 102 |
| C6_D8 M Qualitative D8 | 45 | 80 | 22 | 4 | 73 | 21 | 14 | 99 |

## Per-scene RETRIEVE_NOW counts (out of five seeds)

| Scene | Target | C0 | C1 | C2_D4 | C2_D8 | C3_D4 | C3_D8 | C4_D4 | C4_D8 | C5_D4 | C5_D8 | C6_D4 | C6_D8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_fc_blocked_v01 | 64 | 5 | 4 | 5 | 0 | 3 | 3 | 2 | 4 | 2 | 5 | 0 | 5 |
| scene_fc_blocked_v02 | 53 | 4 | 4 | 4 | 2 | 4 | 3 | 3 | 3 | 1 | 5 | 2 | 2 |
| scene_fc_blocked_v03 | 12 | 5 | 3 | 5 | 5 | 5 | 5 | 3 | 4 | 5 | 5 | 5 | 5 |
| scene_fc_blocked_v04 | 89 | 5 | 2 | 2 | 0 | 5 | 5 | 5 | 0 | 0 | 2 | 3 | 5 |
| scene_fc_blocked_v05 | 59 | 0 | 0 | 0 | 0 | 5 | 5 | 0 | 3 | 0 | 2 | 3 | 4 |
| scene_fc_clear_v01 | 53 | 5 | 2 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_clear_v02 | 29 | 5 | 5 | 5 | 5 | 4 | 5 | 3 | 0 | 3 | 5 | 5 | 5 |
| scene_fc_clear_v03 | 95 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 5 |
| scene_fc_clear_v04 | 61 | 5 | 4 | 5 | 5 | 5 | 5 | 2 | 3 | 5 | 5 | 2 | 5 |
| scene_fc_clear_v05 | 62 | 5 | 4 | 5 | 4 | 3 | 5 | 2 | 4 | 1 | 3 | 2 | 2 |
| scene_lift_and_relocate_v01 | 59 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| scene_lift_and_relocate_v02 | 70 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v03 | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v04 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v05 | 43 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_rotate_v01 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_rotate_v02 | 38 | 0 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 4 | 2 | 0 | 0 |
| scene_rotate_v03 | 83 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_rotate_v04 | 47 | 1 | 0 | 0 | 0 | 2 | 2 | 2 | 0 | 2 | 2 | 0 | 0 |
| scene_rotate_v05 | 44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_translate_v01 | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_translate_v02 | 45 | 2 | 5 | 0 | 0 | 1 | 2 | 2 | 2 | 4 | 1 | 2 | 0 |
| scene_translate_v03 | 18 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 1 | 0 | 2 |
| scene_translate_v04 | 13 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_translate_v05 | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
