# C2 D4 pre-reason latent cosine analysis

## Design

- 25 scenes, one deterministic point per scene; no answer generation and no sampling seed.
- Fixed assistant prefix: `{"blocking_reason": "`
- Readout: immediately before the first reason value token.
- Primary comparison: decoder block 24 vs final RMSNorm.

## Main result

### FC_CLEAR family (5 scenes)

- Block 24: within clear 0.999321; clear vs other 0.998419; gap 0.000902.
- Final: within clear 0.990699; clear vs other 0.985283; gap 0.005416.
- Gap change, final minus block 24: 0.004513.

### Semantic clear / GT NONE (4 scenes)

- Block 24: within clear 0.999363; clear vs other 0.998448; gap 0.000915.
- Final: within clear 0.997376; clear vs other 0.984311; gap 0.013065.
- Gap change, final minus block 24: 0.012151.

## Reproducibility and limits

- Hidden state archive SHA-256: `fef78c125e2db1017b180d71a48e9343b3346a4df6acf1bf51869285aa92fc43`
- The five historical v19 seeds are metadata only; they do not create five latent points.
- This is descriptive representation analysis, not proof of the model's causal reasoning.
