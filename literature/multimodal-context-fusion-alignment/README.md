# Multimodal Context Fusion / Alignment

[문헌 목록](../README.md) · [연구 개요](../../README.md) · [노션 원본](https://www.notion.so/6be98779ee7148b5b820590f18300785)

**Vision–Language와 F/T·tactile·proprioception처럼 통계적·의미적 성격이 다른 모달리티를 로봇 매니퓰레이션 모델이 어디서, 어떤 방식으로 연결·정렬·융합하는지 비교한다.**

정리 기준일: 2026-09-26. 노션의 5개 행과 상세 본문을 가져왔다. 기존 분류·보고 수치는 snapshot으로 보존하고, 현재 sub-goal 연구와의 연결은 논문별 첫 섹션에 구분했다.

## 읽는 관점

- **연결 위치:** encoder 내부, shared Transformer 입력, VLM 출력 이후 중 어디에서 정보를 교환하는가?
- **정렬 학습:** 행동 loss에 의존하는가, contrastive·generative·reconstruction 목표를 추가하는가?
- **융합·선택:** concat, attention, FiLM, gating, routing이 각각 어떤 역할인가?
- **현재 목표와의 거리:** 저수준 행동 개선이 접촉 상태 이해나 sub-goal 생성 능력까지 검증한 것인가?

`Implicit`은 별도 명시적 정렬 loss 없이 task learning으로 관계를 학습하는 분류다. 의미 정렬이 입증됐다는 뜻은 아니다. `Modalities`에는 원본 분류에 따라 출력 Action도 포함돼 있다. FuSe의 Audio는 상세 본문에서 다루지만 원본 테이블 태그에는 없다. Adaptive Vision–Torque의 F/T 태그는 실제 관측인 joint external torque와 구별해 읽는다.

## 융합·정렬 비교

| 논문 | Modalities | Fusion Stage | Cross-modal Interaction | Alignment Type | Context Strategy |
| --- | --- | --- | --- | --- | --- |
| [ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](papers/forcevla.md) | Vision, Language, F/T, Proprioception, Action | Post-VLM / Late | Concat, Self-Attention, MoE Routing, Prompt/Token | Implicit | VLM-first로 VL context 보존 → 6D F/T를 1 force token으로 projection → post-VLM concat → self-attention으로 cross-modal contextualization → Top-1 MoE routing으로 task/phase-dependent specialization |
| [Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation](papers/adaptive-vision-torque.md) | Vision, F/T, Proprioception, Action | Hybrid | Concat, Contact Gating, CFG Guidance | Implicit | Vision을 base planning context로 유지 → joint external torque threshold로 contact 판별 → non-contact에서는 real torque를 learned f\*로 대체 → contact에서 torque expert 활성화 → vision+gated torque로 scale predictor가 w_torque 결정 → CFG-style로 vision prediction을 torque correction이 수정 |
| [Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding](papers/fuse.md) | Vision, Language, Tactile, Proprioception, Action | Early | Self-Attention, Contrastive Alignment, Prompt/Token | Explicit | Natural language as common cross-modal grounding; modality-specific encoders → shared pretrained Transformer; CLIP-style contrastive alignment + observation-to-language generation for available modality combinations. |
| [ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Visuomotor Learning](papers/vitas.md) | Vision, Tactile, Action | Mid | Concat, Contrastive Alignment | Explicit | Separate CNN encoders → alternating cross-modal Top-K neighborhood transfer (Vision→Tactile / Tactile→Vision) → feature concatenation → CVAE current-image reconstruction as complementarity regularizer → PPO or Diffusion Policy. |
| [TacFiLM: Tactile Modality Fusion for Vision-Language-Action Models](papers/tacfilm.md) | Vision, Language, Tactile, Action | Mid | FiLM Conditioning | Implicit | Pretrained tactile encoder → pooled z → block-wise MLP → channel scale/shift inside DINOv2 & SigLIP (after normalization, before self-attention) → projector → language integration → action. |

## 실험 근거와 해석 범위

| 논문 | Task | Key Result | Ablation Evidence | Strength (This View) | Limitation (This View) |
| --- | --- | --- | --- | --- | --- |
| [ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](papers/forcevla.md) | Insertion, Surface Contact, Peeling, Pumping | 평균 success 60.5%; π0-base w/o F 대비 +23.2 percentage points. Visual Occlusion 90%. | baseline 45%; linear before VLM 55%; MoE before VLM 0%; concat after VLM 60%; ForceVLA 80% | Pretrained VL representation을 보존하면서 F/T를 late-inject하고 self-attention + MoE로 context-dependent interaction을 학습. Fusion timing ablation 및 router analysis 제공. | Explicit alignment loss/metric 없음. Self-attention-only vs MoE-only 분해 ablation 없음. Pre-VLM baselines 구현 상세 부족. Linear-before-VLM 55%라 early fusion 일반 부정은 어려움. |
| [Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation](papers/adaptive-vision-torque.md) | Surface Contact, Disassembly, Force Discrimination | Average success 82%; strongest baseline Torque Gating 68% 대비 +14 percentage points. 가장 큰 jump는 Feature Concatenation 30% → Torque Gating 68%로, modality relevance selection의 중요성을 보여줌. | Vision-only 30%; Feature Concatenation 30%; Torque Gating 68%; Auxiliary Goals 28%; MoE 24%; MoE w/o torque encoding 54%; Ours 82%; Torque-Gated MoE 12/20 vs Ours 16/20 | Fusion 전에 modality relevance를 contact state로 구조화하고, contact에서만 learned CFG correction을 적용. 동일 Diffusion Policy backbone에서 concat/gating/auxiliary/MoE를 controlled comparison하며 router/weight analysis까지 제공. | Explicit alignment loss/metric 없음. Contact detection이 predefined torque threshold에 의존. Binary contact state, 3 real-world tasks 및 적은 trial 수. Passive F/T reasoning만 검증하고 active force control은 다루지 않음. |
| [Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding](papers/fuse.md) | — | FuSe reports \>20% higher success than considered baselines overall; gains are especially clear in partially observable Shopping Bag, and the recipe transfers from Octo to a PaliGemma-based 3B VLA. | Shopping Bag ablation: full FuSe with both contrastive and generative auxiliary losses outperforms removing either loss or both, especially on unseen test objects. | Explicitly connects new heterogeneous sensors to pretrained semantic knowledge through language, rather than relying on BC/action loss alone; supports multimodal and compositional cross-modal prompting. | 0.4 s observation history; added training cost; modality-specific language annotations required. Paper does not clearly specify timestep-level masking/gating of tactile/audio semantic losses before the sensor becomes informative. |
| [ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Visuomotor Learning](papers/vitas.md) | — | Simulation 12-task avg 91.4; simulation IL avg DP+ViTaS 60.4 vs DP+CNN 38.2 / Transformer 33.4; real-world avg 46.0 vs DP 30.0. | Table V avg: ViTaS 92.5; w/o tactile 60.9; unified encoder 27.1; w/o soft fusion contrastive 54.7; w/o CVAE 63.2; time contrastive 70.6; K=1 78.8; K=20 68.1; K=50 67.3. | Separate heterogeneous modalities with modality-specific encoders, explicitly align cross-modal neighborhood structure, then fuse features; CVAE adds complementarity-oriented representation supervision. | No language-semantic modality; no direct representation-level alignment metric; CVAE reconstructs the current image rather than a clean de-occluded target; natural vision-touch timestep correspondence may not transfer to Language–F/T. |
| [TacFiLM: Tactile Modality Fusion for Vision-Language-Action Models](papers/tacfilm.md) | Insertion | Table 1 reported averages: ID success 86.67% vs Concat 71.11%; ID peak-force mean 8.34 vs 10.29 N. OOD success 86.67% vs 73.33%. USB +30 percentage points; not best on every task/metric. | Table 2: All/Early/Middle/Late FiLM; 80% dimming and 50% frame updates. Table 3: tactile classifier average T3 83.04, IJEPA 93.56, MAE 96.64, DINO 97.72. | Encoder-internal tactile conditioning without extra LLM tokens; separate pretrained tactile representation; task-driven connection without explicit cross-modal alignment loss. | No explicit semantic-alignment evidence, patch-contact correspondence, F/T/proprioception fusion or high-level sub-goal evaluation. Camera tests are dimming/stale frames, not arm-induced spatial occlusion; insertion-only setup. |

## 비교의 경계

- 연구별 데이터·과제·모델·평가 조건이 달라 성공률을 논문 간 성능 순위로 해석하지 않는다.
- ViTaS는 정렬 학습 후에도 concat을 사용한다. Concat과 alignment는 배타적인 선택지가 아니다.
- 접촉 상태별 유효성 선택과 모달리티 의미 정렬은 별도 설계 축이다.
- 현재 공개 VLM 실험 결과는 [experiments](../../experiments/README.md)에 기록한다. 본 문헌의 결과를 자체 실험 결과와 혼합하지 않는다.

[출처·이전 범위](SOURCES.md) · [구조화된 비교 데이터](comparison.json)
