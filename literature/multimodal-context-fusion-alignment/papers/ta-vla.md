# TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models

[문헌 비교표](../README.md) · [논문](https://proceedings.mlr.press/v305/zhang25k.html) · [노션 원본](https://app.notion.com/p/3e9952c9e27381ebbc71f4f90fe032ee)

> 2026-09-29 노션 정리의 GitHub snapshot. 원본 최종 수정: 2026-09-28T11:40:06.752Z. 아래 보고 수치와 해석은 노션 원본을 보존했으며, 이번 이전에서 논문 전체를 재검증하지 않았다.

## 현재 연구 목표와의 연결

Torque history의 압축 위치와 future physical response 예측을 접촉 정보 기반 sub-goal 생성·갱신 모델의 설계 후보로 검토한다. 이 논문의 출력은 저수준 action이며, sub-goal의 행동 종류·대상·방향·거리 결정이나 가려짐 상황의 고수준 계획을 직접 검증한 것은 아니다. Motor-current 기반 joint torque는 손목 6축 F/T·tactile과 구별한다.

위 연결은 현재 연구로의 확장 가설이며, 아래 논문 정리의 실험 결과와 구분한다.

---

> **문헌 정리 관점:** Vision–Language이 형성하는 semantic/context representation과 joint torque가 제공하는 physical interaction signal을 **어디서, 어떤 형태로 연결하고, 어떤 학습 목표로 의미 있게 쓰는가?**
> **핵심 분류 — Decoder-side late token fusion + implicit proprioceptive alignment + future-torque auxiliary objective**
> TA-VLA는 명시적인 Vision–Language–Torque alignment loss를 두지 않는다. 대신 **torque를 proprioception/action 쪽에 가까운 신호로 보고 decoder/action expert에 별도 token으로 late-inject**하며, **과거 2초 torque history를 1 token으로 압축**하고 **future torque prediction을 auxiliary objective**로 추가한다.
> **중요한 구분**
> 논문의 HSIC 분석은 torque와 joint angle feature가 강하게 통계적으로 연관되어 있음을 보여주는 **분석/진단 도구**이지, 학습 시 사용하는 explicit alignment loss가 아니다. 따라서 이 표에서는 Alignment Type을 **Implicit**로 분류한다.

## 1. 이 논문에서 볼 핵심 질문

TA-VLA는 새로운 VLA backbone 자체를 제안하기보다, pretrained VLA에 joint torque를 붙일 때의 **design space**를 체계적으로 실험한다.

| 축 | 질문 | 비교 |
| --- | --- | --- |
| **Where** | Torque를 어디에 넣을까? | Encoder vs Decoder |
| **How** | History를 어떤 token 구조로 넣을까? | Single-token vs Multi-token |
| **When** | 현재/과거만 볼까, 미래 physical response도 학습할까? | Immediate / History / Future |

이 관점에서 핵심은 **“F/T를 입력에 추가했다”**가 아니라, pretrained VLA의 기존 표현을 깨뜨리지 않으면서 **physical modality를 action-generation 쪽에 정렬하고 활용하는 학습 설계**다.

## 2. Paper Figure 1 — Torque가 왜 별도 physical context인가

![Paper Figure 1 — Torque response, joint mapping, design space](https://arxiv.org/html/2509.07962v1/teaser.png)

Figure 1은 논문의 전체 논리를 압축한다.

- **Before Contact:** torque 변화가 비교적 작다.
- **Contact without Insertion:** 접촉은 발생했지만 성공하지 못한 torque pattern이 나타난다.
- **Successful Insertion:** 성공적으로 삽입될 때 더 크고 뚜렷한 torque spike가 나타난다.
- 우측 design space는 **Immediate / History / Future**, **Encoder / Decoder**, **Single / Multi-token**을 모두 실험 대상으로 둔다.

> **이 표 관점에서의 의미**
> Vision/Language가 semantic·spatial context를 제공한다면, torque는 RGB만으로 잘 보이지 않는 **contact state / interaction dynamics context**를 제공한다. TA-VLA의 질문은 이 둘을 처음부터 동등하게 섞는 것이 아니라, torque가 가장 자연스럽게 들어갈 representation 위치를 찾는 것이다.

## 3. 물리적 근거 — 왜 joint torque가 contact를 담는가

논문은 end-effector 외력이 joint torque로 투영되는 관계를 다음과 같이 정리한다.

$$
\tau_{\mathrm{ext}} = J^{\top}(q)F_{\mathrm{ext}}
$$

그리고 측정 torque는 개념적으로 다음처럼 분해된다.

$$
\tau_{\mathrm{measured}}
=
\tau_{\mathrm{model}}
+
J^{\top}(q)F_{\mathrm{ext}}
$$

즉 외부 접촉이 생기면 end-effector wrench가 kinematic chain을 통해 joint torque 변화로 나타난다. 이 때문에 torque history는 단순한 low-dimensional state가 아니라 **contact dynamics를 반영하는 physical observation**으로 취급된다.

> 이 논문은 별도 6D F/T sensor를 사용하지 않는다. Cobot Magic ALOHA의 motor current와 current-to-torque constant를 이용해 joint torque를 실시간 추정한다: τ = k_t i.

## 4. Fusion Stage — 왜 Decoder-side late fusion인가

원래 π0에서는 **RGB images + instruction → VLM/Conditioning Encoder → Vision–Language conditioning context**가 먼저 형성되고, joint state와 noisy action chunk가 action expert/decoder에 들어간다.
TA-VLA는 torque를 세 위치로 비교한다.

| 방식 | Torque 처리 | 위치 |
| --- | --- | --- |
| Enc | MLP adapter → torque token | Image/Language와 함께 conditioning encoder |
| DePre | Torque 값을 기존 state zero-padding에 직접 삽입 | Decoder state token 내부 |
| **DePost** | MLP adapter → **별도 torque token** | **Decoder/action expert state input 앞에 추가** |

논문 결론은 **DePost**가 가장 안정적이라는 것이다. Torque를 image/language 쪽 encoder에 넣기보다 proprioception과 action generation이 있는 decoder 쪽에 두고, 기존 state token 자체를 바꾸기보다 **별도 token**으로 추가한다.

## 5. Paper Table 1 — Encoder vs Decoder

| Task | π0 | Enc | DePre | **DePost** |
| --- | --- | --- | --- | --- |
| Button Pushing | 5/20 | 7/20 | 8/20 | **10/20** |
| Charger Plugging | 0/20 | 8/20 | 11/20 | **12/20** |

### 5.1 HSIC — torque는 joint angle 쪽과 가장 강하게 정렬됨

![Paper Figure 3 — Normalized HSIC between modality hidden states](https://arxiv.org/html/2509.07962v1/hsic.png)

Paper Figure 3에서 angle–torque normalized HSIC는 **0.957**로 매우 높다. Torque–image는 0.300, torque–text는 0.114다.

> **정확한 해석**
> 이 결과는 “Vision–Language와 Torque가 semantic하게 align됐다”는 의미가 아니다. 오히려 torque가 **joint angle/proprioception representation과 더 가까운 신호**임을 보여주며, 그래서 decoder/action side에 넣는 설계를 뒷받침한다.

### Paper Table 2 — Decoder sensitivity test

| Task | π0 | Enc-Noised | Dec-Noised |
| --- | --- | --- | --- |
| Bottle Pick and Place | 14/20 | 12/20 | 8/20 |
| Button Pushing | 5/20 | 4/20 | 0/20 |

Decoder input에 noise를 넣었을 때 성능 저하가 더 크므로, 저자들은 decoder가 **fine-grained input variation에 더 민감**하다고 해석한다. Torque처럼 접촉 순간의 작은 변화가 중요한 신호에는 이 sensitivity가 오히려 유용하다는 논리다.

## 6. Torque History — 과거 2초를 1 token으로 압축

![Paper Figure 4 — Architectures for embedding torque history](https://arxiv.org/html/2509.07962v1/history.png)

Appendix의 구현은 매우 구체적이다.

- 과거 **2초**에서 현재 frame을 포함해 **10개 frame**을 균일 샘플링
- 각 frame의 effort는 **14-dimensional**
- H-token 방식: 10개 frame을 각각 MLP로 token화
- 1-token 방식: 10 × 14 = **140D**로 flatten/concatenate → MLP → **single history token**

따라서 논문에서 말하는 “single token”은 single-frame이 아니라, **2초 history 전체를 요약한 하나의 token**이다.

### Paper Table 3 — History encoding architecture

| Task | π0 | Enc-1 | Enc-H | **Dec-1** | Dec-H |
| --- | --- | --- | --- | --- | --- |
| Button Pushing | 5/20 | 1/20 | 4/20 | **15/20** | 9/20 |
| Charger Plugging | 0/20 | 3/20 | 6/20 | **16/20** | 7/20 |

### Paper Table 4 — Input-pattern disruption

| Task | π0 | Enc-Disrupted | Dec-Disrupted |
| --- | --- | --- | --- |
| Bottle Pick and Place | 14/20 | 13/20 | 8/20 |
| Button Pushing | 5/20 | 5/20 | 2/20 |

저자들의 해석은 **decoder는 pretrained input pattern 변화에 민감**하므로 많은 history token을 추가하면 정보량은 늘어도 기존 state/action token pattern을 방해할 수 있다는 것이다. 그래서 **history information + original decoder pattern preservation**의 절충으로 Dec-1이 가장 좋다.

## 7. Future Torque를 학습 목표로 추가 — Observation만 보지 않고 physical consequence도 예측

![Paper Figure 5 — Action–Torque Diffusion](https://arxiv.org/html/2509.07962v1/future.png)

Section 5의 핵심은 torque를 input observation으로만 쓰지 않고, **future torque를 action과 함께 예측**하게 하는 것이다.
Action chunk와 torque chunk를 합친 joint target은 다음과 같이 정의된다.

$$
Z_t = [A_t; T_t]
$$

최종 학습 objective는 다음과 같다.

$$
\mathcal{L}_{\mathrm{joint}}
=
\mathcal{L}_{\mathrm{action}}
+
\beta\mathcal{L}_{\mathrm{torque}}
$$

- Action과 torque는 **별도 loss**를 유지하지만 diffusion/flow prediction weight를 공유한다.
- 별도 torque head를 크게 추가하기보다 **single linear layer가 concatenated action+torque output을 함께 출력**하고 다시 분리한다.
- Appendix에서 future effort sequence는 action chunk와 동일하게 **H = 50 steps**이다.
- 논문의 의도는 torque prediction 자체가 최종 목적이라기보다, action → physical response 관계를 학습시켜 **physically grounded latent representation**을 유도하는 것이다.

> **Fusion/Alignment 관점의 핵심**
> TA-VLA는 sensor semantics를 language space에 직접 맞추지 않는다. 대신 **action과 torque를 공동 예측**하여 physical modality를 motor command와 연결한다. 따라서 semantic alignment보다 **proprioceptive / dynamics alignment**에 가깝다.

## 8. 학습 데이터와 Fine-tuning Setup

이 논문은 실제 robot demonstration을 수집하고 pretrained VLA를 fine-tuning한다.

| 항목 | 설정 |
| --- | --- |
| Robot | Cobot Magic ALOHA, dual arm, arm당 7-DoF |
| Vision | D435 3대 — top/front + left wrist + right wrist 계열 입력 |
| Torque source | Motor current 기반 joint torque 추정, τ = k_t i |
| Demonstrations | **각 task당 400 teleoperation demonstrations** |
| π0 training | Public pretrained checkpoint, encoder+decoder LoRA fine-tuning, **30k gradient steps**, 4 × NVIDIA L20 |
| RDT training | 4 × NVIDIA L20, full-parameter training, **40k gradient steps** |
| Inference | RTX 4090; π0 action horizon 50, RDT 64 |

> **이 논문의 연구 초점**
> 데이터 수집 자체의 novelty보다 **동일한 torque-containing demonstrations를 가지고 pretrained VLA를 어떻게 fine-tuning해야 physical feedback을 잘 쓰는가**를 비교하는 design study에 가깝다.

## 9. Main Result — Contact-rich task에서 큰 차이

### Paper Table 5 — Contact-rich tasks

| Method | Button Pushing | Charger Plugging | USB Plugging | Socket Unplugging | Door Handle Turning |
| --- | --- | --- | --- | --- | --- |
| ACT | 2/20 | 0/20 | 0/20 | 12/20 | 0/20 |
| RDT | 4/20 | 1/20 | 0/20 | 10/20 | 0/20 |
| π0 | 5/20 | 0/20 | 0/20 | 16/20 | 2/20 |
| π0 + obs | 15/20 | 16/20 | 15/20 | 19/20 | 13/20 |
| π0 + obj | 11/20 | 10/20 | 12/20 | 19/20 | 12/20 |
| **π0 + obs + obj** | **18/20** | **17/20** | **17/20** | **19/20** | **15/20** |

**표에서 직접 계산한 평균:** contact-rich 5 tasks에서 π0는 23/100 = **23%**, π0+obs+obj는 86/100 = **86%**로 **+63 percentage points**다.

### Paper Table 5 — Regular tasks

| Method | Bottle Pick & Place | Liquid Pouring | Stacking Cubes | Push-to-Position | Opening a Drawer |
| --- | --- | --- | --- | --- | --- |
| π0 | 17/20 | 16/20 | 17/20 | 16/20 | 19/20 |
| π0 + obs | 18/20 | 16/20 | 18/20 | 16/20 | 19/20 |
| π0 + obj | 17/20 | 16/20 | 17/20 | 16/20 | 18/20 |
| π0 + obs + obj | 19/20 | 17/20 | 17/20 | 18/20 | 18/20 |

Regular 5 tasks는 π0 85/100 = **85%**, full model 89/100 = **89%**다. 즉 이 논문의 torque design 효과는 특히 **contact-rich manipulation에서 훨씬 크게 나타난다.**

## 10. Paper Figure 7 — Failure detection → Retry behavior

![Paper Figure 7 — Task visualization and retry behavior](https://arxiv.org/html/2509.07962v1/visualization.png)

이 그림에서 중요한 것은 단순 success rate가 아니라 **closed-loop behavior**다.

- Button Pushing: 첫 접촉 misalignment → retreat → second press → success
- Door Handle Turning: 첫 press/turn 실패 → second turn attempt → door opened
- 저자들은 abnormal torque 변화로 failed attempt를 감지하고 retry가 나타난다고 설명한다.

> **이 표 관점에서의 operational evidence**
> Torque가 단순히 representation에 들어갔다는 것보다, contact failure 이후 행동 수정으로 이어졌다는 점이 physical modality가 policy decision에 실제로 사용되었다는 중요한 정성적 근거다.

## 11. Appendix Figure 9 — Torque pattern과 failed/successful interaction

![Paper Figure 9 — Contact-rich task execution with torque traces](https://arxiv.org/html/2509.07962v1/app_effort_task.png)

Figure 9는 Button Pushing, Charger Plugging, USB Plugging, Socket Unplugging에서 execution sequence와 selected joint torque trace를 함께 보여준다. 실패 시점과 성공 시점의 torque 변화가 다르게 나타나며, **temporal torque history가 contact state 판별에 유용한 이유**를 시각적으로 뒷받침한다.

## 12. Cross-model / Cross-embodiment generalization

### Paper Table 6 — RDT에도 동일 recipe 적용

| Method | Button Pushing | Charger Plugging | Bottle Pick & Place |
| --- | --- | --- | --- |
| RDT | 4/20 | 1/20 | 17/20 |
| **RDT + obs + obj** | **16/20** | **15/20** | **19/20** |

3-task 합계는 22/60에서 50/60으로 증가한다. Appendix에서 RDT effort projector는 2-layer MLP로 2048D에 맞추고, projected effort token을 state token 뒤에 concat한다.

### Paper Figure 8 — Cross Embodiment

![Paper Figure 8 — ROKAE SR cross-embodiment charging insertion](https://arxiv.org/html/2509.07962v1/cross_embodiment.png)

ROKAE SR arm에서도 misaligned initial contact를 torque feedback으로 감지한 뒤 두 번째 시도에서 charging connector insertion을 완료한다.

## 13. 추가 Ablation — β, history aggregator, efficiency

### Paper Table 7 — Training time

| Model | Training Time (s/iter) |
| --- | --- |
| π0 | 1.703 |
| π0 + obs | 1.628 |
| π0 + obj | 1.648 |
| π0 + obs + obj | 1.640 |

### Paper Table 8 — Inference time

| Model | Inference Time (ms) |
| --- | --- |
| π0 | 90.70 |
| π0 + obs | 90.81 |
| π0 + obj | 90.61 |
| π0 + obs + obj | 93.96 |

즉 torque-aware design이 inference latency를 크게 증가시키지는 않는다.

### Paper Table 9 — Auxiliary torque loss weight β

| β | 0.01 | 0.1 | 0.2 | 0.5 | 1 |
| --- | --- | --- | --- | --- | --- |
| π0 + obj | 6/20 | 8/20 | 10/20 | 9/20 | 11/20 |
| π0 + obs + obj | 14/20 | **18/20** | **18/20** | 15/20 | 12/20 |

최종 설정은 +obj에서 β=1, +obs+obj에서 β=0.1을 사용한다.

### Paper Table 10 — Torque-history aggregation

| Aggregation | MLP | RNN | Attention |
| --- | --- | --- | --- |
| π0 + obs | **15/20** | 7/20 | 13/20 |
| π0 + obs + obj | **18/20** | 10/20 | 17/20 |

제한된 fine-tuning data에서는 복잡한 RNN/attention보다 단순 MLP aggregation이 가장 좋았다.

## 14. Alignment / Fusion 관점의 정확한 분류

| 항목 | TA-VLA |
| --- | --- |
| Physical signal | Motor-current 기반 **joint torque / effort** |
| Explicit VL–torque alignment objective | **없음** |
| Alignment diagnostic | Normalized HSIC — torque ↔ joint-angle feature dependence 분석 |
| Fusion stage | **Decoder / action expert side late fusion** |
| Fusion unit | MLP-projected torque token |
| Temporal context | Past **2 s / 10 frames → 140D → 1 token** |
| Predictive physical objective | **Future torque H=50** jointly predicted with action |
| Alignment Type | **Implicit — proprioceptive / dynamics-side** |

## 15. 이 관점에서의 강점

- **Where / How / When**을 분리한 controlled design-space study라서 physical modality fusion 설계 근거가 명확하다.
- Encoder vs Decoder, PreConcat vs PostConcat, 1-token vs H-token을 실제 robot task에서 비교한다.
- HSIC로 **torque가 Vision/Language보다 proprioception에 더 가까운 신호**라는 근거를 제시한다.
- Extra noise-token experiment로 pretrained decoder의 **input-pattern sensitivity**를 검증한다.
- History를 단순 추가하는 데서 끝나지 않고 **future torque auxiliary prediction**으로 dynamics learning까지 확장한다.
- π0뿐 아니라 RDT와 다른 robot embodiment에도 적용한다.
- 별도 external F/T sensor 없이 motor current에서 torque를 추정한다.

## 16. 이 관점에서의 한계 / 연구 공백

- **Explicit Vision–Language–Torque semantic alignment objective가 없다.**
- HSIC가 높다는 것은 torque와 joint angle feature의 dependence이지, vision/text와 physical signal의 의미적 alignment를 직접 증명하는 것은 아니다.
- History window가 **2초 / 10 samples로 고정**되어 있고 temporal context length ablation은 없다.
- Joint torque 추정은 motor calibration, noise, thermal drift의 영향을 받을 수 있다.
- Contact 여부에 따라 torque의 relevance를 직접 gating하는 구조는 아니다.
- Tactile/temperature 등 더 많은 physical modality로 확장할 때 shared token budget과 integration scalability는 미검증이다.
- 400 demonstrations/task의 fine-tuning 설정에서 얻은 결론이 훨씬 더 적거나 더 큰 dataset에서도 동일한지는 추가 검증이 필요하다.

## 17. ForceVLA / FuSe와 나란히 보면

| 관점 | TA-VLA | ForceVLA | FuSe / Beyond Sight |
| --- | --- | --- | --- |
| 핵심 질문 | **Torque를 어디에/어떻게 넣고 무엇을 예측할까?** | VL context 뒤에서 F/T를 어떻게 contextualize할까? | 새 sensor가 무엇을 의미하는지 language로 어떻게 grounding할까? |
| Alignment | Implicit proprioceptive alignment | Implicit contextualization | Explicit semantic alignment |
| 대표 mechanism | Decoder single-history-token + future torque objective | Post-VLM self-attention + MoE | Contrastive + language generation |
| Temporal physical context | **2 s torque history** | Force token / task-phase interaction | Short multimodal observation history |

> **세 논문을 연결해서 보면**
> FuSe는 “**이 sensor가 무엇을 의미하는가**”, ForceVLA는 “**어디서 어떤 context-dependent interaction을 만들 것인가**”, TA-VLA는 “**physical signal을 action side에 어떤 temporal representation과 predictive objective로 학습할 것인가**”에 더 강하게 초점을 둔다.

## 18. Paper / Source

- [PMLR — CoRL 2025 Proceedings](https://proceedings.mlr.press/v305/zhang25k.html)
- [arXiv — 2509.07962](https://arxiv.org/abs/2509.07962)
- [Project Page](https://zzongzheng0918.github.io/Torque-Aware-VLA.github.io/)
- [Code — GitHub](https://github.com/ZZongzheng0918/TA-VLA)

[TA-VLA Paper PDF](https://arxiv.org/pdf/2509.07962)
