# Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding

[문헌 비교표](../README.md) · [논문](https://arxiv.org/abs/2501.04693) · [노션 원본](https://www.notion.so/3e4952c9e27381739056cffb6421c87f)

> 2026-09-26 노션 정리의 GitHub 스냅샷. 논문별 수치·해석은 원본 정리 기준이며, 이번 이전 작업에서 모든 논문을 재검증한 것은 아니다.

## 현재 연구 목표와의 연결

센서 관측을 언어적 의미와 연결하는 학습 목표를 참고한다. 접촉 사건·현재 행동·sub-goal의 대응을 학습하려면 시간별 감독 범위와 접촉 전 비정보성 입력 처리를 추가로 설계해야 한다.

이 문헌의 결과를 접촉 기반 sub-goal 생성의 입증으로 간주하지 않는다. 아래 이전 연구 시사점은 작성 당시 맥락을 보존한 것이며 현재의 확정 아키텍처가 아니다.

---

> **관점:** 이 논문은 heterogeneous sensor를 단순히 policy input에 추가하는 방법이 아니라, **pretrained robot policy가 이미 가진 language-semantic knowledge를 공통 grounding space로 사용해 Vision–Touch–Audio를 명시적으로 align하는 방법**으로 보는 것이 핵심이다.

> **한 줄 분류 — Language-grounded explicit multimodal alignment**
> FuSe는 Vision·Touch·Audio를 같은 Transformer에 넣는 것만으로 끝내지 않고, **CLIP-style multimodal contrastive loss + sensory-grounded language generation loss**를 추가해 새 sensor modality가 기존 semantic knowledge와 연결되도록 강제한다.


| Input modality | Encoder / representation | Shared processing |
| --- | --- | --- |
| Workspace + Wrist RGB | Vision encoder → visual tokens | FuSe Multimodal Transformer |
| Two DIGIT tactile images | Pretrained TVL tactile encoder | FuSe Multimodal Transformer |
| Audio waveform | Spectrogram → ResNet26 | FuSe Multimodal Transformer |
| Language | Language tokens | FuSe Multimodal Transformer |

### 3.1 Fusion stage
각 modality는 modality-specific encoder를 거친 뒤 **observation token으로 Transformer 입력에 함께 들어간다.** 따라서 fusion 위치만 놓고 보면 token-level early fusion에 가깝다.
그러나 FuSe의 차별점은 fusion 위치 자체보다 **fusion된 representation에 explicit semantic alignment objective를 추가했다는 점**이다.
### 3.2 Tactile encoder
- Small finetuning dataset 때문에 tactile encoder를 scratch로 학습하지 않는다.
- **TVL encoder**를 사용한다.
- TVL은 Vision–Language–Touch pairwise contrastive pretraining을 거친 encoder다.
- 양쪽 gripper의 두 DIGIT tactile image는 동일한 TVL encoder를 weight-sharing하여 별도로 처리한다.
### 3.3 Audio encoder
- Raw waveform은 high-dimensional + noisy하므로 **spectrogram**으로 변환한다.
- Spectrogram을 일반 image처럼 취급해 **ResNet26**으로 encoding한다.
## 4. 왜 BC loss만으로는 부족한가

| 단계 | 무슨 일이 생기는가 |
| --- | --- |
| 1 | Pretrained policy는 이미 Vision을 잘 사용한다. |
| 2 | Touch / Audio를 새 input으로 추가한다. |
| 3 | BC loss만 쓰면 목표는 action prediction만 맞추는 것이다. |
| 4 | Vision만으로 action을 어느 정도 맞힐 수 있으면 Touch / Audio를 무시하는 shortcut이 가능하다. |

즉 BC는 **action을 잘 맞추는지**만 확인하므로, 새 sensor를 의미 있게 쓰도록 강제하지 않는다. FuSe는 그래서 action supervision과 별도로 **semantic supervision**을 추가한다.
## 5. Loss 1 — Multimodal Contrastive Loss
FuSe는 observation embedding과 language instruction을 CLIP-style contrastive learning으로 align한다.
$$
\mathcal{L}_{contrast}
$$
개념적으로 같은 의미의 pair는 가까워지고, 틀린 의미의 pair는 멀어진다.

| Sensor observation | Positive language semantics |
| --- | --- |
| Soft-object tactile pattern | “feels squishy” |
| Button audio | “plays piano” |
| RGB observation | “red” / “round” |

논문은 가능한 modality combination마다 observation embedding을 만들고 해당 instruction과 contrastive loss를 계산한 뒤 평균한다.

> **Alignment 관점**
> 이 항은 FuSe를 단순 fusion과 구분하는 가장 직접적인 요소다. Vision과 Touch를 단순 concat하는 것이 아니라 **sensor observation ↔ language semantics correspondence 자체를 objective로 둔다.**

## 6. Loss 2 — Multimodal Generative Loss
두 번째 auxiliary objective는 observation에서 **language를 직접 생성**하게 한다.
예:
- Touch → “the object feels soft and squishy”
- Audio → “the button plays piano”
- Vision + Touch → “the object looks round and feels soft”
가능한 각 modality combination에서 observation embedding을 만들고, shared generative Transformer head로 language instruction을 예측한다.
$$
\mathcal{L}_{gen}=CE(\hat{y}_{lang}, y_{lang})
$$
Contrastive loss가 **representation correspondence**를 학습한다면, generative loss는 sensor representation 안에 **언어로 복원 가능한 high-level semantics**가 들어가도록 압박한다.
## 7. 최종 Training Objective
$$
\mathcal{L}=\mathcal{L}_{BC}+\beta\mathcal{L}_{gen}+\lambda\mathcal{L}_{contrast}
$$
논문은 모든 실험에서 다음 값을 사용한다.
$$
\beta=1,\quad \lambda=1
$$

| Loss | 학습하는 것 | 역할 |
| --- | --- | --- |
| L_BC | Observation → Robot Action | Imitation / control behavior |
| L_contrast | Observation ↔ Language | Cross-modal semantic alignment |
| L_gen | Observation → Language | Sensor feature에 explicit semantics 주입 |


> **가장 간단한 이해**
> BC는 **“어떻게 움직일지”**를 가르치고, 두 auxiliary loss는 **“새 sensor가 무엇을 의미하는지”**를 가르친다.

## 8. Language Rephrasing — language supervision은 어떻게 만든가
Robot trajectory에는 데이터 수집 후 **after-the-fact language annotation**을 붙인다.
예:
- Vision: red
- Touch: squishy
- Vision + Touch: “the object feels squishy and is red”
- Touch + Audio: “the object feels metallic and sounds clinking”
이후 template 표현에 과적합되지 않도록 **ChatGPT를 사용해 semantic meaning을 유지하는 rephrasing**을 생성한다. 각 가능한 modality combination마다 20개 language template을 사용한다.

| Step | Process |
| --- | --- |
| 1 | Teleoperation으로 Vision / Touch / Audio / Action trajectory 수집 |
| 2 | After-the-fact templated language annotation |
| 3 | ChatGPT rephrasing으로 free-form 표현 다양화 |
| 4 | BC + contrastive + generative objectives로 FuSe finetuning |

중요한 점은 ChatGPT가 online controller가 아니라 **training instruction augmentation 도구**라는 것이다.
## 9. Paper Fig. 3 — Sensor / Robot Setup
![Paper Fig. 3 — Robot sensor setup](https://fuse-model.github.io/static/images/fuse-robot.png)
- Robot: **WidowX 250, 6-DoF**
- Control: delta end-effector position command, **5 Hz**
- Vision: third-person RGB + wrist RGB
- Touch: **two DIGIT tactile sensors**
- Audio: standard microphone
- Additional sensing: 9-DoF IMU
- Visual resolution: **640×480**
- DIGIT tactile: **320×240**
- Audio: most recent **1 s**, 44.1 kHz
- Dataset: **26,866 teleoperated trajectories**
Tactile observation은 zero-deformation background를 빼서 contact deformation을 강조한다.
## 10. Evaluation Tasks — modality가 왜 필요한가

| Task | 주요 modality | Vision만으로 부족한 이유 |
| --- | --- | --- |
| Tabletop Grasping | Vision + Touch | 유사 외형 object의 tactile property 구별 |
| Shopping Bag | Vision + Touch | Bag 내부에서 occlusion + poor lighting |
| Button Pressing | Vision + Audio | button identity를 sound semantics로 구별 |
| Compositional Task | Vision + Audio + Language | sound → button → visual attribute → action 연결 |

## 11. Main Results
### Tabletop Grasping
![Project result — Tabletop Grasping](https://fuse-model.github.io/static/images/bar_chart_Tabletop%20Grasping.png)
### Shopping Bag — 가장 중요한 partial-observation case
![Project result — Shopping Bag](https://fuse-model.github.io/static/images/bar_chart_Shopping%20Bag.png)
Shopping Bag에서는 gripper가 bag 안으로 들어갈수록 visual cue가 약해진다. 이 환경에서 FuSe와 Vision-only FT 사이 격차가 크게 나타나므로, 이 결과는 **new modality가 실제 policy decision에 기여했다는 operational evidence**로 볼 수 있다.
### Average
![Project result — Average](https://fuse-model.github.io/static/images/bar_chart_average.png)
논문은 FuSe가 고려한 baseline 대비 real-world success rate를 **20% 이상 향상**시킨다고 보고한다.
## 12. Multimodal Prompting — 한 modality로 애매한 object를 구별
대표 instruction:
> “grab the round object that feels squishy”
이 명령에서는 round는 Vision, squishy는 Touch에서 판단한다. 즉 language가 modality-specific constraint를 하나의 query로 묶고, policy는 여러 modality evidence를 함께 사용해야 한다.
## 13. Compositional Cross-modal Reasoning
### Simple compositional task
![Project result — Compositional Simple](https://fuse-model.github.io/static/images/bar_chart_Compositional%20Task%20-%20Simple.png)
대표 instruction:
> “grab the object that has the same color as the button that plays piano”
필요한 reasoning은 **Audio semantic → corresponding button → button color → same-color object → grasp** 순으로 연결된다.
### Multi-step compositional task
![Project result — Compositional Multi-step](https://fuse-model.github.io/static/images/bar_chart_Compositional%20Task%20-%20Multi-step.png)
1. unseen button을 visual instruction으로 press
2. resulting sound를 generative head에 입력
3. sound를 language instruction으로 변환
4. training environment에서 같은 sound의 button을 찾아 press
이 실험은 **language generation head가 단순 auxiliary regularizer를 넘어 cross-modal subtask bridge로 사용될 수 있음**을 보여준다.
## 14. Ablation — 두 auxiliary loss가 필요한가
Shopping Bag에서 Full FuSe와 다음을 비교한다.
- No generative loss
- No CLIP / contrastive loss
- No auxiliary losses
결과적으로 **두 auxiliary loss를 모두 사용하는 FuSe가 가장 좋고**, 특히 unseen test objects에서 loss 제거 시 성능이 감소한다.

> **핵심 ablation 읽기**
> 단순히 multimodal sensor를 input에 포함한 것만으로는 충분하지 않다. 성능 향상은 **sensor availability + language-grounded feature learning**의 조합에서 나온다.

## 15. Alignment 관점의 정확한 분류

| 항목 | FuSe |
| --- | --- |
| Explicit alignment objective | **있음 — CLIP-style contrastive loss** |
| Generative semantic objective | **있음 — observation → language** |
| Common grounding space | Natural Language |
| Fusion | Modality tokens → shared pretrained Transformer |
| Cross-modal interaction | Shared Transformer + attention + language objectives |
| Alignment Type | **Explicit semantic alignment** |
| Temporal correspondence | Synchronized robot trajectory observations |

## 16. 중요하게 비판해서 볼 부분 — Touch/Audio는 항상 informative한가?

> **아래는 논문의 직접 명시가 아니라, method description을 읽을 때 생기는 중요한 해석/공백이다.**
> Touch와 Audio는 continuous semantic signal이 아니라 **contact/event 이후에 강해지는 sparse modality**다. 그런데 본문은 auxiliary semantic loss를 정확히 어느 timestep에서 활성화하거나 마스킹하는지 명확히 설명하지 않는다.


| Phase | Vision | Touch | Audio |
| --- | --- | --- | --- |
| Approach | Informative | Mostly background / zero deformation | Often uninformative |
| Contact | Informative | Deformation signal appears | Task dependent |
| Interaction / button press | Informative or partially occluded | Informative | Sound event may appear |

따라서 다음 질문이 남는다.
- Contact 이전 tactile observation에도 soft / squishy semantic target을 동일하게 적용하는가?
- Audio event 발생 전에도 audio-language alignment loss를 계산하는가?
- Sensor informativeness / availability mask가 존재하는가?
- trajectory-level annotation을 timestep-level supervision으로 어떻게 배분하는가?
논문 결론에서도 현재 observation history가 **0.4 s**로 제한되며, 더 긴 context가 tactile처럼 sparse한 signal reasoning에 도움이 될 수 있다고 언급한다.
## 17. 이 관점에서의 강점
- **Language를 common semantic grounding으로 명시적으로 사용**
- 새 modality를 BC action loss에만 의존하지 않고 explicit representation objective로 연결
- Small multimodal dataset에서도 pretrained policy knowledge를 활용
- Vision–Touch–Audio를 공통 semantic query space로 통합
- Multimodal prompting뿐 아니라 **compositional cross-modal reasoning** 평가
- Octo뿐 아니라 **PaliGemma-based 3B VLA**에도 적용
- Auxiliary losses에 대한 ablation 제공
## 18. 이 관점에서의 한계 / 연구 공백
- Touch/Audio의 **temporal informativeness와 loss gating 방식이 충분히 설명되지 않음**
- **0.4 s observation history**는 sparse event reasoning에 짧음
- Modality-specific language annotation과 template design 필요
- 새 modality 추가 시 encoder와 auxiliary branch로 training resource 증가
- Alignment가 semantic language space에 집중되어 있어 **metric force magnitude / geometry / contact mode**를 직접 정렬하는 목적은 아님
- embedding similarity나 cross-modal retrieval처럼 **representation alignment 자체를 정량 분석한 결과는 제한적**
## 19. ForceVLA / Contact-aware Fusion과 나란히 보면

| 관점 | FuSe | ForceVLA | Adaptive Vision–Torque Fusion |
| --- | --- | --- | --- |
| 핵심 문제 | 새 sensor를 pretrained semantics에 연결 | VL context를 보존하며 F/T를 late contextualize | contact 상태에 따라 torque relevance 결정 |
| Alignment | **Explicit language grounding** | Implicit self-attention/MoE | Implicit operational alignment |
| Fusion timing | Shared Transformer input token level | Post-VLM / late | Hybrid gating + late correction |
| 핵심 mechanism | Contrastive + Generative loss | Self-attention + MoE routing | Contact gating + CFG-style guidance |
| 핵심 질문 | What does this sensor mean? | Where/how should physical token interact? | When is this modality valid? |


> **세 논문을 연결해서 보면**
> FuSe는 **semantic alignment**, ForceVLA는 **fusion/contextualization**, Adaptive Vision–Torque Fusion은 **modality relevance/gating**에 더 강하게 초점이 있다. 서로 대체 관계라기보다 multimodal manipulation의 서로 다른 축을 해결한다.

## 20. 초기 VLA–DRL 구상에 대한 해석 (이전 연구 맥락)

> **아래는 논문의 직접 주장보다 현재 연구 관점에서의 시사점이다.**
> Physical modality를 VLA에 붙일 때는 하나의 fusion layer만 고민하기보다 **(1) semantic meaning을 어떻게 맞출지, (2) 언제 modality가 유효한지, (3) policy decision에 어느 강도로 반영할지**를 분리해 설계할 수 있다.

1. **Language grounding loss**로 heterogeneous modality의 semantic compatibility를 먼저 확보한다.
2. Touch/F/T처럼 sparse한 modality에는 별도로 **contact/event-aware gating**을 둘 수 있다.
3. BC/action objective와 representation objective를 분리하면 new sensor가 shortcut으로 무시되는 것을 줄일 수 있다.
4. 향후에는 **language semantic alignment + contact-aware relevance + DRL-based physical refinement**의 3단 구조를 실험할 수 있다.
## 21. Paper / Source
- [Project Page — FuSe](https://fuse-model.github.io/)
- [arXiv — 2501.04693](https://arxiv.org/abs/2501.04693)
- [Code — GitHub](https://github.com/fuse-model/FuSe)
- [Dataset / Models — Hugging Face](https://huggingface.co/papers/2501.04693)
[Paper PDF — ICRA 2025 / FuSe](https://fuse-model.github.io/static/FuSe.pdf)
