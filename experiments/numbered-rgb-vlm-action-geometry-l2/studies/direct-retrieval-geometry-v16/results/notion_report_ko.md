<callout icon="🧪" color="blue_bg">
	**완료: 25개 장면 × 12개 Geometry 조건 × 5개 seed = 1,500회.** 현재 직접 회수 판단 프롬프트에서 R 정성형 D8의 장면 설계 일치율이 117/125 = 93.6%로 가장 높았다. RGB only는 103/125 = 82.4%. 그리퍼 정보는 입력하지 않았다.
</callout>

실험일: 2026-09-17 (Asia/Seoul). 버전: `l2_direct_retrieval_geometry_v16`.
원본 명세: <mention-page url="https://app.notion.com/p/3dc952c9e273815db294c6997f868773"/>

<callout icon="⚠️" color="yellow_bg">
	**이 문서의 일치율은 실제 회수 성공률 또는 검증된 물리적 정확도가 아니다.** FC_CLEAR → RETRIEVE_NOW, 나머지 family → REARRANGE_FIRST라는 장면 설계 의도와 응답을 비교한 참고 지표다. 새 L2 질문에 대응하는 실제 grasp·retrieval 실행 GT는 없다. 원본 명세의 FULL + surface gap > 5 cm 규칙 기반 L2 점수와 혼용하지 않는다.
</callout>

## 1. 실험 목적과 범위
사람처럼 목표 확인 → 지금 회수 가능한지 판단 → 방해 물체 판단 → 행동 선택으로 문제를 나누려는 목적에서, L2를 “주변 물체를 그대로 두고 접촉 없이 타깃을 꺼낼 수 있는가?”로 재정의했다.
- 장애물이 없다면 타깃을 잡고 회수할 수 있는 로봇이라고 가정한다.
- 부분 가림이나 단순한 근접만으로 회수 불가라고 단정하지 않는다.
- L2에서는 blocker ID나 행동을 출력하지 않는다.
- 이번 실행은 L2만 대상이다. L1·L3·L4 및 실제 로봇/simulator 회수 동작은 새로 평가하지 않았다.
- 기존 v15 프롬프트의 Geometry information 블록만 변경했다. R/F/M을 서로 조합하지 않고 하나씩 입력했다.

## 2. 고정 실행 조건

<table fit-page-width="true" header-row="true">
	<tr>
		<td>항목</td>
		<td>내용</td>
	</tr>
	<tr>
		<td>모델</td>
		<td>Qwen/Qwen3-VL-30B-A3B-Instruct</td>
	</tr>
	<tr>
		<td>이미지</td>
		<td>동일한 Numbered RGB 25개; family별 5개</td>
	</tr>
	<tr>
		<td>호출 수</td>
		<td>조건당 125회, scene–condition당 5회, 전체 1,500회</td>
	</tr>
	<tr>
		<td>seed</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>생성 설정</td>
		<td>temperature=0.3, top_p=0.9, top_k=0, repetition_penalty=1.0, presence/frequency_penalty=0.0, max_tokens=64</td>
	</tr>
	<tr>
		<td>런타임</td>
		<td>vLLM offline; batch_size=1; max_num_seqs=1; max_model_len=8192; moe_backend=triton</td>
	</tr>
	<tr>
		<td>출력</td>
		<td>JSON Schema 강제, decision 필드 하나; 모든 호출 독립</td>
	</tr>
	<tr>
		<td>입력 일관성</td>
		<td>동일 scene의 image·target ID·ID mapping 고정; system prompt·task 문구·schema 고정</td>
	</tr>
	<tr>
		<td>검증</td>
		<td>고유 run_id 1,500개; 조건별 125개; 모든 scene–condition의 seed 5개 완비</td>
	</tr>
	<tr>
		<td>완료 상태</td>
		<td>schema valid 1,500/1,500; infrastructure error 0; 전부 첫 attempt 완료; finish_reason=stop 1,500개</td>
	</tr>
</table>

## 3. 실제 사용 프롬프트
아래는 실행 config에 저장된 원문이다. Geometry JSON 외에 조건별 추가 설명이나 그리퍼 사양은 넣지 않았다.
### System prompt
```plain text
You assess whether a robot can retrieve a target object from a shelf.

Each numbered badge identifies the physical object underneath it.
Badge positions, sizes, and numbers are identity markers only.

Judge the current scene.
Return exactly one JSON object. Do not include explanations.
```
### User prompt template
```plain text
Target object ID: {target_object_id}

Geometry information:
{geometry_json}

Assume the robot can grasp and retrieve the target when it is
unobstructed.

In the current scene, can the robot pick up the target and take it
out through the shelf opening while leaving all other objects
in place and avoiding contact with them?

Consider whether surrounding objects obstruct access to the target
or its removal.

Partial visibility alone does not mean retrieval is blocked.
Being near another object alone does not mean retrieval is blocked.

Choose one:
- RETRIEVE_NOW: the target can be retrieved without rearranging
  other objects.
- REARRANGE_FIRST: another object must be moved before the target
  can be retrieved.

Return exactly one JSON object with the field "decision".
Do not identify the blocker or propose an action.
```
C0에서는 `{geometry_json}`을 `null`로, 나머지는 각 조건의 JSON 원문으로 치환한다.
### Output schema
```json
{
  "type": "object",
  "properties": {
    "decision": {
      "type": "string",
      "enum": [
        "RETRIEVE_NOW",
        "REARRANGE_FIRST"
      ]
    }
  },
  "required": [
    "decision"
  ],
  "additionalProperties": false
}
```

## 4. Geometry 입력 정의와 출처
- R: target–object 관계. 수치형은 surface gap, center bearing, occupied angular sectors; 정성형은 clearance 범주, center direction, occupied target sides.
- F: 물체 footprint 외곽에서 방향별 주변 자유 공간.
- M: 전체 footprint와 swept region을 고려한 translation·rotation capacity.
- D4/D8은 4방향/8방향 구분이다. C0와 R 수치형은 실제 입력이 같으므로 D4/D8로 복제하지 않았다.
- R 정성형에는 기존 정의의 `CLEARANCE_ZONE_OVERLAP`(surface gap ≤ 5 cm) / `CLEARANCE_ZONE_CLEAR`(> 5 cm)가 포함된다. 그리퍼 치수는 아니지만, 5 cm 기준의 범주화가 있으므로 결과를 “방향 정보만의 효과”라고 해석하면 안 된다.
- Geometry는 `vlm_action_geometry_single_info_maskfix_validation`의 원래 R/F/M 정의를 사용했다. 모든 scene의 이미지 hash와 simulator-instance/display-ID 대응이 v10 평가 이미지와 일치함을 확인했다.
- visible object record만 제공하고 object_id 오름차순 정렬을 유지했다. lift_and_relocate_v04의 완전 가림 물체 ID 16은 11개 non-C0 payload의 record에서 제외했다. 나머지 기하 측정값은 재계산하지 않고 기존 값을 유지했다.
- direct_graspable 정답, blocker ID, best action, retrieval 성공 여부는 입력하지 않았다.

## 5. 전체 조건별 결과
각 행은 125회다. “바로 회수”는 RETRIEVE_NOW, “먼저 정리”는 REARRANGE_FIRST다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>입력</td>
		<td>바로 회수</td>
		<td>먼저 정리</td>
		<td>설계 일치 횟수</td>
		<td>설계 일치율</td>
		<td>5회 만장일치 장면 /25</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>RGB only</td>
		<td>47</td>
		<td>78</td>
		<td>103/125</td>
		<td>82.4%</td>
		<td>22</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>R 수치형</td>
		<td>39</td>
		<td>86</td>
		<td>101/125</td>
		<td>80.8%</td>
		<td>17</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>R 정성형 D4</td>
		<td>41</td>
		<td>84</td>
		<td>109/125</td>
		<td>87.2%</td>
		<td>23</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>R 정성형 D8</td>
		<td>31</td>
		<td>94</td>
		<td>117/125</td>
		<td>93.6%</td>
		<td>23</td>
	</tr>
	<tr>
		<td>C3_D4</td>
		<td>F 수치형 D4</td>
		<td>57</td>
		<td>68</td>
		<td>87/125</td>
		<td>69.6%</td>
		<td>14</td>
	</tr>
	<tr>
		<td>C3_D8</td>
		<td>F 수치형 D8</td>
		<td>56</td>
		<td>69</td>
		<td>94/125</td>
		<td>75.2%</td>
		<td>17</td>
	</tr>
	<tr>
		<td>C4_D4</td>
		<td>F 정성형 D4</td>
		<td>33</td>
		<td>92</td>
		<td>99/125</td>
		<td>79.2%</td>
		<td>16</td>
	</tr>
	<tr>
		<td>C4_D8</td>
		<td>F 정성형 D8</td>
		<td>33</td>
		<td>92</td>
		<td>101/125</td>
		<td>80.8%</td>
		<td>18</td>
	</tr>
	<tr>
		<td>C5_D4</td>
		<td>M 수치형 D4</td>
		<td>38</td>
		<td>87</td>
		<td>98/125</td>
		<td>78.4%</td>
		<td>16</td>
	</tr>
	<tr>
		<td>C5_D8</td>
		<td>M 수치형 D8</td>
		<td>49</td>
		<td>76</td>
		<td>97/125</td>
		<td>77.6%</td>
		<td>17</td>
	</tr>
	<tr>
		<td>C6_D4</td>
		<td>M 정성형 D4</td>
		<td>36</td>
		<td>89</td>
		<td>102/125</td>
		<td>81.6%</td>
		<td>18</td>
	</tr>
	<tr>
		<td>C6_D8</td>
		<td>M 정성형 D8</td>
		<td>45</td>
		<td>80</td>
		<td>99/125</td>
		<td>79.2%</td>
		<td>21</td>
	</tr>
</table>

### Family별 기대 응답 횟수
FC_CLEAR에서는 바로 회수, 나머지 family에서는 먼저 정리 응답을 센다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>FC_CLEAR 바로 회수 /25</td>
		<td>FC_BLOCKED 먼저 정리 /25</td>
		<td>TRANSLATE 먼저 정리 /25</td>
		<td>ROTATE 먼저 정리 /25</td>
		<td>LIFT 먼저 정리 /25</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>25</td>
		<td>6</td>
		<td>23</td>
		<td>24</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>20</td>
		<td>12</td>
		<td>20</td>
		<td>24</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>25</td>
		<td>9</td>
		<td>25</td>
		<td>25</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>24</td>
		<td>18</td>
		<td>25</td>
		<td>25</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C3_D4</td>
		<td>22</td>
		<td>3</td>
		<td>21</td>
		<td>21</td>
		<td>20</td>
	</tr>
	<tr>
		<td>C3_D8</td>
		<td>25</td>
		<td>4</td>
		<td>20</td>
		<td>21</td>
		<td>24</td>
	</tr>
	<tr>
		<td>C4_D4</td>
		<td>16</td>
		<td>12</td>
		<td>23</td>
		<td>23</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C4_D8</td>
		<td>17</td>
		<td>11</td>
		<td>23</td>
		<td>25</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C5_D4</td>
		<td>18</td>
		<td>17</td>
		<td>21</td>
		<td>19</td>
		<td>23</td>
	</tr>
	<tr>
		<td>C5_D8</td>
		<td>23</td>
		<td>6</td>
		<td>22</td>
		<td>21</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C6_D4</td>
		<td>19</td>
		<td>12</td>
		<td>21</td>
		<td>25</td>
		<td>25</td>
	</tr>
	<tr>
		<td>C6_D8</td>
		<td>22</td>
		<td>4</td>
		<td>23</td>
		<td>25</td>
		<td>25</td>
	</tr>
</table>

### 이번 실행 C0와의 대응 비교
- C2 D8: C0 대비 16개 응답이 바뀌었다. 설계 의도 쪽으로 15개, 반대쪽으로 1개가 바뀌어 순증 14/125, +11.2%p.
- C2 D4: 6개 응답이 바뀌었고 모두 설계 의도 쪽이었다. 82.4% → 87.2%, +4.8%p.
- C1: 16개 응답 중 설계 의도 쪽 7개, 반대쪽 9개. 82.4% → 80.8%, −1.6%p.
- 모든 F/M 조건은 이번 C0보다 설계 일치율이 낮았다. Geometry를 추가하는 것 자체가 개선을 보장하지 않았다.

## 6. C1과 C2 D4: 불일치가 나온 전체 장면
- C1 R 수치형: 9/25개 장면에서 한 번 이상 불일치. 총 24/125회 불일치.
- C2 R 정성형 D4: 4/25개 장면에서 한 번 이상 불일치. 총 16/125회 불일치.
- 아래 숫자는 모두 **바로 회수 / 먼저 정리** 횟수다. 장면당 조건별 5회이며, 이미지와 개별 불일치 seed는 8절에 있다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>장면</td>
		<td>타깃 ID</td>
		<td>설계상 기대</td>
		<td>C1 바로/정리</td>
		<td>C2 D4 바로/정리</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v01</td>
		<td>64</td>
		<td>REARRANGE_FIRST</td>
		<td>4 / 1</td>
		<td>5 / 0</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v02</td>
		<td>53</td>
		<td>REARRANGE_FIRST</td>
		<td>4 / 1</td>
		<td>4 / 1</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v03</td>
		<td>12</td>
		<td>REARRANGE_FIRST</td>
		<td>3 / 2</td>
		<td>5 / 0</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v04</td>
		<td>89</td>
		<td>REARRANGE_FIRST</td>
		<td>2 / 3</td>
		<td>2 / 3</td>
	</tr>
	<tr>
		<td>scene_fc_clear_v01</td>
		<td>53</td>
		<td>RETRIEVE_NOW</td>
		<td>2 / 3</td>
		<td>5 / 0</td>
	</tr>
	<tr>
		<td>scene_fc_clear_v04</td>
		<td>61</td>
		<td>RETRIEVE_NOW</td>
		<td>4 / 1</td>
		<td>5 / 0</td>
	</tr>
	<tr>
		<td>scene_fc_clear_v05</td>
		<td>62</td>
		<td>RETRIEVE_NOW</td>
		<td>4 / 1</td>
		<td>5 / 0</td>
	</tr>
	<tr>
		<td>scene_rotate_v03</td>
		<td>83</td>
		<td>REARRANGE_FIRST</td>
		<td>1 / 4</td>
		<td>0 / 5</td>
	</tr>
	<tr>
		<td>scene_translate_v02</td>
		<td>45</td>
		<td>REARRANGE_FIRST</td>
		<td>5 / 0</td>
		<td>0 / 5</td>
	</tr>
</table>

C1은 FC_BLOCKED 외에도 FC_CLEAR를 불필요하게 먼저 정리로 답하고, TRANSLATE v02를 5회 모두 바로 회수로 답했다. C2 D4는 나머지 21개 장면에서 모두 5/5 일치했지만 FC_BLOCKED v01·v03에서는 C1보다 불일치가 늘었다. 따라서 C2 D4의 전체 개선을 FC_BLOCKED 해결로 해석하지 않는다.

## 7. 최고 조건 C2 D8의 잔여 불일치
3개 장면에 걸쳐 총 8/125회 불일치다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>장면</td>
		<td>타깃 ID</td>
		<td>설계상 기대</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v02</td>
		<td>53</td>
		<td>REARRANGE_FIRST</td>
		<td>2 / 3</td>
		<td>28102, 28105</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v03</td>
		<td>12</td>
		<td>REARRANGE_FIRST</td>
		<td>5 / 0</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>scene_fc_clear_v05</td>
		<td>62</td>
		<td>RETRIEVE_NOW</td>
		<td>4 / 1</td>
		<td>28101</td>
	</tr>
</table>

C2 D8에서 FC_BLOCKED의 먼저 정리 응답은 C0 6/25 → 18/25로 증가했고, FC_CLEAR의 바로 회수 응답은 25/25 → 24/25로 대부분 유지됐다. TRANSLATE·ROTATE·LIFT의 먼저 정리 응답은 72/75 → 75/75였다.

## 8. 불일치 사례 원본 이미지와 결과
실제 추론에 사용한 v10 Numbered RGB 원본을 수정 없이 첨부했다. 각 표에는 RGB only 기준인 C0와 C1·C2 D4·C2 D8을 함께 표시했다. 모든 표의 “기대”는 장면 설계 의도이며, 모델 설명은 수집하지 않았으므로 개별 응답의 내부 판단 이유는 확인할 수 없다.

### scene_fc_blocked_v01 — 타깃 ID 64
설계상 기대: `REARRANGE_FIRST`.
<image src="file-upload://3de952c9-e273-8190-944c-00b211428bd9"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>4 / 1</td>
		<td>4</td>
		<td>28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
</table>

### scene_fc_blocked_v02 — 타깃 ID 53
설계상 기대: `REARRANGE_FIRST`.
<image src="file-upload://3de952c9-e273-8118-bf75-00b2ad6eee83"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>4 / 1</td>
		<td>4</td>
		<td>28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>4 / 1</td>
		<td>4</td>
		<td>28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>4 / 1</td>
		<td>4</td>
		<td>28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>2 / 3</td>
		<td>2</td>
		<td>28102, 28105</td>
	</tr>
</table>

### scene_fc_blocked_v03 — 타깃 ID 12
설계상 기대: `REARRANGE_FIRST`.
<image src="file-upload://3de952c9-e273-81f4-a4ce-00b230448a09"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>3 / 2</td>
		<td>3</td>
		<td>28102, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
</table>

### scene_fc_blocked_v04 — 타깃 ID 89
설계상 기대: `REARRANGE_FIRST`.
<image src="file-upload://3de952c9-e273-8169-a14d-00b29d74d053"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>2 / 3</td>
		<td>2</td>
		<td>28102, 28105</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>2 / 3</td>
		<td>2</td>
		<td>28102, 28105</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
</table>

### scene_fc_clear_v01 — 타깃 ID 53
설계상 기대: `RETRIEVE_NOW`.
<image src="file-upload://3de952c9-e273-811c-98d2-00b2c98e8fc9"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>2 / 3</td>
		<td>3</td>
		<td>28101, 28103, 28104</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
</table>

### scene_fc_clear_v04 — 타깃 ID 61
설계상 기대: `RETRIEVE_NOW`.
<image src="file-upload://3de952c9-e273-81cb-b115-00b262d3139e"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>4 / 1</td>
		<td>1</td>
		<td>28101</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
</table>

### scene_fc_clear_v05 — 타깃 ID 62
설계상 기대: `RETRIEVE_NOW`.
<image src="file-upload://3de952c9-e273-813c-8666-00b2b7488be5"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>4 / 1</td>
		<td>1</td>
		<td>28103</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>5 / 0</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>4 / 1</td>
		<td>1</td>
		<td>28101</td>
	</tr>
</table>

### scene_rotate_v03 — 타깃 ID 83
설계상 기대: `REARRANGE_FIRST`.
<image src="file-upload://3de952c9-e273-81f7-9055-00b25477954a"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>1 / 4</td>
		<td>1</td>
		<td>28105</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
</table>

### scene_translate_v02 — 타깃 ID 45
설계상 기대: `REARRANGE_FIRST`.
<image src="file-upload://3de952c9-e273-81c4-bec8-00b2582ec724"></image>
<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>바로 회수 / 먼저 정리</td>
		<td>불일치 /5</td>
		<td>불일치 seed</td>
	</tr>
	<tr>
		<td>C0</td>
		<td>2 / 3</td>
		<td>2</td>
		<td>28102, 28105</td>
	</tr>
	<tr>
		<td>C1</td>
		<td>5 / 0</td>
		<td>5</td>
		<td>28101, 28102, 28103, 28104, 28105</td>
	</tr>
	<tr>
		<td>C2_D4</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
	<tr>
		<td>C2_D8</td>
		<td>0 / 5</td>
		<td>0</td>
		<td>없음</td>
	</tr>
</table>

## 9. 25개 장면 × 12조건 전체 응답표
각 값은 5회 중 RETRIEVE_NOW 횟수다. REARRANGE_FIRST 횟수는 5에서 해당 값을 뺀 값이다. 읽기 쉬운 CSV도 11절에 첨부했다.
<details>
<summary>전체 장면별 RETRIEVE_NOW 횟수 펼치기</summary>
	<table fit-page-width="true" header-row="true">
		<tr>
			<td>scene_id</td>
			<td>target_object_id</td>
			<td>C0</td>
			<td>C1</td>
			<td>C2_D4</td>
			<td>C2_D8</td>
			<td>C3_D4</td>
			<td>C3_D8</td>
			<td>C4_D4</td>
			<td>C4_D8</td>
			<td>C5_D4</td>
			<td>C5_D8</td>
			<td>C6_D4</td>
			<td>C6_D8</td>
		</tr>
		<tr>
			<td>scene_fc_blocked_v01</td>
			<td>64</td>
			<td>5</td>
			<td>4</td>
			<td>5</td>
			<td>0</td>
			<td>3</td>
			<td>3</td>
			<td>2</td>
			<td>4</td>
			<td>2</td>
			<td>5</td>
			<td>0</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_blocked_v02</td>
			<td>53</td>
			<td>4</td>
			<td>4</td>
			<td>4</td>
			<td>2</td>
			<td>4</td>
			<td>3</td>
			<td>3</td>
			<td>3</td>
			<td>1</td>
			<td>5</td>
			<td>2</td>
			<td>2</td>
		</tr>
		<tr>
			<td>scene_fc_blocked_v03</td>
			<td>12</td>
			<td>5</td>
			<td>3</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>3</td>
			<td>4</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_blocked_v04</td>
			<td>89</td>
			<td>5</td>
			<td>2</td>
			<td>2</td>
			<td>0</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>0</td>
			<td>0</td>
			<td>2</td>
			<td>3</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_blocked_v05</td>
			<td>59</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>5</td>
			<td>5</td>
			<td>0</td>
			<td>3</td>
			<td>0</td>
			<td>2</td>
			<td>3</td>
			<td>4</td>
		</tr>
		<tr>
			<td>scene_fc_clear_v01</td>
			<td>53</td>
			<td>5</td>
			<td>2</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>4</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_clear_v02</td>
			<td>29</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>4</td>
			<td>5</td>
			<td>3</td>
			<td>0</td>
			<td>3</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_clear_v03</td>
			<td>95</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>4</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_clear_v04</td>
			<td>61</td>
			<td>5</td>
			<td>4</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>5</td>
			<td>2</td>
			<td>3</td>
			<td>5</td>
			<td>5</td>
			<td>2</td>
			<td>5</td>
		</tr>
		<tr>
			<td>scene_fc_clear_v05</td>
			<td>62</td>
			<td>5</td>
			<td>4</td>
			<td>5</td>
			<td>4</td>
			<td>3</td>
			<td>5</td>
			<td>2</td>
			<td>4</td>
			<td>1</td>
			<td>3</td>
			<td>2</td>
			<td>2</td>
		</tr>
		<tr>
			<td>scene_lift_and_relocate_v01</td>
			<td>59</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_lift_and_relocate_v02</td>
			<td>70</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>1</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_lift_and_relocate_v03</td>
			<td>12</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_lift_and_relocate_v04</td>
			<td>68</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_lift_and_relocate_v05</td>
			<td>43</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>1</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_rotate_v01</td>
			<td>12</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_rotate_v02</td>
			<td>38</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>2</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
			<td>4</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_rotate_v03</td>
			<td>83</td>
			<td>0</td>
			<td>1</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_rotate_v04</td>
			<td>47</td>
			<td>1</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>2</td>
			<td>2</td>
			<td>2</td>
			<td>0</td>
			<td>2</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_rotate_v05</td>
			<td>44</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_translate_v01</td>
			<td>67</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_translate_v02</td>
			<td>45</td>
			<td>2</td>
			<td>5</td>
			<td>0</td>
			<td>0</td>
			<td>1</td>
			<td>2</td>
			<td>2</td>
			<td>2</td>
			<td>4</td>
			<td>1</td>
			<td>2</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_translate_v03</td>
			<td>18</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>3</td>
			<td>2</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>1</td>
			<td>0</td>
			<td>2</td>
		</tr>
		<tr>
			<td>scene_translate_v04</td>
			<td>13</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>1</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
		</tr>
		<tr>
			<td>scene_translate_v05</td>
			<td>54</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>0</td>
			<td>1</td>
			<td>2</td>
			<td>0</td>
		</tr>
	</table>
</details>

## 10. 해석과 한계
1. **관찰된 결과:** 현재 모델·프롬프트·25개 장면에서는 R 정성형, 특히 D8이 설계 의도와 가장 잘 맞았다. F/M은 추가 입력만으로 개선되지 않았다.
2. **인과 해석 제한:** D4/D8 비교는 방향 범주뿐 아니라 JSON token 및 occupied sides 표기도 달라진다. R 정성형의 5 cm clearance 범주도 함께 작용할 수 있다. 특정 요소만의 효과라고 결론내리지 않는다.
3. **GT 불일치:** 원래 명세의 L2는 FULL visibility AND 모든 gap > 5 cm 규칙이다. 이번 질문은 주변 물체를 유지한 비접촉 회수 가능성이다. 서로 다른 task이므로 기존 L2 GT 점수를 그대로 적용하지 않았다.
4. **FC_BLOCKED의 특수성:** 해당 family는 손가락 여유 공간 가정을 포함해 설계됐다. 그리퍼를 제공하지 않는 현재 질문에서 설계 불일치를 곧바로 물리적 오답으로 단정할 수 없다.
5. **표본과 선택:** 25개 장면을 5회 반복한 결과이지 서로 다른 125개 장면 결과가 아니다. 최고 조건도 같은 25개에서 선택했다. 유의성 검정·별도 holdout 검증은 수행하지 않았다.
6. **재현성:** 동일 입력·seed의 C0를 새로 실행했을 때 이전 v15와 4/125개 응답이 달랐다. 이전 C0는 바로 회수 49 / 먼저 정리 76, 이번 C0는 47 / 78이다. 위 Geometry 비교는 전부 이번 C0 기준이다. 차이의 구체적 원인은 이번 실험에서 규명하지 않았다.
7. **현재 결론:** R 정성형 D8을 후속 검증 후보로 볼 근거는 있으나, 실제 회수 판단 문제를 해결했다거나 정확도 93.6%를 달성했다고 주장하지 않는다.
### C0 재실행에서 달라진 응답
<table fit-page-width="true" header-row="true">
	<tr>
		<td>장면</td>
		<td>seed</td>
	</tr>
	<tr>
		<td>scene_rotate_v04</td>
		<td>28102</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v05</td>
		<td>28105</td>
	</tr>
	<tr>
		<td>scene_rotate_v02</td>
		<td>28102</td>
	</tr>
	<tr>
		<td>scene_fc_blocked_v02</td>
		<td>28101</td>
	</tr>
</table>
### 미수행 후속 검토안
- 새 L2 질문에 대응하는 실행 기반 GT 또는 명시적 평가 기준 정리.
- 새로운 장면에서 후보 조건 검증.
- FC_BLOCKED의 그리퍼 가정과 새 질문의 충돌 분리.
위 항목은 제안이며 이번에 실행한 결과가 아니다.

## 11. 첨부 데이터와 재현 경로
### Notion 첨부 파일
로컬 파일 경로만 남기지 않고 결과와 설정을 직접 첨부했다. scene_results에는 모든 scene–condition의 5개 seed별 decision이 포함된다.

<file src="file-upload://3de952c9-e273-81b7-8b65-00b22c449d33"></file>

<file src="file-upload://3de952c9-e273-812a-8a30-00b2bad63173"></file>

<file src="file-upload://3de952c9-e273-81ff-a2cf-00b25d5a36f1"></file>

<file src="file-upload://3de952c9-e273-819b-b0f2-00b2efa78e55"></file>

<file src="file-upload://3de952c9-e273-81f8-bd70-00b24278b7c6"></file>

### 로컬 원자료
아래는 실험 머신의 경로이며 Notion에서 열리는 웹 링크가 아니다.
```plain text
/home/ssu/ShelfScene/
  scripts/run_l2_direct_retrieval_geometry_v16.py
  experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16/
    config/experiment_config.json
    config/frozen_scenes.json
    config/run_table.jsonl
    prompts/system_en.txt
    prompts/l2_template_en.txt
    schemas/l2.json
    geometry/
    logs/runs.jsonl
    logs/attempts.jsonl
    logs/console.log
    audit/preflight_report.json
    audit/completion_report.json
    results/summary.json
    results/scene_results.json
    results/paired_changes_vs_c0.json
    results/retrieve_counts_by_scene.csv
    results/condition_results.csv
    results/report.md

이미지 원본:
experiments/vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10/images/numbered_rgb/

Geometry 원본:
experiments/vlm_action_geometry_single_info_maskfix_validation/geometry/
```
기존 로그로 분석 결과만 다시 생성:
```bash
.qwen3-vl/venv/bin/python scripts/run_l2_direct_retrieval_geometry_v16.py --analyze-only
```
### 기록 범위
2026-09-17 v16 완료 로그와 현재 대화의 해석을 정리했다. 기존 v1 실행 명세 본문은 수정하지 않으며, task 변경과 실제 실행 조건은 이 결과 페이지 안에서 별도 명시한다.
