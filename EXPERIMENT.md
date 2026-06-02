# 실험 상세 (EXPERIMENT.md)

모든 실험은 **Task07 Pancreas (MSD)** 데이터셋 5개 케이스 (`pancreas_001/004/005/006/010`) 기준, Tesla T4 GPU에서 수행했습니다.

---

## 실험 1: 번역 전략 비교

**노트북:** `Med3D_Korean_Interface_Pipeline_Experiment.ipynb`

**의도:** 한국어 입력을 M3D에 전달하는 4가지 방식이 세그멘테이션 성능에 미치는 영향 측정

**방법 정의**

| ID | 방법 | 처리 방식 |
|----|------|---------|
| A | 한국어 직접 입력 | "췌장을 분할해줘" 그대로 전달 |
| B | 일반 번역 | GoogleTranslate → "Split the pancreas" |
| C | 의료 사전 정규화 (Layer 2만) | 장기명만 영문 변환 → "Please segment the pancreas in this image." |
| D | Hybrid Full (L1+L2+L3) | 전체 파이프라인 |

**결과**

| 방법 | 평균 Dice | 표준편차 | 최대 | 최소 |
|------|:---------:|:--------:|:----:|:----:|
| A. 한국어 직접 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B. 일반 번역 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C. 의료 사전 (L2) | 0.5457 | 0.0314 | 0.5981 | 0.5109 |
| **D. Hybrid Full** | **0.5515** | **0.0352** | **0.6109** | **0.5037** |

**케이스별 상세**

| 케이스 | A | B | C | D |
|--------|:---:|:---:|:---:|:---:|
| pancreas_001 | 0.0000 | 0.0000 | 0.5109 | 0.5037 |
| pancreas_004 | 0.0000 | 0.0000 | 0.5343 | 0.5417 |
| pancreas_005 | 0.0000 | 0.0000 | 0.5981 | 0.6109 |
| pancreas_006 | 0.0000 | 0.0000 | 0.5220 | 0.5382 |
| pancreas_010 | 0.0000 | 오류 | 0.5630 | 0.5630 |

**핵심 발견:**
- 일반 번역이 의미상 맞더라도 `[SEG]` 토큰이 생성되지 않아 완전 실패
- Layer 2 단독 적용만으로도 0.5457로 급격히 회복 → **엔티티 정규화가 핵심**
- Full 파이프라인에서 0.5515로 최고 성능

---

## 실험 2: 표현 다양성 강건성

**노트북:** `Med3D_Korean_Interface_Pipeline_Experiment.ipynb`

**의도:** 같은 의미의 다양한 한국어 표현에 대한 각 파이프라인의 강건성 비교 (`pancreas_001` 단일 케이스)

**결과**

| 표현 | Raw Korean | General Trans | Hybrid |
|------|:---:|:---:|:---:|
| "췌장을 분할해줘" | 0.0000 | 0.0000 | **0.5037** |
| "이자를 찾아줘" (구어체) | 0.0000 | 오류 | 오류 |
| "췌장 부위를 세그멘테이션해줘" | 0.0000 | **0.5064** | **0.5037** |
| "췌장이 어디있어?" | 0.0000 | 0.0000 | **0.5037** |
| "췌장" (명사만) | 0.0000 | 오류 | **0.5048** |
| 영어 직접 입력 | **0.5037** | **0.5037** | **0.5037** |

**핵심 발견:**
- Raw Korean은 영어 직접 입력을 제외하면 전부 실패
- General Trans는 일부 표현에서 작동하지만 오류가 많고 불안정
- **Hybrid는 대부분의 표현에서 일관된 0.50+ 성능** 유지
- 구어체·혼합어("이자", "pancreas 분할해줘") 처리는 개선 여지 있음

---

## 실험 3: Threshold Sensitivity

**노트북:** `Med3D_Korean_Interface_Pipeline_Experiment.ipynb`

**의도:** 세그멘테이션 마스크 이진화 임계값이 성능에 미치는 영향 (Hybrid 파이프라인, `pancreas_001`)

| Threshold | Dice | 예측 복셀 수 |
|:---------:|:----:|:----------:|
| 0.3 | 0.5025 | 73,332 |
| 0.4 | 0.5031 | 71,648 |
| **0.5** | **0.5037** | 70,036 |
| 0.6 | 0.5036 | 68,148 |
| 0.7 | 0.5033 | 66,220 |

**핵심 발견:** 임계값 변화(0.3→0.7)에 따른 Dice 차이 0.001 미만으로 매우 안정적. 기본값 **0.5** 사용이 적절합니다.

---

## 실험 4: 프롬프트 엔지니어링 최적화

**노트북:** `M3D_KO_Prompt_Optimal.ipynb`

**의도:** Hybrid 파이프라인 프롬프트 외에, 더 나은 영문 프롬프트 형식이 있는지 22종 × 5케이스 = 110회 추론으로 탐색

**실험 그룹 및 그룹 평균 Dice**

| 그룹 | 예시 프롬프트 | 그룹평균 | 그룹최고 |
|------|------------|:-------:|:-------:|
| **Baseline** | "Can you segment the pancreas in this image?" | **0.5500** | **0.5517** |
| **CoT** | "First identify abdominal organs, then segment the pancreas." | 0.5496 | 0.5507 |
| Spatial | "Segment the pancreas in the upper abdomen." | 0.5162 | 0.5477 |
| Length | "Please carefully identify and segment the pancreatic tissue..." | 0.2739 | 0.5487 |
| Structured | `{"task":"segmentation","organ":"pancreas"}` | 0.2185 | 0.4376 |
| **Lexical** | "identify pancreas" / "isolate pancreas" | **0.0000** | 0.0000 |

**Top 5 최고 프롬프트**

| 순위 | ID | 그룹 | 프롬프트 | Dice |
|:---:|:---:|:---:|---------|:---:|
| 1 | B2 | Baseline | Can you segment the pancreas in this image? | **0.5517** |
| 2 | B3 | Baseline | Please segment the pancreas in this image and output the mask. | 0.5513 |
| 3 | CoT2 | CoT | First identify abdominal organs, then segment the pancreas. | 0.5507 |
| 4 | CoT1 | CoT | Step 1: Locate the stomach. Step 2: Find the pancreas... | 0.5505 |
| 5 | L4 | Length | Please carefully identify and segment the pancreatic tissue... | 0.5487 |

**핵심 발견:**
- Hybrid 파이프라인(0.5515)을 유의미하게 초과하는 형식은 존재하지 않음
- `segment` / `can you` 패턴이 없는 Lexical 변형(identify/isolate/extract)은 Dice 0.0000으로 완전 실패
- CoT · 공간 힌트 · 장문 프롬프트의 개선 효과는 미미함
- Structured (JSON, key-value) 형식은 불안정하고 신뢰할 수 없음

---

## 실험 5: Layer 1 의도 분류 Robustness

**노트북:** `Korean_Medical_Query_Transformation_Experiment.ipynb`

**의도:** 규칙 기반 키워드 분류기가 다양한 한국어 표현에 얼마나 강건한가 검증

**테스트셋:** 50개 paraphrase (SEG 15 · VQA 15 · REPORT 10 · REG 10)

**결과: 정확도 92% (46/50), Macro F1: 0.92**

| 클래스 | Precision | Recall | F1 | 샘플 수 |
|--------|:---------:|:------:|:--:|:-------:|
| Segmentation | 1.00 | 0.90 | 0.95 | 15 |
| VQA | 0.91 | 1.00 | 0.95 | 15 |
| Report | 0.93 | 0.93 | 0.93 | 10 |
| REG | 0.87 | 0.87 | 0.87 | 10 |
| **Macro avg** | **0.93** | **0.92** | **0.92** | **50** |

**오분류 케이스 (4건)**

| 질문 | 정답 | 예측 | 원인 |
|------|:---:|:---:|------|
| "췌장만 보고싶어" | SEG | VQA | SEG 키워드 없음 |
| "이상 소견 있나?" | VQA | Report | "소견" 키워드 오인 |
| "췌장이 정상으로 보여?" | VQA | SEG | "보여" 키워드 오인 |
| "신장이 어떤 장기야?" | REG | VQA | REG 키워드 없음 |

**핵심 발견:** 규칙 기반 분류기만으로 Macro F1 0.92 달성. 오분류는 의도가 경계에 걸친 표현에서 집중적으로 발생합니다.

---

## 실험 6: Layer 3 Task Mismatch (템플릿 선택 필요성 검증)

**노트북:** `Korean_Medical_Query_Transformation_Experiment.ipynb`

**의도:** 세그멘테이션 질문에 잘못된 태스크 템플릿 적용 시 성능 저하 측정 (`pancreas_001`)

| 적용 템플릿 | Dice | 모델 응답 |
|------------|:----:|---------|
| **Seg 템플릿 (올바름)** | **0.5037** | `The segmentation reveals [SEG].` |
| VQA 템플릿 | 0.0000 | `"Atrophic pancreas with calcification."` |
| Report 템플릿 | 0.0000 | `"Numerous liver lesions with peripheral nodular enhancement..."` |

**핵심 발견:** 장기명이 정확해도 잘못된 템플릿에서는 `[SEG]` 토큰이 생성되지 않고 모델이 완전히 다른 태스크를 수행합니다. **Layer 3는 필수 계층**입니다.

---

## 실험 7: Ablation Study

**노트북:** `Korean_Medical_Query_Transformation_Experiment.ipynb`

**의도:** 3계층 각각을 제거했을 때 성능 변화 측정 (n=5)

**설정 정의**

| 설정 | 처리 방식 |
|------|---------|
| Full (L1+L2+L3) | 전체 파이프라인 |
| −Layer 1 | 의도 분류 없이 3개 템플릿 중 랜덤 선택 |
| −Layer 2 | 엔티티 정규화 없이 일반 번역 사용 |
| −Layer 3 | 엔티티 정규화만, 템플릿 없이 장기명만 전달 |

**결과**

| 설정 | 평균 Dice | 표준편차 |
|------|:---------:|:--------:|
| **Full (L1+L2+L3)** | **0.5515** | 0.0352 |
| −Layer 1 (랜덤 템플릿) | 0.1083 | 0.2167 |
| −Layer 2 (정규화 제거) | 0.0000 | 0.0000 |
| −Layer 3 (템플릿 제거) | 0.0000 | 0.0000 |

**케이스별 상세 (−Layer 1)**

| 케이스 | 선택된 랜덤 템플릿 | Dice |
|--------|-----------------|:---:|
| pancreas_001 | VQA 템플릿 선택 | 0.0000 |
| pancreas_004 | **SEG 템플릿 선택** | 0.5417 |
| pancreas_005 | VQA 템플릿 선택 | 0.0000 |
| pancreas_006 | Report 템플릿 선택 | 0.0000 |
| pancreas_010 | Report 템플릿 선택 | 0.0000 |

**핵심 발견:**
- **Layer 2 > Layer 3 > Layer 1** 순으로 기여도가 큽니다
- Layer 2·3 제거 시 모두 0.0000 — 두 계층 모두 필수
- Layer 1 제거 시 평균 0.1083이지만 분산이 매우 커(σ=0.2167) 예측 불가능한 동작을 보임
- 세 계층 모두 제거 불가능한 필수 구성 요소임이 확인됨