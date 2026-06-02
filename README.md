# MedSeg-3D-KO

<div align="center">

### 한국어로 CT를 물어보면, 장기를 세그멘테이션해서 임상 분석 결과를 드립니다

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)
![Model](https://img.shields.io/badge/Model-M3D--LaMed--Phi--3--4B-green)
![License](https://img.shields.io/badge/License-Apache_2.0-yellow)

</div>

---

![Main UI](docs/images/main_ui.png)

---

## 개요

M3D-LaMed는 영어 프롬프트로만 작동합니다. 한국어로 "췌장 분할해줘"라고 물으면 Dice Score **0.0000** — 완전히 실패합니다.

MedSeg-3D-KO는 **3계층 한국어 변환 파이프라인**을 실험으로 설계·검증하고, 이를 기반으로 한국어 질의 기반 CT 세그멘테이션 + 임상 분석 웹 앱을 구현한 프로젝트입니다.

```
"간 분할해줘"  →  [3계층 파이프라인]  →  M3D-LaMed  →  3D 마스크 + 부피(mL) + 임상 평가
```

한국 의료 현장에서 CT 분석 도구는 대부분 영어 전용입니다. 일반 번역기를 써도 M3D는 여전히 실패하며 (Dice 0.000), 의료 도메인 특화 정규화가 필수임을 7종의 실험으로 확인하고 최적 파이프라인을 설계했습니다.

---

## M3D-LaMed

[M3D-LaMed](https://arxiv.org/abs/2404.00578)는 BAAI에서 개발한 3D 의료 영상 특화 VLM(Vision Language Model)입니다. 3D ViT로 CT 볼륨을 인코딩하고 LLM(Phi-3 또는 LLaMA)과 결합해 세그멘테이션, VQA, 보고서 생성 등 8가지 의료 영상 태스크를 지원합니다.

| 구성 요소 | 역할 |
|-----------|------|
| 3D ViT Encoder | CT 볼륨 (32×256×256) → 공간 임베딩 |
| Spatial Pooling Perceiver | 3D 토큰 압축 및 LLM 입력 정렬 |
| Phi-3 LLM (4B) | 언어 이해 + `[SEG]` 토큰 생성 |
| SAM 3D Decoder | `[SEG]` 토큰 → 3D 세그멘테이션 마스크 |

단, 학습 데이터가 영어 기반이라 한국어 입력 시 Dice **0.0000**으로 완전 실패합니다. MedSeg-3D-KO는 이 문제를 3계층 파이프라인으로 해결합니다.

---

## 시스템 아키텍처

```mermaid
flowchart TD
    A["💬 한국어 질의"] --> L1

    subgraph PIPE["3계층 한국어 변환 파이프라인"]
        direction TB
        L1["Layer 1 — 의도 분류\nSEG · VQA · REPORT · REG"]
        L2["Layer 2 — 엔티티 정규화\n췌장→pancreas  콩팥→kidney"]
        L3["Layer 3 — 템플릿 선택\n검증된 M3D 영문 프롬프트 생성"]
        L1 --> L2 --> L3
    end

    L3 --> M["🧠 M3D-LaMed-Phi-3-4B\nVision Encoder + Phi-3 LLM + SAM 3D"]

    M --> R1["3D 세그멘테이션 마스크"]
    M --> R2["VQA 텍스트 답변"]
    M --> R3["CT 소견 보고서"]
    M --> R4["해부학 설명"]

    R1 --> Q["📊 정량 분석\n부피(mL) · 크기(mm) · 정상범위 비교"]
    Q --> UI
    R2 --> UI
    R3 --> UI
    R4 --> UI

    UI["🖥 Gradio 웹 UI\n3방향 CT 뷰어 · 환자 DB · PDF 보고서"]
```

---

## 3계층 한국어 변환 파이프라인

본 프로젝트의 핵심 연구 결과로, 7종의 실험을 통해 도출한 최적 구조입니다.

| 계층 | 역할 | 방법 | 제거 시 |
|------|------|------|--------|
| **Layer 1** 의도 분류 | SEG / VQA / REPORT / REG 판별 | 규칙 기반 키워드 (정확도 92%, F1 0.92) | Dice 0.108, σ=0.22 (불안정) |
| **Layer 2** 엔티티 정규화 ★ | 한국어 장기명 → 영문 canonical form | 104종 의료 용어 사전 | Dice 0.000 (완전 실패) |
| **Layer 3** 템플릿 선택 | 태스크별 최적 M3D 프롬프트 | 검증된 4가지 템플릿 | Dice 0.000 (완전 실패) |

### Layer 2가 핵심인 이유 — 토크나이저 분석

M3D의 토크나이저는 Phi-3 기반으로 한국어에 최적화되어 있지 않습니다. 같은 의미라도 한국어는 영어보다 훨씬 많은 토큰으로 파편화되어 임베딩 공간에서 전혀 다른 표현이 됩니다.

| 한국어 | 영어 | 한국어 토큰 수 | 영어 토큰 수 |
|--------|------|:---:|:---:|
| 간 | liver | 4 | 2 |
| 신장 | kidney | 3 | 2 |
| 췌장 | pancreas | 5 | 3 |
| 콩팥 (구어체) | kidney | **7** | 2 |
| 대동맥 | aorta | **6** | 3 |

### Layer 3 — 태스크별 검증 템플릿

```
SEG    → "Can you segment the {organ} in this image? Please output the mask."
VQA    → "What is the condition of the {organ} in this image?"
REPORT → "What are the main findings in this medical image? Describe any abnormalities
           or notable observations visible in the scan."
REG    → "Describe the appearance and condition of the {organ} visible in this scan.
           Note its size, shape, and any observable abnormalities."
```

---

## 주요 기능

### 1. 한국어 자연어 질의

4가지 M3D 태스크를 한국어로 자유롭게 요청할 수 있습니다. 3계층 파이프라인이 의도를 분류하고 최적 프롬프트로 변환합니다.

```
"간 분할해줘"         → SEG    → Can you segment the liver in this image? Please output the mask.
"췌장 상태 어때?"     → VQA    → What is the condition of the pancreas in this image?
"소견서 써줘"         → REPORT → What are the main findings in this medical image?
"비장 기능 설명해줘"  → REG    → Describe the appearance and condition of the spleen...
```

![Intent Classification](docs/images/intent_badge.png)

---

### 2. 다장기 3D 세그멘테이션

104종 해부학 구조물을 한 번의 질문으로 동시에 세그멘테이션하고, 축상·시상·관상 3방향 슬라이스에 색상별로 오버레이합니다.

- **윈도우 프리셋:** 복부 / 폐 / 뼈 / 뇌 4종 (HU 수동 조절 가능)
- **마스크 다운로드:** NIfTI 형식 (.nii.gz), 원본 affine 정보 보존

![Segmentation](docs/images/segmentation.png)

---

### 3. 임상 정량 분석

세그멘테이션 마스크에서 임상 수치를 자동 계산하고 성인 정상범위와 비교합니다.

| 지표 | 설명 |
|------|------|
| 부피 (mL) | 복셀 크기 × 복셀 수 |
| 크기 (mm) | 축별 바운딩박스 D × H × W |
| 임상 상태 | 정상 ✅ / 초과 ⚠️ / 미만 ⚠️ (연령·성별 보정) |

정상범위는 해부학·영상의학 교과서 기반 성인 기준값을 사용합니다.

| 장기 | 정상 범위 |
|------|-----------|
| 간 | 1,000 ~ 1,500 mL |
| 신장 (편측) | 90 ~ 160 mL |
| 비장 | 100 ~ 314 mL |
| 췌장 | 50 ~ 100 mL |

![Clinical Analysis](docs/images/clinical.png)

---

### 4. VQA / 소견 생성 / 영역 설명

M3D의 멀티태스크 기능을 한국어로 지원합니다. 모델 출력은 한국어로 자동 번역됩니다.

![VQA](docs/images/vqa.png)

---

### 5. 환자 관리 및 종단적 추이

SQLite 기반 환자 정보 저장 및 장기 부피 변화 추이를 인터랙티브 차트로 확인할 수 있습니다.

- 주민등록번호 자동 파싱 (나이/성별), 뒷자리 마스킹 `000000-*******`, 암호화 저장
- CSV 내보내기 (전체 / 환자별)

![Longitudinal](docs/images/longitudinal.png)

---

### 6. PDF 보고서 자동 생성

3방향 CT 이미지 + 장기별 결과표 + 한국어 면책 고지를 포함한 PDF를 자동으로 생성합니다.

![PDF Report](docs/images/pdf.png)

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 베이스 모델 | M3D-LaMed-Phi-3-4B (HuggingFace) |
| 딥러닝 | PyTorch 2.2, MONAI 1.3, Transformers 4.41 |
| 의료 영상 | SimpleITK 2.3, nibabel 5.2 |
| UI | Gradio 4.0+ |
| DB / 보고서 | SQLite3, fpdf2 |
| 시각화 | Plotly, Matplotlib, OpenCV |
| 번역 | deep-translator (GoogleTranslator) |
| 실행 환경 | Google Colab T4 GPU |

---

## 핵심 실험 결과

> 실험 전체 상세 (7종, 방법·결과·분석·종합 결론) → **[EXPERIMENT.md](EXPERIMENT.md)**

### 실험 1: 번역 전략 비교 — 파이프라인이 필요한 이유

> n=5, Task07 췌장 (MSD)

| 방법 | 평균 Dice | 비고 |
|------|:---------:|------|
| 한국어 직접 입력 | 0.0000 | `[SEG]` 토큰 미생성 |
| 일반 번역 (GoogleTranslate) | 0.0000 | "Split the pancreas" → 실패 |
| 의료 사전 정규화 (Layer 2만) | 0.5457 | 장기명 정규화만으로도 급격히 회복 |
| **3계층 Full 파이프라인** | **0.5515** | 최고 성능 |

### 실험 2: Ablation Study — 각 계층의 기여도

> n=5

| 설정 | 평균 Dice | 비고 |
|------|:---------:|------|
| **Full (L1+L2+L3)** | **0.5515** | 최고 성능 |
| −Layer 1 (랜덤 템플릿) | 0.1083 | 불안정 (σ=0.22) |
| −Layer 2 (정규화 제거) | 0.0000 | 완전 실패 |
| −Layer 3 (템플릿 제거) | 0.0000 | 완전 실패 |

**Layer 2 > Layer 3 > Layer 1** 순으로 기여도가 크며, 세 계층 모두 필수입니다.

![Ablation](docs/images/ablation.png)

---

## 디렉토리 구조

```
MedSeg-3D-KO/
├── app/
│   ├── gradio_app.py          # Gradio 웹 앱 메인
│   └── visualization.py       # 3방향 슬라이스 시각화
├── src/
│   ├── translation/
│   │   ├── pipeline.py        # 3계층 변환 파이프라인 (핵심)
│   │   ├── translator.py      # 한↔영 번역 레이어
│   │   └── medical_terms.py   # 104종 의료 용어 사전
│   ├── inference/
│   │   ├── model_loader.py    # M3D 모델 로드
│   │   └── segmentation.py    # 추론 파이프라인
│   ├── analysis/
│   │   ├── volume.py          # 부피·크기 계산
│   │   ├── clinical.py        # 정상범위 비교 / 임상 평가
│   │   └── report.py          # PDF 보고서 생성
│   └── database/
│       ├── db.py              # SQLite 연결 및 초기화
│       └── crud.py            # 환자 CRUD
├── notebooks/
│   ├── m3d_KO_run.ipynb                                      # Colab 실행 스크립트
│   ├── Med3D_Korean_Interface_Pipeline_Experiment.ipynb      # 실험 1~4
│   ├── Korean_Medical_Query_Transformation_Experiment.ipynb  # 실험 5~7
│   └── M3D_KO_Prompt_Optimal.ipynb                           # 실험 4 (프롬프트 최적화)
├── EXPERIMENT.md              # 실험 전체 상세
└── requirements.txt
```

---

## 실행 방법 (Google Colab)

```python
# 1. 레포 클론 및 패키지 설치
!git clone https://github.com/yuji4/MedSeg3D-KO.git
%cd MedSeg3D-KO
!pip install -r requirements.txt
!apt-get install -y fonts-nanum -q   # 한국어 폰트 (PDF 보고서용)

# 2. 모델 로드 (약 5분 소요)
import sys, torch
sys.path.insert(0, '/content/MedSeg3D-KO')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, model_max_length=512,
    padding_side="right", use_fast=False, trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16,
    device_map='auto', trust_remote_code=True,
)

# 3. 앱 실행
from src.inference.segmentation import SegmentationPipeline
import app.gradio_app as app_module

pipeline = SegmentationPipeline()
pipeline.model = model
pipeline.tokenizer = tokenizer
pipeline._device = next(model.parameters()).device
app_module._pipeline = pipeline

app_module.demo.queue()
app_module.demo.launch(share=True)
```

> ⚠️ T4 GPU 환경 권장.

---

## 데이터셋

| 데이터셋 | 설명 | 용도 |
|---------|------|------|
| [Task07 Pancreas (MSD)](http://medicaldecathlon.com/) | 췌장 CT + GT 마스크 281케이스 | 실험 정량 평가 (5케이스 사용) |
| NVIDIA Sample CT | 복부 CT 3케이스 | 데모 및 기능 테스트 |

---

## 참고

- [M3D: Advancing 3D Medical Image Analysis](https://arxiv.org/abs/2404.00578)
- [M3D GitHub](https://github.com/BAAI-DCAI/M3D)
- [M3D-LaMed-Phi-3-4B (HuggingFace)](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)