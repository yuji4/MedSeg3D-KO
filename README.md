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

## 무엇인가요?

M3D-LaMed는 영어 프롬프트로만 작동합니다. 한국어로 "췌장 분할해줘"라고 물으면 Dice Score **0.0000** — 완전히 실패합니다.

MedSeg-3D-KO는 **3계층 한국어 변환 파이프라인**을 설계해 이 문제를 해결하고, 한국어 질의 기반 CT 세그멘테이션 + 임상 분석 웹 앱을 구현한 프로젝트입니다.

```
"간 분할해줘"  →  [3계층 파이프라인]  →  M3D-LaMed  →  3D 마스크 + 부피(mL) + 임상 평가
```

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

## 주요 기능

| | 기능 | 설명 |
|--|------|------|
| 🔍 | **한국어 자연어 질의** | SEG / VQA / REPORT / REG 4가지 태스크 자동 분류 |
| 🫁 | **다장기 세그멘테이션** | 104종 해부학 구조물, 3방향 뷰 오버레이 |
| 📊 | **임상 정량 분석** | 부피(mL)·크기(mm) 자동 계산, 정상범위 비교, 연령·성별 보정 |
| 🏥 | **환자 관리** | SQLite 기반 검사 이력, 종단적 부피 추이 차트 (Plotly) |
| 📄 | **PDF 보고서** | 3방향 이미지 + 결과표 + 한국어 면책 고지 |

![Segmentation](docs/images/segmentation.png)

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

### 실험 1: 번역 전략 비교 — 왜 파이프라인이 필요한가

> **한국어를 어떻게 M3D에 전달해야 하는가?** (n=5, Task07 췌장)

| 방법 | Dice | 비고 |
|------|:----:|------|
| 한국어 직접 입력 | 0.0000 | `[SEG]` 토큰 미생성 |
| 일반 번역 (GoogleTranslate) | 0.0000 | "Split the pancreas" → 실패 |
| 의료 사전 정규화 (Layer 2만) | 0.5457 | 장기명만 맞춰도 급격히 회복 |
| **3계층 Full 파이프라인** | **0.5515** | **최고 성능** |

### 실험 2: Ablation Study — 각 계층이 정말 필요한가

> **계층 하나씩 제거하면 성능이 어떻게 변하는가?** (n=5)

| 설정 | Dice | 해석 |
|------|:----:|------|
| **Full (L1+L2+L3)** | **0.5515** | 최고 성능 |
| −Layer 1 (랜덤 템플릿) | 0.1083 | 불안정 (σ=0.22) |
| −Layer 2 (정규화 제거) | 0.0000 | 완전 실패 |
| −Layer 3 (템플릿 제거) | 0.0000 | 완전 실패 |

> 세 계층 모두 필수. **Layer 2 > Layer 3 > Layer 1** 순으로 기여도가 큽니다.

→ 실험 전체 상세 (7종): **[EXPERIMENT.md](EXPERIMENT.md)**

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
│   ├── m3d_KO_run.ipynb                                  # Colab 실행 스크립트
│   ├── Med3D_Korean_Interface_Pipeline_Experiment.ipynb  # 실험 1~4
│   ├── Korean_Medical_Query_Transformation_Experiment.ipynb  # 실험 5~7
│   └── M3D_KO_Prompt_Optimal.ipynb                       # 실험 4 (프롬프트 최적화)
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

## 참고

- [M3D: Advancing 3D Medical Image Analysis](https://arxiv.org/abs/2404.00578)
- [M3D GitHub](https://github.com/BAAI-DCAI/M3D)
- [M3D-LaMed-Phi-3-4B (HuggingFace)](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)