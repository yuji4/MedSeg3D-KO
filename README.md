# MedSeg3D-KO

## 프로젝트 소개
M3D-LaMed 기반 한국어 지원 3D CT 세그멘테이션 시스템

한국어로 질문하면 CT에서 원하는 장기/종양을 자동으로 세그멘테이션하고 부피·크기를 계산해주는 웹

## 주요 기능
- 한국어 질의 지원 (번역 레이어)
- 다장기 세그멘테이션 (104종, TotalSegmentator 기반)
- 정량적 분석 (장기 부피·크기 자동 계산)
- Gradio 웹 UI

## 베이스 모델
M3D-LaMed-Phi-3-4B

## 데이터셋
- TotalSegmentator
- MSD Task07 (췌장 종양)
- BTCV

