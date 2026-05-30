"""
3계층 한국어 의료 질의 변환 파이프라인.

Layer 1 — 의도 분류   (규칙 기반, 실험 정확도 92%)
Layer 2 — 엔티티 정규화 (없으면 Dice 0.000)
Layer 3 — 템플릿 선택  (잘못된 템플릿 → Dice 0.000)
"""
from __future__ import annotations

import re
from enum import Enum

from src.translation.medical_terms import TERM_KO

# ── 역방향 사전: 한국어 → 영문 ────────────────────────────────────────────────
_KO_TO_EN: dict[str, str] = {v: k for k, v in TERM_KO.items()}

# ── 검증된 템플릿 ─────────────────────────────────────────────────────────────
SEG_TEMPLATE    = "Can you segment the {organ} in this image? Please output the mask."
VQA_TEMPLATE    = "What is the condition of the {organ} in this image?"
REPORT_TEMPLATE = "What are the main findings in this medical image? Describe any abnormalities or notable observations visible in the scan."
REG_TEMPLATE    = "Describe the appearance and condition of the {organ} visible in this scan. Note its size, shape, and any observable abnormalities."


class Intent(str, Enum):
    SEGMENTATION = "Segmentation"
    VQA          = "VQA"
    REPORT       = "Report"
    REG          = "REG"


_INTENT_KO = {
    Intent.SEGMENTATION: "세그멘테이션",
    Intent.VQA:          "VQA (질의응답)",
    Intent.REPORT:       "소견 생성",
    Intent.REG:          "영역 설명",
}

# 우선순위 순서: SEGMENTATION > REPORT > REG > VQA
_INTENT_PATTERNS: list[tuple[Intent, re.Pattern]] = [
    (Intent.SEGMENTATION, re.compile(
        r"(분할|세그|마스크|segmen|segment|mask|찾아줘|표시|보여줘|어디|위치)",
        re.IGNORECASE,
    )),
    (Intent.REPORT, re.compile(
        r"(소견|리포트|report|findings|보고서|진단서|요약|생성해|작성)",
        re.IGNORECASE,
    )),
    (Intent.REG, re.compile(
        r"(설명|기능|역할|구조|작동|특징|어떻게\s*생겼)",
        re.IGNORECASE,
    )),
]


class KoreanMedicalQueryPipeline:
    """3계층 한국어 의료 질의 변환 파이프라인."""

    def transform(self, question_ko: str) -> dict:
        """
        Args:
            question_ko: 사용자 한국어 질문

        Returns:
            {
                "intent":    Intent,       # 분류된 의도 (enum)
                "intent_ko": str,          # 한국어 의도 표시용
                "organ":     str | None,   # 정규화된 장기 영문명 (Layer 2)
                "prompt":    str,          # M3D 영문 프롬프트 (Layer 3)
            }
        """
        intent   = self._classify_intent(question_ko)   # Layer 1
        organ_en = self._extract_organ(question_ko)      # Layer 2
        prompt   = self._select_template(intent, organ_en, question_ko)  # Layer 3
        return {
            "intent":    intent,
            "intent_ko": _INTENT_KO[intent],
            "organ":     organ_en,
            "prompt":    prompt,
        }

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    def _classify_intent(self, question_ko: str) -> Intent:
        for intent, pattern in _INTENT_PATTERNS:
            if pattern.search(question_ko):
                return intent
        return Intent.VQA

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    def _extract_organ(self, question_ko: str) -> str | None:
        """가장 긴 한국어 장기명을 우선 매칭하여 영문명 반환."""
        for ko, en in sorted(_KO_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
            if ko in question_ko:
                return en
        return None

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    def _select_template(
        self,
        intent: Intent,
        organ_en: str | None,
        question_ko: str,
    ) -> str:
        organ = organ_en or "the target region"

        if intent == Intent.SEGMENTATION:
            return SEG_TEMPLATE.format(organ=organ)

        if intent == Intent.REPORT:
            return REPORT_TEMPLATE

        if intent == Intent.REG:
            return REG_TEMPLATE.format(organ=organ)

        # VQA
        if organ_en:
            return VQA_TEMPLATE.format(organ=organ)

        # 장기 없는 VQA → 질문을 직접 영어로 번역
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source="ko", target="en").translate(question_ko)
        except Exception:
            return question_ko
