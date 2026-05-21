import re
from src.translation.medical_terms import get_korean_term, get_korean_description, TERM_KO


class MedicalTranslator:
    """M3D 모델 출력(영문)을 한국어로 변환하는 번역 레이어."""

    # 세그멘테이션 응답 패턴: "It is [SEG]." / "Sure, [SEG]." 등
    _SEG_PATTERN = re.compile(r"\[SEG\]", re.IGNORECASE)

    # 장기명이 포함된 응답 내 영문 장기명 검출용
    _TERM_PATTERN = re.compile(
        "|".join(re.escape(k) for k in sorted(TERM_KO.keys(), key=len, reverse=True)),
        re.IGNORECASE,
    )

    def translate_term(self, term: str) -> str:
        """단일 영문 장기/구조물명 → 한국어."""
        return get_korean_term(term)

    def translate_response(self, response: str) -> str:
        """모델 자유 텍스트 응답에서 장기명을 한국어로 치환."""
        def _replace(m: re.Match) -> str:
            return get_korean_term(m.group(0))

        return self._TERM_PATTERN.sub(_replace, response)

    def translate_segmentation_result(
        self,
        label: str,
        present: bool,
        volume_ml: float | None = None,
    ) -> str:
        """
        세그멘테이션 결과를 한국어 요약 문장으로 변환.

        Args:
            label: 영문 장기/구조물명 (예: "liver")
            present: 해당 구조물이 감지되었는지 여부
            volume_ml: 부피 (mL). None이면 부피 정보를 생략.
        """
        ko_term = get_korean_term(label)

        if not present:
            return f"영상에서 {ko_term}이(가) 감지되지 않았습니다."

        parts = [f"{ko_term} 세그멘테이션이 완료되었습니다."]

        if volume_ml is not None:
            parts.append(f"추정 부피: {volume_ml:.1f} mL")

        description = get_korean_description(label)
        if description:
            parts.append(description)

        return " ".join(parts)

    def translate_vqa_answer(self, answer: str) -> str:
        """VQA(질의응답) 답변의 장기명을 한국어로 치환."""
        return self.translate_response(answer)

    def build_report(
        self,
        results: list[dict],
        patient_id: str | None = None,
    ) -> str:
        """
        여러 세그멘테이션 결과를 한국어 보고서 문자열로 조합.

        Args:
            results: [{"label": str, "present": bool, "volume_ml": float|None}, ...]
            patient_id: 보고서에 포함할 환자 ID (선택)
        """
        lines: list[str] = []

        if patient_id:
            lines.append(f"환자 ID: {patient_id}")
            lines.append("")

        lines.append("=== 3D 의료 영상 분할 보고서 ===")
        lines.append("")

        detected = [r for r in results if r.get("present")]
        not_detected = [r for r in results if not r.get("present")]

        if detected:
            lines.append("[ 감지된 구조물 ]")
            for r in detected:
                vol = r.get("volume_ml")
                ko = get_korean_term(r["label"])
                vol_str = f"  (부피: {vol:.1f} mL)" if vol is not None else ""
                lines.append(f"  • {ko}{vol_str}")
            lines.append("")

        if not_detected:
            lines.append("[ 미감지 구조물 ]")
            for r in not_detected:
                ko = get_korean_term(r["label"])
                lines.append(f"  • {ko}")
            lines.append("")

        return "\n".join(lines)
