from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image as PILImage
from fpdf import FPDF

from src.analysis.clinical import ClinicalAssessment, NormalRange
from src.analysis.volume import VolumeStats
from src.translation.medical_terms import get_korean_term


# 한국어 TTF 폰트 후보 경로 (순서대로 시도)
_FONT_CANDIDATES: list[str] = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",          # Colab: apt install fonts-nanum
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Colab 대체
    "C:/Windows/Fonts/malgun.ttf",                              # Windows
    "C:/Windows/Fonts/malgunbd.ttf",
    "malgun.ttf",
]

_STATUS_COLOR = {
    "normal":  (40,  167, 69),   # 초록
    "high":    (220, 53,  69),   # 빨강
    "low":     (255, 140, 0),    # 주황
    "unknown": (108, 117, 125),  # 회색
}

_STATUS_LABEL = {
    "normal":  "✓ 정상",
    "high":    "↑ 초과",
    "low":     "↓ 미만",
    "unknown": "— 없음",
}


class _PDF(FPDF):
    def __init__(self, patient_id: str = ""):
        super().__init__()
        self.patient_id = patient_id
        self._ko = self._load_korean_font()

    def _load_korean_font(self) -> bool:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    self.add_font("Ko", "", path)
                    return True
                except Exception:
                    continue
        return False

    def _font(self, size: int, bold: bool = False) -> None:
        if self._ko:
            self.set_font("Ko", size=size)
        else:
            self.set_font("Helvetica", style="B" if bold else "", size=size)

    def header(self) -> None:
        self._font(15, bold=True)
        title = "MedSeg-3D-KO 임상 분석 보고서" if self._ko else "MedSeg-3D-KO Clinical Report"
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
        self._font(9)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        pid = f"환자 ID: {self.patient_id}   " if self.patient_id and self._ko else (f"Patient: {self.patient_id}   " if self.patient_id else "")
        date_label = "생성일시: " if self._ko else "Generated: "
        self.cell(0, 6, f"{pid}{date_label}{now}", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)
        self.set_draw_color(150, 150, 150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-14)
        self._font(8)
        self.set_text_color(130, 130, 130)
        note = (
            "본 보고서는 AI 분석 결과이며, 최종 진단은 전문의 판단을 따르십시오. | "
            if self._ko else
            "AI-based analysis only. Final diagnosis requires physician evaluation. | "
        )
        self.cell(0, 8, f"{note}Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _section_title(pdf: _PDF, text_ko: str, text_en: str) -> None:
    pdf._font(12, bold=True)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 9, text_ko if pdf._ko else text_en, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)


def _organ_table(pdf: _PDF, organ_results: list[dict]) -> None:
    col = [52, 28, 45, 28, 37]  # 합계 190mm
    h_ko = ["장기명", "부피(mL)", "정상 범위(mL)", "상태", "비고"]
    h_en = ["Organ", "Vol.(mL)", "Normal Range(mL)", "Status", "Note"]
    headers = h_ko if pdf._ko else h_en

    # 헤더 행
    pdf._font(9, bold=True)
    pdf.set_fill_color(60, 100, 170)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col, headers):
        pdf.cell(w, 8, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    # 데이터 행
    pdf._font(9)
    for i, row in enumerate(organ_results):
        stats: VolumeStats       = row["stats"]
        assessment: ClinicalAssessment = row["assessment"]
        label: str               = row["label"]

        fill_rgb = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_rgb)

        organ_name = get_korean_term(label) if pdf._ko else label
        nr: NormalRange = assessment.normal_range
        range_str = f"{nr.lo:.0f}~{nr.hi:.0f}" if (nr.lo is not None and nr.hi is not None) else "—"
        status_lbl = _STATUS_LABEL.get(assessment.status, "?")
        note_str = nr.note[:10] if nr.note else ""  # 넘치지 않게 자름

        pdf.cell(col[0], 7, organ_name, border=1, fill=True)
        pdf.cell(col[1], 7, f"{stats.volume_ml:.1f}", border=1, fill=True, align="C")
        pdf.cell(col[2], 7, range_str, border=1, fill=True, align="C")

        pdf.set_text_color(*_STATUS_COLOR.get(assessment.status, (0, 0, 0)))
        pdf.cell(col[3], 7, status_lbl, border=1, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col[4], 7, note_str, border=1, fill=True, align="C")
        pdf.ln()


def _findings_section(pdf: _PDF, organ_results: list[dict]) -> None:
    assessments = [r["assessment"] for r in organ_results]
    abnormal = [a for a in assessments if a.status in ("high", "low")]
    normal   = [a for a in assessments if a.status == "normal"]
    unknown  = [a for a in assessments if a.status == "unknown"]

    groups = [
        (abnormal, _STATUS_COLOR["high"],    "[ 이상 소견 ]",       "[ Abnormal Findings ]"),
        (normal,   _STATUS_COLOR["normal"],  "[ 정상 소견 ]",       "[ Normal Findings ]"),
        (unknown,  _STATUS_COLOR["unknown"], "[ 참고범위 없음 ]",    "[ No Reference Range ]"),
    ]
    pdf._font(10)
    for items, color, title_ko, title_en in groups:
        if not items:
            continue
        pdf.set_text_color(*color)
        pdf._font(10, bold=True)
        pdf.cell(0, 7, title_ko if pdf._ko else title_en, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf._font(9)
        for a in items:
            pdf.multi_cell(0, 6, f"  {a.message_ko}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)


def generate_report(
    organ_results: list[dict],
    panel_image: np.ndarray | None = None,
    patient_id: str = "",
    output_path: str | None = None,
) -> str:
    """
    임상 분석 PDF 보고서를 생성하고 파일 경로를 반환.

    Args:
        organ_results: [{"label": str, "stats": VolumeStats, "assessment": ClinicalAssessment}, ...]
        panel_image:   CT 슬라이스 패널 이미지 (H, W, 3) numpy 배열
        patient_id:    보고서에 표시할 환자 ID
        output_path:   저장 경로. None이면 임시파일에 저장.

    Returns:
        생성된 PDF 파일의 절대 경로
    """
    pdf = _PDF(patient_id=patient_id)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # ── 1. 장기별 부피 표 ────────────────────────────────────────────────────
    _section_title(pdf, "1. 장기별 부피 분석", "1. Organ Volume Analysis")
    _organ_table(pdf, organ_results)
    pdf.ln(5)

    # ── 2. 임상 소견 ─────────────────────────────────────────────────────────
    _section_title(pdf, "2. 임상 소견", "2. Clinical Findings")
    _findings_section(pdf, organ_results)

    # ── 3. CT 슬라이스 이미지 ─────────────────────────────────────────────────
    if panel_image is not None:
        pdf.add_page()
        _section_title(pdf, "3. CT 슬라이스 (마스크 오버레이)", "3. CT Slices (Mask Overlay)")

        pil_img = PILImage.fromarray(panel_image.astype(np.uint8))
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)

        available_h = 270 - pdf.get_y()  # 페이지 남은 높이(mm)
        img_w, img_h = pil_img.size
        # 가로 190mm 기준 비율 유지
        display_w = 190.0
        display_h = display_w * img_h / img_w
        if display_h > available_h:
            display_h = available_h
            display_w = display_h * img_w / img_h
        pdf.image(img_buf, x=(210 - display_w) / 2, y=pdf.get_y(), w=display_w)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="medseg_report_")
        output_path = tmp.name
        tmp.close()

    pdf.output(output_path)
    return output_path
