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


_FONT_CANDIDATES: list[str] = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "malgun.ttf",
]

_STATUS_COLOR = {
    "normal":  (40,  167, 69),
    "high":    (220, 53,  69),
    "low":     (255, 140, 0),
    "unknown": (108, 117, 125),
}

_STATUS_LABEL_KO = {
    "normal":  "정상",
    "high":    "초과",
    "low":     "미만",
    "unknown": "없음",
}


class _PDF(FPDF):
    def __init__(self):
        super().__init__()
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
        self.cell(0, 10, "MedSeg-3D-KO 임상 분석 보고서" if self._ko else "MedSeg-3D-KO Clinical Report",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self._font(8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(2)
        self.set_draw_color(150, 150, 150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-14)
        self._font(7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8,
                  "본 보고서는 AI 분석 결과이며, 최종 진단은 전문의 판단을 따르십시오.  |  "
                  f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _section_title(pdf: _PDF, text: str) -> None:
    pdf._font(11, bold=True)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)


def _patient_info_section(
    pdf: _PDF,
    patient_name: str,
    age: int | None,
    sex: str,
    exam_date: str,
    doctor_notes: str,
    panel_image: np.ndarray | None,
) -> None:
    """환자 정보 + CT 이미지를 좌우 2열로 배치."""
    _section_title(pdf, "1. 환자 정보")

    # CT 이미지가 있으면 우측에 배치 (가로 70mm)
    img_w = 70.0
    info_w = 190 - img_w - 5  # 정보 영역 너비

    start_y = pdf.get_y()

    # ── 좌측: 환자 정보 ──
    pdf._font(9)
    rows = [
        ("환자명",   patient_name or "미입력"),
        ("나이/성별", f"{age}세 / {sex}" if age else sex or "미입력"),
        ("검사일",   exam_date or "미입력"),
    ]
    for label, value in rows:
        pdf._font(9, bold=True)
        pdf.set_x(10)
        pdf.cell(25, 7, label, border="B")
        pdf._font(9)
        pdf.cell(info_w - 25, 7, value, border="B", new_x="LMARGIN", new_y="NEXT")

    # 의사 메모
    pdf._font(9, bold=True)
    pdf.set_x(10)
    pdf.cell(0, 7, "의사 메모", new_x="LMARGIN", new_y="NEXT")
    pdf._font(9)
    pdf.set_x(10)
    notes_text = doctor_notes.strip() if doctor_notes and doctor_notes.strip() else "없음"
    pdf.multi_cell(info_w, 6, notes_text, border=1, new_x="LMARGIN", new_y="NEXT")

    info_end_y = pdf.get_y()

    # ── 우측: CT 이미지 ──
    if panel_image is not None:
        pil_img = PILImage.fromarray(panel_image.astype(np.uint8))
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)

        img_h = img_w * pil_img.size[1] / pil_img.size[0]
        pdf.image(img_buf, x=10 + info_w + 5, y=start_y, w=img_w, h=img_h)

        # 이미지 캡션
        pdf._font(7)
        pdf.set_text_color(100, 100, 100)
        pdf.set_xy(10 + info_w + 5, start_y + img_h + 1)
        pdf.cell(img_w, 5, "CT 슬라이스 (마스크 오버레이)", align="C")
        pdf.set_text_color(0, 0, 0)

    # 다음 섹션은 정보/이미지 중 더 아래서 시작
    pdf.set_y(max(info_end_y, start_y + (img_w * panel_image.shape[0] / panel_image.shape[1] if panel_image is not None else 0)) + 5)
    pdf.ln(2)


def _organ_table(pdf: _PDF, organ_results: list[dict]) -> None:
    col = [52, 28, 45, 28, 37]
    headers = ["장기명", "부피(mL)", "정상 범위(mL)", "상태", "비고"]

    pdf._font(9, bold=True)
    pdf.set_fill_color(60, 100, 170)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    pdf._font(9)
    for i, row in enumerate(organ_results):
        stats: VolumeStats = row["stats"]
        assessment: ClinicalAssessment = row["assessment"]
        label: str = row["label"]

        fill_rgb = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_rgb)

        organ_name = get_korean_term(label) if pdf._ko else label
        nr: NormalRange = assessment.normal_range
        range_str = f"{nr.lo:.0f}~{nr.hi:.0f}" if (nr.lo is not None and nr.hi is not None) else "-"
        status_lbl = _STATUS_LABEL_KO.get(assessment.status, "?")
        note_str = (nr.note[:10] if nr.note else "") if pdf._ko else ""

        pdf.cell(col[0], 6, organ_name, border=1, fill=True)
        pdf.cell(col[1], 6, f"{stats.volume_ml:.1f}", border=1, fill=True, align="C")
        pdf.cell(col[2], 6, range_str, border=1, fill=True, align="C")
        pdf.set_text_color(*_STATUS_COLOR.get(assessment.status, (0, 0, 0)))
        pdf.cell(col[3], 6, status_lbl, border=1, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col[4], 6, note_str, border=1, fill=True, align="C")
        pdf.ln()


def _findings_section(pdf: _PDF, organ_results: list[dict]) -> None:
    assessments = [r["assessment"] for r in organ_results]
    abnormal = [a for a in assessments if a.status in ("high", "low")]
    normal   = [a for a in assessments if a.status == "normal"]
    unknown  = [a for a in assessments if a.status == "unknown"]

    groups = [
        (abnormal, _STATUS_COLOR["high"],    "[ 이상 소견 ]"),
        (normal,   _STATUS_COLOR["normal"],  "[ 정상 소견 ]"),
        (unknown,  _STATUS_COLOR["unknown"], "[ 참고범위 없음 ]"),
    ]
    for items, color, title in groups:
        if not items:
            continue
        pdf.set_text_color(*color)
        pdf._font(9, bold=True)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf._font(9)
        for a in items:
            pdf.multi_cell(0, 5, f"  {a.message_ko}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)


def generate_report(
    organ_results: list[dict],
    panel_image: np.ndarray | None = None,
    patient_name: str = "",
    patient_age: int | None = None,
    patient_sex: str = "",
    exam_date: str = "",
    doctor_notes: str = "",
    output_path: str | None = None,
) -> str:
    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # ── 1. 환자 정보 + CT 이미지 ────────────────────────────────────────────────
    _patient_info_section(pdf, patient_name, patient_age, patient_sex,
                          exam_date, doctor_notes, panel_image)

    # ── 2. 장기별 부피 표 ────────────────────────────────────────────────────────
    _section_title(pdf, "2. 장기별 부피 분석")
    _organ_table(pdf, organ_results)
    pdf.ln(4)

    # ── 3. 임상 소견 ─────────────────────────────────────────────────────────────
    _section_title(pdf, "3. 임상 소견")
    _findings_section(pdf, organ_results)

    if output_path is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part = f"_{patient_name.strip()}" if patient_name and patient_name.strip() else ""
        filename = f"medseg{name_part}_{now}.pdf"
        output_path = os.path.join(tempfile.gettempdir(), filename)

    pdf.output(output_path)
    return output_path