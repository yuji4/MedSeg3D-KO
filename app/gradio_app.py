"""
MedSeg-3D-KO Gradio 앱 — 클리니컬 대시보드 버전
"""
from __future__ import annotations

from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import nibabel as nib
import gradio as gr
from PIL import Image

from src.inference.model_loader import ModelConfig, get_colab_config
from src.inference.segmentation import (
    SegmentationPipeline, _detect_organs_en,
    detect_intent, _translate_en_to_ko,
)
from src.analysis.volume import analyze_mask
from src.analysis.clinical import assess_organ, format_clinical_summary
from src.analysis.report import generate_report
from src.database.db import init_db
from src.database.crud import (
    upsert_patient, save_exam, list_patients,
    list_all_exams, get_exam_organs,
    get_organ_trend, get_patient_organ_labels,
    export_all_csv, export_patient_csv,
)
from src.auth.auth import get_auth_list
from src.translation.translator import MedicalTranslator
from src.translation.medical_terms import get_korean_term
from app.visualization import get_slice_views, make_panel, _LABEL_COLORS

init_db()


# ── Colab 한국어 폰트 자동 설치 ────────────────────────────────────────────────
def _setup_korean_font() -> None:
    _NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(_NANUM) or "google.colab" not in sys.modules:
        return
    import subprocess
    print("📦 한국어 폰트 설치 중 (fonts-nanum)...")
    subprocess.run(["apt-get", "install", "-y", "fonts-nanum", "-q"], check=False)
    print("✅ 완료" if os.path.exists(_NANUM) else "⚠️  실패 — PDF 영문 출력")

_setup_korean_font()


# ── 전역 상태 ──────────────────────────────────────────────────────────────────
_pipeline: SegmentationPipeline | None = None
_translator = MedicalTranslator()
_current_volume: np.ndarray | None = None
_current_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
_last_inference: dict = {}
_SEX_MAP = {"남성": "male", "여성": "female", "미입력": "unknown"}
_SEX_KO  = {"male": "남성", "female": "여성", "unknown": "미입력"}


# ── 주민등록번호 파싱 ──────────────────────────────────────────────────────────
def _parse_rrn(rrn: str) -> dict | None:
    """주민등록번호 → {age, sex, birth_year}. 형식 불일치 시 None."""
    import re
    clean = re.sub(r"[\s\-]", "", rrn)
    if len(clean) != 13 or not clean.isdigit():
        return None
    yy, g = int(clean[:2]), int(clean[6])
    if g in (1, 2, 5, 6):
        year = 1900 + yy
    elif g in (3, 4, 7, 8):
        year = 2000 + yy
    elif g in (9, 0):
        year = 1800 + yy
    else:
        return None
    return {
        "birth_year": year,
        "age": datetime.now().year - year,
        "sex": "male" if g % 2 == 1 else "female",
    }


def _mask_rrn(rrn: str) -> str:
    """뒷자리 마스킹: 000000-*"""
    clean = rrn.replace("-", "").replace(" ", "")
    return f"{clean[:6]}-*******" if len(clean) == 13 else ""


# ── 테마 / CSS ─────────────────────────────────────────────────────────────────
_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_background_fill="#1e293b",
    block_label_text_color="#94a3b8",
    block_title_text_color="#e2e8f0",
    body_text_color="#e2e8f0",
    input_background_fill="#0f172a",
    input_border_color="#334155",
    input_placeholder_color="#64748b",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="white",
    button_secondary_background_fill="#334155",
    button_secondary_text_color="#e2e8f0",
    slider_color="#2563eb",
)

_CSS = """
footer { display: none; }
.gradio-container { max-width: 1800px !important; margin: 0 auto !important; }

/* 결과 카드 스크롤 */
.card-scroll { max-height: 420px; overflow-y: auto; padding-right: 4px; }

/* CT 뷰 레이블 */
.view-label { text-align:center; color:#64748b; font-size:0.78rem; padding:2px 0 6px; }

/* 텍스트박스 스크롤 */
#volume_box textarea, #clinical_box textarea, #notes_box textarea {
    overflow-y: auto !important; resize: none;
}

/* 섹션 간격 */
.section-gap { margin-top: 16px; }

/* ── 데이터프레임 ── */
/* Gradio 4.x: .table-wrap 안의 table */
.table-wrap table { border-collapse: collapse; width: 100%; }
.table-wrap thead tr,
.table-wrap thead tr th {
    background: #0f172a !important;
    color: #94a3b8 !important;
    border-color: #334155 !important;
}
.table-wrap tbody tr:nth-child(odd)  td { background: #1e293b !important; color: #e2e8f0 !important; }
.table-wrap tbody tr:nth-child(even) td { background: #1a2744 !important; color: #e2e8f0 !important; }
.table-wrap td, .table-wrap th { border-color: #334155 !important; padding: 6px 10px !important; }
/* 혹시 bare table 도 커버 */
table:not(.options) thead tr th { background: #0f172a !important; color: #94a3b8 !important; }
table:not(.options) tbody tr:nth-child(odd)  td { background: #1e293b !important; color: #e2e8f0 !important; }
table:not(.options) tbody tr:nth-child(even) td { background: #1a2744 !important; color: #e2e8f0 !important; }

/* ── 드롭다운 팝업 ── */
/* Gradio 4.x 드롭다운 옵션 리스트 */
ul.options                { background: #1e293b !important; border: 1px solid #334155 !important; }
ul.options li.item        { color: #e2e8f0 !important; }
ul.options li.item:hover  { background: #334155 !important; color: #ffffff !important; }
ul.options li.item.active { background: #2563eb !important; color: #ffffff !important; }
/* 선택된 값 표시 영역 */
.wrap-inner span, .svelte-select .value-container { color: #e2e8f0 !important; }
.token                    { background: #334155 !important; color: #e2e8f0 !important; }
"""


# ── HTML 생성 헬퍼 ─────────────────────────────────────────────────────────────
def _make_header_html(patient_name: str = "", exam_date: str = "",
                      status: str = "idle") -> str:
    _S = {
        "idle":      ("● 대기 중",    "#64748b", "#1e293b"),
        "loaded":    ("● CT 로드됨",  "#3b82f6", "#172554"),
        "done_ok":   ("● 분석 완료",  "#22c55e", "#052e16"),
        "done_warn": ("● 이상 소견",  "#ef4444", "#2d0a0a"),
    }
    label, color, bg = _S.get(status, _S["idle"])
    name = patient_name.strip() or "환자 미입력"
    date = exam_date.strip() or datetime.now().strftime("%Y-%m-%d")
    return f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:14px 20px;background:#1e293b;border-radius:10px;
            border:1px solid #334155;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:1.8rem;">🏥</span>
    <div>
      <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;">MedSeg-3D-KO</div>
      <div style="font-size:0.78rem;color:#64748b;">M3D 기반 3D 의료 영상 한국어 세그멘테이션</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="text-align:right;">
      <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;">{name}</div>
      <div style="font-size:0.78rem;color:#94a3b8;">검사일: {date}</div>
    </div>
    <div style="background:{bg};color:{color};border:1px solid {color};
                border-radius:20px;padding:4px 14px;font-size:0.78rem;font-weight:600;">
      {label}
    </div>
  </div>
</div>"""


def _make_legend_html(label_names: dict) -> str:
    labels = {k: v for k, v in label_names.items() if k != 0}
    if not labels:
        return ("<div style='color:#475569;font-size:0.8rem;padding:4px 0;'>"
                "세그멘테이션 후 색상 범례가 표시됩니다</div>")
    items = []
    for lbl, name in sorted(labels.items()):
        r, g, b = _LABEL_COLORS[lbl % len(_LABEL_COLORS)]
        hex_c = f"#{r:02x}{g:02x}{b:02x}"
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin:2px 10px 2px 0;">'
            f'<span style="width:11px;height:11px;background:{hex_c};border-radius:2px;'
            f'display:inline-block;margin-right:5px;"></span>'
            f'<span style="color:#cbd5e1;font-size:0.82rem;">{name}</span></span>'
        )
    return ('<div style="display:flex;flex-wrap:wrap;align-items:center;padding:6px 0;">'
            + "".join(items) + "</div>")


def _make_results_html(assessments: list) -> str:
    if not assessments:
        return ("<div style='color:#475569;padding:32px 16px;text-align:center;"
                "font-size:0.9rem;line-height:2;'>"
                "세그멘테이션을 실행하면<br>장기별 분석 결과가 여기에 표시됩니다</div>")
    _COLOR = {
        "high":    ("#ef4444", "#3f0c0c"),
        "low":     ("#f97316", "#3f1a08"),
        "normal":  ("#22c55e", "#052e16"),
        "unknown": ("#64748b", "#1e293b"),
    }
    _BADGE = {
        "high": "↑ 정상 초과", "low": "↓ 정상 미만",
        "normal": "✓ 정상", "unknown": "참고범위없음",
    }
    ordered = (
        [a for a in assessments if a.status in ("high", "low")] +
        [a for a in assessments if a.status == "normal"] +
        [a for a in assessments if a.status == "unknown"]
    )
    cards = []
    for a in ordered:
        border, badge_bg = _COLOR.get(a.status, _COLOR["unknown"])
        badge = _BADGE.get(a.status, "—")
        nr = a.normal_range
        range_str = (f"{nr.lo:.0f}~{nr.hi:.0f} mL"
                     if nr.lo is not None and nr.hi is not None else "기준 없음")
        note = (f"<br><span style='color:#64748b;font-size:0.74rem;'>({nr.note})</span>"
                if nr.note else "")
        cards.append(f"""
<div style="background:#1e293b;border-radius:8px;padding:12px 15px;margin-bottom:8px;
            border:1px solid #334155;border-left:4px solid {border};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
    <span style="color:#e2e8f0;font-weight:600;font-size:0.93rem;">{get_korean_term(a.label)}</span>
    <span style="background:{badge_bg};color:{border};border:1px solid {border};
                 padding:2px 9px;border-radius:10px;font-size:0.72rem;font-weight:700;">
      {badge}</span>
  </div>
  <div style="color:#94a3b8;font-size:0.84rem;line-height:1.6;">
    측정: <b style="color:#e2e8f0;">{a.volume_ml:.1f} mL</b>
    &nbsp;|&nbsp; 정상범위: <span style="color:#cbd5e1;">{range_str}</span>{note}
  </div>
</div>""")
    return "<div>" + "".join(cards) + "</div>"


def _translate_and_clean(answer_en: str) -> str:
    """영문 M3D 출력 → 한국어 번역 → 의학 용어 정리."""
    ko = _translate_en_to_ko(answer_en)
    return _translator.translate_response(ko)


_ANSWER_MODE = {
    "vqa":     ("💬 VQA 답변",   "#3b82f6", "#172554"),
    "caption": ("📋 소견 생성",  "#8b5cf6", "#2e1065"),
    "reg":     ("🔬 장기 설명",  "#06b6d4", "#083344"),
}


def _make_answer_html(mode: str, text: str) -> str:
    icon, color, bg = _ANSWER_MODE.get(mode, _ANSWER_MODE["vqa"])
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"<div style='background:{bg};border:1px solid {color};border-radius:8px;"
        f"padding:14px 16px;margin-bottom:8px;'>"
        f"<div style='color:{color};font-size:0.8rem;font-weight:700;margin-bottom:8px;'>"
        f"{icon}</div>"
        f"<div style='color:#e2e8f0;font-size:0.88rem;line-height:1.8;"
        f"white-space:pre-wrap;'>{safe}</div></div>"
    )


def _make_exam_detail_html(organs: list[dict]) -> str:
    if not organs:
        return "<div style='color:#475569;padding:16px;'>장기 데이터 없음</div>"
    _COLOR = {
        "high":    ("#ef4444", "#3f0c0c"),
        "low":     ("#f97316", "#3f1a08"),
        "normal":  ("#22c55e", "#052e16"),
        "unknown": ("#64748b", "#1e293b"),
    }
    _BADGE = {"high": "↑ 정상 초과", "low": "↓ 정상 미만",
              "normal": "✓ 정상", "unknown": "참고범위없음"}
    ordered = (
        [o for o in organs if o["status"] in ("high", "low")] +
        [o for o in organs if o["status"] == "normal"] +
        [o for o in organs if o["status"] == "unknown"]
    )
    cards = []
    for o in ordered:
        st = o.get("status", "unknown")
        border, badge_bg = _COLOR.get(st, _COLOR["unknown"])
        badge = _BADGE.get(st, "—")
        lo, hi = o.get("range_lo"), o.get("range_hi")
        range_str = f"{lo:.0f}~{hi:.0f} mL" if (lo is not None and hi is not None) else "기준 없음"
        note = o.get("note") or ""
        note_html = (f"<br><span style='color:#64748b;font-size:0.74rem;'>({note})</span>"
                     if note else "")
        cards.append(f"""
<div style="background:#1e293b;border-radius:8px;padding:10px 13px;margin-bottom:6px;
            border:1px solid #334155;border-left:4px solid {border};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
    <span style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{get_korean_term(o['label'])}</span>
    <span style="background:{badge_bg};color:{border};border:1px solid {border};
                 padding:1px 8px;border-radius:10px;font-size:0.71rem;font-weight:700;">{badge}</span>
  </div>
  <div style="color:#94a3b8;font-size:0.82rem;line-height:1.6;">
    측정: <b style="color:#e2e8f0;">{o['volume_ml']:.1f} mL</b>
    &nbsp;|&nbsp; 정상범위: <span style="color:#cbd5e1;">{range_str}</span>{note_html}
  </div>
</div>""")
    return "<div>" + "".join(cards) + "</div>"


# ── 종단적 차트 ───────────────────────────────────────────────────────────────
def _make_trend_chart(patient_id: int, label: str):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    trend = get_organ_trend(patient_id, label)
    if len(trend) < 2:
        return None

    dates = [r["exam_date"] or r["analyzed_at"][:10] for r in trend]
    vols  = [r["volume_ml"] for r in trend]
    lo    = trend[0]["range_lo"]
    hi    = trend[0]["range_hi"]

    fig = go.Figure()
    if lo is not None and hi is not None:
        fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(34,197,94,0.08)",
                      line_width=0, annotation_text="정상범위",
                      annotation_position="top left",
                      annotation_font_color="#22c55e")
    fig.add_trace(go.Scatter(
        x=dates, y=vols,
        mode="lines+markers+text",
        text=[f"{v:.1f}" for v in vols],
        textposition="top center",
        textfont=dict(color="#e2e8f0", size=11),
        name=get_korean_term(label),
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=9, color=[
            "#ef4444" if r["status"] in ("high", "low") else "#22c55e"
            for r in trend
        ]),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        margin=dict(l=50, r=20, t=50, b=40),
        title=dict(text=f"{get_korean_term(label)} 부피 추이 (mL)",
                   font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(gridcolor="#334155", title="검사일"),
        yaxis=dict(gridcolor="#334155", title="부피 (mL)"),
        showlegend=False,
    )
    return fig


# ── 초기화 ────────────────────────────────────────────────────────────────────
def _get_pipeline() -> SegmentationPipeline:
    global _pipeline
    if _pipeline is None:
        in_colab = "google.colab" in sys.modules
        config = get_colab_config() if in_colab else ModelConfig(precision="bf16")
        _pipeline = SegmentationPipeline(config)
        _pipeline.load()
    return _pipeline


def _views_to_pil(views: dict) -> tuple:
    return (Image.fromarray(views["axial"]),
            Image.fromarray(views["sagittal"]),
            Image.fromarray(views["coronal"]))


# ── 파일 로드 ─────────────────────────────────────────────────────────────────
def load_file(file_obj, patient_name, exam_date):
    global _current_volume, _current_spacing
    _hdr = lambda s: _make_header_html(patient_name, exam_date, s)
    _blank = (None, None, None)

    if file_obj is None:
        return (*_blank, "파일을 업로드해주세요.", _hdr("idle"), _make_legend_html({}))

    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        if path.endswith(".nii.gz") or path.endswith(".nii"):
            nii = nib.load(path)
            zooms = nii.header.get_zooms()
            _current_spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
            _current_volume = nii.get_fdata().astype(np.float32).transpose(2, 1, 0)
        elif path.endswith(".npy"):
            arr = np.load(path)
            _current_volume = arr[0] if arr.ndim == 4 else arr
            _current_spacing = (1.0, 1.0, 1.0)
        else:
            return (*_blank, f"지원하지 않는 형식: {os.path.basename(path)}",
                    _hdr("idle"), _make_legend_html({}))

        views = get_slice_views(_current_volume)
        D, H, W = _current_volume.shape
        sp = _current_spacing
        msg = (f"✅ {os.path.basename(path)} | {D}×{H}×{W} voxels | "
               f"간격 {sp[0]:.2f}×{sp[1]:.2f}×{sp[2]:.2f} mm")
        return (*_views_to_pil(views), msg, _hdr("loaded"), _make_legend_html({}))

    except Exception as e:
        return (*_blank, f"❌ 로드 실패: {e}", _hdr("idle"), _make_legend_html({}))


# ── 추론 ──────────────────────────────────────────────────────────────────────
def run_reg_fn(organ_ko_label: str):
    """선택된 장기 영역 설명 (REG)."""
    if not organ_ko_label or _current_volume is None:
        return ""
    if not _last_inference.get("label_names"):
        return "세그멘테이션을 먼저 실행해주세요."
    # 레이블 이름에서 영문 추출 "간 (liver)" → "liver"
    organ_en = organ_ko_label.split("(")[-1].rstrip(")").strip() if "(" in organ_ko_label else organ_ko_label
    try:
        pipeline = _get_pipeline()
        answer_en = pipeline.run_reg(_current_volume, organ_en)
        answer_ko = _translate_and_clean(answer_en)
        return _make_answer_html("reg", answer_ko)
    except Exception as e:
        return f"오류: {e}"


def on_rrn_change(rrn: str):
    """주민등록번호 입력 시 나이·성별 자동 채우기."""
    parsed = _parse_rrn(rrn)
    if parsed:
        sex_ko = _SEX_KO.get(parsed["sex"], "미입력")
        return parsed["age"], sex_ko
    return gr.update(), gr.update()


def run_inference(question_ko, slice_idx, alpha, wl, ww, mask_on,
                  age, sex_ko, patient_name, exam_date, doctor_notes, rrn):
    global _current_volume, _current_spacing, _last_inference

    _hdr  = lambda s: _make_header_html(patient_name, exam_date, s)
    _empty_views = (None, None, None)
    _blank = (*_empty_views, _make_legend_html({}), _make_results_html([]))

    if _current_volume is None:
        return (*_blank, _hdr("idle"), "", "", "", gr.update(choices=[]))
    if not question_ko.strip():
        return (*_blank, _hdr("loaded"), "", "", "", gr.update(choices=[]))

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return (*_blank, _hdr("idle"), "", "", f"모델 로드 실패: {e}", gr.update(choices=[]))

    intent = detect_intent(question_ko)

    # ── VQA 분기 ────────────────────────────────────────────────────────────
    if intent == "vqa":
        try:
            answer_en = pipeline.run_vqa(_current_volume, question_ko)
            answer_ko = _translate_and_clean(answer_en)
        except Exception as e:
            answer_ko = f"VQA 오류: {e}"
        D = _current_volume.shape[0]
        views = get_slice_views(_current_volume, None,
                                slice_indices={"axial": D // 2,
                                               "sagittal": _current_volume.shape[2] // 2,
                                               "coronal": _current_volume.shape[1] // 2})
        return (
            *_views_to_pil(views),
            _make_legend_html({}),
            _make_answer_html("vqa", answer_ko),
            _hdr("done_ok"),
            "", answer_ko, "",
            gr.update(choices=[]),
        )

    # ── 소견 생성 분기 ────────────────────────────────────────────────────────
    if intent == "caption":
        try:
            answer_en = pipeline.run_caption(_current_volume)
            answer_ko = _translate_and_clean(answer_en)
        except Exception as e:
            answer_ko = f"소견 생성 오류: {e}"
        D = _current_volume.shape[0]
        views = get_slice_views(_current_volume, None,
                                slice_indices={"axial": D // 2,
                                               "sagittal": _current_volume.shape[2] // 2,
                                               "coronal": _current_volume.shape[1] // 2})
        return (
            *_views_to_pil(views),
            _make_legend_html({}),
            _make_answer_html("caption", answer_ko),
            _hdr("done_ok"),
            "", answer_ko, "",
            gr.update(choices=[]),
        )

    # ── 세그멘테이션 분기 ────────────────────────────────────────────────────
    organs = _detect_organs_en(question_ko)
    try:
        if not organs:
            results = [pipeline.run(_current_volume, question_ko)]
            organs = [results[0]["organ_label"] or ""]
        else:
            image_pt, original = pipeline._prepare_image_pt(_current_volume)
            results = [pipeline._infer(image_pt, original, org, question_ko) for org in organs]
    except Exception as e:
        return (*_blank, _hdr("loaded"), "", "", f"추론 오류: {e}", gr.update(choices=[]))

    combined_mask = np.zeros(_current_volume.shape, dtype=np.uint8)
    label_names: dict[int, str] = {}
    for lbl_idx, (org, res) in enumerate(zip(organs, results), start=1):
        if res["mask"].any():
            combined_mask[res["mask"]] = lbl_idx
        label_names[lbl_idx] = get_korean_term(org) if org else f"구조물 {lbl_idx}"

    sex = _SEX_MAP.get(sex_ko, "unknown")
    volume_lines, organ_results, assessments = [], [], []
    any_detected = False

    for lbl_idx, (org, res) in enumerate(zip(organs, results), start=1):
        organ_mask = combined_mask == lbl_idx
        stats = analyze_mask(organ_mask, label=org, voxel_spacing_mm=_current_spacing)
        present = stats.voxel_count > 0
        if present:
            any_detected = True
        vol = stats.volume_ml if present else 0.0
        assessment = assess_organ(org, vol, age=int(age), sex=sex)
        assessments.append(assessment)
        organ_results.append({"label": org, "stats": stats, "assessment": assessment})
        volume_lines.append(
            stats.summary_ko() if present else f"[{get_korean_term(org)}] 마스크 미감지"
        )

    has_abnormal = any(a.status in ("high", "low") for a in assessments)
    status = "done_warn" if has_abnormal else "done_ok"

    D = _current_volume.shape[0]
    auto_idx = _best_slice_index(combined_mask > 0) if any_detected else D // 2
    chosen_idx = max(0, min(auto_idx if slice_idx == 0 else slice_idx, D - 1))

    eff_mask = combined_mask if mask_on else None
    views = get_slice_views(
        _current_volume, eff_mask,
        slice_indices={"axial": chosen_idx,
                       "sagittal": _current_volume.shape[2] // 2,
                       "coronal": _current_volume.shape[1] // 2},
        alpha=alpha, wl=wl, ww=ww, label_names=None,
    )
    panel_arr = make_panel(views)

    _last_inference.update({
        "organ_results": organ_results,
        "panel_image": panel_arr,
        "combined_mask": combined_mask,
        "label_names": label_names,
    })

    # DB 저장
    try:
        parsed = _parse_rrn(rrn or "")
        birth_year = parsed["birth_year"] if parsed else None
        pid = upsert_patient(patient_name or "미입력", sex,
                             rrn=rrn or "", birth_year=birth_year)
        save_exam(pid,
                  exam_date or datetime.now().strftime("%Y-%m-%d"),
                  int(age), organ_results, notes=doctor_notes or "")
    except Exception:
        pass

    # REG용 장기 드롭다운 선택지
    organ_choices = [f"{get_korean_term(org)} ({org})" for org in organs if org]

    return (
        *_views_to_pil(views),
        _make_legend_html(label_names) if mask_on else _make_legend_html({}),
        _make_results_html(assessments),
        _hdr(status),
        "\n\n".join(volume_lines),
        format_clinical_summary(assessments, age=int(age), sex=sex),
        "",
        gr.update(choices=organ_choices, value=organ_choices[0] if organ_choices else None),
    )


def _best_slice_index(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def generate_pdf_report() -> tuple:
    if not _last_inference.get("organ_results"):
        return None, "⚠️ 먼저 세그멘테이션을 실행해주세요."
    path = generate_report(
        organ_results=_last_inference["organ_results"],
        panel_image=_last_inference.get("panel_image"),
        patient_id="",
    )
    return path, f"✅ PDF 생성: {os.path.basename(path)}"


def update_preview(slice_idx, alpha, wl, ww, mask_on):
    if _current_volume is None:
        return None, None, None
    D = _current_volume.shape[0]
    idx = max(0, min(int(slice_idx), D - 1))
    eff_mask = _last_inference.get("combined_mask") if mask_on else None
    views = get_slice_views(
        _current_volume, eff_mask,
        slice_indices={"axial": idx,
                       "sagittal": _current_volume.shape[2] // 2,
                       "coronal": _current_volume.shape[1] // 2},
        wl=wl, ww=ww, alpha=alpha, label_names=None,
    )
    return _views_to_pil(views)


def on_sagittal_click(evt: gr.SelectData) -> int:
    if _current_volume is None:
        return 0
    D = _current_volume.shape[0]
    return max(0, min(int(evt.index[1] / 256 * D), D - 1))


def on_coronal_click(evt: gr.SelectData) -> int:
    if _current_volume is None:
        return 0
    D = _current_volume.shape[0]
    return max(0, min(int(evt.index[1] / 256 * D), D - 1))


# ── 환자 이력 탭 핸들러 ────────────────────────────────────────────────────────
def _exams_to_df(exams: list[dict]) -> list[list]:
    rows = []
    for e in exams:
        ab = e.get("n_abnormal") or 0
        n  = e.get("n_organs") or 0
        rows.append([
            str(e.get("id", "")),
            e.get("patient_name", "-"),
            _SEX_KO.get(e.get("sex", ""), "미입력"),
            e.get("exam_date", "-"),
            str(e.get("age_at_exam", "-")),
            e.get("analyzed_at", "")[:16].replace("T", " "),
            f"{n}개" + (f" (이상 {ab})" if ab else ""),
        ])
    return rows


def refresh_history_fn():
    exams = list_all_exams()
    patients = list_patients()
    choices = [f"{p['name']} (ID:{p['id']})" for p in patients]
    return _exams_to_df(exams), exams, gr.update(choices=choices, value=None)


def on_exam_select_fn(evt: gr.SelectData, exams: list) -> tuple:
    if not exams or evt.index[0] >= len(exams):
        return "<div style='color:#475569;padding:16px;'>선택 오류</div>", None
    exam = exams[evt.index[0]]
    organs = get_exam_organs(exam["id"])
    # 의사 메모 표시
    notes_html = ""
    if exam.get("notes"):
        notes_html = (
            f"<div style='margin-top:8px;padding:8px 12px;background:#172554;"
            f"border-radius:6px;color:#93c5fd;font-size:0.83rem;'>"
            f"📝 의사 메모: {exam['notes']}</div>"
        )
    rrn_str = _mask_rrn(exam.get("rrn") or "")
    rrn_html = f" | {rrn_str}" if rrn_str else ""
    detail_html = (
        f"<div style='color:#94a3b8;font-size:0.83rem;padding:6px 10px;"
        f"background:#0f172a;border-radius:6px;margin-bottom:8px;'>"
        f"👤 {exam.get('patient_name','-')}{rrn_html} | "
        f"검사일: {exam.get('exam_date','-')} | "
        f"{exam.get('age_at_exam','-')}세 | "
        f"분석: {str(exam.get('analyzed_at',''))[:16].replace('T',' ')}"
        f"</div>"
        + notes_html
        + _make_exam_detail_html(organs)
    )
    return detail_html, exam["id"]


def select_patient_fn(patient_str: str):
    if not patient_str:
        return gr.update(choices=[], value=None)
    try:
        pid = int(patient_str.split("ID:")[-1].rstrip(")"))
    except Exception:
        return gr.update(choices=[], value=None)
    labels = get_patient_organ_labels(pid)
    choices = [f"{get_korean_term(l)} ({l})" for l in labels]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def draw_trend_fn(patient_str: str, organ_ko_str: str):
    if not patient_str or not organ_ko_str:
        return None
    try:
        pid   = int(patient_str.split("ID:")[-1].rstrip(")"))
        label = organ_ko_str.split("(")[-1].rstrip(")")
    except Exception:
        return None
    return _make_trend_chart(pid, label)


def export_all_fn():
    path = export_all_csv()
    return gr.update(value=path, visible=True) if path else gr.update(visible=False)


def export_patient_fn(patient_str: str):
    if not patient_str:
        return gr.update(visible=False)
    try:
        pid = int(patient_str.split("ID:")[-1].rstrip(")"))
    except Exception:
        return gr.update(visible=False)
    path = export_patient_csv(pid)
    return gr.update(value=path, visible=True) if path else gr.update(visible=False)


# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES_Q = [
    "간을 분할해줘", "비장을 세그멘테이션해줘", "좌측 폐를 분할해줘",
    "신장 마스크를 보여줘", "췌장이 어디있어?",
]

with gr.Blocks(title="MedSeg-3D-KO", theme=_THEME, css=_CSS) as demo:

    with gr.Tabs():

        # ━━ 분석 탭 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🔬 분석"):

            header_html = gr.HTML(value=_make_header_html())

            with gr.Row(equal_height=False):

                # ── 좌측 사이드바 ───────────────────────────────────────────────
                with gr.Column(scale=1, min_width=290):

                    with gr.Group():
                        gr.Markdown("#### 👤 환자 정보")
                        patient_name_input = gr.Textbox(
                            label="환자 이름", placeholder="홍길동", lines=1,
                        )
                        rrn_input = gr.Textbox(
                            label="주민등록번호",
                            placeholder="000000-0000000",
                            lines=1,
                            type="password",
                            info="입력 시 나이·성별 자동 입력 / 환자 고유 식별자",
                        )
                        exam_date_input = gr.Textbox(
                            label="검사일 (YYYY-MM-DD)",
                            value=datetime.now().strftime("%Y-%m-%d"),
                            lines=1,
                        )
                        with gr.Row():
                            age_input = gr.Number(
                                label="나이 (세)", value=30, minimum=0, maximum=120, step=1,
                            )
                            sex_input = gr.Radio(
                                ["남성", "여성", "미입력"], label="성별", value="미입력",
                            )
                        doctor_notes_input = gr.Textbox(
                            label="📝 의사 메모",
                            placeholder="임상 소견, 특이사항 등 자유롭게 입력",
                            lines=3,
                            elem_id="notes_box",
                        )

                    gr.Markdown("---")

                    with gr.Group():
                        gr.Markdown("#### 📂 CT 파일 업로드")
                        file_input = gr.File(
                            label="파일 선택 (.nii.gz / .nii / .npy)",
                            file_types=[".nii", ".gz", ".npy"],
                        )
                        load_status = gr.Textbox(
                            label="로드 상태", interactive=False, lines=2,
                        )

                # ── 중앙: CT 뷰어 ──────────────────────────────────────────────
                with gr.Column(scale=3):

                    with gr.Group():
                        with gr.Row(equal_height=True):
                            axial_img    = gr.Image(show_label=False, type="pil", height=270)
                            sagittal_img = gr.Image(show_label=False, type="pil", height=270)
                            coronal_img  = gr.Image(show_label=False, type="pil", height=270)
                        with gr.Row():
                            gr.HTML("<div class='view-label'>축상면 (Axial)</div>")
                            gr.HTML("<div class='view-label'>시상면 (Sagittal) ← 클릭</div>")
                            gr.HTML("<div class='view-label'>관상면 (Coronal) ← 클릭</div>")
                        legend_html = gr.HTML(value=_make_legend_html({}))

                    gr.Markdown("---")

                    with gr.Accordion("🎛️ 뷰어 컨트롤", open=False):
                        with gr.Row():
                            mask_toggle = gr.Checkbox(
                                label="마스크 오버레이", value=True, scale=1,
                            )
                            slice_slider = gr.Slider(
                                0, 31, value=0, step=1, label="축상 슬라이스 (0=자동)", scale=3,
                            )
                        with gr.Row():
                            wl_slider = gr.Slider(-200, 400, value=40, step=10, label="윈도우 레벨 (HU)")
                            ww_slider = gr.Slider(100, 2000, value=400, step=50, label="윈도우 너비 (HU)")
                        alpha_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="마스크 불투명도")

                    gr.Markdown("---")

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="한국어 질문",
                            placeholder="예: 간을 분할해줘  /  신장이랑 비장 찾아줘",
                            lines=1, scale=4,
                        )
                        run_btn   = gr.Button("🔬 실행", variant="primary", scale=1)
                        clear_btn = gr.ClearButton(
                            [file_input, question_input, rrn_input, doctor_notes_input],
                            value="🗑️ 초기화", scale=1,
                        )

                    gr.Examples(examples=EXAMPLES_Q, inputs=[question_input], label="예시 질문")

                # ── 우측: 결과 패널 ─────────────────────────────────────────────
                with gr.Column(scale=2, min_width=280):

                    gr.Markdown("#### 📊 장기별 분석 결과")
                    results_html = gr.HTML(
                        value=_make_results_html([]),
                        elem_classes=["card-scroll"],
                    )

                    gr.Markdown("---")

                    with gr.Accordion("📈 부피 상세 통계", open=False):
                        volume_box = gr.Textbox(
                            label=None, lines=5, interactive=False, show_copy_button=True,
                            elem_id="volume_box",
                        )
                    with gr.Accordion("🩺 임상 소견", open=False):
                        clinical_box = gr.Textbox(
                            label=None, lines=7, interactive=False, show_copy_button=True,
                            elem_id="clinical_box",
                        )

                    gr.Markdown("---")

                    gr.Markdown("#### 🔬 장기 설명 요청 (REG)")
                    with gr.Row():
                        organ_select = gr.Dropdown(
                            label="세그멘테이션된 장기 선택",
                            choices=[], interactive=True, scale=3,
                        )
                        reg_btn = gr.Button("설명 요청", variant="secondary", scale=1)
                    reg_answer_html = gr.HTML(value="")

                    gr.Markdown("---")

                    with gr.Row():
                        pdf_btn    = gr.Button("📄 PDF 생성", variant="secondary", scale=1)
                        pdf_output = gr.File(label="PDF 다운로드", scale=2)
                    pdf_status = gr.Textbox(label=None, interactive=False, lines=1,
                                            placeholder="PDF 생성 상태")

            # ── 이벤트 연결 ─────────────────────────────────────────────────────
            _LOAD_OUT = [axial_img, sagittal_img, coronal_img, load_status, header_html, legend_html]
            _RUN_OUT  = [axial_img, sagittal_img, coronal_img,
                         legend_html, results_html, header_html,
                         volume_box, clinical_box, reg_answer_html, organ_select]
            _VIEW_OUT = [axial_img, sagittal_img, coronal_img]
            _CTRL_IN  = [slice_slider, alpha_slider, wl_slider, ww_slider, mask_toggle]

            # RRN 입력 → 나이/성별 자동 채우기
            rrn_input.change(
                fn=on_rrn_change,
                inputs=[rrn_input],
                outputs=[age_input, sex_input],
            )

            file_input.change(
                fn=load_file,
                inputs=[file_input, patient_name_input, exam_date_input],
                outputs=_LOAD_OUT,
            )
            for inp in [patient_name_input, exam_date_input]:
                inp.change(
                    fn=lambda n, d: _make_header_html(n, d, "idle"),
                    inputs=[patient_name_input, exam_date_input],
                    outputs=[header_html],
                )
            run_btn.click(
                fn=run_inference,
                inputs=[question_input, slice_slider, alpha_slider, wl_slider, ww_slider,
                        mask_toggle, age_input, sex_input,
                        patient_name_input, exam_date_input, doctor_notes_input, rrn_input],
                outputs=_RUN_OUT,
            )
            pdf_btn.click(
                fn=generate_pdf_report,
                outputs=[pdf_output, pdf_status],
            )
            reg_btn.click(
                fn=run_reg_fn,
                inputs=[organ_select],
                outputs=[reg_answer_html],
            )
            for ctrl in [slice_slider, alpha_slider, wl_slider, ww_slider, mask_toggle]:
                ctrl.change(fn=update_preview, inputs=_CTRL_IN, outputs=_VIEW_OUT)
            sagittal_img.select(fn=on_sagittal_click, outputs=[slice_slider]).then(
                fn=update_preview, inputs=_CTRL_IN, outputs=_VIEW_OUT,
            )
            coronal_img.select(fn=on_coronal_click, outputs=[slice_slider]).then(
                fn=update_preview, inputs=_CTRL_IN, outputs=_VIEW_OUT,
            )

        # ━━ 환자 이력 탭 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("📁 환자 이력"):

            exams_state = gr.State([])

            with gr.Row():
                refresh_hist_btn   = gr.Button("🔄 새로고침", variant="secondary", scale=1)
                export_all_btn     = gr.Button("📊 전체 CSV", variant="secondary", scale=1)
                export_patient_btn = gr.Button("📊 환자별 CSV", variant="secondary", scale=1)
                with gr.Column(scale=3):
                    pass

            exams_df = gr.Dataframe(
                headers=["ID", "환자명", "성별", "검사일", "나이", "분석일시", "장기"],
                label="검사 기록 — 행 클릭 시 상세 보기",
                interactive=False,
                wrap=True,
                elem_id="exams_df",
            )

            with gr.Row(equal_height=False):

                with gr.Column(scale=3):
                    gr.Markdown("#### 검사 상세")
                    exam_detail_html = gr.HTML(
                        value=("<div style='color:#475569;padding:24px;text-align:center;"
                               "font-size:0.9rem;'>행을 클릭하면 상세 내용이 표시됩니다</div>")
                    )
                    selected_exam_id = gr.State(None)

                with gr.Column(scale=2):
                    gr.Markdown("#### 📈 장기 부피 추이")
                    patient_dropdown = gr.Dropdown(
                        label="환자 선택", choices=[], interactive=True,
                    )
                    organ_dropdown = gr.Dropdown(
                        label="장기 선택", choices=[], interactive=True,
                    )
                    trend_plot = gr.Plot(label=None)

            csv_download = gr.File(label="📥 CSV 다운로드", visible=False)

            # 이벤트
            refresh_hist_btn.click(
                fn=refresh_history_fn,
                outputs=[exams_df, exams_state, patient_dropdown],
            )
            exams_df.select(
                fn=on_exam_select_fn,
                inputs=[exams_state],
                outputs=[exam_detail_html, selected_exam_id],
            )
            patient_dropdown.change(
                fn=select_patient_fn,
                inputs=[patient_dropdown],
                outputs=[organ_dropdown],
            )
            organ_dropdown.change(
                fn=draw_trend_fn,
                inputs=[patient_dropdown, organ_dropdown],
                outputs=[trend_plot],
            )
            export_all_btn.click(fn=export_all_fn, outputs=[csv_download])
            export_patient_btn.click(
                fn=export_patient_fn,
                inputs=[patient_dropdown],
                outputs=[csv_download],
            )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share=True,
        auth=get_auth_list(),
        auth_message="MedSeg-3D-KO — 의사 계정으로 로그인하세요",
    )
