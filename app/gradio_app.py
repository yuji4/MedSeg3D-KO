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
from src.inference.segmentation import SegmentationPipeline, _detect_organs_en
from src.analysis.volume import analyze_mask
from src.analysis.clinical import assess_organ, format_clinical_summary
from src.analysis.report import generate_report
from src.database.db import init_db
from src.database.crud import (
    upsert_patient, save_exam, list_patients,
    list_all_exams, get_patient_exams, get_exam_organs,
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
.card-scroll { max-height: 480px; overflow-y: auto; padding-right: 4px; }
.view-label { text-align:center; color:#64748b; font-size:0.78rem; padding:3px 0; }
#volume_box textarea { overflow-y: auto !important; resize: none; }
#clinical_box textarea { overflow-y: auto !important; resize: none; }
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
            border:1px solid #334155;margin-bottom:4px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:1.8rem;">🏥</span>
    <div>
      <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px;">
        MedSeg-3D-KO</div>
      <div style="font-size:0.78rem;color:#64748b;">
        M3D 기반 3D 의료 영상 한국어 세그멘테이션</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="text-align:right;">
      <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;">{name}</div>
      <div style="font-size:0.78rem;color:#94a3b8;">검사일: {date}</div>
    </div>
    <div style="background:{bg};color:{color};border:1px solid {color};
                border-radius:20px;padding:4px 14px;
                font-size:0.78rem;font-weight:600;white-space:nowrap;">
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
            f'display:inline-block;margin-right:5px;flex-shrink:0;"></span>'
            f'<span style="color:#cbd5e1;font-size:0.82rem;">{name}</span></span>'
        )
    return ('<div style="display:flex;flex-wrap:wrap;align-items:center;padding:4px 0;">'
            + "".join(items) + "</div>")


def _make_results_html(assessments: list) -> str:
    if not assessments:
        return ("<div style='color:#475569;padding:32px 16px;text-align:center;"
                "font-size:0.9rem;line-height:1.8;'>"
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
        note = (f" <span style='color:#64748b;font-size:0.74rem;'>({nr.note})</span>"
                if nr.note else "")
        cards.append(f"""
<div style="background:#1e293b;border-radius:8px;padding:11px 14px;margin-bottom:7px;
            border:1px solid #334155;border-left:4px solid {border};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="color:#e2e8f0;font-weight:600;font-size:0.92rem;">{get_korean_term(a.label)}</span>
    <span style="background:{badge_bg};color:{border};border:1px solid {border};
                 padding:2px 9px;border-radius:10px;font-size:0.72rem;font-weight:700;">
      {badge}</span>
  </div>
  <div style="color:#94a3b8;font-size:0.83rem;">
    측정: <b style="color:#e2e8f0;">{a.volume_ml:.1f} mL</b>
    &nbsp;|&nbsp; 정상범위: <span style="color:#cbd5e1;">{range_str}</span>{note}
  </div>
</div>""")
    return "<div>" + "".join(cards) + "</div>"


def _make_exam_detail_html(organs: list[dict]) -> str:
    """DB에서 가져온 장기 결과 dict 리스트를 카드 HTML로 변환."""
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
        note = o.get("note", "") or ""
        note_html = (f" <span style='color:#64748b;font-size:0.74rem;'>({note})</span>"
                     if note else "")
        cards.append(f"""
<div style="background:#1e293b;border-radius:8px;padding:10px 13px;margin-bottom:6px;
            border:1px solid #334155;border-left:4px solid {border};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
    <span style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{get_korean_term(o['label'])}</span>
    <span style="background:{badge_bg};color:{border};border:1px solid {border};
                 padding:1px 8px;border-radius:10px;font-size:0.71rem;font-weight:700;">{badge}</span>
  </div>
  <div style="color:#94a3b8;font-size:0.82rem;">
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
        margin=dict(l=50, r=20, t=40, b=40),
        title=dict(
            text=f"{get_korean_term(label)} 부피 추이 (mL)",
            font=dict(size=14, color="#e2e8f0"),
        ),
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
def run_inference(question_ko, slice_idx, alpha, wl, ww, mask_on,
                  age, sex_ko, patient_name, exam_date):
    global _current_volume, _current_spacing, _last_inference

    _hdr = lambda s: _make_header_html(patient_name, exam_date, s)
    _blank = (None, None, None, _make_legend_html({}), _make_results_html([]))

    if _current_volume is None:
        return (*_blank, _hdr("idle"), "", "먼저 CT 파일을 업로드해주세요.")
    if not question_ko.strip():
        return (*_blank, _hdr("loaded"), "", "질문을 입력해주세요.")

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return (*_blank, _hdr("idle"), "", f"모델 로드 실패: {e}")

    organs = _detect_organs_en(question_ko)
    try:
        if not organs:
            results = [pipeline.run(_current_volume, question_ko)]
            organs = [results[0]["organ_label"] or ""]
        else:
            image_pt, original = pipeline._prepare_image_pt(_current_volume)
            results = [pipeline._infer(image_pt, original, org, question_ko) for org in organs]
    except Exception as e:
        return (*_blank, _hdr("loaded"), "", f"추론 오류: {e}")

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
        "patient_id": "",
        "combined_mask": combined_mask,
        "label_names": label_names,
    })

    # DB 저장
    try:
        pid = upsert_patient(patient_name or "미입력", sex)
        save_exam(pid, exam_date or datetime.now().strftime("%Y-%m-%d"),
                  int(age), organ_results)
    except Exception:
        pass

    return (
        *_views_to_pil(views),
        _make_legend_html(label_names) if mask_on else _make_legend_html({}),
        _make_results_html(assessments),
        _hdr(status),
        "\n\n".join(volume_lines),
        format_clinical_summary(assessments, age=int(age), sex=sex),
    )


def _best_slice_index(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def generate_pdf_report(patient_id_str: str = "") -> tuple:
    if not _last_inference.get("organ_results"):
        return None, "⚠️ 먼저 세그멘테이션을 실행해주세요."
    path = generate_report(
        organ_results=_last_inference["organ_results"],
        panel_image=_last_inference.get("panel_image"),
        patient_id=patient_id_str,
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


# ── 환자 이력 탭 이벤트 핸들러 ────────────────────────────────────────────────
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
    patient_choices = [f"{p['name']} (ID:{p['id']})" for p in patients]
    return _exams_to_df(exams), exams, gr.update(choices=patient_choices)


def on_exam_select_fn(evt: gr.SelectData, exams: list) -> tuple:
    if not exams or evt.index[0] >= len(exams):
        return "<div style='color:#475569;padding:16px;'>선택 오류</div>", None
    exam = exams[evt.index[0]]
    exam_id = exam["id"]
    organs = get_exam_organs(exam_id)
    detail_html = (
        f"<div style='color:#94a3b8;font-size:0.83rem;padding:6px 10px;"
        f"background:#0f172a;border-radius:6px;margin-bottom:8px;'>"
        f"👤 {exam.get('patient_name','-')} | "
        f"검사일: {exam.get('exam_date','-')} | "
        f"{exam.get('age_at_exam','-')}세 | "
        f"분석: {str(exam.get('analyzed_at',''))[:16].replace('T',' ')}"
        f"</div>"
        + _make_exam_detail_html(organs)
    )
    return detail_html, exam_id


def load_patient_trend_fn(patient_str: str, organ_label: str):
    if not patient_str or not organ_label:
        return None, gr.update(choices=[])
    try:
        pid = int(patient_str.split("ID:")[-1].rstrip(")"))
    except Exception:
        return None, gr.update(choices=[])
    labels = get_patient_organ_labels(pid)
    ko_choices = [f"{get_korean_term(l)} ({l})" for l in labels]
    fig = _make_trend_chart(pid, organ_label)
    return fig, gr.update(choices=ko_choices)


def select_patient_fn(patient_str: str):
    if not patient_str:
        return gr.update(choices=[])
    try:
        pid = int(patient_str.split("ID:")[-1].rstrip(")"))
    except Exception:
        return gr.update(choices=[])
    labels = get_patient_organ_labels(pid)
    return gr.update(choices=[f"{get_korean_term(l)} ({l})" for l in labels], value=None)


def draw_trend_fn(patient_str: str, organ_ko_str: str):
    if not patient_str or not organ_ko_str:
        return None
    try:
        pid = int(patient_str.split("ID:")[-1].rstrip(")"))
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
                with gr.Column(scale=1, min_width=270):

                    with gr.Group():
                        gr.Markdown("#### 👤 환자 정보")
                        patient_name_input = gr.Textbox(
                            label="환자 이름", placeholder="홍길동 (선택)", lines=1,
                        )
                        exam_date_input = gr.Textbox(
                            label="검사일",
                            placeholder=datetime.now().strftime("%Y-%m-%d"),
                            lines=1,
                        )
                        with gr.Row():
                            age_input = gr.Number(
                                label="나이 (세)", value=30, minimum=0, maximum=120, step=1,
                            )
                            sex_input = gr.Radio(
                                ["남성", "여성", "미입력"], label="성별", value="미입력",
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
                            axial_img    = gr.Image(show_label=False, type="pil", height=260)
                            sagittal_img = gr.Image(show_label=False, type="pil", height=260)
                            coronal_img  = gr.Image(show_label=False, type="pil", height=260)
                        with gr.Row():
                            gr.HTML("<div class='view-label'>축상면 (Axial)</div>")
                            gr.HTML("<div class='view-label'>시상면 (Sagittal) ← 클릭</div>")
                            gr.HTML("<div class='view-label'>관상면 (Coronal) ← 클릭</div>")
                        legend_html = gr.HTML(value=_make_legend_html({}))

                    gr.Markdown("---")

                    with gr.Group():
                        gr.Markdown("#### 🎛️ 뷰어 컨트롤")
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

                # ── 우측: 결과 패널 ─────────────────────────────────────────────
                with gr.Column(scale=2):

                    gr.Markdown("#### 📊 장기별 분석 결과")
                    results_html = gr.HTML(
                        value=_make_results_html([]),
                        elem_classes=["card-scroll"],
                    )
                    with gr.Accordion("📈 부피 상세 통계", open=False):
                        volume_box = gr.Textbox(
                            label=None, lines=6, interactive=False, show_copy_button=True,
                            elem_id="volume_box",
                        )
                    with gr.Accordion("🩺 임상 소견 상세", open=False):
                        clinical_box = gr.Textbox(
                            label=None, lines=8, interactive=False, show_copy_button=True,
                            elem_id="clinical_box",
                        )

            gr.Markdown("---")
            with gr.Row():
                question_input = gr.Textbox(
                    label="한국어 질문",
                    placeholder="예: 간을 분할해줘  /  신장이랑 비장 찾아줘",
                    lines=1, scale=4,
                )
                run_btn   = gr.Button("🔬 세그멘테이션 실행", variant="primary", scale=1)
                clear_btn = gr.ClearButton([file_input, question_input], value="🗑️ 초기화", scale=1)

            gr.Examples(examples=EXAMPLES_Q, inputs=[question_input], label="예시 질문")

            with gr.Row():
                patient_id_input = gr.Textbox(
                    label="환자 ID (PDF용)", placeholder="P-20240522", scale=3,
                )
                pdf_btn = gr.Button("📄 PDF 보고서 생성", variant="secondary", scale=1)
            pdf_status = gr.Textbox(label="PDF 상태", interactive=False, lines=1)
            pdf_output = gr.File(label="📥 PDF 다운로드")

            # ── 이벤트 연결 ─────────────────────────────────────────────────────
            _LOAD_OUT = [axial_img, sagittal_img, coronal_img, load_status, header_html, legend_html]
            _RUN_OUT  = [axial_img, sagittal_img, coronal_img,
                         legend_html, results_html, header_html,
                         volume_box, clinical_box]
            _VIEW_OUT = [axial_img, sagittal_img, coronal_img]
            _CTRL_IN  = [slice_slider, alpha_slider, wl_slider, ww_slider, mask_toggle]

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
                        mask_toggle, age_input, sex_input, patient_name_input, exam_date_input],
                outputs=_RUN_OUT,
            )
            pdf_btn.click(
                fn=generate_pdf_report,
                inputs=[patient_id_input],
                outputs=[pdf_output, pdf_status],
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
                with gr.Column(scale=2):
                    pass

            exams_df = gr.Dataframe(
                headers=["ID", "환자명", "성별", "검사일", "나이", "분석일시", "장기"],
                label="검사 기록 — 행 클릭 시 상세 보기",
                interactive=False,
                wrap=True,
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
            export_all_btn.click(
                fn=export_all_fn,
                outputs=[csv_download],
            )
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
