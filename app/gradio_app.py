"""
MedSeg-3D-KO Gradio 앱
CT 업로드 → 한국어 질문 → 세그멘테이션 → 마스크 오버레이 + 부피 수치
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import nibabel as nib
import gradio as gr
from PIL import Image

from src.inference.model_loader import ModelConfig, get_colab_config
from src.inference.segmentation import SegmentationPipeline, preprocess_volume, _detect_organs_en
from src.analysis.volume import analyze_mask
from src.analysis.clinical import assess_organ, format_clinical_summary
from src.analysis.report import generate_report
from src.translation.translator import MedicalTranslator
from src.translation.medical_terms import get_korean_term
from app.visualization import get_slice_views, make_panel, _LABEL_COLORS


# ── Colab 한국어 폰트 자동 설치 ────────────────────────────────────────────────
def _setup_korean_font() -> None:
    """Colab 환경에서 fonts-nanum 자동 설치 (PDF 한국어 출력용)."""
    _NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(_NANUM_PATH):
        return  # 이미 설치됨
    if "google.colab" not in sys.modules:
        return  # Colab 아님
    import subprocess
    print("📦 한국어 폰트 설치 중 (fonts-nanum)...")
    subprocess.run(["apt-get", "install", "-y", "fonts-nanum", "-q"], check=False)
    if os.path.exists(_NANUM_PATH):
        print("✅ 한국어 폰트 설치 완료 — PDF에서 한국어가 출력됩니다.")
    else:
        print("⚠️  폰트 설치 실패 — PDF는 영문으로 출력됩니다.")


_setup_korean_font()


# ── 전역 상태 ──────────────────────────────────────────────────────────────────
_pipeline: SegmentationPipeline | None = None
_translator = MedicalTranslator()
_current_volume: np.ndarray | None = None
_current_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
_last_inference: dict = {}


# ── 테마 ──────────────────────────────────────────────────────────────────────
_THEME = gr.themes.Soft()

_CSS = """
footer { display: none; }
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; }
.tab-nav button { font-size: 0.92rem; font-weight: 600; }
h1, h2, h3 { color: #e2e8f0 !important; }
.legend-wrap { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; }
.info-tag { background: #1e3a5f; color: #93c5fd; padding: 2px 8px; border-radius: 4px;
            font-size: 0.8rem; font-weight: 600; margin-right: 6px; }
"""

_HEADER = """
<div style="padding: 18px 0 10px; border-bottom: 1px solid #334155; margin-bottom: 16px;">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:2rem;">🏥</span>
    <div>
      <h1 style="margin:0; font-size:1.6rem; color:#e2e8f0;">MedSeg-3D-KO</h1>
      <p style="margin:4px 0 0; color:#94a3b8; font-size:0.9rem;">
        M3D-LaMed 기반 3D 의료 영상 한국어 세그멘테이션 &nbsp;|&nbsp;
        CT 업로드 → 한국어 질문 → 마스크 오버레이 + 부피·임상 분석
      </p>
    </div>
  </div>
</div>
"""


# ── 색상 범례 HTML ──────────────────────────────────────────────────────────────
def _make_legend_html(label_names: dict[int, str]) -> str:
    labels = {k: v for k, v in label_names.items() if k != 0}
    if not labels:
        return (
            "<div class='legend-wrap' style='color:#64748b; font-size:0.85rem;'>"
            "세그멘테이션 실행 후 장기별 색상 범례가 표시됩니다</div>"
        )
    items = []
    for lbl, name in sorted(labels.items()):
        r, g, b = _LABEL_COLORS[lbl % len(_LABEL_COLORS)]
        hex_c = f"#{r:02x}{g:02x}{b:02x}"
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin:3px 12px 3px 0;">'
            f'<span style="width:13px;height:13px;background:{hex_c};display:inline-block;'
            f'margin-right:6px;border-radius:3px;flex-shrink:0;"></span>'
            f'<span style="color:#e2e8f0;font-size:0.88rem;font-weight:500;">{name}</span>'
            f'</span>'
        )
    return (
        "<div class='legend-wrap' style='display:flex;flex-wrap:wrap;align-items:center;'>"
        + "".join(items)
        + "</div>"
    )


# ── 초기화 ────────────────────────────────────────────────────────────────────
def _get_pipeline() -> SegmentationPipeline:
    global _pipeline
    if _pipeline is None:
        in_colab = "google.colab" in sys.modules
        config = get_colab_config() if in_colab else ModelConfig(precision="bf16")
        _pipeline = SegmentationPipeline(config)
        _pipeline.load()
    return _pipeline


def _views_to_pil(views: dict) -> tuple[Image.Image, Image.Image, Image.Image]:
    return (
        Image.fromarray(views["axial"]),
        Image.fromarray(views["sagittal"]),
        Image.fromarray(views["coronal"]),
    )


# ── 파일 로드 ─────────────────────────────────────────────────────────────────
def load_file(file_obj):
    global _current_volume, _current_spacing
    empty = (None, None, None, "", _make_legend_html({}))

    if file_obj is None:
        return (*empty[:3], "파일을 업로드해주세요.", empty[-1])

    path: str = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        if path.endswith(".nii.gz") or path.endswith(".nii"):
            nii = nib.load(path)
            zooms = nii.header.get_zooms()
            _current_spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
            arr = nii.get_fdata().astype(np.float32)
            _current_volume = arr.transpose(2, 1, 0)
        elif path.endswith(".npy"):
            arr = np.load(path)
            _current_volume = arr[0] if arr.ndim == 4 else arr
            _current_spacing = (1.0, 1.0, 1.0)
        else:
            return (*empty[:3], f"지원하지 않는 형식: {os.path.basename(path)}", empty[-1])

        views = get_slice_views(_current_volume)
        D, H, W = _current_volume.shape
        sp = _current_spacing
        msg = (
            f"✅  {os.path.basename(path)}  |  "
            f"{D}×{H}×{W} voxels  |  "
            f"간격 {sp[0]:.2f}×{sp[1]:.2f}×{sp[2]:.2f} mm"
        )
        return (*_views_to_pil(views), msg, _make_legend_html({}))

    except Exception as e:
        return (*empty[:3], f"❌ 로드 실패: {e}", empty[-1])


# ── 추론 ──────────────────────────────────────────────────────────────────────
def run_inference(question_ko, slice_idx, alpha, wl, ww):
    global _current_volume, _current_spacing, _last_inference
    if _current_volume is None:
        return None, None, None, _make_legend_html({}), "", "먼저 CT 파일을 업로드해주세요.", ""
    if not question_ko.strip():
        return None, None, None, _make_legend_html({}), "", "질문을 입력해주세요.", ""

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return None, None, None, _make_legend_html({}), "", f"모델 로드 실패: {e}", ""

    organs = _detect_organs_en(question_ko)

    try:
        if not organs:
            results = [pipeline.run(_current_volume, question_ko)]
            organs = [results[0]["organ_label"] or ""]
        else:
            image_pt, original = pipeline._prepare_image_pt(_current_volume)
            results = [pipeline._infer(image_pt, original, org, question_ko) for org in organs]
    except Exception as e:
        return None, None, None, _make_legend_html({}), "", f"추론 오류: {e}", ""

    vol_shape = _current_volume.shape
    combined_mask = np.zeros(vol_shape, dtype=np.uint8)
    label_names: dict[int, str] = {}
    for lbl_idx, (org, res) in enumerate(zip(organs, results), start=1):
        if res["mask"].any():
            combined_mask[res["mask"]] = lbl_idx
        label_names[lbl_idx] = get_korean_term(org) if org else f"구조물 {lbl_idx}"

    volume_lines, answer_lines, organ_results, assessments = [], [], [], []
    any_detected = False

    for lbl_idx, (org, res) in enumerate(zip(organs, results), start=1):
        organ_mask = combined_mask == lbl_idx
        stats = analyze_mask(organ_mask, label=org, voxel_spacing_mm=_current_spacing)
        present = stats.voxel_count > 0
        if present:
            any_detected = True
        assessment = assess_organ(org, stats.volume_ml) if present else assess_organ(org, 0.0)
        assessments.append(assessment)
        organ_results.append({"label": org, "stats": stats, "assessment": assessment})
        volume_lines.append(stats.summary_ko() if present else f"[{get_korean_term(org)}] 마스크 미감지")
        answer_lines.append(
            _translator.translate_segmentation_result(
                org, present=present, volume_ml=stats.volume_ml if present else None
            )
        )

    D = _current_volume.shape[0]
    auto_idx = _best_slice_index(combined_mask > 0) if any_detected else D // 2
    chosen_idx = max(0, min(auto_idx if slice_idx == 0 else slice_idx, D - 1))

    views = get_slice_views(
        _current_volume, combined_mask,
        slice_indices={
            "axial": chosen_idx,
            "sagittal": _current_volume.shape[2] // 2,
            "coronal": _current_volume.shape[1] // 2,
        },
        alpha=alpha, wl=wl, ww=ww, label_names=label_names,
    )
    panel_arr = make_panel(views)

    _last_inference.update({
        "organ_results": organ_results,
        "panel_image": panel_arr,
        "patient_id": "",
        "combined_mask": combined_mask,
        "label_names": label_names,
    })

    return (
        *_views_to_pil(views),
        _make_legend_html(label_names),
        "\n\n".join(volume_lines),
        "\n".join(answer_lines),
        format_clinical_summary(assessments),
    )


def _best_slice_index(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def generate_pdf_report(patient_id: str = "") -> tuple:
    if not _last_inference.get("organ_results"):
        return None, "⚠️ 먼저 세그멘테이션을 실행해주세요."
    _last_inference["patient_id"] = patient_id
    path = generate_report(
        organ_results=_last_inference["organ_results"],
        panel_image=_last_inference.get("panel_image"),
        patient_id=patient_id,
    )
    return path, f"✅ PDF 생성 완료: {os.path.basename(path)}"


def update_preview(slice_idx, alpha, wl, ww):
    if _current_volume is None:
        return None, None, None
    D = _current_volume.shape[0]
    idx = max(0, min(int(slice_idx), D - 1))
    combined_mask = _last_inference.get("combined_mask")
    label_names = _last_inference.get("label_names") if combined_mask is not None else None
    views = get_slice_views(
        _current_volume, combined_mask,
        slice_indices={
            "axial": idx,
            "sagittal": _current_volume.shape[2] // 2,
            "coronal": _current_volume.shape[1] // 2,
        },
        wl=wl, ww=ww, alpha=alpha, label_names=label_names,
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


# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES_Q = [
    "간을 분할해줘",
    "비장을 세그멘테이션해줘",
    "좌측 폐를 분할해줘",
    "신장 마스크를 보여줘",
    "췌장이 어디있어?",
]

with gr.Blocks(title="MedSeg-3D-KO", theme=_THEME, css=_CSS) as demo:

    gr.HTML(_HEADER)

    with gr.Row(equal_height=False):

        # ── 왼쪽: 입력 패널 ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=290):

            with gr.Group():
                gr.Markdown("#### 📂 CT 파일 업로드")
                file_input = gr.File(
                    label="파일 선택 (.nii.gz / .nii / .npy)",
                    file_types=[".nii", ".gz", ".npy"],
                )
                load_status = gr.Textbox(
                    label="로드 상태", interactive=False, lines=2, show_copy_button=False,
                )

            gr.Markdown("---")

            with gr.Group():
                gr.Markdown("#### 💬 한국어 질문")
                question_input = gr.Textbox(
                    label=None,
                    placeholder="예: 간을 분할해줘",
                    lines=2,
                )
                gr.Examples(
                    examples=EXAMPLES_Q,
                    inputs=[question_input],
                    label="예시 질문",
                )

            gr.Markdown("---")

            with gr.Accordion("⚙️ 시각화 설정", open=False):
                slice_slider = gr.Slider(0, 31, value=0, step=1, label="축상 슬라이스 (0=자동)")
                alpha_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="마스크 불투명도")
                wl_slider = gr.Slider(-200, 400, value=40, step=10, label="윈도우 레벨 (HU)")
                ww_slider = gr.Slider(100, 2000, value=400, step=50, label="윈도우 너비 (HU)")

            gr.Markdown("---")

            with gr.Row():
                clear_btn = gr.ClearButton(
                    [file_input, question_input], value="🗑️ 초기화",
                )
                run_btn = gr.Button("🔬 세그멘테이션 실행", variant="primary")

        # ── 오른쪽: 출력 패널 ─────────────────────────────────────────────
        with gr.Column(scale=3):

            # CT 뷰어
            with gr.Group():
                gr.Markdown(
                    "#### 🖼️ CT 슬라이스 뷰어 "
                    "<span style='color:#64748b;font-size:0.8rem;font-weight:400;'>"
                    "시상/관상면 클릭 → 해당 깊이로 축상 슬라이스 이동</span>"
                )
                with gr.Row(equal_height=True):
                    axial_img    = gr.Image(label="축상면 (Axial)",              type="pil", height=260)
                    sagittal_img = gr.Image(label="시상면 (Sagittal) ← 클릭",   type="pil", height=260)
                    coronal_img  = gr.Image(label="관상면 (Coronal) ← 클릭",    type="pil", height=260)
                legend_html = gr.HTML(value=_make_legend_html({}))

            gr.Markdown("---")

            # 결과 탭
            with gr.Tabs():

                with gr.Tab("📊 분석 결과"):
                    with gr.Row():
                        with gr.Column():
                            answer_box = gr.Textbox(
                                label="한국어 분석 결과",
                                lines=3,
                                interactive=False,
                                show_copy_button=True,
                            )
                        with gr.Column():
                            volume_box = gr.Textbox(
                                label="부피·크기 통계",
                                lines=3,
                                interactive=False,
                                show_copy_button=True,
                            )

                with gr.Tab("🩺 임상 평가"):
                    clinical_box = gr.Textbox(
                        label="정상 범위 비교 (성인 기준)",
                        lines=10,
                        interactive=False,
                        show_copy_button=True,
                    )

                with gr.Tab("📄 PDF 보고서"):
                    gr.Markdown(
                        "세그멘테이션 실행 후 환자 ID를 입력하고 보고서를 생성하세요.\n\n"
                        "보고서에는 장기 부피 표, 임상 소견, CT 슬라이스 이미지가 포함됩니다."
                    )
                    with gr.Row():
                        patient_id_input = gr.Textbox(
                            label="환자 ID (선택)", placeholder="예: P-20240522", scale=2,
                        )
                        pdf_btn = gr.Button("📄 보고서 생성", variant="secondary", scale=1)
                    pdf_status = gr.Textbox(
                        label="상태", interactive=False, lines=1, show_copy_button=False,
                    )
                    pdf_output = gr.File(label="📥 PDF 다운로드")

    # ── 이벤트 연결 ───────────────────────────────────────────────────────────
    file_input.change(
        fn=load_file,
        inputs=[file_input],
        outputs=[axial_img, sagittal_img, coronal_img, load_status, legend_html],
    )

    run_btn.click(
        fn=run_inference,
        inputs=[question_input, slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[
            axial_img, sagittal_img, coronal_img,
            legend_html,
            volume_box, answer_box, clinical_box,
        ],
    )

    pdf_btn.click(
        fn=generate_pdf_report,
        inputs=[patient_id_input],
        outputs=[pdf_output, pdf_status],
    )

    for slider in [slice_slider, alpha_slider, wl_slider, ww_slider]:
        slider.change(
            fn=update_preview,
            inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
            outputs=[axial_img, sagittal_img, coronal_img],
        )

    sagittal_img.select(fn=on_sagittal_click, outputs=[slice_slider]).then(
        fn=update_preview,
        inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[axial_img, sagittal_img, coronal_img],
    )

    coronal_img.select(fn=on_coronal_click, outputs=[slice_slider]).then(
        fn=update_preview,
        inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[axial_img, sagittal_img, coronal_img],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
