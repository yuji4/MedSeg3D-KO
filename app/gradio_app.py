"""
MedSeg-3D-KO Gradio 앱
CT 업로드 → 한국어 질문 → 세그멘테이션 → 마스크 오버레이 + 부피 수치
"""
from __future__ import annotations

import sys
import os

# Colab에서 클론 후 sys.path 설정
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
from app.visualization import get_slice_views, make_panel


# ── 전역 상태 ──────────────────────────────────────────────────────────────────
_pipeline: SegmentationPipeline | None = None
_translator = MedicalTranslator()

_current_volume: np.ndarray | None = None
_current_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

# 마지막 추론 결과 (슬라이더 갱신 + PDF 생성용)
_last_inference: dict = {}


# ── 초기화 ────────────────────────────────────────────────────────────────────
def _get_pipeline() -> SegmentationPipeline:
    global _pipeline
    if _pipeline is None:
        in_colab = "google.colab" in sys.modules
        config = get_colab_config() if in_colab else ModelConfig(precision="bf16")
        _pipeline = SegmentationPipeline(config)
        _pipeline.load()
    return _pipeline


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def _views_to_pil(views: dict) -> tuple[Image.Image, Image.Image, Image.Image]:
    return (
        Image.fromarray(views["axial"]),
        Image.fromarray(views["sagittal"]),
        Image.fromarray(views["coronal"]),
    )


# ── 파일 로드 ─────────────────────────────────────────────────────────────────
def load_file(file_obj) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str]:
    """업로드된 CT 파일을 로드하고 3방향 슬라이스 미리보기를 반환."""
    global _current_volume, _current_spacing

    if file_obj is None:
        return None, None, None, "파일을 업로드해주세요."

    path: str = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        if path.endswith(".nii.gz") or path.endswith(".nii"):
            nii = nib.load(path)
            zooms = nii.header.get_zooms()
            _current_spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
            arr = nii.get_fdata().astype(np.float32)
            # NIfTI는 (W, H, D) → (D, H, W) 로 전치
            _current_volume = arr.transpose(2, 1, 0)
        elif path.endswith(".npy"):
            arr = np.load(path)
            # (1, D, H, W) or (D, H, W)
            _current_volume = arr[0] if arr.ndim == 4 else arr
            _current_spacing = (1.0, 1.0, 1.0)
        else:
            return None, None, None, f"지원하지 않는 형식입니다: {os.path.basename(path)}\n(.nii.gz / .nii / .npy 만 가능)"

        views = get_slice_views(_current_volume)
        D, H, W = _current_volume.shape
        msg = (
            f"로드 완료: {os.path.basename(path)} | "
            f"크기: {D}×{H}×{W} | "
            f"간격: {_current_spacing[0]:.2f}×{_current_spacing[1]:.2f}×{_current_spacing[2]:.2f} mm"
        )
        return (*_views_to_pil(views), msg)

    except Exception as e:
        return None, None, None, f"파일 로드 실패: {e}"


# ── 추론 ──────────────────────────────────────────────────────────────────────
def run_inference(
    question_ko: str,
    slice_idx: int,
    alpha: float,
    wl: float,
    ww: float,
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str, str, str]:
    """단일/복수 장기 세그멘테이션 수행. 3방향 슬라이스 이미지 + 분석 텍스트 반환."""
    global _current_volume, _current_spacing, _last_inference

    if _current_volume is None:
        return None, None, None, "", "먼저 CT 파일을 업로드해주세요.", ""
    if not question_ko.strip():
        return None, None, None, "", "질문을 입력해주세요.", ""

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return None, None, None, "", f"모델 로드 실패: {e}", ""

    organs = _detect_organs_en(question_ko)

    # ── 장기별 추론 ──────────────────────────────────────────────────────────
    try:
        if not organs:
            results = [pipeline.run(_current_volume, question_ko)]
            organs = [results[0]["organ_label"] or ""]
        else:
            image_pt, original = pipeline._prepare_image_pt(_current_volume)
            results = [
                pipeline._infer(image_pt, original, org, question_ko)
                for org in organs
            ]
    except Exception as e:
        return None, None, None, "", f"추론 오류: {e}", ""

    # ── 마스크 합성 ──────────────────────────────────────────────────────────
    vol_shape = _current_volume.shape
    combined_mask = np.zeros(vol_shape, dtype=np.uint8)
    label_names: dict[int, str] = {}
    for lbl_idx, (org, res) in enumerate(zip(organs, results), start=1):
        if res["mask"].any():
            combined_mask[res["mask"]] = lbl_idx
        label_names[lbl_idx] = get_korean_term(org) if org else f"구조물 {lbl_idx}"

    # ── 부피 통계 + 임상 평가 ────────────────────────────────────────────────
    volume_lines: list[str] = []
    answer_lines: list[str] = []
    organ_results: list[dict] = []
    assessments = []
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

        volume_lines.append(
            stats.summary_ko() if present else f"[{get_korean_term(org)}] 마스크가 감지되지 않았습니다."
        )
        answer_lines.append(
            _translator.translate_segmentation_result(
                org, present=present, volume_ml=stats.volume_ml if present else None
            )
        )

    volume_report = "\n\n".join(volume_lines)
    answer_ko = "\n".join(answer_lines)
    clinical_text = format_clinical_summary(assessments)

    # ── 시각화 ──────────────────────────────────────────────────────────────
    D = _current_volume.shape[0]
    auto_idx = _best_slice_index(combined_mask > 0) if any_detected else D // 2
    chosen_idx = max(0, min(auto_idx if slice_idx == 0 else slice_idx, D - 1))

    views = get_slice_views(
        _current_volume,
        combined_mask,
        slice_indices={
            "axial": chosen_idx,
            "sagittal": _current_volume.shape[2] // 2,
            "coronal": _current_volume.shape[1] // 2,
        },
        alpha=alpha,
        wl=wl,
        ww=ww,
        label_names=label_names,
    )
    panel_arr = make_panel(views)  # PDF용 단일 패널 유지

    _last_inference.update({
        "organ_results": organ_results,
        "panel_image": panel_arr,
        "patient_id": "",
        "combined_mask": combined_mask,
        "label_names": label_names,
    })

    return (*_views_to_pil(views), volume_report, answer_ko, clinical_text)


def _best_slice_index(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def generate_pdf_report(patient_id: str = "") -> str | None:
    if not _last_inference.get("organ_results"):
        return None
    _last_inference["patient_id"] = patient_id
    return generate_report(
        organ_results=_last_inference["organ_results"],
        panel_image=_last_inference.get("panel_image"),
        patient_id=patient_id,
    )


def update_preview(
    slice_idx: int, alpha: float, wl: float, ww: float
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None]:
    """슬라이더 변경 시 저장된 마스크와 함께 3방향 뷰 갱신."""
    if _current_volume is None:
        return None, None, None
    D = _current_volume.shape[0]
    idx = max(0, min(int(slice_idx), D - 1))
    combined_mask = _last_inference.get("combined_mask")
    label_names = _last_inference.get("label_names") if combined_mask is not None else None
    views = get_slice_views(
        _current_volume,
        combined_mask,
        slice_indices={
            "axial": idx,
            "sagittal": _current_volume.shape[2] // 2,
            "coronal": _current_volume.shape[1] // 2,
        },
        wl=wl,
        ww=ww,
        alpha=alpha,
        label_names=label_names,
    )
    return _views_to_pil(views)


# ── 클릭 네비게이션 ───────────────────────────────────────────────────────────
def on_sagittal_click(evt: gr.SelectData) -> int:
    """시상면 클릭 → y좌표를 축상 슬라이스 인덱스로 변환."""
    if _current_volume is None:
        return 0
    D = _current_volume.shape[0]
    return max(0, min(int(evt.index[1] / 256 * D), D - 1))


def on_coronal_click(evt: gr.SelectData) -> int:
    """관상면 클릭 → y좌표를 축상 슬라이스 인덱스로 변환."""
    if _current_volume is None:
        return 0
    D = _current_volume.shape[0]
    return max(0, min(int(evt.index[1] / 256 * D), D - 1))


# ── UI ────────────────────────────────────────────────────────────────────────
TITLE = "# MedSeg-3D-KO\n### M3D 기반 3D 의료 영상 한국어 세그멘테이션"

EXAMPLES_Q = [
    "간을 분할해줘",
    "비장을 세그멘테이션해줘",
    "좌측 폐를 분할해줘",
    "신장 마스크를 보여줘",
    "췌장이 어디있어?",
]

with gr.Blocks(title="MedSeg-3D-KO", theme=gr.themes.Soft()) as demo:
    gr.Markdown(TITLE)

    with gr.Row():
        # ── 왼쪽: 입력 패널 ───────────────────────────────────────────────
        with gr.Column(scale=1):
            file_input = gr.File(
                label="CT 파일 업로드 (.nii.gz / .nii / .npy)",
                file_types=[".nii", ".gz", ".npy"],
            )
            load_status = gr.Textbox(label="로드 상태", interactive=False, lines=2)

            question_input = gr.Textbox(
                label="한국어 질문",
                placeholder="예: 간을 분할해줘",
                lines=2,
            )
            gr.Examples(examples=EXAMPLES_Q, inputs=[question_input], label="예시 질문")

            with gr.Accordion("시각화 설정", open=False):
                slice_slider = gr.Slider(0, 31, value=0, step=1, label="축상 슬라이스 번호 (0=자동)")
                alpha_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="마스크 불투명도")
                wl_slider = gr.Slider(-200, 400, value=40, step=10, label="윈도우 레벨 (HU)")
                ww_slider = gr.Slider(100, 2000, value=400, step=50, label="윈도우 너비 (HU)")

            with gr.Row():
                clear_btn = gr.ClearButton([file_input, question_input], value="초기화")
                run_btn = gr.Button("세그멘테이션 실행", variant="primary")

        # ── 오른쪽: 출력 패널 ─────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("**CT 슬라이스** — 시상/관상 뷰를 클릭하면 해당 깊이로 축상 슬라이스가 이동합니다")
            with gr.Row():
                axial_img = gr.Image(label="축상면 (Axial)", type="pil")
                sagittal_img = gr.Image(label="시상면 (Sagittal) ← 클릭", type="pil")
                coronal_img = gr.Image(label="관상면 (Coronal) ← 클릭", type="pil")
            answer_box = gr.Textbox(label="한국어 분석 결과", lines=3, interactive=False)
            volume_box = gr.Textbox(label="부피·크기 통계", lines=6, interactive=False)
            clinical_box = gr.Textbox(label="임상 평가 (정상 범위 비교)", lines=8, interactive=False)
            with gr.Row():
                patient_id_input = gr.Textbox(
                    label="환자 ID (선택)", placeholder="예: P-20240522", scale=2
                )
                pdf_btn = gr.Button("📄 PDF 보고서 생성", scale=1)
            pdf_output = gr.File(label="PDF 다운로드", visible=False)

    # ── 이벤트 연결 ───────────────────────────────────────────────────────────
    file_input.change(
        fn=load_file,
        inputs=[file_input],
        outputs=[axial_img, sagittal_img, coronal_img, load_status],
    )

    run_btn.click(
        fn=run_inference,
        inputs=[question_input, slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[axial_img, sagittal_img, coronal_img, volume_box, answer_box, clinical_box],
    )

    pdf_btn.click(
        fn=lambda pid: (generate_pdf_report(pid), gr.File(visible=True)),
        inputs=[patient_id_input],
        outputs=[pdf_output, pdf_output],
    )

    for slider in [slice_slider, alpha_slider, wl_slider, ww_slider]:
        slider.change(
            fn=update_preview,
            inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
            outputs=[axial_img, sagittal_img, coronal_img],
        )

    # 시상면 클릭 → 축상 슬라이스 업데이트
    sagittal_img.select(
        fn=on_sagittal_click,
        outputs=[slice_slider],
    ).then(
        fn=update_preview,
        inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[axial_img, sagittal_img, coronal_img],
    )

    # 관상면 클릭 → 축상 슬라이스 업데이트
    coronal_img.select(
        fn=on_coronal_click,
        outputs=[slice_slider],
    ).then(
        fn=update_preview,
        inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[axial_img, sagittal_img, coronal_img],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
