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
from src.inference.segmentation import SegmentationPipeline, preprocess_volume
from src.analysis.volume import analyze_mask
from src.translation.translator import MedicalTranslator
from src.translation.medical_terms import get_korean_term
from app.visualization import get_slice_views, make_panel


# ── 전역 상태 ──────────────────────────────────────────────────────────────────
_pipeline: SegmentationPipeline | None = None
_translator = MedicalTranslator()

# 현재 로드된 볼륨 (전역 캐시)
_current_volume: np.ndarray | None = None
_current_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


# ── 초기화 ────────────────────────────────────────────────────────────────────
def _get_pipeline() -> SegmentationPipeline:
    global _pipeline
    if _pipeline is None:
        # Colab이면 4-bit, 로컬이면 기본 설정
        in_colab = "google.colab" in sys.modules
        config = get_colab_config() if in_colab else ModelConfig(precision="bf16")
        _pipeline = SegmentationPipeline(config)
        _pipeline.load()
    return _pipeline


# ── 파일 로드 ─────────────────────────────────────────────────────────────────
def load_file(file_obj) -> tuple[Image.Image | None, str]:
    """업로드된 CT 파일을 로드하고 중앙 축상 슬라이스 미리보기를 반환."""
    global _current_volume, _current_spacing

    if file_obj is None:
        return None, "파일을 업로드해주세요."

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
            return None, f"지원하지 않는 형식입니다: {os.path.basename(path)}\n(.nii.gz / .nii / .npy 만 가능)"

        views = get_slice_views(_current_volume)
        panel = make_panel(views)
        preview = Image.fromarray(panel)
        D, H, W = _current_volume.shape
        msg = f"로드 완료: {os.path.basename(path)} | 크기: {D}×{H}×{W} | 간격: {_current_spacing[0]:.2f}×{_current_spacing[1]:.2f}×{_current_spacing[2]:.2f} mm"
        return preview, msg

    except Exception as e:
        return None, f"파일 로드 실패: {e}"


# ── 추론 ──────────────────────────────────────────────────────────────────────
def run_inference(
    question_ko: str,
    slice_idx: int,
    alpha: float,
    wl: float,
    ww: float,
) -> tuple[Image.Image | None, str, str]:
    """
    Returns:
        (panel_image, volume_report, model_answer_ko)
    """
    global _current_volume, _current_spacing

    if _current_volume is None:
        return None, "", "먼저 CT 파일을 업로드해주세요."
    if not question_ko.strip():
        return None, "", "질문을 입력해주세요."

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return None, "", f"모델 로드 실패: {e}"

    try:
        result = pipeline.run(_current_volume, question_ko)
    except Exception as e:
        return None, "", f"추론 오류: {e}"

    mask = result["mask"]           # (D, H, W) bool
    organ_label = result["organ_label"] or ""
    answer_en = result["answer_en"]

    # 부피 계산
    stats = analyze_mask(mask, label=organ_label, voxel_spacing_mm=_current_spacing)
    volume_report = stats.summary_ko() if stats.voxel_count > 0 else "마스크가 감지되지 않았습니다."

    # 한국어 번역 응답
    answer_ko = _translator.translate_segmentation_result(
        organ_label, present=stats.voxel_count > 0, volume_ml=stats.volume_ml
    )

    # 시각화: 마스크가 있는 슬라이스 자동 선택
    D = _current_volume.shape[0]
    auto_idx = _best_slice_index(mask) if stats.voxel_count > 0 else D // 2
    chosen_idx = auto_idx if slice_idx == 0 else slice_idx

    label_names = {1: get_korean_term(organ_label)} if organ_label else None
    views = get_slice_views(
        _current_volume,
        mask.astype(np.uint8),
        slice_indices={"axial": chosen_idx, "sagittal": _current_volume.shape[2] // 2, "coronal": _current_volume.shape[1] // 2},
        alpha=alpha,
        wl=wl,
        ww=ww,
        label_names=label_names,
    )
    panel = make_panel(views)
    panel_img = Image.fromarray(panel)

    return panel_img, volume_report, answer_ko


def _best_slice_index(mask: np.ndarray) -> int:
    """마스크 면적이 가장 큰 축상 슬라이스 인덱스."""
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def update_preview(slice_idx: int, alpha: float, wl: float, ww: float) -> Image.Image | None:
    """슬라이더 변경 시 마스크 없이 미리보기 갱신."""
    if _current_volume is None:
        return None
    D = _current_volume.shape[0]
    idx = max(0, min(int(slice_idx), D - 1))
    views = get_slice_views(_current_volume, slice_indices={"axial": idx, "sagittal": _current_volume.shape[2] // 2, "coronal": _current_volume.shape[1] // 2}, wl=wl, ww=ww)
    return Image.fromarray(make_panel(views))


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
                slice_slider = gr.Slider(0, 31, value=0, step=1, label="축상 슬라이스 번호 (-1=자동)")
                alpha_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="마스크 불투명도")
                wl_slider = gr.Slider(-200, 400, value=40, step=10, label="윈도우 레벨 (HU)")
                ww_slider = gr.Slider(100, 2000, value=400, step=50, label="윈도우 너비 (HU)")

            with gr.Row():
                clear_btn = gr.ClearButton([file_input, question_input], value="초기화")
                run_btn = gr.Button("세그멘테이션 실행", variant="primary")

        # ── 오른쪽: 출력 패널 ─────────────────────────────────────────────
        with gr.Column(scale=2):
            preview_img = gr.Image(label="CT 슬라이스 (축상 / 시상 / 관상)", type="pil")
            answer_box = gr.Textbox(label="한국어 분석 결과", lines=3, interactive=False)
            volume_box = gr.Textbox(label="부피·크기 통계", lines=6, interactive=False)

    # ── 이벤트 연결 ───────────────────────────────────────────────────────────
    file_input.change(
        fn=load_file,
        inputs=[file_input],
        outputs=[preview_img, load_status],
    )

    run_btn.click(
        fn=run_inference,
        inputs=[question_input, slice_slider, alpha_slider, wl_slider, ww_slider],
        outputs=[preview_img, volume_box, answer_box],
    )

    for slider in [slice_slider, alpha_slider, wl_slider, ww_slider]:
        slider.change(
            fn=update_preview,
            inputs=[slice_slider, alpha_slider, wl_slider, ww_slider],
            outputs=[preview_img],
        )


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
