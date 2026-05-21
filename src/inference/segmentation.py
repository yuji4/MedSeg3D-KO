from __future__ import annotations

import re

import numpy as np
import torch
from monai.transforms import Resize

from src.inference.model_loader import ModelConfig, PROJ_OUT_NUM, load_model
from src.translation.medical_terms import TERM_KO

# 모델 입력 크기 고정값
TARGET_SHAPE = (32, 256, 256)

# 한국어 장기명 → 영문 (TERM_KO 역방향)
_KO_TO_EN: dict[str, str] = {v: k for k, v in TERM_KO.items()}

# 세그멘테이션 의도 패턴
_SEG_INTENT = re.compile(
    r"(분할|세그|마스크|segmen|segment|mask|찾아|표시|보여|어디|위치)",
    re.IGNORECASE,
)


def _detect_organs_en(question_ko: str) -> list[str]:
    """한국어 질문에서 장기 영문명을 모두 추출 (긴 이름 우선, 중복 제거)."""
    found: list[str] = []
    seen: set[str] = set()
    for ko, en in sorted(_KO_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
        if ko in question_ko and en not in seen:
            found.append(en)
            seen.add(en)
    return found


def _detect_organ_en(question_ko: str) -> str | None:
    """하위 호환: 첫 번째 장기만 반환."""
    organs = _detect_organs_en(question_ko)
    return organs[0] if organs else None


def _translate_ko_to_en(text: str) -> str:
    """deep-translator로 한국어 → 영어 번역. 실패 시 원문 반환."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="ko", target="en").translate(text)
    except Exception:
        return text


def _build_english_prompt(question_ko: str, organ_en: str | None = None) -> str:
    """
    한국어 질문을 M3D 모델이 이해하는 영문 프롬프트로 변환.

    Args:
        question_ko: 원문 한국어 질문 (의도 판단용)
        organ_en:    명시적 장기명. None이면 question_ko에서 자동 감지.
    """
    if organ_en is None:
        organ_en = _detect_organ_en(question_ko)
    is_seg = bool(_SEG_INTENT.search(question_ko))

    if organ_en and is_seg:
        return f"Can you segment the {organ_en} in this image? Please output the mask."
    if organ_en:
        return f"What is {organ_en} in this image? Please output the segmentation mask."

    # 폴백: 번역 후 직접 전달
    translated = _translate_ko_to_en(question_ko)
    return translated


def preprocess_volume(
    image_np: np.ndarray,
    target_shape: tuple[int, int, int] = TARGET_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NIfTI/NumPy 볼륨을 모델 입력 형식으로 전처리.

    Args:
        image_np: (D, H, W) 또는 (1, D, H, W) 배열
        target_shape: 리사이즈 목표 (D, H, W)

    Returns:
        (preprocessed_np, original_np)
        preprocessed_np: float32 (1, D, H, W), 값 범위 [0, 1]
    """
    if image_np.ndim == 4:
        arr = image_np  # (C, D, H, W)
    else:
        arr = image_np[np.newaxis]  # (1, D, H, W)

    # 정규화: HU → [0, 1] (전체 범위 min-max)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = arr.astype(np.float32)

    resize = Resize(spatial_size=target_shape, mode="bilinear")
    import torch as _torch
    tensor = _torch.from_numpy(arr)
    resized = resize(tensor).numpy()  # (1, D, H, W)

    return resized, image_np


class SegmentationPipeline:
    """
    M3D-LaMed 기반 3D 세그멘테이션 추론 파이프라인.

    Usage:
        pipeline = SegmentationPipeline()
        pipeline.load()
        result = pipeline.run(image_np, "간을 분할해줘")
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self._device: torch.device | None = None

    def load(self) -> None:
        """모델과 토크나이저를 로드. 처음 한 번만 호출."""
        self.model, self.tokenizer = load_model(self.config)
        self._device = next(self.model.parameters()).device

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _prepare_image_pt(
        self, image_np: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        """전처리 + GPU 텐서 변환. 여러 장기 추론 시 한 번만 호출하도록 분리."""
        preprocessed, original = preprocess_volume(image_np)
        dtype = next(self.model.parameters()).dtype
        image_pt = (
            torch.from_numpy(preprocessed)
            .unsqueeze(0)
            .to(dtype=dtype, device=self._device)
        )
        return image_pt, original

    def _infer(
        self,
        image_pt: torch.Tensor,
        original: np.ndarray,
        organ_en: str,
        question_ko: str = "",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> dict:
        """전처리된 텐서로 단일 장기 추론. run / run_single 공통 내부 로직."""
        question_en = _build_english_prompt(question_ko, organ_en=organ_en)
        prompt = "<im_patch>" * PROJ_OUT_NUM + question_en
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device)

        with torch.no_grad():
            generation, seg_logit = self.model.generate(
                image_pt,
                input_ids,
                seg_enable=True,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )

        answer_en = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
        mask_model = (torch.sigmoid(seg_logit) > 0.5).squeeze().cpu().numpy().astype(bool)
        orig_shape = original.shape[-3:] if original.ndim == 4 else original.shape
        mask = self._resize_mask(mask_model, orig_shape)

        return {
            "question_en": question_en,
            "answer_en": answer_en,
            "organ_label": organ_en,
            "mask": mask,
            "mask_model": mask_model,
        }

    def run_single(
        self,
        image_np: np.ndarray,
        organ_en: str,
        question_ko: str = "",
        **kwargs,
    ) -> dict:
        """
        단일 장기명을 명시하여 추론.
        여러 장기를 순차적으로 처리할 때 gradio_app에서 호출.
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. pipeline.load()를 먼저 호출하세요.")
        image_pt, original = self._prepare_image_pt(image_np)
        return self._infer(image_pt, original, organ_en, question_ko, **kwargs)

    def run(
        self,
        image_np: np.ndarray,
        question_ko: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> dict:
        """
        CT 볼륨과 한국어 질문을 받아 세그멘테이션 수행 (단일 장기 / 폴백).

        Returns:
            {
                "question_en": str,
                "answer_en": str,
                "organ_label": str | None,
                "mask": np.ndarray,       # bool (D, H, W)
                "mask_model": np.ndarray, # bool (32, 256, 256)
            }
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. pipeline.load()를 먼저 호출하세요.")

        organ_en = _detect_organ_en(question_ko)
        image_pt, original = self._prepare_image_pt(image_np)

        if organ_en:
            return self._infer(image_pt, original, organ_en, question_ko,
                               max_new_tokens=max_new_tokens, do_sample=do_sample,
                               top_p=top_p, temperature=temperature)

        # 장기 미탐지 시 번역 폴백
        question_en = _build_english_prompt(question_ko)
        prompt = "<im_patch>" * PROJ_OUT_NUM + question_en
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device)
        with torch.no_grad():
            generation, seg_logit = self.model.generate(
                image_pt, input_ids, seg_enable=True,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                top_p=top_p, temperature=temperature,
            )
        answer_en = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
        mask_model = (torch.sigmoid(seg_logit) > 0.5).squeeze().cpu().numpy().astype(bool)
        orig_shape = original.shape[-3:] if original.ndim == 4 else original.shape
        mask = self._resize_mask(mask_model, orig_shape)
        return {"question_en": question_en, "answer_en": answer_en,
                "organ_label": None, "mask": mask, "mask_model": mask_model}

    @staticmethod
    def _resize_mask(
        mask: np.ndarray,
        target_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """bool 마스크를 target_shape (D, H, W) 로 nearest 보간하여 복원."""
        if mask.shape == target_shape:
            return mask

        try:
            resize = Resize(spatial_size=target_shape, mode="nearest")
            t = torch.from_numpy(mask[np.newaxis].astype(np.float32))
            resized = resize(t).numpy()[0].astype(bool)
            return resized
        except Exception:
            # MONAI 없을 때 단순 반복으로 근사
            return mask
