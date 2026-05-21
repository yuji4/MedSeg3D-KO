from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 장기별 고정 색상표 (RGB). 인덱스 0은 배경(투명).
_LABEL_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0 배경
    (255, 80, 80),    # 1 빨강
    (80, 200, 80),    # 2 초록
    (80, 120, 255),   # 3 파랑
    (255, 200, 60),   # 4 노랑
    (200, 80, 255),   # 5 보라
    (60, 220, 220),   # 6 청록
    (255, 140, 60),   # 7 주황
    (180, 255, 100),  # 8 연두
    (255, 100, 200),  # 9 분홍
    (100, 200, 255),  # 10 하늘
    (255, 230, 130),  # 11 황금
    (160, 100, 255),  # 12 라벤더
    (80, 255, 180),   # 13 민트
    (255, 160, 160),  # 14 살구
    (140, 180, 255),  # 15 연파랑
]


def _normalize_slice(arr: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """CT HU값을 윈도우 레벨로 정규화하여 0-255 uint8 반환."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def _label_to_color(label_idx: int) -> tuple[int, int, int]:
    return _LABEL_COLORS[label_idx % len(_LABEL_COLORS)]


def overlay_mask_on_slice(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.4,
    wl: float = 40.0,
    ww: float = 400.0,
    label_names: dict[int, str] | None = None,
) -> np.ndarray:
    """
    CT 슬라이스 위에 세그멘테이션 마스크를 오버레이.

    Args:
        ct_slice:    2D 배열 (H, W) — CT 값 (HU 또는 정규화된 값)
        mask_slice:  2D 정수 배열 (H, W) — 레이블 인덱스 (0=배경)
        alpha:       마스크 불투명도 (0.0~1.0)
        wl:          CT 윈도우 레벨 (HU 단위)
        ww:          CT 윈도우 너비 (HU 단위)
        label_names: {레이블 인덱스: 한국어 이름} 매핑 (범례용)

    Returns:
        RGB uint8 배열 (H, W, 3)
    """
    gray = _normalize_slice(ct_slice, wl, ww)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    unique_labels = np.unique(mask_slice)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        color = np.array(_label_to_color(lbl), dtype=np.float32)
        region = mask_slice == lbl
        rgb[region] = rgb[region] * (1 - alpha) + color * alpha

    result = np.clip(rgb, 0, 255).astype(np.uint8)

    if label_names:
        img = Image.fromarray(result)
        draw = ImageDraw.Draw(img)
        _draw_legend(draw, unique_labels, label_names, img.size)
        result = np.array(img)

    return result


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    unique_labels: np.ndarray,
    label_names: dict[int, str],
    img_size: tuple[int, int],
) -> None:
    """이미지 우측 상단에 범례 삽입."""
    try:
        font = ImageFont.truetype("malgun.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    x, y = img_size[0] - 160, 8
    box_size = 12
    gap = 4

    for lbl in unique_labels:
        if lbl == 0:
            continue
        name = label_names.get(int(lbl), f"레이블 {lbl}")
        color = _label_to_color(int(lbl))
        draw.rectangle([x, y, x + box_size, y + box_size], fill=color)
        draw.text((x + box_size + gap, y), name, fill=(255, 255, 255), font=font)
        y += box_size + gap + 2


def get_slice_views(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    slice_indices: dict[str, int] | None = None,
    alpha: float = 0.4,
    wl: float = 40.0,
    ww: float = 400.0,
    label_names: dict[int, str] | None = None,
) -> dict[str, np.ndarray]:
    """
    3D 볼륨에서 축상(Axial), 시상(Sagittal), 관상(Coronal) 슬라이스를 추출하고
    선택적으로 마스크를 오버레이하여 반환.

    Args:
        volume:        3D 배열 (D, H, W)
        mask:          3D 정수 배열 (D, H, W) — None이면 마스크 없이 반환
        slice_indices: {"axial": int, "sagittal": int, "coronal": int} — 없으면 중앙 슬라이스
        alpha:         마스크 불투명도
        wl, ww:        CT 윈도우 파라미터
        label_names:   범례용 레이블→한국어 이름 매핑

    Returns:
        {"axial": ndarray(H,W,3), "sagittal": ndarray(D,W,3), "coronal": ndarray(D,H,3)}
    """
    D, H, W = volume.shape
    if slice_indices is None:
        slice_indices = {"axial": D // 2, "sagittal": W // 2, "coronal": H // 2}

    def _get_mask_slice(plane: str, idx: int) -> np.ndarray | None:
        if mask is None:
            return None
        if plane == "axial":
            return mask[idx, :, :]
        if plane == "sagittal":
            return mask[:, :, idx]
        return mask[:, idx, :]

    def _get_ct_slice(plane: str, idx: int) -> np.ndarray:
        if plane == "axial":
            return volume[idx, :, :]
        if plane == "sagittal":
            return volume[:, :, idx]
        return volume[:, idx, :]

    views: dict[str, np.ndarray] = {}
    for plane, idx in slice_indices.items():
        ct_sl = _get_ct_slice(plane, idx)
        msk_sl = _get_mask_slice(plane, idx)

        if msk_sl is not None:
            views[plane] = overlay_mask_on_slice(
                ct_sl, msk_sl, alpha=alpha, wl=wl, ww=ww, label_names=label_names
            )
        else:
            gray = _normalize_slice(ct_sl, wl, ww)
            views[plane] = np.stack([gray, gray, gray], axis=-1)

    return views


def make_panel(
    views: dict[str, np.ndarray],
    titles: dict[str, str] | None = None,
    padding: int = 4,
) -> np.ndarray:
    """
    axial / sagittal / coronal 뷰를 가로로 이어붙인 패널 이미지 반환.

    Args:
        views:   get_slice_views() 결과
        titles:  {"axial": "축상면", "sagittal": "시상면", "coronal": "관상면"}
        padding: 슬라이스 간 여백(픽셀)

    Returns:
        RGB uint8 배열
    """
    if titles is None:
        titles = {"axial": "축상면", "sagittal": "시상면", "coronal": "관상면"}

    order = ["axial", "sagittal", "coronal"]
    imgs = [views[p] for p in order if p in views]

    if not imgs:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    max_h = max(im.shape[0] for im in imgs)
    padded: list[np.ndarray] = []
    for im in imgs:
        h, w = im.shape[:2]
        canvas = np.zeros((max_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = im
        padded.append(canvas)

    sep = np.zeros((max_h, padding, 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, im in enumerate(padded):
        parts.append(im)
        if i < len(padded) - 1:
            parts.append(sep)
    panel = np.concatenate(parts, axis=1)

    # 제목 텍스트 추가
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("malgun.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    x_offset = 0
    for i, plane in enumerate([p for p in order if p in views]):
        w = views[plane].shape[1]
        label = titles.get(plane, plane)
        draw.text((x_offset + 4, 2), label, fill=(255, 255, 0), font=font)
        x_offset += w + padding

    return np.array(pil)
