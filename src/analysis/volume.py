from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VolumeStats:
    label: str
    voxel_count: int
    volume_ml: float
    bbox_min: tuple[int, int, int]   # (d_min, h_min, w_min)
    bbox_max: tuple[int, int, int]   # (d_max, h_max, w_max)
    bbox_size_mm: tuple[float, float, float]  # (depth_mm, height_mm, width_mm)
    centroid_vox: tuple[float, float, float]  # (d, h, w) in voxel coords

    def summary_ko(self) -> str:
        from src.translation.medical_terms import get_korean_term
        ko = get_korean_term(self.label)
        lines = [
            f"[{ko}]",
            f"  추정 부피  : {self.volume_ml:.1f} mL",
            f"  복셀 수    : {self.voxel_count:,}",
            f"  크기 (mm)  : D={self.bbox_size_mm[0]:.1f}  H={self.bbox_size_mm[1]:.1f}  W={self.bbox_size_mm[2]:.1f}",
            f"  무게중심   : ({self.centroid_vox[0]:.1f}, {self.centroid_vox[1]:.1f}, {self.centroid_vox[2]:.1f})",
        ]
        return "\n".join(lines)


def compute_volume_ml(
    mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    바이너리 마스크의 부피를 mL 단위로 계산.

    Args:
        mask: bool 또는 0/1 배열 (D, H, W)
        voxel_spacing_mm: 각 축의 복셀 간격 (mm). 기본 1mm 등방성.

    Returns:
        부피 (mL). 1 mL = 1000 mm³
    """
    voxel_volume_mm3 = float(voxel_spacing_mm[0]) * float(voxel_spacing_mm[1]) * float(voxel_spacing_mm[2])
    return int(np.sum(mask > 0)) * voxel_volume_mm3 / 1000.0


def compute_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    마스크가 있는 영역의 축별 최소/최대 인덱스를 반환.

    Returns:
        (bbox_min, bbox_max) — 각각 (d, h, w) 정수 튜플.
        마스크가 비어 있으면 ((0,0,0), (0,0,0)) 반환.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 0), (0, 0, 0)
    mn = tuple(int(v) for v in coords.min(axis=0))
    mx = tuple(int(v) for v in coords.max(axis=0))
    return mn, mx  # type: ignore[return-value]


def compute_centroid(mask: np.ndarray) -> tuple[float, float, float]:
    """마스크 무게중심 (voxel 단위)."""
    coords = np.argwhere(mask > 0).astype(float)
    if len(coords) == 0:
        return (0.0, 0.0, 0.0)
    c = coords.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def analyze_mask(
    mask: np.ndarray,
    label: str = "",
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> VolumeStats:
    """
    세그멘테이션 마스크에서 부피·크기·위치 통계를 한 번에 계산.

    Args:
        mask:              bool (D, H, W)
        label:             영문 장기명 (번역용)
        voxel_spacing_mm:  (d_mm, h_mm, w_mm) 복셀 간격

    Returns:
        VolumeStats 객체
    """
    voxel_count = int(np.sum(mask > 0))
    volume_ml = compute_volume_ml(mask, voxel_spacing_mm)
    bbox_min, bbox_max = compute_bbox(mask)
    centroid = compute_centroid(mask)

    size_mm = (
        (bbox_max[0] - bbox_min[0]) * voxel_spacing_mm[0],
        (bbox_max[1] - bbox_min[1]) * voxel_spacing_mm[1],
        (bbox_max[2] - bbox_min[2]) * voxel_spacing_mm[2],
    )

    return VolumeStats(
        label=label,
        voxel_count=voxel_count,
        volume_ml=volume_ml,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_size_mm=size_mm,
        centroid_vox=centroid,
    )
