from __future__ import annotations

from dataclasses import dataclass

from src.translation.medical_terms import get_korean_term


@dataclass
class NormalRange:
    lo: float | None  # mL
    hi: float | None  # mL
    note: str = ""


@dataclass
class ClinicalAssessment:
    label: str
    volume_ml: float
    status: str          # "normal" | "high" | "low" | "unknown"
    message_ko: str
    normal_range: NormalRange


# 성인 기준 장기 정상 부피 범위 (mL)
# 출처: 일반 해부학·영상의학 교과서 참고값
NORMAL_RANGES: dict[str, NormalRange] = {
    # 복부 장기
    "liver":                    NormalRange(1000, 1500),
    "spleen":                   NormalRange(100,  314),
    "pancreas":                 NormalRange(50,   100),
    "gallbladder":              NormalRange(20,   70,  note="충만 시"),
    "gall bladder":             NormalRange(20,   70,  note="충만 시"),
    "stomach":                  NormalRange(None, None, note="상태에 따라 가변"),
    "duodenum":                 NormalRange(None, None, note="부피 기준 없음"),
    "esophagus":                NormalRange(None, None, note="부피 기준 없음"),
    "colon":                    NormalRange(None, None, note="부피 기준 없음"),
    "small intestin":           NormalRange(None, None, note="부피 기준 없음"),
    "biliary system":           NormalRange(None, None, note="부피 기준 없음"),

    # 신장 / 부신
    "kidney":                   NormalRange(90,  160),
    "right kidney":             NormalRange(90,  160),
    "left kidney":              NormalRange(90,  160),
    "kidney left":              NormalRange(90,  160),
    "kidney right":             NormalRange(90,  160),
    "kidneys":                  NormalRange(180, 320,  note="양측 합산"),
    "kidney tumor":             NormalRange(None, None, note="종양 부피 기준 없음"),
    "kidney cyst":              NormalRange(None, None, note="낭종 부피 기준 없음"),
    "right adrenal gland":      NormalRange(2,   8),
    "left adrenal gland":       NormalRange(2,   8),
    "adrenal gland left":       NormalRange(2,   8),
    "adrenal gland right":      NormalRange(2,   8),
    "suprarenal gland":         NormalRange(2,   8),

    # 비뇨생식기
    "bladder":                  NormalRange(300, 500,  note="충만 시"),
    "prostate or uterus":       NormalRange(15,  30,   note="전립선 기준"),
    "uterus":                   NormalRange(40,  120,  note="가임기 여성 기준"),

    # 혈관 (부피 기준 없음)
    "aorta":                    NormalRange(None, None, note="부피 기준 없음"),
    "inferior vena cava":       NormalRange(None, None, note="부피 기준 없음"),
    "portal vein":              NormalRange(None, None, note="부피 기준 없음"),
    "portal vein and splenic vein": NormalRange(None, None, note="부피 기준 없음"),
    "pulmonary artery":         NormalRange(None, None, note="부피 기준 없음"),
    "renal vein":               NormalRange(None, None, note="부피 기준 없음"),
    "renal artery":             NormalRange(None, None, note="부피 기준 없음"),
    "carotid artery left":      NormalRange(None, None, note="부피 기준 없음"),
    "carotid artery right":     NormalRange(None, None, note="부피 기준 없음"),
    "iliac artery left":        NormalRange(None, None, note="부피 기준 없음"),
    "iliac artery right":       NormalRange(None, None, note="부피 기준 없음"),

    # 폐 / 심장
    "left lung":                NormalRange(1500, 3000),
    "right lung":               NormalRange(1700, 3500),
    "lungs":                    NormalRange(3200, 6500, note="양측 합산"),
    "heart":                    NormalRange(400,  900),
    "heart atrium left":        NormalRange(None, None, note="부피 기준 없음"),
    "heart atrium right":       NormalRange(None, None, note="부피 기준 없음"),
    "heart myocardium":         NormalRange(None, None, note="부피 기준 없음"),
    "heart ventricle left":     NormalRange(None, None, note="부피 기준 없음"),
    "heart ventricle right":    NormalRange(None, None, note="부피 기준 없음"),

    # 두경부
    "brain":                    NormalRange(1100, 1500),
    "thyroid":                  NormalRange(10,  25),
    "spinal cord":              NormalRange(None, None, note="부피 기준 없음"),
    "pituitary gland":          NormalRange(0.3, 0.9,  note="단위: mL"),
    "brainstem":                NormalRange(None, None, note="부피 기준 없음"),

    # 종양 / 병변
    "tumor":                    NormalRange(None, None, note="종양 부피 기준 없음"),
    "liver tumor":              NormalRange(None, None, note="종양 부피 기준 없음"),
    "liver cyst":               NormalRange(None, None, note="낭종 부피 기준 없음"),
}


def assess_organ(label: str, volume_ml: float) -> ClinicalAssessment:
    """
    장기 부피를 정상 범위와 비교하여 임상 평가를 반환.

    Args:
        label:     영문 장기명
        volume_ml: 측정 부피 (mL)
    """
    key = label.strip().lower()
    nr = NORMAL_RANGES.get(key)
    ko_name = get_korean_term(label)

    # 참고 범위 없음
    if nr is None or (nr.lo is None and nr.hi is None):
        note_part = f" ({nr.note})" if (nr and nr.note) else ""
        return ClinicalAssessment(
            label=label,
            volume_ml=volume_ml,
            status="unknown",
            message_ko=f"— {ko_name}: {volume_ml:.1f} mL — 참고범위 없음{note_part}",
            normal_range=nr or NormalRange(None, None),
        )

    range_str = f"{nr.lo:.0f}~{nr.hi:.0f} mL"

    if nr.lo is not None and volume_ml < nr.lo:
        status = "low"
        msg = f"⚠️ {ko_name}: {volume_ml:.1f} mL ↓ 정상 미만 (정상: {range_str})"
    elif nr.hi is not None and volume_ml > nr.hi:
        status = "high"
        msg = f"⚠️ {ko_name}: {volume_ml:.1f} mL ↑ 정상 초과 (정상: {range_str})"
    else:
        status = "normal"
        msg = f"✅ {ko_name}: {volume_ml:.1f} mL — 정상 범위 (정상: {range_str})"

    if nr.note:
        msg += f" [{nr.note}]"

    return ClinicalAssessment(
        label=label,
        volume_ml=volume_ml,
        status=status,
        message_ko=msg,
        normal_range=nr,
    )


def format_clinical_summary(assessments: list[ClinicalAssessment]) -> str:
    """임상 평가 목록을 Textbox 표시용 문자열로 변환."""
    if not assessments:
        return "임상 평가 결과 없음"

    abnormal = [a for a in assessments if a.status in ("high", "low")]
    normal   = [a for a in assessments if a.status == "normal"]
    unknown  = [a for a in assessments if a.status == "unknown"]

    lines = ["=== 임상 평가 (성인 정상 범위 기준) ===", ""]

    if abnormal:
        lines.append("[ 이상 소견 ]")
        lines += [f"  {a.message_ko}" for a in abnormal]
        lines.append("")

    if normal:
        lines.append("[ 정상 소견 ]")
        lines += [f"  {a.message_ko}" for a in normal]
        lines.append("")

    if unknown:
        lines.append("[ 참고범위 없음 ]")
        lines += [f"  {a.message_ko}" for a in unknown]

    return "\n".join(lines)
