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


# 성인 기준 장기 정상 부피 범위 (mL) — 기준: 18-65세 평균 성인
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


# ── 나이·성별 보정 ────────────────────────────────────────────────────────────

# 나이별 성인 대비 스케일 팩터 (단순 추정값 — BSA 비례)
# (상한 나이, 팩터) 리스트: 해당 나이 미만이면 적용
_AGE_FACTORS: list[tuple[int, float]] = [
    (1,   0.15),   # 영아 (0-1세)
    (5,   0.30),   # 유아 (1-5세)
    (10,  0.50),   # 초등 (5-10세)
    (15,  0.70),   # 중등 (10-15세)
    (18,  0.85),   # 고등 (15-18세)
    (65,  1.00),   # 성인 (18-65세)
    (999, 0.90),   # 노년 (65세+, 장기 위축 반영)
]

# 성별별 조정 계수 (장기명 → 팩터)
_SEX_FACTORS: dict[str, dict[str, float]] = {
    "male": {
        "liver": 1.10, "heart": 1.10,
        "left lung": 1.10, "right lung": 1.10, "lungs": 1.10,
        "kidney": 1.05, "right kidney": 1.05, "left kidney": 1.05,
        "kidney left": 1.05, "kidney right": 1.05, "kidneys": 1.05,
        "spleen": 1.05, "brain": 1.05,
    },
    "female": {
        "liver": 0.92, "heart": 0.90,
        "left lung": 0.88, "right lung": 0.88, "lungs": 0.88,
        "kidney": 0.95, "right kidney": 0.95, "left kidney": 0.95,
        "kidney left": 0.95, "kidney right": 0.95, "kidneys": 0.95,
        "spleen": 0.95, "brain": 0.97,
    },
}

# 성별 전용 장기 (반대 성별이면 "해당없음" 처리)
_MALE_ONLY: set[str] = {"prostate or uterus"}
_FEMALE_ONLY: set[str] = {"uterus"}


def _age_factor(age: int) -> float:
    for threshold, factor in _AGE_FACTORS:
        if age < threshold:
            return factor
    return 0.90


def get_normal_range(organ: str, age: int = 30, sex: str = "unknown") -> NormalRange:
    """나이·성별을 반영한 정상 범위 반환."""
    key = organ.strip().lower()
    base = NORMAL_RANGES.get(key)
    if base is None:
        return NormalRange(None, None)
    if base.lo is None and base.hi is None:
        return base

    # 성별 전용 장기 처리
    if sex == "male" and key in _FEMALE_ONLY:
        return NormalRange(None, None, note="해당없음 (남성)")
    if sex == "female" and key in _MALE_ONLY:
        return NormalRange(None, None, note="해당없음 (여성)")

    af = _age_factor(age)
    sf = _SEX_FACTORS.get(sex, {}).get(key, 1.0)
    factor = af * sf

    lo = base.lo * factor if base.lo is not None else None
    hi = base.hi * factor if base.hi is not None else None

    note_parts = [base.note] if base.note else []
    if af != 1.0:
        group = (
            "영아" if age < 1 else
            "유아" if age < 5 else
            "초등" if age < 10 else
            "청소년" if age < 18 else
            "노년"
        )
        note_parts.append(f"{group} 기준 보정")
    if sf != 1.0:
        note_parts.append("성별 보정")

    return NormalRange(lo, hi, note=", ".join(note_parts))


def assess_organ(
    label: str,
    volume_ml: float,
    age: int = 30,
    sex: str = "unknown",
) -> ClinicalAssessment:
    """
    장기 부피를 나이·성별을 반영한 정상 범위와 비교하여 임상 평가를 반환.

    Args:
        label:     영문 장기명
        volume_ml: 측정 부피 (mL)
        age:       환자 나이 (세)
        sex:       "male" | "female" | "unknown"
    """
    key = label.strip().lower()
    nr = get_normal_range(key, age, sex)
    ko_name = get_korean_term(label)

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


def format_clinical_summary(
    assessments: list[ClinicalAssessment],
    age: int = 30,
    sex: str = "unknown",
) -> str:
    """임상 평가 목록을 Textbox 표시용 문자열로 변환."""
    if not assessments:
        return "임상 평가 결과 없음"

    sex_ko = {"male": "남성", "female": "여성"}.get(sex, "미입력")
    age_group = (
        "영아" if age < 1 else
        "유아" if age < 5 else
        "초등" if age < 10 else
        "청소년" if age < 18 else
        "노년" if age >= 65 else
        "성인"
    )

    abnormal = [a for a in assessments if a.status in ("high", "low")]
    normal   = [a for a in assessments if a.status == "normal"]
    unknown  = [a for a in assessments if a.status == "unknown"]

    lines = [
        f"=== 임상 평가 — {age}세 ({age_group}) / {sex_ko} ===",
        "",
    ]

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
