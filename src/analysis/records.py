from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _ensure_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def save_record(
    patient_name: str,
    patient_id: str,
    exam_date: str,
    age: int,
    sex: str,
    organ_results: list[dict],
) -> str:
    """분석 결과를 results/ 폴더에 JSON으로 저장. 파일 경로 반환."""
    _ensure_dir()
    now = datetime.now()
    safe = (patient_name.strip() or "unnamed").replace(" ", "_")
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe}.json"
    record = {
        "patient_name": patient_name.strip(),
        "patient_id": (patient_id or "").strip(),
        "exam_date": exam_date.strip() or now.strftime("%Y-%m-%d"),
        "age": age,
        "sex": sex,
        "analyzed_at": now.isoformat(),
        "organs": [
            {
                "label": r["label"],
                "volume_ml": round(r["stats"].volume_ml, 2),
                "status": r["assessment"].status,
                "range_lo": r["assessment"].normal_range.lo,
                "range_hi": r["assessment"].normal_range.hi,
                "note": r["assessment"].normal_range.note,
            }
            for r in organ_results
        ],
    }
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return str(path)


def list_records() -> list[dict]:
    """results/ 폴더의 JSON 레코드를 최신순으로 반환."""
    _ensure_dir()
    records = []
    for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            rec["_filename"] = p.name
            records.append(rec)
        except Exception:
            continue
    return records


def export_records_csv(records: list[dict]) -> str:
    """레코드 목록을 CSV 임시파일로 저장하고 경로 반환."""
    if not records:
        return ""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, prefix="medseg_records_",
        mode="w", encoding="utf-8-sig", newline="",
    )
    writer = csv.writer(tmp)
    writer.writerow([
        "환자명", "환자ID", "검사일", "나이", "성별", "분석일시",
        "장기", "부피(mL)", "상태", "정상범위(mL)", "비고",
    ])
    for rec in records:
        base = [
            rec.get("patient_name", ""),
            rec.get("patient_id", ""),
            rec.get("exam_date", ""),
            rec.get("age", ""),
            rec.get("sex", ""),
            rec.get("analyzed_at", ""),
        ]
        for organ in rec.get("organs", []):
            lo = organ.get("range_lo")
            hi = organ.get("range_hi")
            range_str = f"{lo:.0f}~{hi:.0f}" if (lo is not None and hi is not None) else "-"
            writer.writerow(base + [
                organ.get("label", ""),
                organ.get("volume_ml", ""),
                organ.get("status", ""),
                range_str,
                organ.get("note", ""),
            ])
    tmp.close()
    return tmp.name
