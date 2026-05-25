from __future__ import annotations

import csv
import tempfile

from src.database.db import get_conn


# ── 환자 ──────────────────────────────────────────────────────────────────────

def upsert_patient(name: str, sex: str, birth_year: int | None = None) -> int:
    """같은 이름의 환자가 있으면 ID 반환, 없으면 새로 생성."""
    name = name.strip() or "미입력"
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM patients WHERE name = ?", (name,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE patients SET sex = ? WHERE id = ?",
                (sex, row["id"]),
            )
            conn.commit()
            return row["id"]
        cur = conn.execute(
            "INSERT INTO patients (name, sex, birth_year) VALUES (?, ?, ?)",
            (name, sex, birth_year),
        )
        conn.commit()
        return cur.lastrowid


def list_patients() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.id, p.name, p.sex, p.birth_year,
                      COUNT(e.id) as n_exams
               FROM patients p
               LEFT JOIN exams e ON e.patient_id = p.id
               GROUP BY p.id
               ORDER BY p.name"""
        ).fetchall()
        return [dict(r) for r in rows]


# ── 검사 ──────────────────────────────────────────────────────────────────────

def save_exam(
    patient_id: int,
    exam_date: str,
    age: int,
    organ_results: list[dict],
    notes: str = "",
) -> int:
    """검사 + 장기 결과 저장, exam_id 반환."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO exams (patient_id, exam_date, age_at_exam, notes) VALUES (?, ?, ?, ?)",
            (patient_id, exam_date, age, notes.strip()),
        )
        exam_id = cur.lastrowid
        for r in organ_results:
            nr = r["assessment"].normal_range
            conn.execute(
                """INSERT INTO organ_results
                   (exam_id, label, volume_ml, status, range_lo, range_hi, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    exam_id,
                    r["label"],
                    round(r["stats"].volume_ml, 2),
                    r["assessment"].status,
                    nr.lo,
                    nr.hi,
                    nr.note,
                ),
            )
        conn.commit()
        return exam_id


def list_all_exams() -> list[dict]:
    """모든 검사 목록 (최신순)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.id, p.name AS patient_name, p.sex,
                      e.exam_date, e.age_at_exam, e.analyzed_at,
                      COUNT(o.id) AS n_organs,
                      SUM(CASE WHEN o.status IN ('high','low') THEN 1 ELSE 0 END) AS n_abnormal
               FROM exams e
               JOIN patients p ON p.id = e.patient_id
               LEFT JOIN organ_results o ON o.exam_id = e.id
               GROUP BY e.id
               ORDER BY e.analyzed_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_patient_exams(patient_id: int) -> list[dict]:
    """환자의 검사 이력 (최신순)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.id, e.exam_date, e.age_at_exam, e.analyzed_at,
                      COUNT(o.id) AS n_organs,
                      SUM(CASE WHEN o.status IN ('high','low') THEN 1 ELSE 0 END) AS n_abnormal
               FROM exams e
               LEFT JOIN organ_results o ON o.exam_id = e.id
               WHERE e.patient_id = ?
               GROUP BY e.id
               ORDER BY e.analyzed_at DESC""",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_exam_organs(exam_id: int) -> list[dict]:
    """특정 검사의 장기별 결과."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM organ_results WHERE exam_id = ? ORDER BY label",
            (exam_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── 종단적 추이 ───────────────────────────────────────────────────────────────

def get_organ_trend(patient_id: int, label: str) -> list[dict]:
    """특정 환자의 특정 장기 부피 추이 (날짜순)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.exam_date, e.analyzed_at, o.volume_ml, o.status,
                      o.range_lo, o.range_hi
               FROM organ_results o
               JOIN exams e ON e.id = o.exam_id
               WHERE e.patient_id = ? AND o.label = ?
               ORDER BY e.analyzed_at""",
            (patient_id, label),
        ).fetchall()
        return [dict(r) for r in rows]


def get_patient_organ_labels(patient_id: int) -> list[str]:
    """환자가 검사받은 장기 레이블 목록."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT o.label
               FROM organ_results o
               JOIN exams e ON e.id = o.exam_id
               WHERE e.patient_id = ?
               ORDER BY o.label""",
            (patient_id,),
        ).fetchall()
        return [r["label"] for r in rows]


# ── CSV 내보내기 ──────────────────────────────────────────────────────────────

def export_all_csv() -> str:
    """전체 검사 결과를 CSV 임시파일로 저장하고 경로 반환."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.name, p.sex, e.exam_date, e.age_at_exam, e.analyzed_at,
                      o.label, o.volume_ml, o.status, o.range_lo, o.range_hi, o.note
               FROM organ_results o
               JOIN exams e ON e.id = o.exam_id
               JOIN patients p ON p.id = e.patient_id
               ORDER BY e.analyzed_at DESC, p.name, o.label"""
        ).fetchall()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, prefix="medseg_all_",
        mode="w", encoding="utf-8-sig", newline="",
    )
    writer = csv.writer(tmp)
    writer.writerow([
        "환자명", "성별", "검사일", "나이", "분석일시",
        "장기", "부피(mL)", "상태", "정상범위_하한(mL)", "정상범위_상한(mL)", "비고",
    ])
    for r in rows:
        writer.writerow([
            r["name"], r["sex"], r["exam_date"], r["age_at_exam"],
            r["analyzed_at"], r["label"], r["volume_ml"],
            r["status"], r["range_lo"], r["range_hi"], r["note"],
        ])
    tmp.close()
    return tmp.name


def export_patient_csv(patient_id: int) -> str:
    """특정 환자의 결과만 CSV로 내보내기."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.name, p.sex, e.exam_date, e.age_at_exam, e.analyzed_at,
                      o.label, o.volume_ml, o.status, o.range_lo, o.range_hi, o.note
               FROM organ_results o
               JOIN exams e ON e.id = o.exam_id
               JOIN patients p ON p.id = e.patient_id
               WHERE e.patient_id = ?
               ORDER BY e.analyzed_at, o.label""",
            (patient_id,),
        ).fetchall()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, prefix="medseg_patient_",
        mode="w", encoding="utf-8-sig", newline="",
    )
    writer = csv.writer(tmp)
    writer.writerow([
        "환자명", "성별", "검사일", "나이", "분석일시",
        "장기", "부피(mL)", "상태", "정상범위_하한(mL)", "정상범위_상한(mL)", "비고",
    ])
    for r in rows:
        writer.writerow([
            r["name"], r["sex"], r["exam_date"], r["age_at_exam"],
            r["analyzed_at"], r["label"], r["volume_ml"],
            r["status"], r["range_lo"], r["range_hi"], r["note"],
        ])
    tmp.close()
    return tmp.name
