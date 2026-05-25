CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    sex        TEXT    DEFAULT 'unknown',
    birth_year INTEGER,
    rrn        TEXT    UNIQUE,
    created_at TEXT    DEFAULT (datetime('now', 'localtime'))
)"""

CREATE_EXAMS = """
CREATE TABLE IF NOT EXISTS exams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL REFERENCES patients(id),
    exam_date   TEXT,
    age_at_exam INTEGER,
    notes       TEXT,
    analyzed_at TEXT DEFAULT (datetime('now', 'localtime'))
)"""

CREATE_ORGAN_RESULTS = """
CREATE TABLE IF NOT EXISTS organ_results (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id   INTEGER NOT NULL REFERENCES exams(id),
    label     TEXT    NOT NULL,
    volume_ml REAL,
    status    TEXT,
    range_lo  REAL,
    range_hi  REAL,
    note      TEXT
)"""

ALL_TABLES = [CREATE_PATIENTS, CREATE_EXAMS, CREATE_ORGAN_RESULTS]
