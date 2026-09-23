"""SQLite persistence layer for StudyMate AI."""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "studymate.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    explanation_style TEXT NOT NULL DEFAULT 'simple',
    depth_level TEXT NOT NULL DEFAULT 'standard'
);

CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto',   -- auto | manual
    summary TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'unseen'  -- unseen | learning | cleared | flagged
);

CREATE TABLE IF NOT EXISTS flashcards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL REFERENCES concepts(id),
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    box_level INTEGER NOT NULL DEFAULT 1,
    next_review_at TEXT,
    source TEXT NOT NULL DEFAULT 'generated',  -- generated | drill
    attempt INTEGER NOT NULL DEFAULT 0          -- remediation attempt number that created this card (0 = original)
);

CREATE TABLE IF NOT EXISTS concept_remediations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL REFERENCES concepts(id),
    attempt INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL REFERENCES concepts(id),
    question TEXT NOT NULL,
    options TEXT NOT NULL,              -- JSON list
    correct_index INTEGER NOT NULL,
    distractor_notes TEXT               -- JSON: misconception per wrong option
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES quiz_questions(id),
    concept_id INTEGER NOT NULL REFERENCES concepts(id),
    selected_index INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    attempted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pyq_solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    question TEXT NOT NULL,
    matched_concept TEXT,
    model_answer TEXT NOT NULL,
    expand_hints TEXT   -- JSON list
);
"""


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)
    conn.close()


def _migrate(conn):
    """Adds columns introduced after a dev database already existed, so an
    existing local studymate.db doesn't break when the schema grows."""
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
    if "explanation_style" not in existing_cols:
        conn.execute("ALTER TABLE documents ADD COLUMN explanation_style TEXT NOT NULL DEFAULT 'simple'")
    if "depth_level" not in existing_cols:
        conn.execute("ALTER TABLE documents ADD COLUMN depth_level TEXT NOT NULL DEFAULT 'standard'")

    flashcard_cols = {row["name"] for row in conn.execute("PRAGMA table_info(flashcards)")}
    if "source" not in flashcard_cols:
        conn.execute("ALTER TABLE flashcards ADD COLUMN source TEXT NOT NULL DEFAULT 'generated'")
    if "attempt" not in flashcard_cols:
        conn.execute("ALTER TABLE flashcards ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def now_iso():
    return datetime.utcnow().isoformat()


_EMPTY_SUMMARY = {
    "from_notes": "", "explained_further": "", "comparisons_and_limits": "", "analogy": "",
    "exam_answer_example": "", "expand_hints": [], "why_it_matters": "", "key_terms": [],
}


def parse_summary(raw: str) -> dict:
    """Concept summaries are stored as JSON (from_notes, explained_further, analogy,
    exam_answer_example, expand_hints, why_it_matters, key_terms). Falls back
    gracefully for empty/legacy values."""
    if not raw:
        return dict(_EMPTY_SUMMARY)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            merged = dict(_EMPTY_SUMMARY)
            merged.update(data)
            # legacy documents stored a flat "explanation" field -- surface it
            # as explained_further so old data doesn't just vanish
            if not merged["explained_further"] and data.get("explanation"):
                merged["explained_further"] = data["explanation"]
            return merged
    except (json.JSONDecodeError, TypeError):
        pass
    merged = dict(_EMPTY_SUMMARY)
    merged["explained_further"] = raw
    return merged


# ---- documents ----

def create_document(title: str, raw_text: str, explanation_style: str = "simple",
                     depth_level: str = "standard") -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO documents (title, raw_text, uploaded_at, explanation_style, depth_level) "
        "VALUES (?, ?, ?, ?, ?)",
        (title, raw_text, now_iso(), explanation_style, depth_level),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def list_documents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document(doc_id: int):
    conn = get_conn()
    concept_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM concepts WHERE document_id=?", (doc_id,)
    ).fetchall()]
    for cid in concept_ids:
        question_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM quiz_questions WHERE concept_id=?", (cid,)
        ).fetchall()]
        for qid in question_ids:
            conn.execute("DELETE FROM quiz_attempts WHERE question_id=?", (qid,))
        conn.execute("DELETE FROM quiz_questions WHERE concept_id=?", (cid,))
        conn.execute("DELETE FROM flashcards WHERE concept_id=?", (cid,))
        conn.execute("DELETE FROM concept_remediations WHERE concept_id=?", (cid,))
    conn.execute("DELETE FROM concepts WHERE document_id=?", (doc_id,))
    conn.execute("DELETE FROM pyq_solutions WHERE document_id=?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()


# ---- concepts ----

def create_concept(document_id: int, name: str, source: str = "auto", summary: str = "") -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO concepts (document_id, name, source, summary) VALUES (?, ?, ?, ?)",
        (document_id, name, source, summary),
    )
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def list_concepts(document_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM concepts WHERE document_id=? ORDER BY id", (document_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_concept(concept_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM concepts WHERE id=?", (concept_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_concept_summary(concept_id: int, summary: str):
    conn = get_conn()
    conn.execute("UPDATE concepts SET summary=? WHERE id=?", (summary, concept_id))
    conn.commit()
    conn.close()


def add_concept_remediation(concept_id: int, attempt: int, summary: str) -> int:
    """Stores a re-explanation from the adaptive loop SEPARATELY from the
    concept's original summary, instead of overwriting it -- so both the
    original explanation and each remediation attempt stay visible, distinctly
    labeled, rather than the remediation silently replacing what was there."""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO concept_remediations (concept_id, attempt, summary, created_at) VALUES (?, ?, ?, ?)",
        (concept_id, attempt, summary, now_iso()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def list_concept_remediations(concept_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM concept_remediations WHERE concept_id=? ORDER BY attempt", (concept_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_concept_status(concept_id: int, status: str, increment_attempt: bool = False):
    conn = get_conn()
    if increment_attempt:
        conn.execute(
            "UPDATE concepts SET status=?, attempts=attempts+1 WHERE id=?",
            (status, concept_id),
        )
    else:
        conn.execute("UPDATE concepts SET status=? WHERE id=?", (status, concept_id))
    conn.commit()
    conn.close()


# ---- flashcards ----

def add_flashcard(concept_id: int, front: str, back: str, source: str = "generated", attempt: int = 0) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO flashcards (concept_id, front, back, box_level, next_review_at, source, attempt) "
        "VALUES (?, ?, ?, 1, ?, ?, ?)",
        (concept_id, front, back, now_iso(), source, attempt),
    )
    conn.commit()
    fid = cur.lastrowid
    conn.close()
    return fid


def list_flashcards(document_id: int):
    conn = get_conn()
    rows = conn.execute(
        """SELECT f.* FROM flashcards f
           JOIN concepts c ON f.concept_id = c.id
           WHERE c.document_id = ? ORDER BY f.id""",
        (document_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


LEITNER_INTERVALS_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 14}


def reschedule_flashcards_for_concept(concept_id: int, correct: bool):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, box_level FROM flashcards WHERE concept_id=?", (concept_id,)
    ).fetchall()
    for r in rows:
        box = r["box_level"]
        if correct:
            box = min(box + 1, 5)
        else:
            box = 1
        days = LEITNER_INTERVALS_DAYS.get(box, 0)
        next_at = (datetime.utcnow() + timedelta(days=days)).isoformat()
        conn.execute(
            "UPDATE flashcards SET box_level=?, next_review_at=? WHERE id=?",
            (box, next_at, r["id"]),
        )
    conn.commit()
    conn.close()


# ---- quiz ----

def add_quiz_question(concept_id: int, question: str, options: list, correct_index: int,
                       distractor_notes: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO quiz_questions (concept_id, question, options, correct_index, distractor_notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (concept_id, question, json.dumps(options), correct_index, json.dumps(distractor_notes)),
    )
    conn.commit()
    qid = cur.lastrowid
    conn.close()
    return qid


def list_quiz_questions(document_id: int, concept_id: int = None):
    conn = get_conn()
    if concept_id:
        rows = conn.execute(
            "SELECT * FROM quiz_questions WHERE concept_id=? ORDER BY id", (concept_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT q.* FROM quiz_questions q
               JOIN concepts c ON q.concept_id = c.id
               WHERE c.document_id = ? ORDER BY q.id""",
            (document_id,),
        ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        options = json.loads(d["options"])
        d["options"] = options if isinstance(options, list) else []
        notes = json.loads(d["distractor_notes"]) if d["distractor_notes"] else {}
        d["distractor_notes"] = notes if isinstance(notes, dict) else {}
        out.append(d)
    return out


def record_quiz_attempt(question_id: int, concept_id: int, selected_index: int, correct: bool):
    conn = get_conn()
    conn.execute(
        "INSERT INTO quiz_attempts (question_id, concept_id, selected_index, correct, attempted_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (question_id, concept_id, selected_index, int(correct), now_iso()),
    )
    conn.commit()
    conn.close()


# ---- PYQ (previous year question paper) solutions ----

def add_pyq_solution(document_id: int, question: str, matched_concept: str,
                      model_answer: str, expand_hints: list) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO pyq_solutions (document_id, question, matched_concept, model_answer, expand_hints) "
        "VALUES (?, ?, ?, ?, ?)",
        (document_id, question, matched_concept, model_answer, json.dumps(expand_hints)),
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def list_pyq_solutions(document_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM pyq_solutions WHERE document_id=? ORDER BY id", (document_id,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["expand_hints"] = json.loads(d["expand_hints"]) if d["expand_hints"] else []
        out.append(d)
    return out


def update_pyq_solution(solution_id: int, model_answer: str, expand_hints: list):
    conn = get_conn()
    conn.execute(
        "UPDATE pyq_solutions SET model_answer=?, expand_hints=? WHERE id=?",
        (model_answer, json.dumps(expand_hints), solution_id),
    )
    conn.commit()
    conn.close()
