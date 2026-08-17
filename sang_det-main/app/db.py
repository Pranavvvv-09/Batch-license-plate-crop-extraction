"""Job store supporting SQLite and Supabase (PostgreSQL).

Holds the video queue, per-video progress, saved-crop provenance and an event
log. When configured with DATABASE_URL or Supabase PostgreSQL, all state lives
in the cloud so runs continue uninterrupted when local machines disconnect.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from typing import Any, Iterable

from .config import ROOT, load as load_config
from .logging_setup import get

log = get("db")

DB_PATH = ROOT / "data" / "sang_det.db"

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

_local = threading.local()
_write_lock = threading.Lock()

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    url               TEXT NOT NULL,
    title             TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    error             TEXT,
    frames_processed  INTEGER NOT NULL DEFAULT 0,
    plates_saved      INTEGER NOT NULL DEFAULT 0,
    detections_seen   INTEGER NOT NULL DEFAULT 0,
    rejected_conf     INTEGER NOT NULL DEFAULT 0,
    rejected_size     INTEGER NOT NULL DEFAULT 0,
    rejected_blur     INTEGER NOT NULL DEFAULT 0,
    rejected_dupe     INTEGER NOT NULL DEFAULT 0,
    duration_s        REAL,
    position_s        REAL NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 0,
    added_at          REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL,
    heartbeat_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);

CREATE TABLE IF NOT EXISTS plates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     INTEGER NOT NULL,
    filename     TEXT NOT NULL,
    frame_index  INTEGER NOT NULL,
    timestamp_s  REAL NOT NULL,
    confidence   REAL NOT NULL,
    box          TEXT NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    blur_score   REAL,
    phash        TEXT,
    bytes        INTEGER,
    saved_at     REAL NOT NULL,
    plate_text   TEXT,
    ocr_status   TEXT NOT NULL DEFAULT 'unlabeled',
    labeled_at   REAL,
    storage_url  TEXT,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);
CREATE INDEX IF NOT EXISTS idx_plates_video ON plates(video_id);
CREATE INDEX IF NOT EXISTS idx_plates_saved ON plates(saved_at);
CREATE INDEX IF NOT EXISTS idx_plates_status ON plates(ocr_status);
CREATE INDEX IF NOT EXISTS idx_plates_filename ON plates(filename);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id  INTEGER,
    level     TEXT NOT NULL,
    message   TEXT NOT NULL,
    at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);

CREATE TABLE IF NOT EXISTS state (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id                BIGSERIAL PRIMARY KEY,
    url               TEXT NOT NULL,
    title             TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    error             TEXT,
    frames_processed  BIGINT NOT NULL DEFAULT 0,
    plates_saved      BIGINT NOT NULL DEFAULT 0,
    detections_seen   BIGINT NOT NULL DEFAULT 0,
    rejected_conf     BIGINT NOT NULL DEFAULT 0,
    rejected_size     BIGINT NOT NULL DEFAULT 0,
    rejected_blur     BIGINT NOT NULL DEFAULT 0,
    rejected_dupe     BIGINT NOT NULL DEFAULT 0,
    duration_s        DOUBLE PRECISION,
    position_s        DOUBLE PRECISION NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 0,
    added_at          DOUBLE PRECISION NOT NULL,
    started_at        DOUBLE PRECISION,
    finished_at       DOUBLE PRECISION,
    heartbeat_at      DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);

CREATE TABLE IF NOT EXISTS plates (
    id           BIGSERIAL PRIMARY KEY,
    video_id     BIGINT REFERENCES videos(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    frame_index  BIGINT NOT NULL DEFAULT 0,
    timestamp_s  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    confidence   DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    box          TEXT NOT NULL DEFAULT '[0,0,0,0]',
    width        INTEGER NOT NULL DEFAULT 0,
    height       INTEGER NOT NULL DEFAULT 0,
    blur_score   DOUBLE PRECISION,
    phash        TEXT,
    bytes        BIGINT,
    saved_at     DOUBLE PRECISION NOT NULL,
    plate_text   TEXT,
    ocr_status   TEXT NOT NULL DEFAULT 'unlabeled',
    labeled_at   DOUBLE PRECISION,
    storage_url  TEXT
);
CREATE INDEX IF NOT EXISTS idx_plates_video ON plates(video_id);
CREATE INDEX IF NOT EXISTS idx_plates_saved ON plates(saved_at);
CREATE INDEX IF NOT EXISTS idx_plates_status ON plates(ocr_status);
CREATE INDEX IF NOT EXISTS idx_plates_filename ON plates(filename);

CREATE TABLE IF NOT EXISTS events (
    id        BIGSERIAL PRIMARY KEY,
    video_id  BIGINT,
    level     TEXT NOT NULL,
    message   TEXT NOT NULL,
    at        DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);

CREATE TABLE IF NOT EXISTS state (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""


def is_postgres() -> bool:
    """Check if PostgreSQL / Supabase connection is configured."""
    cfg = load_config()
    db_url = (
        str(cfg.get("supabase.db_url") or "").strip()
        or os.environ.get("DATABASE_URL", "").strip()
        or os.environ.get("SUPABASE_DB_URL", "").strip()
    )
    return bool(db_url and ("postgres://" in db_url or "postgresql://" in db_url))


def _get_postgres_url() -> str:
    cfg = load_config()
    db_url = (
        str(cfg.get("supabase.db_url") or "").strip()
        or os.environ.get("DATABASE_URL", "").strip()
        or os.environ.get("SUPABASE_DB_URL", "").strip()
    )
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url


def connect_sqlite() -> sqlite3.Connection:
    conn = getattr(_local, "sqlite_conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.sqlite_conn = conn
    return conn


def connect_postgres():
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise RuntimeError(
            "psycopg2 is required for PostgreSQL / Supabase. Please install psycopg2-binary."
        )

    conn = getattr(_local, "pg_conn", None)
    if conn is None or conn.closed:
        url = _get_postgres_url()
        conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
        conn.autocommit = True
        _local.pg_conn = conn
    return conn


def _translate_sql(sql: str) -> str:
    """Translate '?' parameter placeholders to '%s' for PostgreSQL."""
    if not is_postgres():
        return sql
    # Replace ? with %s for Postgres
    return sql.replace("?", "%s")


class DBConnectionWrapper:
    """Wrapper that provides uniform execution across SQLite and PostgreSQL."""

    def __init__(self):
        self.use_pg = is_postgres()

    def execute(self, sql: str, params: tuple | list = ()):
        sql = _translate_sql(sql)
        params = tuple(params) if params is not None else ()
        if self.use_pg:
            conn = connect_postgres()
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur
        else:
            conn = connect_sqlite()
            return conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple | list = ()) -> dict[str, Any] | None:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        cur = self.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def insert_get_id(self, sql: str, params: tuple | list = (), returning: str = "id") -> int:
        if self.use_pg:
            pg_sql = _translate_sql(sql)
            if "RETURNING" not in pg_sql.upper():
                pg_sql = f"{pg_sql.rstrip(';')} RETURNING {returning}"
            conn = connect_postgres()
            cur = conn.cursor()
            cur.execute(pg_sql, tuple(params))
            row = cur.fetchone()
            return row[returning] if row else 0
        else:
            conn = connect_sqlite()
            cur = conn.execute(sql, tuple(params))
            return cur.lastrowid or 0


def db_wrapper() -> DBConnectionWrapper:
    return DBConnectionWrapper()


def _migrate_schema() -> None:
    """Add new columns to existing databases if missing."""
    if is_postgres():
        try:
            conn = connect_postgres()
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE plates ADD COLUMN IF NOT EXISTS plate_text TEXT;"
                    "ALTER TABLE plates ADD COLUMN IF NOT EXISTS ocr_status TEXT NOT NULL DEFAULT 'unlabeled';"
                    "ALTER TABLE plates ADD COLUMN IF NOT EXISTS labeled_at DOUBLE PRECISION;"
                    "ALTER TABLE plates ADD COLUMN IF NOT EXISTS storage_url TEXT;"
                )
        except Exception as exc:
            log.warning("Postgres schema migration note: %s", exc)
    else:
        try:
            conn = connect_sqlite()
            cur = conn.execute("PRAGMA table_info(plates)")
            cols = {row["name"] for row in cur.fetchall()}
            if cols:
                if "plate_text" not in cols:
                    conn.execute("ALTER TABLE plates ADD COLUMN plate_text TEXT")
                if "ocr_status" not in cols:
                    conn.execute("ALTER TABLE plates ADD COLUMN ocr_status TEXT NOT NULL DEFAULT 'unlabeled'")
                if "labeled_at" not in cols:
                    conn.execute("ALTER TABLE plates ADD COLUMN labeled_at REAL")
                if "storage_url" not in cols:
                    conn.execute("ALTER TABLE plates ADD COLUMN storage_url TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_plates_status ON plates(ocr_status)")
        except Exception:
            pass


def init() -> None:
    if is_postgres():
        log.info("Initializing Supabase / PostgreSQL database connection...")
        try:
            conn = connect_postgres()
            with conn.cursor() as cur:
                cur.execute(PG_SCHEMA)
            _migrate_schema()
            log.info("Supabase / PostgreSQL initialized successfully.")
        except Exception as exc:
            log.error("Failed to connect/initialize Supabase PostgreSQL: %s", exc)
            raise
    else:
        conn = connect_sqlite()
        with _write_lock:
            _migrate_schema()
            conn.executescript(SQLITE_SCHEMA)
            _migrate_schema()

    reclaim_orphans()


def reclaim_orphans() -> int:
    with _write_lock:
        db = db_wrapper()
        cur = db.execute(
            "UPDATE videos SET status=?, started_at=NULL WHERE status=?",
            (STATUS_PENDING, STATUS_PROCESSING),
        )
        return cur.rowcount if hasattr(cur, "rowcount") and cur.rowcount is not None else 0


# --------------------------------------------------------------------- state


def set_state(key: str, value: Any) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, json.dumps(value)),
        )


def get_state(key: str, default: Any = None) -> Any:
    db = db_wrapper()
    row = db.fetchone("SELECT value FROM state WHERE key=?", (key,))
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return default


# -------------------------------------------------------------------- videos


def add_videos(urls: Iterable[str]) -> tuple[int, int]:
    """Insert URLs, skipping ones already queued or completed. -> (added, skipped)"""
    added = skipped = 0
    now = time.time()
    db = db_wrapper()

    with _write_lock:
        for url in urls:
            url = url.strip()
            if not url:
                continue
            existing = db.fetchone(
                "SELECT id FROM videos WHERE url=? AND status IN (?,?,?)",
                (url, STATUS_PENDING, STATUS_PROCESSING, STATUS_DONE),
            )
            if existing:
                skipped += 1
                continue
            db.execute(
                "INSERT INTO videos(url, status, added_at) VALUES(?,?,?)",
                (url, STATUS_PENDING, now),
            )
            added += 1
    return added, skipped


def claim_next_video() -> dict[str, Any] | None:
    """Atomically take the oldest pending job. None if the queue is empty."""
    now = time.time()
    db = db_wrapper()

    with _write_lock:
        row = db.fetchone(
            "SELECT * FROM videos WHERE status=? ORDER BY id LIMIT 1",
            (STATUS_PENDING,),
        )
        if row is None:
            return None
        db.execute(
            "UPDATE videos SET status=?, started_at=COALESCE(started_at,?), "
            "heartbeat_at=?, attempts=attempts+1, error=NULL WHERE id=?",
            (STATUS_PROCESSING, now, now, row["id"]),
        )

    claimed = dict(row)
    claimed["status"] = STATUS_PROCESSING
    claimed["attempts"] = int(claimed.get("attempts") or 0) + 1
    return claimed



def update_video(video_id: int, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "title", "status", "error", "frames_processed", "plates_saved",
        "detections_seen", "rejected_conf", "rejected_size", "rejected_blur",
        "rejected_dupe", "duration_s", "position_s", "started_at",
        "finished_at", "heartbeat_at",
    }
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return
    sql = "UPDATE videos SET " + ", ".join(f"{k}=?" for k in cols) + " WHERE id=?"
    with _write_lock:
        db = db_wrapper()
        db.execute(sql, (*cols.values(), video_id))


def bump_video(video_id: int, **deltas: int) -> None:
    """Increment counters without a read-modify-write race."""
    allowed = {
        "frames_processed", "plates_saved", "detections_seen", "rejected_conf",
        "rejected_size", "rejected_blur", "rejected_dupe",
    }
    cols = {k: v for k, v in deltas.items() if k in allowed and v}
    if not cols:
        return
    sql = (
        "UPDATE videos SET "
        + ", ".join(f"{k}={k}+?" for k in cols)
        + ", heartbeat_at=? WHERE id=?"
    )
    with _write_lock:
        db = db_wrapper()
        db.execute(sql, (*cols.values(), time.time(), video_id))


def get_video(video_id: int) -> dict[str, Any] | None:
    db = db_wrapper()
    return db.fetchone("SELECT * FROM videos WHERE id=?", (video_id,))


def list_videos(limit: int = 500) -> list[dict[str, Any]]:
    db = db_wrapper()
    sql = (
        "SELECT * FROM videos ORDER BY "
        "CASE status WHEN 'processing' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, id LIMIT ?"
    )
    return db.fetchall(sql, (limit,))


def requeue_video(video_id: int, reset_progress: bool = False) -> None:
    with _write_lock:
        db = db_wrapper()
        if reset_progress:
            db.execute(
                "UPDATE videos SET status=?, error=NULL, finished_at=NULL, "
                "position_s=0, frames_processed=0, attempts=0 WHERE id=?",
                (STATUS_PENDING, video_id),
            )
        else:
            db.execute(
                "UPDATE videos SET status=?, error=NULL, finished_at=NULL WHERE id=?",
                (STATUS_PENDING, video_id),
            )


def requeue_all_errored() -> int:
    with _write_lock:
        db = db_wrapper()
        cur = db.execute(
            "UPDATE videos SET status=?, error=NULL, finished_at=NULL, attempts=0 "
            "WHERE status=?",
            (STATUS_PENDING, STATUS_ERROR),
        )
        return cur.rowcount if hasattr(cur, "rowcount") and cur.rowcount is not None else 0


def cancel_video(video_id: int) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute(
            "UPDATE videos SET status=?, finished_at=? WHERE id=? AND status IN (?,?)",
            (STATUS_CANCELLED, time.time(), video_id, STATUS_PENDING, STATUS_PROCESSING),
        )


def delete_video(video_id: int) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute("DELETE FROM plates WHERE video_id=?", (video_id,))
        db.execute("DELETE FROM videos WHERE id=?", (video_id,))


# -------------------------------------------------------------------- plates


def record_plate(
    video_id: int,
    filename: str,
    frame_index: int,
    timestamp_s: float,
    confidence: float,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    blur_score: float | None,
    phash: str | None,
    nbytes: int | None,
    storage_url: str | None = None,
) -> int:
    with _write_lock:
        db = db_wrapper()
        sql = (
            "INSERT INTO plates(video_id, filename, frame_index, timestamp_s, "
            "confidence, box, width, height, blur_score, phash, bytes, saved_at, storage_url) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        return db.insert_get_id(
            sql,
            (
                video_id, filename, frame_index, timestamp_s, confidence,
                json.dumps(list(box)), width, height, blur_score, phash,
                nbytes, time.time(), storage_url,
            ),
        )


def recent_plates(limit: int = 40) -> list[dict[str, Any]]:
    db = db_wrapper()
    return db.fetchall(
        "SELECT p.*, v.url AS source_url FROM plates p "
        "LEFT JOIN videos v ON v.id = p.video_id "
        "ORDER BY p.id DESC LIMIT ?",
        (limit,),
    )


def get_plate(plate_id: int) -> dict[str, Any] | None:
    db = db_wrapper()
    return db.fetchone(
        "SELECT p.*, v.url AS source_url FROM plates p "
        "LEFT JOIN videos v ON v.id = p.video_id "
        "WHERE p.id=?",
        (plate_id,),
    )


def get_plate_by_filename(filename: str) -> dict[str, Any] | None:
    db = db_wrapper()
    return db.fetchone(
        "SELECT p.*, v.url AS source_url FROM plates p "
        "LEFT JOIN videos v ON v.id = p.video_id "
        "WHERE p.filename=?",
        (filename,),
    )


def update_plate_label(
    plate_id: int, plate_text: str | None, ocr_status: str = "labeled"
) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute(
            "UPDATE plates SET plate_text=?, ocr_status=?, labeled_at=? WHERE id=?",
            (plate_text, ocr_status, time.time(), plate_id),
        )


def list_plates(
    limit: int = 100,
    offset: int = 0,
    status: str | None = None,
    search: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return (items, total_count) for paginated plate labeling table."""
    db = db_wrapper()
    conditions: list[str] = []
    params: list[Any] = []

    if status and status != "all":
        conditions.append("p.ocr_status = ?")
        params.append(status)

    if search:
        s = f"%{search.strip()}%"
        conditions.append("(p.plate_text LIKE ? OR p.filename LIKE ?)")
        params.extend([s, s])

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    count_row = db.fetchone(
        f"SELECT COUNT(*) AS c FROM plates p {where_clause}",
        params,
    )
    total = count_row["c"] if count_row else 0

    rows = db.fetchall(
        f"SELECT p.*, v.url AS source_url, v.title AS source_title FROM plates p "
        f"LEFT JOIN videos v ON v.id = p.video_id "
        f"{where_clause} "
        f"ORDER BY p.id DESC LIMIT ? OFFSET ?",
        (*params, max(1, min(500, limit)), max(0, offset)),
    )

    return rows, total


def get_unlabeled_plates(limit: int = 100) -> list[dict[str, Any]]:
    db = db_wrapper()
    return db.fetchall(
        "SELECT p.*, v.url AS source_url FROM plates p "
        "LEFT JOIN videos v ON v.id = p.video_id "
        "WHERE p.ocr_status = 'unlabeled' OR p.plate_text IS NULL OR p.plate_text = '' "
        "ORDER BY p.id ASC LIMIT ?",
        (limit,),
    )


def get_all_plates_for_export() -> list[dict[str, Any]]:
    db = db_wrapper()
    return db.fetchall(
        "SELECT p.*, v.url AS source_url, v.title AS source_title FROM plates p "
        "LEFT JOIN videos v ON v.id = p.video_id "
        "ORDER BY p.id ASC",
        (),
    )


def get_label_stats() -> dict[str, Any]:
    db = db_wrapper()
    row = db.fetchone(
        "SELECT "
        " COUNT(*) AS total_plates,"
        " SUM(CASE WHEN ocr_status = 'labeled' AND plate_text IS NOT NULL AND plate_text != '' THEN 1 ELSE 0 END) AS labeled,"
        " SUM(CASE WHEN ocr_status = 'unlabeled' OR plate_text IS NULL OR plate_text = '' THEN 1 ELSE 0 END) AS unlabeled,"
        " SUM(CASE WHEN ocr_status = 'error' THEN 1 ELSE 0 END) AS error "
        " FROM plates"
    )
    if not row:
        return {"total_plates": 0, "labeled": 0, "unlabeled": 0, "error": 0}
    return {
        "total_plates": int(row["total_plates"] or 0),
        "labeled": int(row["labeled"] or 0),
        "unlabeled": int(row["unlabeled"] or 0),
        "error": int(row["error"] or 0),
    }


def clear_all_plates() -> int:
    """Delete all plate records, clear manual upload video entries, and reset sequence counters."""
    with _write_lock:
        db = db_wrapper()
        row = db.fetchone("SELECT COUNT(*) AS c FROM plates")
        count = row["c"] if row else 0
        db.execute("DELETE FROM plates")
        db.execute("DELETE FROM videos WHERE url='manual_upload'")
        if not is_postgres():
            try:
                connect_sqlite().execute("DELETE FROM sqlite_sequence WHERE name IN ('plates')")
            except Exception:
                pass
        return count


def reset_all_plate_labels() -> int:
    """Reset all plate labels back to unlabeled without deleting crop records."""
    with _write_lock:
        db = db_wrapper()
        cur = db.execute(
            "UPDATE plates SET plate_text = NULL, ocr_status = 'unlabeled', labeled_at = NULL"
        )
        return cur.rowcount if hasattr(cur, "rowcount") and cur.rowcount is not None else 0


def record_custom_plate(
    filename: str,
    width: int = 0,
    height: int = 0,
    nbytes: int = 0,
    plate_text: str | None = None,
    ocr_status: str = "unlabeled",
    storage_url: str | None = None,
) -> int:
    """Record a manually uploaded plate image."""
    now = time.time()
    with _write_lock:
        db = db_wrapper()
        cur = db.fetchone("SELECT id FROM videos WHERE url='manual_upload' LIMIT 1")
        if cur:
            video_id = cur["id"]
        else:
            video_id = db.insert_get_id(
                "INSERT INTO videos(url, title, status, added_at) VALUES(?,?,?,?)",
                ("manual_upload", "Manual Uploads", STATUS_DONE, now),
            )

        plate_id = db.insert_get_id(
            "INSERT INTO plates(video_id, filename, frame_index, timestamp_s, "
            "confidence, box, width, height, blur_score, phash, bytes, saved_at, "
            "plate_text, ocr_status, labeled_at, storage_url) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                video_id, filename, 0, 0.0, 1.0, "[0,0,0,0]",
                width, height, None, None, nbytes, now,
                plate_text, ocr_status, (now if plate_text else None), storage_url,
            ),
        )
        return plate_id


def batch_update_labels(records: list[dict[str, Any]]) -> int:
    """Bulk update plate labels by id or filename."""
    updated = 0
    now = time.time()
    db = db_wrapper()

    with _write_lock:
        for rec in records:
            plate_id = rec.get("id")
            filename = rec.get("filename")
            plate_text = rec.get("plate_text") or rec.get("text") or rec.get("plate")
            if plate_text is not None:
                plate_text = str(plate_text).strip()
            if plate_id:
                cur = db.execute(
                    "UPDATE plates SET plate_text=?, ocr_status='labeled', labeled_at=? WHERE id=?",
                    (plate_text, now, plate_id),
                )
                updated += cur.rowcount if hasattr(cur, "rowcount") and cur.rowcount else 1
            elif filename:
                cur = db.execute(
                    "UPDATE plates SET plate_text=?, ocr_status='labeled', labeled_at=? WHERE filename=?",
                    (plate_text, now, filename),
                )
                updated += cur.rowcount if hasattr(cur, "rowcount") and cur.rowcount else 1

    return updated


def delete_plate(plate_id: int) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute("DELETE FROM plates WHERE id=?", (plate_id,))


def totals() -> dict[str, Any]:
    db = db_wrapper()
    sql = (
        "SELECT "
        " COUNT(*) AS videos,"
        " SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,"
        " SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,"
        " SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) AS processing,"
        " SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errored,"
        " SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled,"
        " COALESCE(SUM(frames_processed),0) AS frames,"
        " COALESCE(SUM(plates_saved),0) AS plates,"
        " COALESCE(SUM(detections_seen),0) AS detections,"
        " COALESCE(SUM(rejected_conf),0) AS rejected_conf,"
        " COALESCE(SUM(rejected_size),0) AS rejected_size,"
        " COALESCE(SUM(rejected_blur),0) AS rejected_blur,"
        " COALESCE(SUM(rejected_dupe),0) AS rejected_dupe"
        " FROM videos"
    )
    row = db.fetchone(sql)
    out = {k: (row[k] or 0) for k in (row or {}).keys()}
    size = db.fetchone("SELECT COALESCE(SUM(bytes),0) AS b FROM plates")
    out["output_bytes"] = size["b"] if size and "b" in size else 0
    return out


# -------------------------------------------------------------------- events


def log_event(level: str, message: str, video_id: int | None = None) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute(
            "INSERT INTO events(video_id, level, message, at) VALUES(?,?,?,?)",
            (video_id, level, message[:2000], time.time()),
        )


def recent_events(limit: int = 100) -> list[dict[str, Any]]:
    db = db_wrapper()
    return db.fetchall("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


def prune_events(keep: int = 5000) -> None:
    with _write_lock:
        db = db_wrapper()
        db.execute(
            "DELETE FROM events WHERE id NOT IN "
            "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
            (keep,),
        )
