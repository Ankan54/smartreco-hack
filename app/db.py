"""SQLite access. WAL is mandatory: batched background writes from event ingest
will throw 'database is locked' without it."""

import sqlite3
import threading
from contextlib import contextmanager

import numpy as np

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL DEFAULT 'user',
  onboarded     INTEGER NOT NULL DEFAULT 0,
  -- Opt-in, default off. Signup addresses are unverified, so mailing everyone
  -- who registered would be unsolicited email to strangers.
  digest_opt_in INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
  id           INTEGER PRIMARY KEY,
  title        TEXT NOT NULL,
  description  TEXT NOT NULL,
  category     TEXT NOT NULL,
  level        TEXT,
  price        REAL NOT NULL DEFAULT 0,
  provider     TEXT,
  skills       TEXT,
  rating       REAL,
  prereq_ids   TEXT NOT NULL DEFAULT '[]',
  content_hash TEXT NOT NULL,
  is_active    INTEGER NOT NULL DEFAULT 1,
  updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_products_cat ON products(category, is_active);

-- Hybrid retrieval half. Dense embeddings miss rare proper nouns ("PySpark",
-- "CISSP"); BM25 nails them. FTS5 ships with SQLite, so this costs no dependency.
CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
  title, skills, description, tokenize='porter'
);

CREATE TABLE IF NOT EXISTS events (
  id         TEXT PRIMARY KEY,      -- client UUID; INSERT OR IGNORE dedupes
  user_id    INTEGER NOT NULL REFERENCES users(id),
  session_id TEXT NOT NULL,
  type       TEXT NOT NULL,
  product_id INTEGER,
  query      TEXT,
  value      REAL,
  ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts DESC);

CREATE TABLE IF NOT EXISTS user_state (
  user_id                 INTEGER PRIMARY KEY REFERENCES users(id),
  intent_vec              BLOB,
  intent_vec_at_last_reco BLOB,
  events_since_reco       INTEGER NOT NULL DEFAULT 0,
  last_reco_at            TEXT,
  last_event_at           TEXT,
  updated_at              TEXT
);

CREATE TABLE IF NOT EXISTS recommendations (
  id         INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id),
  headline   TEXT NOT NULL,
  narrative  TEXT NOT NULL,
  items_json TEXT NOT NULL,
  trigger    TEXT NOT NULL,
  drift      REAL,
  trace_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  delivered_at TEXT,
  is_current INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_reco_user ON recommendations(user_id, id DESC);

CREATE TABLE IF NOT EXISTS embed_cache (
  text_hash  TEXT PRIMARY KEY,
  model      TEXT NOT NULL,
  vec        BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Append-only: every version kept so the UI can diff v(n) against v(n-1).
CREATE TABLE IF NOT EXISTS dossier_versions (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id),
  version     INTEGER NOT NULL,
  claims_json TEXT NOT NULL,
  prose       TEXT NOT NULL,
  source      TEXT NOT NULL,          -- 'reflection' | 'user_edit'
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dossier_ver ON dossier_versions(user_id, version);
"""

_local = threading.local()


def conn() -> sqlite3.Connection:
    """One connection per thread. FastAPI's threadpool reuses threads, so this
    is effectively pooled."""
    c = getattr(_local, "conn", None)
    if c is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(config.DB_PATH, check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return c


@contextmanager
def tx():
    c = conn()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise


# Columns added after the first schema shipped. SQLite has no "ADD COLUMN IF NOT
# EXISTS", and a dev database that predates them should not need deleting.
MIGRATIONS = [
    ("users", "digest_opt_in", "INTEGER NOT NULL DEFAULT 0"),
    ("recommendations", "delivered_at", "TEXT"),
]


def init() -> None:
    with tx() as c:
        c.executescript(SCHEMA)
        for table, column, decl in MIGRATIONS:
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def q(sql: str, args=()) -> list[sqlite3.Row]:
    return conn().execute(sql, args).fetchall()


def q1(sql: str, args=()) -> sqlite3.Row | None:
    return conn().execute(sql, args).fetchone()


# --- vector <-> blob -------------------------------------------------------

def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    if not b:
        return None
    return np.frombuffer(b, dtype=np.float32)
