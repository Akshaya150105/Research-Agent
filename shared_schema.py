"""
shared_schema.py
================
Single source of truth for the SQLite schema.
Both reader_agent.py and kg_population.py import from here.

WHY THIS EXISTS:
  The reader and kg_population previously defined their own _SCHEMA strings
  independently, causing schema mismatches:

  Reader's entities table:
    entity_id TEXT PK (UUID), paper_id FK, text, text_norm, entity_type,
    section_type, confidence, source

  KG's entities table:
    entity_id TEXT PK ("method::lstm"), entity_type, canonical_text,
    raw_variants JSON, papers_seen_in JSON
    ← NO paper_id column → crash: "table entities has no column named canonical_text"

  Fix: ONE schema, both agents import it. The KG's richer schema wins
  because it's what Phase 2 specifies and what downstream agents query.

USAGE:
  from shared_schema import FULL_SCHEMA, ensure_schema, EDGE_TYPE_MAP

  conn = sqlite3.connect("shared_memory/research.db")
  ensure_schema(conn)
"""

import sqlite3

# ── Canonical edge type for each entity type ─────────────────
EDGE_TYPE_MAP = {
    "method":  "uses",
    "dataset": "evaluates_on",
    "metric":  "measures_with",
    "task":    "addresses",
}

# ── Full unified schema ───────────────────────────────────────
FULL_SCHEMA = """
-- ── Papers ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,
    title         TEXT DEFAULT '',
    authors       TEXT DEFAULT '[]',   -- JSON list of author name strings
    year          INTEGER,
    venue         TEXT DEFAULT '',
    doi           TEXT DEFAULT '',
    abstract      TEXT DEFAULT '',
    -- Reader-specific fields (NULL when paper added by KG only)
    source_path   TEXT DEFAULT '',     -- path to claims_output.json on disk
    n_claims      INTEGER DEFAULT 0,
    n_entities    INTEGER DEFAULT 0,
    n_limitations INTEGER DEFAULT 0,
    coverage_gain REAL    DEFAULT 0.0, -- fraction of new entities this paper added
    action        TEXT    DEFAULT 'read', -- 'read' | 'skip'
    processed_at  TEXT    DEFAULT (datetime('now'))
);

-- ── Canonical entities (KG format) ──────────────────────────
-- entity_id format: "{entity_type}::{normalize(canonical_text)}"
-- e.g.  "method::long short-term memory"
-- raw_variants: JSON list of all surface strings that map here
-- papers_seen_in: JSON list of paper_ids that mention this entity
CREATE TABLE IF NOT EXISTS entities (
    entity_id       TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,          -- method | dataset | metric | task
    canonical_text  TEXT NOT NULL,          -- LLM-chosen display form
    raw_variants    TEXT DEFAULT '[]',      -- JSON list
    papers_seen_in  TEXT DEFAULT '[]'       -- JSON list
);

-- ── Paper → Entity typed edges ───────────────────────────────
-- Edge types: uses | evaluates_on | measures_with | addresses
CREATE TABLE IF NOT EXISTS paper_entity_relationships (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    section     TEXT    DEFAULT '',
    confidence  REAL    DEFAULT 1.0,
    UNIQUE(paper_id, entity_id, edge_type)
);

-- ── Limitation statements ─────────────────────────────────────
-- Graph node_type: LimitationStatement | edge: has_limitation
CREATE TABLE IF NOT EXISTS limitation_statements (
    stmt_id           TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    text              TEXT NOT NULL,
    section           TEXT    DEFAULT '',
    confidence        REAL    DEFAULT 1.0,
    entities_involved TEXT    DEFAULT '[]'  -- JSON list
);

-- ── Future work statements ────────────────────────────────────
-- Gap Detector (Phase 6) queries these as direct gap signals
CREATE TABLE IF NOT EXISTS future_work_statements (
    stmt_id           TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    text              TEXT NOT NULL,
    section           TEXT    DEFAULT '',
    confidence        REAL    DEFAULT 1.0,
    entities_involved TEXT    DEFAULT '[]'
);

-- ── Claims ────────────────────────────────────────────────────
-- Comparator Agent (Phase 4) adds contradiction edges on top of these
CREATE TABLE IF NOT EXISTS claims (
    claim_id          TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    claim_type        TEXT    DEFAULT '',   -- performance | comparative | methodological
    description       TEXT    DEFAULT '',
    value             TEXT,                 -- numeric value if present, else NULL
    confidence        REAL    DEFAULT 1.0,
    section           TEXT    DEFAULT '',
    entities_involved TEXT    DEFAULT '[]'
);

-- ── Comparator outputs ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY,
    paper_a       TEXT NOT NULL,
    paper_b       TEXT NOT NULL,
    finding_type  TEXT NOT NULL,
    severity      TEXT DEFAULT 'LOW',
    metric        TEXT DEFAULT '',
    dataset       TEXT DEFAULT '',
    value_a       REAL,
    value_b       REAL,
    diff          REAL,
    rationale     TEXT DEFAULT '',
    confidence    REAL DEFAULT 1.0,
    source        TEXT DEFAULT 'heuristic',
    llm_used      INTEGER DEFAULT 0,
    setup_used    INTEGER DEFAULT 0,
    generated_at  TEXT NOT NULL
);

-- ── Session / action log ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL DEFAULT 'default',
    timestamp      TEXT DEFAULT (datetime('now')),
    agent          TEXT DEFAULT '',
    action         TEXT DEFAULT '',
    detail         TEXT DEFAULT '',      -- KG uses this for free-text notes
    input_summary  TEXT DEFAULT '',      -- Reader/comparator use this
    output_summary TEXT DEFAULT '',
    confidence     REAL    DEFAULT 1.0,
    duration_ms    INTEGER DEFAULT 0
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Create all tables if they don't exist, then add any missing columns
    to tables that were created by an older version of the code.
    Safe to call multiple times — fully idempotent.
    """
    conn.executescript(FULL_SCHEMA)

    # ── Patch tables that may have been created by older schemas ─
    # ALTER TABLE IF NOT EXISTS column syntax is not supported in older SQLite,
    # so we use try/except per column.
    _add_column_if_missing = [
        # papers table — columns added in v2 that older DBs won't have
        ("papers", "doi",           "TEXT DEFAULT ''"),
        ("papers", "source_path",   "TEXT DEFAULT ''"),
        ("papers", "n_claims",      "INTEGER DEFAULT 0"),
        ("papers", "n_entities",    "INTEGER DEFAULT 0"),
        ("papers", "n_limitations", "INTEGER DEFAULT 0"),
        ("papers", "coverage_gain", "REAL DEFAULT 0.0"),
        ("papers", "action",        "TEXT DEFAULT 'read'"),
        ("papers", "processed_at",  "TEXT DEFAULT (datetime('now'))"),
        # session_log — columns that reader and KG add independently
        ("session_log", "detail",         "TEXT DEFAULT ''"),
        ("session_log", "input_summary",  "TEXT DEFAULT ''"),
        ("session_log", "output_summary", "TEXT DEFAULT ''"),
        ("session_log", "confidence",     "REAL DEFAULT 1.0"),
        ("session_log", "duration_ms",    "INTEGER DEFAULT 0"),
    ]

    for table, column, col_def in _add_column_if_missing:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()