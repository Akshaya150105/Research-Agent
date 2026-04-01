"""
reader_agent.py  v1.1.0
========================
Person 3 — AI Agents  |  Branch: feat/agents  |  Folder: agents/

CHANGES FROM v1.0.0:
  - Fixed new_methods/new_datasets/new_metrics accumulation bug.
    Old code had local total_new_* variables that were never incremented —
    the counts went into report.new_* directly inside _process_one but
    the local totals were always 0. Now report fields are accumulated
    directly and consistently.
  - Fixed abstract extraction: claims_output.json stores abstract under
    metadata.abstract, not top-level abstract. Added fallback chain.
  - entities field in claims_output.json is named "entities" but some
    papers may use "llm_entities". Added both fallbacks.
  - DB_PATH is now read from state["db_path"] if provided, else default.
    This lets the Planner override the path for testing.

Usage (standalone):
  python agents/reader_agent.py --verbose
  python agents/reader_agent.py --memory-dir memory --verbose

Usage (from Planner / LangGraph):
  from reader_agent import ReaderAgent
  agent = ReaderAgent(verbose=True)
  updated_state = agent.run(state)
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

_AGENTS_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from entity_resolver import get_typed_entities

# ── defaults (can be overridden via state dict) ───────────────
MEMORY_DIR              = pathlib.Path("memory")
SHARED_MEMORY           = pathlib.Path("shared_memory")
DB_PATH                 = SHARED_MEMORY / "research.db"
COVERAGE_GAIN_THRESHOLD = 0.10


# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class PaperRecord:
    paper_id:      str
    title:         str
    authors:       str        # JSON list
    year:          Optional[int]
    venue:         str
    abstract:      str
    source_path:   str
    n_claims:      int
    n_entities:    int
    n_limitations: int
    coverage_gain: float
    action:        str        # "read" | "skip"
    processed_at:  str


@dataclass
class ReaderReport:
    agent:                str   = "reader_agent"
    agent_version:        str   = "1.1.0"
    session_id:           str   = ""
    papers_found:         int   = 0
    papers_read:          int   = 0
    papers_skipped:       int   = 0
    papers_already_in_db: int   = 0
    new_methods:          int   = 0
    new_datasets:         int   = 0
    new_metrics:          int   = 0
    coverage_score:       float = 0.0
    paper_ids_read:       list  = field(default_factory=list)
    react_trace:          list  = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  DB SCHEMA
# ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,
    title         TEXT DEFAULT '',
    authors       TEXT DEFAULT '[]',
    year          INTEGER,
    venue         TEXT DEFAULT '',
    abstract      TEXT DEFAULT '',
    source_path   TEXT DEFAULT '',
    n_claims      INTEGER DEFAULT 0,
    n_entities    INTEGER DEFAULT 0,
    n_limitations INTEGER DEFAULT 0,
    coverage_gain REAL DEFAULT 0.0,
    action        TEXT DEFAULT 'read',
    processed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id    TEXT PRIMARY KEY,
    paper_id     TEXT NOT NULL,
    text         TEXT NOT NULL,
    text_norm    TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    section_type TEXT DEFAULT '',
    confidence   REAL DEFAULT 1.0,
    source       TEXT DEFAULT 'llm',
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE TABLE IF NOT EXISTS session_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    agent          TEXT NOT NULL,
    action         TEXT NOT NULL,
    input_summary  TEXT DEFAULT '',
    output_summary TEXT DEFAULT '',
    confidence     REAL DEFAULT 1.0,
    duration_ms    INTEGER DEFAULT 0
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Handle existing DBs that are missing the new columns
    try:
        conn.execute("ALTER TABLE entities ADD COLUMN text_norm TEXT DEFAULT ''")
    except Exception:
        pass
    
    try:
        conn.execute("ALTER TABLE session_log ADD COLUMN input_summary TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE session_log ADD COLUMN output_summary TEXT DEFAULT ''")
    except Exception:
        pass
    conn.commit()


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _already_processed(paper_id: str, conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return row is not None


def _known_entities(conn: sqlite3.Connection) -> dict[str, set]:
    """All entity text_norm values in DB, grouped by entity_type."""
    known: dict[str, set] = {
        "method": set(), "dataset": set(),
        "metric": set(), "task":    set(),
    }
    for text_norm, etype in conn.execute(
        "SELECT text_norm, entity_type FROM entities"
    ).fetchall():
        if etype in known:
            known[etype].add(text_norm)
    return known


def _compute_coverage_gain(
    paper: dict,
    known: dict[str, set],
) -> tuple[float, dict[str, int]]:
    """
    What fraction of this paper's entities are not yet in the DB?
    Returns (gain_score 0-1, counts_by_type).
    """
    typed      = get_typed_entities(paper)
    new_counts = {"method": 0, "dataset": 0, "metric": 0, "task": 0}
    total      = 0

    for etype in ("method", "dataset", "metric", "task"):
        for norm in typed.get(etype + "s", set()):
            total += 1
            if norm not in known[etype]:
                new_counts[etype] += 1

    if total == 0:
        return 0.0, new_counts

    return round(sum(new_counts.values()) / total, 4), new_counts


def _extract_abstract(paper: dict) -> str:
    """Try multiple locations for the abstract text."""
    # 1. metadata.abstract (most common in claims_output.json)
    meta = paper.get("metadata", {})
    ab   = meta.get("abstract", "")
    if ab:
        return ab[:1000]
    # 2. paper.summary (some formats)
    s = paper.get("summary", {})
    if isinstance(s, str) and s:
        return s[:1000]
    # 3. sections.abstract
    sections = paper.get("sections", {})
    if isinstance(sections, dict):
        ab = sections.get("abstract", "")
        if ab:
            return ab[:1000]
    return ""


def _write_paper_to_db(
    paper:  dict,
    record: PaperRecord,
    conn:   sqlite3.Connection,
) -> None:
    """Write paper record + all its entities to SQLite."""
    conn.execute(
        """INSERT OR REPLACE INTO papers
           (paper_id, title, authors, year, venue, abstract, source_path,
            n_claims, n_entities, n_limitations, coverage_gain, action, processed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            record.paper_id, record.title, record.authors,
            record.year, record.venue, record.abstract,
            record.source_path, record.n_claims, record.n_entities,
            record.n_limitations, record.coverage_gain,
            record.action, record.processed_at,
        ),
    )

    # Write entities — paper may use "entities" or "llm_entities"
    all_ents = paper.get("entities", []) or paper.get("llm_entities", [])
    for ent in all_ents:
        text      = (ent.get("text") or "").strip()
        text_norm = text.lower()
        if not text_norm:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO entities
               (entity_id, paper_id, text, text_norm, entity_type,
                section_type, confidence, source)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()), record.paper_id,
                text, text_norm,
                ent.get("entity_type", ""),
                ent.get("section_type", ""),
                float(ent.get("confidence", 1.0)),
                ent.get("source", "llm"),
            ),
        )
    conn.commit()


def _log_action(
    conn:           sqlite3.Connection,
    session_id:     str,
    agent:          str,
    action:         str,
    input_summary:  str,
    output_summary: str,
    confidence:     float = 1.0,
) -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO session_log
           (session_id, timestamp, agent, action,
            input_summary, output_summary, confidence, duration_ms)
           VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, now, agent, action,
         input_summary, output_summary, confidence, 0),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────
#  READER AGENT
# ─────────────────────────────────────────────────────────────

class ReaderAgent:

    VERSION = "1.1.0"

    def __init__(
        self,
        memory_dir: pathlib.Path = MEMORY_DIR,
        db_path:    pathlib.Path = DB_PATH,
        verbose:    bool         = False,
    ):
        self.memory_dir = memory_dir
        self.db_path    = db_path
        self.verbose    = verbose
        self.trace:     list[str] = []

    def _think(self, msg: str):
        e = f"[THINK] {msg}"; self.trace.append(e)
        if self.verbose: print(f"  {e}")

    def _act(self, msg: str):
        e = f"[ACT]   {msg}"; self.trace.append(e)
        if self.verbose: print(f"  {e}")

    def _observe(self, msg: str):
        e = f"[OBS]   {msg}"; self.trace.append(e)
        if self.verbose: print(f"  {e}")

    # ── LangGraph / Planner entry point ───────────────────────

    def run(self, state: dict) -> dict:
        """
        Called by Planner. Receives full state dict, returns updated state.

        State keys read:
          session_id, memory_dir, coverage_gain_threshold, db_path (optional)
        State keys written:
          reader_report, coverage_score
        """
        session_id = state.get("session_id", str(uuid.uuid4())[:8])
        mem_dir    = pathlib.Path(state.get("memory_dir", str(self.memory_dir)))
        threshold  = float(state.get("coverage_gain_threshold", COVERAGE_GAIN_THRESHOLD))
        db_path    = pathlib.Path(state.get("db_path", str(self.db_path)))

        report = self._run_internal(session_id, mem_dir, threshold, db_path)
        return {**state, "reader_report": asdict(report)}

    # ── Core logic ────────────────────────────────────────────

    def _run_internal(
        self,
        session_id: str,
        memory_dir: pathlib.Path,
        threshold:  float,
        db_path:    pathlib.Path,
    ) -> ReaderReport:

        report = ReaderReport(session_id=session_id)
        self.trace = []   # reset trace for this run

        # Ensure DB directory and schema exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        _ensure_schema(conn)

        # Step 1: Find all claims_output.json files
        self._think("Scanning memory/ for paper folders")
        paper_files = self._discover_papers(memory_dir)
        report.papers_found = len(paper_files)
        self._observe(
            f"Found {len(paper_files)} paper(s): "
            f"{[p.parent.name for p in paper_files]}"
        )

        if not paper_files:
            self._observe("No papers found — nothing to do.")
            conn.close()
            report.react_trace = self.trace
            return report

        # Step 2: Get current entity snapshot for coverage computation
        self._act("Querying SQLite for known entities")
        known = _known_entities(conn)
        self._observe(
            f"DB entities: {len(known['method'])} methods, "
            f"{len(known['dataset'])} datasets, "
            f"{len(known['metric'])} metrics, "
            f"{len(known['task'])} tasks"
        )

        # Step 3: Process each paper
        for paper_path in paper_files:
            processed = self._process_one(
                paper_path, conn, known, session_id, threshold, report, db_path,
            )
            if processed:
                # Refresh known so next paper's gain is relative to updated DB
                known = _known_entities(conn)

        # Step 4: Compute final coverage score
        # coverage_score = fraction of found papers that were read (not skipped/already done)
        total_available = report.papers_found
        report.coverage_score = round(
            report.papers_read / max(total_available, 1), 3
        )

        self._observe(
            f"Reader done — "
            f"read={report.papers_read}, "
            f"skipped={report.papers_skipped}, "
            f"already_in_db={report.papers_already_in_db}, "
            f"new_methods={report.new_methods}, "
            f"new_datasets={report.new_datasets}, "
            f"coverage={report.coverage_score:.2f}"
        )

        _log_action(
            conn, session_id, "reader_agent", "session_complete",
            f"papers_found={report.papers_found}",
            f"read={report.papers_read}, coverage={report.coverage_score:.2f}",
        )

        conn.close()
        report.react_trace = self.trace
        return report

    def _discover_papers(self, memory_dir: pathlib.Path) -> list[pathlib.Path]:
        """Return paths to all claims_output.json files under memory/."""
        if not memory_dir.exists():
            return []
        results = []
        for folder in sorted(memory_dir.iterdir()):
            if not folder.is_dir():
                continue
            p = folder / "claims_output.json"
            if p.exists():
                results.append(p)
        return results

    def _process_one(
        self,
        paper_path: pathlib.Path,
        conn:       sqlite3.Connection,
        known:      dict[str, set],
        session_id: str,
        threshold:  float,
        report:     ReaderReport,
        db_path:    pathlib.Path,
    ) -> bool:
        """
        Process one paper.
        Returns True if paper was written to DB (new), False otherwise.
        """
        try:
            with open(paper_path, encoding="utf-8") as f:
                paper = json.load(f)
        except Exception as e:
            self._observe(f"⚠ Cannot load {paper_path}: {e}")
            return False

        paper_id = paper.get("paper_id") or paper_path.parent.name
        meta     = paper.get("metadata", {})

        self._think(f"Evaluating '{paper_id}'")

        # Already in DB → count it but don't re-process
        if _already_processed(paper_id, conn):
            self._observe(f"  '{paper_id}' already in DB — counting as available")
            report.papers_already_in_db += 1
            return False

        # Compute coverage gain
        gain, new_counts = _compute_coverage_gain(paper, known)
        self._observe(
            f"  '{paper_id}' gain={gain:.2f} "
            f"(+{new_counts['method']} methods, "
            f"+{new_counts['dataset']} datasets, "
            f"+{new_counts['metric']} metrics)"
        )

        # ── READ-OR-SKIP DECISION ─────────────────────────────
        # TODO (Phase 8 / Person 4):
        #   Replace this with:
        #   state_vec = [gain, len(paper.get('entities',[])), ...]
        #   action = reader_policy.predict(state_vec)
        db_is_empty = conn.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0] == 0
        action = "read" if (gain >= threshold or db_is_empty) else "skip"
        # ─────────────────────────────────────────────────────

        self._act(f"  → {action.upper()} '{paper_id}' (gain={gain:.2f})")

        now    = datetime.datetime.now(datetime.timezone.utc).isoformat()
        claims = paper.get("claims", [])
        ents   = paper.get("entities", []) or paper.get("llm_entities", [])
        lims   = paper.get("limitations", [])

        record = PaperRecord(
            paper_id      = paper_id,
            title         = meta.get("title", ""),
            authors       = json.dumps(meta.get("authors", [])),
            year          = meta.get("year"),
            venue         = meta.get("venue", ""),
            abstract      = _extract_abstract(paper),
            source_path   = str(paper_path),
            n_claims      = len(claims),
            n_entities    = len(ents),
            n_limitations = len(lims),
            coverage_gain = gain,
            action        = action,
            processed_at  = now,
        )

        # Always write to DB (even skipped papers — we record the decision)
        _write_paper_to_db(paper, record, conn)

        # ── FIX: accumulate counts directly into report ───────
        if action == "read":
            report.papers_read    += 1
            report.paper_ids_read.append(paper_id)
            report.new_methods    += new_counts["method"]    # ← was broken in v1.0.0
            report.new_datasets   += new_counts["dataset"]
            report.new_metrics    += new_counts["metric"]
        else:
            report.papers_skipped += 1

        _log_action(
            conn, session_id, "reader_agent", action,
            f"paper={paper_id}",
            f"gain={gain:.2f}, new={sum(new_counts.values())}",
            confidence=gain,
        )

        return True   # written to DB

    # ── Standalone CLI ────────────────────────────────────────

    def print_report(self, report: ReaderReport) -> None:
        print("\n" + "═" * 56)
        print(f"  READER AGENT REPORT  (v{self.VERSION})")
        print("═" * 56)
        print(f"  Papers found        : {report.papers_found}")
        print(f"  Already in DB       : {report.papers_already_in_db}")
        print(f"  Read (new)          : {report.papers_read}")
        print(f"  Skipped (redundant) : {report.papers_skipped}")
        print(f"  New methods         : {report.new_methods}")
        print(f"  New datasets        : {report.new_datasets}")
        print(f"  New metrics         : {report.new_metrics}")
        print(f"  Coverage score      : {report.coverage_score:.2f}")
        if report.paper_ids_read:
            print(f"  Papers read         : {', '.join(report.paper_ids_read)}")
        print("═" * 56 + "\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Reader Agent v{ReaderAgent.VERSION}")
    parser.add_argument("--memory-dir", default=str(MEMORY_DIR))
    parser.add_argument("--db-path",    default=str(DB_PATH))
    parser.add_argument("--threshold",  type=float, default=COVERAGE_GAIN_THRESHOLD)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    agent = ReaderAgent(
        memory_dir = pathlib.Path(args.memory_dir),
        db_path    = pathlib.Path(args.db_path),
        verbose    = args.verbose,
    )
    state = {
        "session_id":              str(uuid.uuid4())[:8],
        "memory_dir":              args.memory_dir,
        "db_path":                 args.db_path,
        "coverage_gain_threshold": args.threshold,
    }
    updated = agent.run(state)
    report  = ReaderReport(**updated["reader_report"])
    agent.print_report(report)


if __name__ == "__main__":
    main()