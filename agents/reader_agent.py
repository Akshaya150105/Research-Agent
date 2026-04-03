"""
reader_agent.py  v2.0.0
========================
Person 3 — AI Agents  |  Branch: feat/agents  |  Folder: agents/

WHAT CHANGED FROM v1.1.0:
--------------------------
The reader now does the FULL extraction pipeline for each new PDF before
deciding read/skip. Previously it only scanned memory/ for already-processed
claims_output.json files — meaning you had to run GROBID, NER, and the
claim extractor manually on the command line before the reader could do anything.

Now the reader owns the full flow:

  For each PDF in papers_dir/ (default: Data/):
    1. Check if memory/{paper_folder}/claims_output.json already exists.
       If yes → skip extraction (already processed), go straight to step 5.
    2. Run GROBID parser → memory/{paper_folder}/sections.json
    3. Run SciBERT NER  → memory/{paper_folder}/enriched_entities.json
    4. Run LLM Claim Extractor (Ollama qwen2.5) → memory/{paper_folder}/claims_output.json
    5. Compute coverage_gain from claims_output.json entities vs what's already in DB.
    6. Read-or-skip decision based on coverage_gain threshold.
    7. Write paper record + entities to SQLite.

CLAIM EXTRACTOR CHANGE:
  Old command used --api-key (Gemini backend).
  New command uses --ollama-host (Ollama/qwen2.5 backend):
    python -m claim_extractor.cli \\
        --grobid-dir memory/{folder}/ \\
        --ner-dir memory/{folder}/ \\
        --ollama-host http://localhost:11434

PDF FOLDER NAMING:
  PDFs are placed in papers_dir/ (default: Data/).
  Each PDF gets its own subfolder in memory/:
    Data/attention_is_all_you_need.pdf
      → memory/attention_is_all_you_need_2023/
         ├── sections.json          (GROBID output)
         ├── enriched_entities.json (NER output)
         └── claims_output.json     (LLM claim extractor output)

  The folder name is derived from the PDF filename (no extension).
  If GROBID extracts a year from the paper, it's appended: {name}_{year}.

WHAT THE READER DOES NOT DO:
  The reader does NOT run GROBID docker itself — it calls the CLI
  via subprocess. You must have GROBID running on port 8070:
    docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0

  If GROBID is not reachable, extraction is skipped for that paper
  and the reader logs a warning. The paper is not added to the DB.

Usage (standalone):
  python agents/reader_agent.py --papers-dir Data/ --verbose
  python agents/reader_agent.py --papers-dir Data/ --no-extraction --verbose
  python agents/reader_agent.py --papers-dir Data/ --ollama-host http://localhost:11434 --verbose

Usage (from Planner / LangGraph):
  from reader_agent import ReaderAgent
  agent = ReaderAgent(verbose=True)
  updated_state = agent.run(state)
  # state can include:
  #   papers_dir    (str) — where raw PDFs live, default "Data"
  #   memory_dir    (str) — where output folders go, default "memory"
  #   ollama_host   (str) — Ollama URL, default "http://localhost:11434"
  #   no_extraction (bool)— if True, skip pipeline, only scan existing JSONs
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

_AGENTS_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from entity_resolver import get_typed_entities

# ── defaults ──────────────────────────────────────────────────
PAPERS_DIR              = pathlib.Path("data_1/papers")
MEMORY_DIR              = pathlib.Path("memory")
SHARED_MEMORY           = pathlib.Path("shared_memory")
DB_PATH                 = SHARED_MEMORY / "research.db"
COVERAGE_GAIN_THRESHOLD = 0.10
DEFAULT_OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GROBID_URL              = "http://localhost:8070"


# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class PaperRecord:
    paper_id:      str
    title:         str
    authors:       str
    year:          Optional[int]
    venue:         str
    abstract:      str
    source_path:   str
    n_claims:      int
    n_entities:    int
    n_limitations: int
    coverage_gain: float
    action:        str
    processed_at:  str


@dataclass
class ReaderReport:
    agent:                str   = "reader_agent"
    agent_version:        str   = "2.0.0"
    session_id:           str   = ""
    papers_found:         int   = 0
    papers_extracted:     int   = 0   # new: ran the full pipeline
    papers_skipped_extract: int = 0   # new: already had claims_output.json
    papers_read:          int   = 0
    papers_skipped:       int   = 0
    papers_already_in_db: int   = 0
    extraction_failed:    int   = 0   # new: pipeline failed for this paper
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
    for alter in [
        "ALTER TABLE entities ADD COLUMN text_norm TEXT DEFAULT ''",
        "ALTER TABLE session_log ADD COLUMN input_summary TEXT DEFAULT ''",
        "ALTER TABLE session_log ADD COLUMN output_summary TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(alter)
        except Exception:
            pass
    conn.commit()


# ─────────────────────────────────────────────────────────────
#  EXTRACTION PIPELINE RUNNER
#  Wraps the 4 CLI commands as subprocess calls.
#  Each step checks if its output already exists before running.
# ─────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Runs GROBID → NER → Claim Extractor → KG Population for one PDF.

    Each step is idempotent: if its output file already exists,
    it is skipped. This means re-running the reader on the same
    paper set is safe and fast.

    Subprocess approach:
      We call the existing CLI entry points via subprocess rather than
      importing the modules directly. This keeps the reader decoupled from
      Person 1's and 2's code — they can change their internals without
      breaking the reader, as long as the CLI interface stays the same.

    Error handling:
      If any step fails (non-zero return code or exception), the pipeline
      stops for that paper and returns False. The paper is not added to the DB.
      The failure is logged to the react trace.
    """

    def __init__(
        self,
        pdf_path:     pathlib.Path,
        memory_dir:   pathlib.Path,
        ollama_host:  str,
        db_path:      pathlib.Path,
        gexf_path:    pathlib.Path,
        verbose:      bool = False,
    ):
        self.pdf_path    = pdf_path
        self.memory_dir  = memory_dir
        self.ollama_host = ollama_host
        self.db_path     = db_path
        self.gexf_path   = gexf_path
        self.verbose     = verbose

        # Output folder name = PDF filename without extension
        self.folder_name = pdf_path.stem
        self.output_dir  = memory_dir / self.folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sections_json(self) -> pathlib.Path:
        return self.output_dir / "sections.json"

    @property
    def enriched_entities_json(self) -> pathlib.Path:
        return self.output_dir / "enriched_entities.json"

    @property
    def claims_output_json(self) -> pathlib.Path:
        return self.output_dir / "claims_output.json"

    def _run_cmd(self, cmd: list[str], step_name: str) -> bool:
        """
        Run a subprocess command. Returns True on success, False on failure.
        Prints stdout/stderr if verbose.
        """
        if self.verbose:
            print(f"    [pipeline:{step_name}] Running: {' '.join(str(c) for c in cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                cwd=str(_PROJECT_ROOT),   # run from project root so module paths resolve
            )
            if result.returncode != 0:
                print(
                    f"    [pipeline:{step_name}] ⚠ Failed (exit {result.returncode})",
                    file=sys.stderr,
                )
                if result.stderr and not self.verbose:
                    # Always show first 300 chars of stderr on failure even in quiet mode
                    print(f"    stderr: {result.stderr[:300]}", file=sys.stderr)
                return False

            if self.verbose:
                print(f"    [pipeline:{step_name}] ✅ Done")
            return True

        except FileNotFoundError as e:
            print(f"    [pipeline:{step_name}] ⚠ Command not found: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"    [pipeline:{step_name}] ⚠ Exception: {e}", file=sys.stderr)
            return False

    def run(self) -> bool:
        """
        Run all 4 steps. Returns True if claims_output.json exists at the end.

        Steps:
          1. GROBID parser   → sections.json
          2. SciBERT NER     → enriched_entities.json
          3. Claim extractor → claims_output.json  (uses Ollama)
          4. KG population   → research.db + knowledge_graph.gexf

        Each step is skipped if its output already exists.
        Step 4 (KG) is run even if claims_output.json already existed —
        it's idempotent and ensures the graph stays current.
        """

        # ── Step 1: GROBID ────────────────────────────────────
        if self.sections_json.exists():
            if self.verbose:
                print(f"    [pipeline:grobid] Skipping — {self.sections_json.name} exists")
        else:
            ok = self._run_cmd(
                [
                    sys.executable, "-m", "grobid_parser.cli",
                    str(self.pdf_path),
                    "-o", str(self.output_dir),
                ],
                step_name="grobid",
            )
            if not ok or not self.sections_json.exists():
                print(
                    f"    [pipeline] ⚠ GROBID failed for '{self.pdf_path.name}'. "
                    "Is GROBID running on port 8070? "
                    "Run: docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0",
                    file=sys.stderr,
                )
                return False

        # ── Step 2: SciBERT NER ───────────────────────────────
        if self.enriched_entities_json.exists():
            if self.verbose:
                print(f"    [pipeline:ner] Skipping — {self.enriched_entities_json.name} exists")
        else:
            ok = self._run_cmd(
                [
                    sys.executable, "-m", "ner_pipeline.cli",
                    str(self.output_dir),
                ],
                step_name="ner",
            )
            if not ok or not self.enriched_entities_json.exists():
                print(
                    f"    [pipeline] ⚠ NER failed for '{self.pdf_path.name}'.",
                    file=sys.stderr,
                )
                return False

        # ── Step 3: LLM Claim Extractor (Ollama) ─────────────
        if self.claims_output_json.exists():
            if self.verbose:
                print(f"    [pipeline:claims] Skipping — {self.claims_output_json.name} exists")
        else:
            ok = self._run_cmd(
                [
                    sys.executable, "-m", "claim_extractor.cli",
                    "--grobid-dir", str(self.output_dir),
                    "--ner-dir",    str(self.output_dir),
                    "--ollama-host", self.ollama_host,
                ],
                step_name="claims",
            )
            if not ok or not self.claims_output_json.exists():
                print(
                    f"    [pipeline] ⚠ Claim extraction failed for '{self.pdf_path.name}'. "
                    f"Is Ollama running at {self.ollama_host}?",
                    file=sys.stderr,
                )
                return False

        # ── Step 4: KG Population ─────────────────────────────
        # Always run this — it's idempotent (INSERT OR IGNORE/REPLACE)
        # and ensures the graph reflects the latest claims_output.json.
        if self.db_path.exists() or True:   # always attempt
            ok = self._run_cmd(
                [
                    sys.executable, "kg_population/kg_population.py",
                    "--inputs",  str(self.claims_output_json),
                    "--db",      str(self.db_path),
                    "--gexf",    str(self.gexf_path),
                ],
                step_name="kg",
            )
            # KG failure is non-fatal — claims_output.json still usable
            if not ok and self.verbose:
                print("    [pipeline:kg] ⚠ KG population failed — continuing anyway")

        return self.claims_output_json.exists()


# ─────────────────────────────────────────────────────────────
#  DB HELPERS
# ─────────────────────────────────────────────────────────────

def _already_processed(paper_id: str, conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone() is not None


def _known_entities(conn: sqlite3.Connection) -> dict[str, set]:
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
    meta = paper.get("metadata", {})
    ab   = meta.get("abstract", "")
    if ab:
        return ab[:1000]
    s = paper.get("summary", {})
    if isinstance(s, str) and s:
        return s[:1000]
    sections = paper.get("sections", {})
    if isinstance(sections, dict):
        ab = sections.get("abstract", "")
        if ab:
            return ab[:1000]
    return ""


def _write_paper_to_db(paper: dict, record: PaperRecord, conn: sqlite3.Connection) -> None:
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
    conn: sqlite3.Connection, session_id: str, agent: str,
    action: str, input_summary: str, output_summary: str,
    confidence: float = 1.0,
) -> None:
    conn.execute(
        """INSERT INTO session_log
           (session_id, timestamp, agent, action,
            input_summary, output_summary, confidence, duration_ms)
           VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, datetime.datetime.now(datetime.timezone.utc).isoformat(),
         agent, action, input_summary, output_summary, confidence, 0),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────
#  READER AGENT
# ─────────────────────────────────────────────────────────────

class ReaderAgent:
    """
    The Reader Agent owns the full paper ingestion pipeline.

    For each PDF in papers_dir/:
      1. Run GROBID → NER → Claim Extractor → KG (if not already done)
      2. Compute coverage gain from the resulting claims_output.json
      3. Decide read/skip based on gain threshold
      4. Write paper + entities to SQLite

    The planner calls agent.run(state). State flows through the LangGraph.
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        papers_dir:   pathlib.Path = PAPERS_DIR,
        memory_dir:   pathlib.Path = MEMORY_DIR,
        db_path:      pathlib.Path = DB_PATH,
        ollama_host:  str          = DEFAULT_OLLAMA_HOST,
        no_extraction: bool        = False,
        verbose:      bool         = False,
    ):
        self.papers_dir    = papers_dir
        self.memory_dir    = memory_dir
        self.db_path       = db_path
        self.ollama_host   = ollama_host
        self.no_extraction = no_extraction
        self.verbose       = verbose
        self.trace:        list[str] = []

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
        Called by the Planner. Reads state, runs pipeline, returns updated state.

        State keys read:
          session_id              (str)
          papers_dir              (str) — where raw PDFs live, default "Data"
          memory_dir              (str) — where output folders go, default "memory"
          db_path                 (str) — optional, overrides default DB path
          ollama_host             (str) — Ollama URL for claim extractor
          coverage_gain_threshold (float)
          no_extraction           (bool) — skip pipeline, only scan existing JSONs
          use_llm                 (bool) — if False, disables claim extraction LLM call

        State keys written:
          reader_report    (dict)
          coverage_score   (float)
        """
        session_id    = state.get("session_id", str(uuid.uuid4())[:8])
        papers_dir    = pathlib.Path(state.get("papers_dir",  str(self.papers_dir)))
        mem_dir       = pathlib.Path(state.get("memory_dir",  str(self.memory_dir)))
        db_path       = pathlib.Path(state.get("db_path",     str(self.db_path)))
        ollama_host   = state.get("ollama_host",  self.ollama_host)
        threshold     = float(state.get("coverage_gain_threshold", COVERAGE_GAIN_THRESHOLD))
        # no_extraction=True means: don't run the pipeline, just scan existing files
        # This is used for testing or when papers are already pre-processed
        no_extract    = state.get("no_extraction", self.no_extraction)
        # If use_llm=False, also skip extraction (extraction requires Ollama)
        if not state.get("use_llm", True):
            no_extract = True

        report = self._run_internal(
            session_id, papers_dir, mem_dir, db_path,
            ollama_host, threshold, no_extract,
        )
        return {**state, "reader_report": asdict(report)}

    # ── Core logic ────────────────────────────────────────────

    def _run_internal(
        self,
        session_id:  str,
        papers_dir:  pathlib.Path,
        memory_dir:  pathlib.Path,
        db_path:     pathlib.Path,
        ollama_host: str,
        threshold:   float,
        no_extract:  bool,
    ) -> ReaderReport:

        report = ReaderReport(session_id=session_id)
        self.trace = []

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        _ensure_schema(conn)

        gexf_path = db_path.parent / "knowledge_graph.gexf"

        # ── Step 1: Discover PDFs ──────────────────────────────
        self._think("Discovering PDFs in papers_dir/")
        pdf_files = self._discover_pdfs(papers_dir)
        self._observe(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

        if not pdf_files:
            # Fallback: if no PDFs found, look for existing memory/ folders
            self._observe(
                f"No PDFs in '{papers_dir}'. "
                "Falling back to scanning memory/ for existing claims_output.json files."
            )
            report.react_trace = self.trace
            return self._scan_memory_only(
                memory_dir, conn, session_id, threshold, report, db_path
            )

        report.papers_found = len(pdf_files)

        # ── Step 2: Load known entities snapshot ─────────────
        self._act("Querying SQLite for known entities")
        known = _known_entities(conn)
        self._observe(
            f"DB entities: {len(known['method'])} methods, "
            f"{len(known['dataset'])} datasets, "
            f"{len(known['metric'])} metrics"
        )

        # ── Step 3: Process each PDF ───────────────────────────
        for pdf_path in pdf_files:
            self._process_one_pdf(
                pdf_path, memory_dir, db_path, gexf_path,
                ollama_host, conn, known, session_id,
                threshold, no_extract, report,
            )
            # Refresh known after each paper so next paper's gain is relative
            known = _known_entities(conn)

        # ── Step 4: Coverage score ────────────────────────────
        report.coverage_score = round(
            report.papers_read / max(report.papers_found, 1), 3
        )

        self._observe(
            f"Reader done — "
            f"extracted={report.papers_extracted}, "
            f"read={report.papers_read}, "
            f"skipped={report.papers_skipped}, "
            f"already_in_db={report.papers_already_in_db}, "
            f"failed={report.extraction_failed}, "
            f"coverage={report.coverage_score:.2f}"
        )

        _log_action(
            conn, session_id, "reader_agent", "session_complete",
            f"pdfs={report.papers_found}, extracted={report.papers_extracted}",
            f"read={report.papers_read}, coverage={report.coverage_score:.2f}",
        )

        conn.close()
        report.react_trace = self.trace
        return report

    def _discover_pdfs(self, papers_dir: pathlib.Path) -> list[pathlib.Path]:
        """Find all PDF files in papers_dir (non-recursive)."""
        if not papers_dir.exists():
            return []
        return sorted(papers_dir.glob("*.pdf"))

    def _find_claims_json(
        self,
        pdf_path:   pathlib.Path,
        memory_dir: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        """
        Find the claims_output.json for a given PDF.

        Looks in:
          memory/{pdf_stem}/claims_output.json             (exact match)
          memory/{pdf_stem}_*/claims_output.json           (with year suffix)
        """
        stem = pdf_path.stem

        # Direct match
        direct = memory_dir / stem / "claims_output.json"
        if direct.exists():
            return direct

        # Match with year suffix: e.g. memory/attention_is_all_you_need_2017/
        for folder in sorted(memory_dir.glob(f"{stem}*")):
            if not folder.is_dir():
                continue
            p = folder / "claims_output.json"
            if p.exists():
                return p

        return None

    def _process_one_pdf(
        self,
        pdf_path:    pathlib.Path,
        memory_dir:  pathlib.Path,
        db_path:     pathlib.Path,
        gexf_path:   pathlib.Path,
        ollama_host: str,
        conn:        sqlite3.Connection,
        known:       dict[str, set],
        session_id:  str,
        threshold:   float,
        no_extract:  bool,
        report:      ReaderReport,
    ) -> None:

        self._think(f"Processing '{pdf_path.name}'")

        # Check if claims_output.json already exists for this PDF
        existing_claims = self._find_claims_json(pdf_path, memory_dir)

        if existing_claims:
            self._observe(f"  '{pdf_path.name}' already has {existing_claims}  — skipping extraction")
            report.papers_skipped_extract += 1
            claims_json = existing_claims
        elif no_extract:
            self._observe(
                f"  '{pdf_path.name}' has no claims_output.json "
                "and no_extraction=True — skipping this paper"
            )
            report.extraction_failed += 1
            return
        else:
            # ── Run the full extraction pipeline ──────────────
            self._act(f"  Running extraction pipeline: GROBID → NER → Claims → KG")
            pipeline = ExtractionPipeline(
                pdf_path    = pdf_path,
                memory_dir  = memory_dir,
                ollama_host = ollama_host,
                db_path     = db_path,
                gexf_path   = gexf_path,
                verbose     = self.verbose,
            )
            success = pipeline.run()

            if not success:
                self._observe(f"  '{pdf_path.name}' extraction FAILED — skipping")
                report.extraction_failed += 1
                return

            claims_json = pipeline.claims_output_json
            report.papers_extracted += 1
            self._observe(f"  '{pdf_path.name}' extraction complete → {claims_json}")

        # ── Load claims_output.json ───────────────────────────
        try:
            with open(claims_json, encoding="utf-8") as f:
                paper = json.load(f)
        except Exception as e:
            self._observe(f"  ⚠ Cannot load {claims_json}: {e}")
            report.extraction_failed += 1
            return

        paper_id = paper.get("paper_id") or claims_json.parent.name
        meta     = paper.get("metadata", {})

        # Already in DB?
        if _already_processed(paper_id, conn):
            self._observe(f"  '{paper_id}' already in SQLite — skipping DB write")
            report.papers_already_in_db += 1
            return

        # ── Coverage gain ─────────────────────────────────────
        gain, new_counts = _compute_coverage_gain(paper, known)
        self._observe(
            f"  '{paper_id}' gain={gain:.2f} "
            f"(+{new_counts['method']} methods, "
            f"+{new_counts['dataset']} datasets, "
            f"+{new_counts['metric']} metrics)"
        )

        # ── READ-OR-SKIP DECISION ─────────────────────────────
        # TODO (Phase 8 / Person 4):
        #   state_vec = [gain, len(ents), n_claims, coverage_so_far, ...]
        #   action = reader_policy.predict(state_vec)
        db_is_empty = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        action      = "read" if (gain >= threshold or db_is_empty) else "skip"
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
            source_path   = str(claims_json),
            n_claims      = len(claims),
            n_entities    = len(ents),
            n_limitations = len(lims),
            coverage_gain = gain,
            action        = action,
            processed_at  = now,
        )

        _write_paper_to_db(paper, record, conn)

        if action == "read":
            report.papers_read    += 1
            report.paper_ids_read.append(paper_id)
            report.new_methods    += new_counts["method"]
            report.new_datasets   += new_counts["dataset"]
            report.new_metrics    += new_counts["metric"]
        else:
            report.papers_skipped += 1

        _log_action(
            conn, session_id, "reader_agent", action,
            f"pdf={pdf_path.name}, paper_id={paper_id}",
            f"gain={gain:.2f}, new={sum(new_counts.values())}",
            confidence=gain,
        )

    def _scan_memory_only(
        self,
        memory_dir: pathlib.Path,
        conn:       sqlite3.Connection,
        session_id: str,
        threshold:  float,
        report:     ReaderReport,
        db_path:    pathlib.Path,
    ) -> ReaderReport:
        """
        Fallback: no PDFs found in papers_dir.
        Scan memory/ for existing claims_output.json files and process them.
        This preserves the v1.1.0 behaviour for users who pre-process papers
        manually and put them directly in memory/.
        """
        self._think("Fallback: scanning memory/ for existing claims_output.json")
        known = _known_entities(conn)

        paper_files = []
        if memory_dir.exists():
            for folder in sorted(memory_dir.iterdir()):
                if not folder.is_dir():
                    continue
                p = folder / "claims_output.json"
                if p.exists():
                    paper_files.append(p)

        report.papers_found = len(paper_files)
        self._observe(f"Found {len(paper_files)} existing claims_output.json file(s)")

        for paper_path in paper_files:
            try:
                with open(paper_path, encoding="utf-8") as f:
                    paper = json.load(f)
            except Exception as e:
                self._observe(f"⚠ Cannot load {paper_path}: {e}")
                continue

            paper_id = paper.get("paper_id") or paper_path.parent.name
            meta     = paper.get("metadata", {})
            self._think(f"Evaluating '{paper_id}' (memory scan)")

            if _already_processed(paper_id, conn):
                self._observe(f"  '{paper_id}' already in DB")
                report.papers_already_in_db += 1
                known = _known_entities(conn)
                continue

            gain, new_counts = _compute_coverage_gain(paper, known)
            db_is_empty = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
            action      = "read" if (gain >= threshold or db_is_empty) else "skip"

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
            _write_paper_to_db(paper, record, conn)

            if action == "read":
                report.papers_read    += 1
                report.paper_ids_read.append(paper_id)
                report.new_methods    += new_counts["method"]
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
            known = _known_entities(conn)

        report.coverage_score = round(
            report.papers_read / max(report.papers_found, 1), 3
        )
        _log_action(
            conn, session_id, "reader_agent", "session_complete",
            f"papers_found={report.papers_found}",
            f"read={report.papers_read}, coverage={report.coverage_score:.2f}",
        )
        conn.close()
        report.react_trace = self.trace
        return report

    # ── Standalone CLI ────────────────────────────────────────

    def print_report(self, report: ReaderReport) -> None:
        print("\n" + "═" * 60)
        print(f"  READER AGENT REPORT  (v{self.VERSION})")
        print("═" * 60)
        print(f"  PDFs found           : {report.papers_found}")
        print(f"  Extraction ran       : {report.papers_extracted}")
        print(f"  Already extracted    : {report.papers_skipped_extract}")
        print(f"  Extraction failed    : {report.extraction_failed}")
        print(f"  Already in DB        : {report.papers_already_in_db}")
        print(f"  Read (added to DB)   : {report.papers_read}")
        print(f"  Skipped (redundant)  : {report.papers_skipped}")
        print(f"  New methods          : {report.new_methods}")
        print(f"  New datasets         : {report.new_datasets}")
        print(f"  New metrics          : {report.new_metrics}")
        print(f"  Coverage score       : {report.coverage_score:.2f}")
        if report.paper_ids_read:
            print(f"  Papers read          : {', '.join(report.paper_ids_read)}")
        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=f"Reader Agent v{ReaderAgent.VERSION} — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline — reads PDFs from Data/, extracts, loads to DB
  python agents/reader_agent.py --papers-dir Data/ --verbose

  # Custom Ollama host (Kaggle ngrok tunnel)
  python agents/reader_agent.py --papers-dir Data/ \\
      --ollama-host https://<ngrok>.ngrok-free.app --verbose

  # Skip extraction — only scan existing memory/ JSONs (v1.x behaviour)
  python agents/reader_agent.py --no-extraction --verbose

  # Different memory dir
  python agents/reader_agent.py --papers-dir data_1/ --memory-dir memory/ --verbose
        """,
    )
    parser.add_argument("--papers-dir",   default=str(PAPERS_DIR),
                        help=f"Folder containing raw PDF files (default: {PAPERS_DIR})")
    parser.add_argument("--memory-dir",   default=str(MEMORY_DIR),
                        help=f"Output folder for extracted JSONs (default: {MEMORY_DIR})")
    parser.add_argument("--db-path",      default=str(DB_PATH))
    parser.add_argument("--ollama-host",  default=DEFAULT_OLLAMA_HOST,
                        help=f"Ollama base URL for claim extractor (default: {DEFAULT_OLLAMA_HOST})")
    parser.add_argument("--threshold",    type=float, default=COVERAGE_GAIN_THRESHOLD,
                        help=f"Coverage gain threshold for read/skip (default: {COVERAGE_GAIN_THRESHOLD})")
    parser.add_argument("--no-extraction", action="store_true",
                        help="Skip pipeline steps — only scan existing memory/ JSONs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    agent = ReaderAgent(
        papers_dir    = pathlib.Path(args.papers_dir),
        memory_dir    = pathlib.Path(args.memory_dir),
        db_path       = pathlib.Path(args.db_path),
        ollama_host   = args.ollama_host,
        no_extraction = args.no_extraction,
        verbose       = args.verbose,
    )
    state = {
        "session_id":              str(uuid.uuid4())[:8],
        "papers_dir":              args.papers_dir,
        "memory_dir":              args.memory_dir,
        "db_path":                 args.db_path,
        "ollama_host":             args.ollama_host,
        "coverage_gain_threshold": args.threshold,
        "no_extraction":           args.no_extraction,
        "use_llm":                 True,
    }
    updated = agent.run(state)
    report  = ReaderReport(**updated["reader_report"])
    agent.print_report(report)


if __name__ == "__main__":
    main()