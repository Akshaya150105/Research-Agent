import argparse
import json
import os
import re
import requests
import time
import sqlite3
import string
import unicodedata
from pathlib import Path
from typing import Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Normalisation (Tier-1 exact match -- always runs)
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("-", ""))


def normalize_entity_text(raw: str) -> str:
    """
    Tier-1 normalisation: unicode NFC -> lowercase -> strip punctuation
    (keep hyphens) -> collapse whitespace.
    Returns the canonical lookup key.
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# LLM-based entity clustering (Tier-2 -- runs once before ingestion)
# ---------------------------------------------------------------------------

CLUSTER_PROMPT = """You are a scientific entity deduplication assistant for NLP/ML research papers.

I will give you a list of entity strings of type "{entity_type}" extracted from multiple research papers.
Your job is to group ONLY TRUE SYNONYMS and SURFACE VARIANTS of the same concept.

STRICT RULES:
- ONLY merge if the strings refer to the exact same real-world concept/method/dataset.
- DO NOT merge concepts that are intentionally distinct:
    "Transformer" vs "4-layer Transformer" vs "Transformer (big)" -> KEEP SEPARATE (different model sizes)
    "self-attention" vs "Self-Attention (restricted)" -> KEEP SEPARATE (different mechanisms)
    "LSTM" vs "ConvLSTM" -> KEEP SEPARATE (different architectures)
    "machine translation" vs "machine translation tasks" -> KEEP SEPARATE (task vs task group)
- DO merge only obvious abbreviation expansions and trivial surface variants:
    "LSTM", "Long Short-Term Memory", "Long Short-Term Memory (LSTM) networks" -> MERGE
    "RNNs", "Recurrent Neural Networks", "Recurrent Neural Network" -> MERGE
    "BLEU", "BLEU score" -> MERGE
    "perplexity", "evaluation perplexity" -> MERGE (same metric, one is more specific label)
- Choose the MOST DESCRIPTIVE and COMPLETE form as the canonical name.
- If unsure whether to merge, DO NOT merge. Keeping them separate is safer.

Entity strings (type: {entity_type}):
{entity_list}

Respond ONLY with a valid JSON object mapping each input string to its canonical form.
Every input string must appear as a key. If a string stays as-is, map it to itself.
Example format:
{{
  "LSTM": "Long Short-Term Memory",
  "Long Short-Term Memory": "Long Short-Term Memory",
  "RNNs": "Recurrent Neural Networks",
  "Recurrent Neural Networks": "Recurrent Neural Networks"
}}
No explanation, no markdown, just the JSON object."""


def _call_llm_with_retry(
    url: str,
    payload: dict,
    max_retries: int = 6,
) -> str:
    """
    Call the Gemini API with exponential backoff on 429 rate limit errors.
    Returns the raw text response. Raises on non-retryable errors.
    """
    import time
    wait = 30  # start with 30s wait -- free tier needs more breathing room
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 429:
            print(f"[LLM Cluster] Rate limited. Waiting {wait}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            wait = min(wait * 2, 120)  # exponential backoff, cap at 2 min
            continue
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"LLM API still rate limited after {max_retries} retries.")


def llm_cluster_entities(
    entity_type: str,
    entity_strings: list[str],
    api_key: str,
    model: str = "gemini-2.5-flash",
    batch_size: int = 50,
) -> dict[str, str]:
    """
    Send entity strings to the LLM in batches and get back a merge map:
    {raw_string -> canonical_string}

    Batches of batch_size (default 50) to stay within token limits and
    avoid rate limits. Includes exponential backoff retry on 429 errors.
    Falls back to identity map (no merging) on unrecoverable errors.
    """
    import time

    if not entity_strings or not api_key:
        return {s: s for s in entity_strings}

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")

    merged: dict[str, str] = {}

    # Split into batches of batch_size
    batches = [entity_strings[i:i+batch_size]
               for i in range(0, len(entity_strings), batch_size)]

    for batch_idx, batch in enumerate(batches):
        if len(batches) > 1:
            print(f"[LLM Cluster]   Batch {batch_idx+1}/{len(batches)} "
                  f"({len(batch)} entities)...")

        entity_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        prompt = CLUSTER_PROMPT.format(
            entity_type=entity_type,
            entity_list=entity_list,
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
            },
        }

        try:
            raw = _call_llm_with_retry(url, payload)

            # Strip markdown code fences if present
            raw = re.sub(r"^```json\s*", "", raw.strip())
            raw = re.sub(r"```$", "", raw.strip())

            batch_map = json.loads(raw)

            # Validate: every input in this batch must be present
            for s in batch:
                if s not in batch_map:
                    batch_map[s] = s

            merged.update(batch_map)

        except Exception as e:
            print(f"[LLM Cluster] Warning: batch {batch_idx+1} failed: {e}")
            print("[LLM Cluster] Using identity map for this batch.")
            for s in batch:
                merged[s] = s

        # Small delay between batches to be polite to the API
        if batch_idx < len(batches) - 1:
            time.sleep(15)  # 15s between batches for free tier

    return merged


def build_merge_maps(
    json_paths: list[str],
    api_key: Optional[str],
) -> dict[str, dict[str, str]]:
    """
    Step 1: Collect all unique entity strings per type across all input papers.
    Step 2: Call LLM once per entity type to get canonical groupings.
    Returns: {entity_type: {raw_string -> canonical_string}}
    """
    # Collect all unique raw strings per type
    all_entities: dict[str, set[str]] = {
        "method": set(), "dataset": set(), "metric": set(), "task": set()
    }

    for json_path in json_paths:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        entity_index = data.get("entity_index", {})
        for etype, entities_dict in entity_index.items():
            if etype not in all_entities:
                continue
            for raw_key, mentions in entities_dict.items():
                best = max(mentions, key=lambda m: m.get("confidence", 0))
                all_entities[etype].add(best.get("text", raw_key))

    # Call LLM per type
    merge_maps: dict[str, dict[str, str]] = {}
    for etype, strings in all_entities.items():
        string_list = sorted(strings)  # sorted for determinism
        if not string_list:
            merge_maps[etype] = {}
            continue

        print(f"[LLM Cluster] Clustering {len(string_list)} {etype} entities...")
        merge_map = llm_cluster_entities(etype, string_list, api_key)

        # Log what got merged
        merged_count = sum(1 for k, v in merge_map.items() if k != v)
        print(f"[LLM Cluster] {etype}: {merged_count} entities will be merged into canonical forms")
        for k, v in sorted(merge_map.items()):
            if k != v:
                print(f"  '{k}' -> '{v}'")

        merge_maps[etype] = merge_map

    return merge_maps


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Papers
CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT PRIMARY KEY,
    title       TEXT,
    authors     TEXT,
    year        INTEGER,
    venue       TEXT,
    doi         TEXT,
    abstract    TEXT
);

-- Canonical entities
-- entity_id format: "{entity_type}::{canonical_key}"
-- raw_variants: all raw strings that mapped here (for "also referred to as...")
-- papers_seen_in: which papers mention this entity
CREATE TABLE IF NOT EXISTS entities (
    entity_id       TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    canonical_text  TEXT NOT NULL,
    raw_variants    TEXT,
    papers_seen_in  TEXT
);

-- Paper to entity typed edges
CREATE TABLE IF NOT EXISTS paper_entity_relationships (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    section     TEXT,
    confidence  REAL,
    UNIQUE(paper_id, entity_id, edge_type)
);

-- Limitation statements (things paper admits it cannot do)
-- Graph node_type: LimitationStatement | edge: has_limitation
CREATE TABLE IF NOT EXISTS limitation_statements (
    stmt_id           TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    text              TEXT NOT NULL,
    section           TEXT,
    confidence        REAL,
    entities_involved TEXT
);

-- Future work statements (things authors plan to do)
-- Graph node_type: FutureWork | edge: has_future_work
-- Kept separate: Gap Detector (Phase 6) queries these as direct gap signals
CREATE TABLE IF NOT EXISTS future_work_statements (
    stmt_id           TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    text              TEXT NOT NULL,
    section           TEXT,
    confidence        REAL,
    entities_involved TEXT
);

-- Claims (performance/comparative/methodological)
-- Not graph nodes yet -- Comparator Agent (Phase 4) adds contradiction edges
CREATE TABLE IF NOT EXISTS claims (
    claim_id          TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    claim_type        TEXT,
    description       TEXT,
    value             TEXT,
    confidence        REAL,
    section           TEXT,
    entities_involved TEXT
);

-- Session / action log
CREATE TABLE IF NOT EXISTS session_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL DEFAULT 'default', -- Added default to avoid null errors
    timestamp      TEXT DEFAULT (datetime('now')),
    agent          TEXT,
    action         TEXT,
    detail         TEXT,
    input_summary  TEXT DEFAULT '',                 -- Fixed: Permanent column
    output_summary TEXT DEFAULT '',                 -- Fixed: Permanent column
    confidence     REAL DEFAULT 1.0, 
    duration_ms    INTEGER DEFAULT 0 
);
"""

EDGE_TYPE_MAP = {
    "method":  "uses",
    "dataset": "evaluates_on",
    "metric":  "measures_with",
    "task":    "addresses",
}


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Reads Phase-1 paper JSONs and populates SQLite + NetworkX graph.

    Call build_merge_maps() first to get LLM-based entity clusters,
    then pass them into the constructor. If no merge_maps provided,
    falls back to Tier-1 exact match only.
    """

    def __init__(
        self,
        db_path: str = "shared_memory/research.db",
        gexf_path: str = "shared_memory/knowledge_graph.gexf",
        merge_maps: Optional[dict] = None,
    ):
        self.db_path = Path(db_path)
        self.gexf_path = Path(gexf_path)
        # merge_maps: {entity_type: {raw_text -> canonical_text}}
        self.merge_maps = merge_maps or {}

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.gexf_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

        if self.gexf_path.exists():
            self.graph = nx.read_gexf(str(self.gexf_path))
            self._log("KGBuilder", "load_graph",
                      f"Loaded existing graph: {self.graph.number_of_nodes()} nodes, "
                      f"{self.graph.number_of_edges()} edges")
        else:
            self.graph = nx.DiGraph()
            self._log("KGBuilder", "init_graph", "Created new empty graph")

        # In-memory canonical key set for fast exact-match lookup
        self._entity_key_cache: dict[str, set[str]] = {
            t: set() for t in ("method", "dataset", "metric", "task")
        }
        self._warm_cache()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def _warm_cache(self):
        cur = self.conn.execute("SELECT entity_id, entity_type FROM entities")
        for row in cur:
            etype = row["entity_type"]
            ckey = row["entity_id"].split("::", 1)[1]
            if etype in self._entity_key_cache:
                self._entity_key_cache[etype].add(ckey)

    def _log(self, agent: str, action: str, detail: str, input_sum: str = "", output_sum: str = ""):
        self.conn.execute(
            """INSERT INTO session_log 
               (session_id, agent, action, detail, input_summary, output_summary) 
               VALUES (?,?,?,?,?,?)""",
            ("kg_builder_session", agent, action, detail, input_sum, output_sum),
        )
        self.conn.commit()
        print(f"[{agent}] {action}: {detail}")

    # -------------------------------------------------------------------------

    def ingest_paper(self, json_path: str) -> str:
        """
        Load one Phase-1 claims_output JSON and populate everything.
        merge_maps must be pre-built before calling this.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        paper_id = data["paper_id"]
        self._log("KGBuilder", "ingest_start", f"paper_id={paper_id}")

        # Step 1: Paper metadata
        meta = data.get("metadata", {})
        self.conn.execute(
            """INSERT OR REPLACE INTO papers
               (paper_id, title, authors, year, venue, doi, abstract)
               VALUES (?,?,?,?,?,?,?)""",
            (
                paper_id,
                meta.get("title", ""),
                json.dumps(meta.get("authors", [])),
                meta.get("year"),
                meta.get("venue", ""),
                meta.get("doi", ""),
                meta.get("abstract", ""),
            ),
        )
        self.conn.commit()

        if paper_id not in self.graph:
            self.graph.add_node(
                paper_id,
                node_type="Paper",
                label=meta.get("title", paper_id),
                year=meta.get("year", 0),
                venue=meta.get("venue", ""),
            )

        # Steps 2 & 3: Entities + typed edges
        entity_index = data.get("entity_index", {})
        for etype, entities_dict in entity_index.items():
            if etype not in EDGE_TYPE_MAP:
                continue
            edge_label = EDGE_TYPE_MAP[etype]
            for raw_key, mentions in entities_dict.items():
                best_mention = max(mentions, key=lambda m: m.get("confidence", 0))
                raw_text = best_mention.get("text", raw_key)

                # Apply LLM merge map if available
                canonical_text = self.merge_maps.get(etype, {}).get(raw_text, raw_text)

                entity_id = self._resolve_entity(
                    raw_text=raw_text,
                    canonical_text=canonical_text,
                    entity_type=etype,
                    paper_id=paper_id,
                )
                self._add_paper_entity_edge(
                    paper_id=paper_id,
                    entity_id=entity_id,
                    edge_type=edge_label,
                    section=best_mention.get("section_heading", ""),
                    confidence=best_mention.get("confidence", 1.0),
                )

        # Step 4a: Limitation statements
        for idx, lim in enumerate(data.get("limitations", [])):
            stmt_id = f"{paper_id}::lim::{idx}"
            self.conn.execute(
                """INSERT OR IGNORE INTO limitation_statements
                   (stmt_id, paper_id, text, section, confidence, entities_involved)
                   VALUES (?,?,?,?,?,?)""",
                (
                    stmt_id, paper_id,
                    lim.get("text", ""),
                    lim.get("section_heading", ""),
                    lim.get("confidence", 1.0),
                    json.dumps(lim.get("entities_involved", [])),
                ),
            )
            if stmt_id not in self.graph:
                self.graph.add_node(stmt_id, node_type="LimitationStatement",
                                    label=lim.get("text", "")[:80])
            if not self.graph.has_edge(paper_id, stmt_id):
                self.graph.add_edge(paper_id, stmt_id, edge_type="has_limitation")

        # Step 4b: Future work statements
        for idx, fw in enumerate(data.get("future_work", [])):
            stmt_id = f"{paper_id}::fw::{idx}"
            self.conn.execute(
                """INSERT OR IGNORE INTO future_work_statements
                   (stmt_id, paper_id, text, section, confidence, entities_involved)
                   VALUES (?,?,?,?,?,?)""",
                (
                    stmt_id, paper_id,
                    fw.get("text", ""),
                    fw.get("section_heading", ""),
                    fw.get("confidence", 1.0),
                    json.dumps(fw.get("entities_involved", [])),
                ),
            )
            if stmt_id not in self.graph:
                self.graph.add_node(stmt_id, node_type="FutureWork",
                                    label=fw.get("text", "")[:80])
            if not self.graph.has_edge(paper_id, stmt_id):
                self.graph.add_edge(paper_id, stmt_id, edge_type="has_future_work")

        # Step 4c: Claims
        for idx, claim in enumerate(data.get("claims", [])):
            claim_id = f"{paper_id}::claim::{idx}"
            self.conn.execute(
                """INSERT OR IGNORE INTO claims
                   (claim_id, paper_id, claim_type, description, value,
                    confidence, section, entities_involved)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    claim_id, paper_id,
                    claim.get("claim_type", ""),
                    claim.get("description", ""),
                    str(claim.get("value", "")) if claim.get("value") else None,
                    claim.get("confidence", 1.0),
                    claim.get("section_heading", ""),
                    json.dumps(claim.get("entities_involved", [])),
                ),
            )
        self.conn.commit()

        summary = data.get("summary", {})
        self._log(
            "KGBuilder", "ingest_done",
            f"paper_id={paper_id} | entities={summary.get('total_llm_entities','?')} "
            f"| claims={summary.get('total_claims','?')} "
            f"| limitations={summary.get('total_limitations','?')} "
            f"| future_work={summary.get('total_future_work','?')}",
        )
        return paper_id

    # -------------------------------------------------------------------------

    def _resolve_entity(
        self,
        raw_text: str,
        canonical_text: str,
        entity_type: str,
        paper_id: str,
    ) -> str:
        """
        Map raw_text to a canonical entity_id using the pre-built canonical_text.
        canonical_text comes from the LLM merge map (or equals raw_text if no merge).
        entity_id = "{entity_type}::{normalize(canonical_text)}"
        """
        canonical_key = normalize_entity_text(canonical_text)
        entity_id = f"{entity_type}::{canonical_key}"

        if canonical_key in self._entity_key_cache.get(entity_type, set()):
            # Entity already exists -- just add this raw variant
            self._update_entity_variants(entity_id, raw_text, paper_id)
            return entity_id

        # New entity
        self.conn.execute(
            """INSERT INTO entities
               (entity_id, entity_type, canonical_text, raw_variants, papers_seen_in)
               VALUES (?,?,?,?,?)""",
            (
                entity_id, entity_type,
                canonical_text,          # LLM-chosen display form
                json.dumps([raw_text]),
                json.dumps([paper_id]),
            ),
        )
        self.conn.commit()
        self._entity_key_cache[entity_type].add(canonical_key)

        if entity_id not in self.graph:
            self.graph.add_node(
                entity_id,
                node_type=entity_type.capitalize(),
                label=canonical_text,    # use canonical as display label
            )
        return entity_id

    def _update_entity_variants(self, entity_id: str, raw_text: str, paper_id: str):
        row = self.conn.execute(
            "SELECT raw_variants, papers_seen_in FROM entities WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
        if not row:
            return
        variants = json.loads(row["raw_variants"])
        papers   = json.loads(row["papers_seen_in"])
        if raw_text not in variants:
            variants.append(raw_text)
        if paper_id not in papers:
            papers.append(paper_id)
        self.conn.execute(
            "UPDATE entities SET raw_variants=?, papers_seen_in=? WHERE entity_id=?",
            (json.dumps(variants), json.dumps(papers), entity_id),
        )
        self.conn.commit()

    def _add_paper_entity_edge(self, paper_id, entity_id, edge_type, section="", confidence=1.0):
        self.conn.execute(
            """INSERT OR IGNORE INTO paper_entity_relationships
               (paper_id, entity_id, edge_type, section, confidence)
               VALUES (?,?,?,?,?)""",
            (paper_id, entity_id, edge_type, section, confidence),
        )
        self.conn.commit()
        if not self.graph.has_edge(paper_id, entity_id):
            self.graph.add_edge(paper_id, entity_id,
                                edge_type=edge_type, section=section, confidence=confidence)

    def save_gexf(self):
        nx.write_gexf(self.graph, str(self.gexf_path))
        self._log("KGBuilder", "save_gexf",
                  f"Saved {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges -> {self.gexf_path}")

    def close(self):
        self.save_gexf()
        self.conn.close()

    # -------------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------------

    def papers_using_method(self, method_text: str) -> list:
        entity_id = f"method::{normalize_entity_text(method_text)}"
        cur = self.conn.execute(
            """SELECT p.paper_id, p.title, p.year, r.section, r.confidence
               FROM paper_entity_relationships r
               JOIN papers p ON r.paper_id = p.paper_id
               WHERE r.entity_id = ? AND r.edge_type = 'uses'""",
            (entity_id,),
        )
        return [dict(row) for row in cur]

    def papers_on_dataset(self, dataset_text: str) -> list:
        entity_id = f"dataset::{normalize_entity_text(dataset_text)}"
        cur = self.conn.execute(
            """SELECT p.paper_id, p.title, p.year
               FROM paper_entity_relationships r
               JOIN papers p ON r.paper_id = p.paper_id
               WHERE r.entity_id = ? AND r.edge_type = 'evaluates_on'""",
            (entity_id,),
        )
        return [dict(row) for row in cur]

    def get_graph_summary(self) -> dict:
        node_counts = {}
        for ntype in ("Paper", "Method", "Dataset", "Metric", "Task",
                      "LimitationStatement", "FutureWork"):
            node_counts[ntype] = sum(
                1 for _, d in self.graph.nodes(data=True)
                if d.get("node_type") == ntype
            )
        edge_counts: dict = {}
        for _, _, d in self.graph.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_counts[et] = edge_counts.get(et, 0) + 1
        return {"node_counts": node_counts, "edge_counts": edge_counts}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 -- Knowledge Graph Population")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to Phase-1 claims_output JSON files")
    parser.add_argument("--db",   default="shared_memory/research.db")
    parser.add_argument("--gexf", default="shared_memory/knowledge_graph.gexf")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key for LLM entity clustering. "
                             "Can also set GEMINI_API_KEY env variable.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] No API key provided. Running without LLM clustering (exact match only).")

    # Step 1: LLM clustering across all papers BEFORE ingestion
    merge_maps = build_merge_maps(args.inputs, api_key)

    # Step 2: Ingest each paper using the merge maps
    builder = KnowledgeGraphBuilder(
        db_path=args.db,
        gexf_path=args.gexf,
        merge_maps=merge_maps,
    )

    for json_file in args.inputs:
        builder.ingest_paper(json_file)

    summary = builder.get_graph_summary()
    print("\n-- Node counts --")
    for k, v in summary["node_counts"].items():
        print(f"  {k:25s}: {v}")
    print("\n-- Edge counts --")
    for k, v in summary["edge_counts"].items():
        print(f"  {k:25s}: {v}")

    builder.close()
    print(f"\n  DB   -> {args.db}")
    print(f"  GEXF -> {args.gexf}")


if __name__ == "__main__":
    main()