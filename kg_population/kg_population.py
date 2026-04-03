"""
kg_population.py  v2.1.0
=========================
Phase 2 — Knowledge Graph Population

CHANGES FROM v2.0.0:
  - Imports schema from shared_schema.py instead of defining its own.
    This eliminates the mismatch where reader and KG had different
    entities table structures.

  - entities table is now the KG-canonical format:
      entity_id = "method::long short-term memory"
      canonical_text, raw_variants (JSON), papers_seen_in (JSON)
    Reader no longer writes to entities directly — it calls kg_population
    via subprocess, which owns entity writes.

  - papers table now has all reader columns (doi, source_path, n_claims,
    etc.) via shared_schema. KG writes only its own fields (paper_id,
    title, authors, year, venue, doi, abstract) using INSERT OR IGNORE so
    it never overwrites reader's richer fields.

  - session_log INSERT now includes detail column (KG logs go here) and
    leaves input_summary/output_summary empty (reader's fields).

Usage (unchanged):
  python kg_population/kg_population.py \\
      --inputs memory/paper1/claims_output.json \\
      --db shared_memory/research.db \\
      --gexf shared_memory/knowledge_graph.gexf \\
      --ollama-host https://<ngrok>.ngrok-free.app
"""

import argparse
import json
import os
import re
import requests
import string
import unicodedata
from pathlib import Path
from typing import Optional
import sqlite3

import networkx as nx

# Import unified schema — single source of truth
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared_schema import FULL_SCHEMA, ensure_schema, EDGE_TYPE_MAP


# ─────────────────────────────────────────────────────────────
#  NORMALISATION (Tier-1)
# ─────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("-", ""))


def normalize_entity_text(raw: str) -> str:
    """
    Tier-1 normalisation: unicode NFC → lowercase → strip punctuation
    (keep hyphens) → collapse whitespace.
    Returns the canonical lookup key used in entity_id.
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
#  LLM ENTITY CLUSTERING (Tier-2, Ollama)
# ─────────────────────────────────────────────────────────────

CLUSTER_PROMPT = """You are a scientific entity deduplication assistant for NLP/ML research papers.

I will give you a list of entity strings of type "{entity_type}" extracted from multiple research papers.
Your job is to group ONLY TRUE SYNONYMS and SURFACE VARIANTS of the same concept.

STRICT RULES:
- ONLY merge if the strings refer to the exact same real-world concept/method/dataset.
- DO NOT merge concepts that are intentionally distinct:
    "Transformer" vs "4-layer Transformer" vs "Transformer (big)" -> KEEP SEPARATE
    "self-attention" vs "Self-Attention (restricted)" -> KEEP SEPARATE
    "LSTM" vs "ConvLSTM" -> KEEP SEPARATE
    "machine translation" vs "machine translation tasks" -> KEEP SEPARATE
- DO merge only obvious abbreviation expansions and trivial surface variants:
    "LSTM", "Long Short-Term Memory", "Long Short-Term Memory (LSTM) networks" -> MERGE
    "RNNs", "Recurrent Neural Networks", "Recurrent Neural Network" -> MERGE
    "BLEU", "BLEU score" -> MERGE
    "perplexity", "evaluation perplexity" -> MERGE
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


def _call_ollama(prompt: str, ollama_host: str, model: str = "qwen2.5") -> str:
    url = f"{ollama_host}/api/generate"
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 4096},
    }
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama] Error: {e}")
        return "{}"


def llm_cluster_entities(
    entity_type:    str,
    entity_strings: list[str],
    ollama_host:    str,
    model:          str = "qwen2.5",
    batch_size:     int = 40,
) -> dict[str, str]:
    if not entity_strings or not ollama_host:
        return {s: s for s in entity_strings}

    merged: dict[str, str] = {}
    batches = [
        entity_strings[i:i + batch_size]
        for i in range(0, len(entity_strings), batch_size)
    ]

    for batch_idx, batch in enumerate(batches):
        print(f"[Ollama Cluster] Batch {batch_idx+1}/{len(batches)} ({len(batch)} entities)...")
        prompt = CLUSTER_PROMPT.format(
            entity_type=entity_type,
            entity_list="\n".join(f"{i+1}. {s}" for i, s in enumerate(batch)),
        )
        raw = _call_ollama(prompt, ollama_host, model)
        try:
            raw_clean = re.sub(r"^```json\s*", "", raw.strip())
            raw_clean = re.sub(r"```$", "", raw_clean.strip())
            batch_map = json.loads(raw_clean)
            for s in batch:
                if s not in batch_map:
                    batch_map[s] = s
            merged.update(batch_map)
        except Exception as e:
            print(f"[Ollama Cluster] Batch {batch_idx+1} parse failed ({e}) — identity fallback")
            for s in batch:
                merged[s] = s

    return merged


def build_merge_maps(
    json_paths:  list[str],
    ollama_host: Optional[str],
) -> dict[str, dict[str, str]]:
    """
    Collect all unique entity strings per type across all papers,
    then cluster via LLM once per type.
    Returns {entity_type: {raw_string -> canonical_string}}.
    """
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

    merge_maps: dict[str, dict[str, str]] = {}
    for etype, strings in all_entities.items():
        string_list = sorted(strings)
        if not string_list:
            merge_maps[etype] = {}
            continue

        print(f"[LLM Cluster] Clustering {len(string_list)} {etype} entities...")
        merge_map = llm_cluster_entities(etype, string_list, ollama_host)

        merged_count = sum(1 for k, v in merge_map.items() if k != v)
        print(f"[LLM Cluster] {etype}: {merged_count} entities merged into canonical forms")
        for k, v in sorted(merge_map.items()):
            if k != v:
                print(f"  '{k}' -> '{v}'")

        merge_maps[etype] = merge_map

    return merge_maps


# ─────────────────────────────────────────────────────────────
#  KNOWLEDGE GRAPH BUILDER
# ─────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:

    def __init__(
        self,
        db_path:    str = "shared_memory/research.db",
        gexf_path:  str = "shared_memory/knowledge_graph.gexf",
        merge_maps: Optional[dict] = None,
    ):
        self.db_path   = Path(db_path)
        self.gexf_path = Path(gexf_path)
        self.merge_maps = merge_maps or {}

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.gexf_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # ── USE UNIFIED SCHEMA ───────────────────────────────
        # This is the only place the schema is created/migrated.
        # Both reader_agent.py and kg_population.py call this same function.
        ensure_schema(self.conn)

        if self.gexf_path.exists():
            self.graph = nx.read_gexf(str(self.gexf_path))
            self._log("KGBuilder", "load_graph",
                      f"Loaded existing graph: {self.graph.number_of_nodes()} nodes, "
                      f"{self.graph.number_of_edges()} edges")
        else:
            self.graph = nx.DiGraph()
            self._log("KGBuilder", "init_graph", "Created new empty graph")

        # In-memory cache of canonical keys already in entities table
        self._entity_key_cache: dict[str, set[str]] = {
            t: set() for t in ("method", "dataset", "metric", "task")
        }
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Load all existing entity_ids from DB into memory for fast lookup."""
        cur = self.conn.execute("SELECT entity_id, entity_type FROM entities")
        for row in cur:
            etype = row["entity_type"]
            # entity_id format: "method::long short-term memory"
            ckey = row["entity_id"].split("::", 1)[1] if "::" in row["entity_id"] else row["entity_id"]
            if etype in self._entity_key_cache:
                self._entity_key_cache[etype].add(ckey)

    def _log(self, agent: str, action: str, detail: str,
             input_sum: str = "", output_sum: str = "") -> None:
        self.conn.execute(
            """INSERT INTO session_log
               (session_id, agent, action, detail, input_summary, output_summary)
               VALUES (?,?,?,?,?,?)""",
            ("kg_builder_session", agent, action, detail, input_sum, output_sum),
        )
        self.conn.commit()
        print(f"[{agent}] {action}: {detail}")

    # ── Main ingestion ────────────────────────────────────────

    def ingest_paper(self, json_path: str) -> str:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        paper_id = data["paper_id"]
        self._log("KGBuilder", "ingest_start", f"paper_id={paper_id}")

        meta = data.get("metadata", {})

        # ── Step 1: Paper metadata ────────────────────────────
        # INSERT OR IGNORE so we never overwrite reader's richer fields
        # (source_path, n_claims, coverage_gain, action, etc.)
        self.conn.execute(
            """INSERT OR IGNORE INTO papers
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

        # ── Steps 2 & 3: Entities + typed edges ───────────────
        entity_index = data.get("entity_index", {})
        for etype, entities_dict in entity_index.items():
            if etype not in EDGE_TYPE_MAP:
                continue
            edge_label = EDGE_TYPE_MAP[etype]

            for raw_key, mentions in entities_dict.items():
                best_mention = max(mentions, key=lambda m: m.get("confidence", 0))
                raw_text     = best_mention.get("text", raw_key)

                # Apply LLM merge map if available
                canonical_text = self.merge_maps.get(etype, {}).get(raw_text, raw_text)

                entity_id = self._resolve_entity(
                    raw_text       = raw_text,
                    canonical_text = canonical_text,
                    entity_type    = etype,
                    paper_id       = paper_id,
                )
                self._add_paper_entity_edge(
                    paper_id   = paper_id,
                    entity_id  = entity_id,
                    edge_type  = edge_label,
                    section    = best_mention.get("section_heading", ""),
                    confidence = best_mention.get("confidence", 1.0),
                )

        # ── Step 4a: Limitation statements ───────────────────
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

        # ── Step 4b: Future work statements ──────────────────
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

        # ── Step 4c: Claims ───────────────────────────────────
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
            f"paper_id={paper_id} | "
            f"entities={summary.get('total_llm_entities','?')} | "
            f"claims={summary.get('total_claims','?')} | "
            f"limitations={summary.get('total_limitations','?')} | "
            f"future_work={summary.get('total_future_work','?')}",
        )
        return paper_id

    # ── Entity resolution ─────────────────────────────────────

    def _resolve_entity(
        self,
        raw_text:       str,
        canonical_text: str,
        entity_type:    str,
        paper_id:       str,
    ) -> str:
        """
        Map raw_text to a canonical entity_id.
        entity_id = "{entity_type}::{normalize(canonical_text)}"

        If the entity already exists, just add this raw_text as a variant
        and add paper_id to papers_seen_in. Otherwise create new entity.
        """
        canonical_key = normalize_entity_text(canonical_text)
        entity_id     = f"{entity_type}::{canonical_key}"

        if canonical_key in self._entity_key_cache.get(entity_type, set()):
            self._update_entity_variants(entity_id, raw_text, paper_id)
            return entity_id

        # New canonical entity
        self.conn.execute(
            """INSERT INTO entities
               (entity_id, entity_type, canonical_text, raw_variants, papers_seen_in)
               VALUES (?,?,?,?,?)""",
            (
                entity_id,
                entity_type,
                canonical_text,
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
                label=canonical_text,
            )
        return entity_id

    def _update_entity_variants(
        self, entity_id: str, raw_text: str, paper_id: str
    ) -> None:
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

    def _add_paper_entity_edge(
        self,
        paper_id:   str,
        entity_id:  str,
        edge_type:  str,
        section:    str   = "",
        confidence: float = 1.0,
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO paper_entity_relationships
               (paper_id, entity_id, edge_type, section, confidence)
               VALUES (?,?,?,?,?)""",
            (paper_id, entity_id, edge_type, section, confidence),
        )
        self.conn.commit()
        if not self.graph.has_edge(paper_id, entity_id):
            self.graph.add_edge(paper_id, entity_id,
                                edge_type=edge_type,
                                section=section,
                                confidence=confidence)

    # ── Graph I/O ─────────────────────────────────────────────

    def save_gexf(self) -> None:
        nx.write_gexf(self.graph, str(self.gexf_path))
        self._log("KGBuilder", "save_gexf",
                  f"Saved {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges → {self.gexf_path}")

    def close(self) -> None:
        self.save_gexf()
        self.conn.close()

    # ── Query helpers (for downstream agents) ─────────────────

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
        node_counts: dict[str, int] = {}
        for ntype in ("Paper", "Method", "Dataset", "Metric", "Task",
                      "LimitationStatement", "FutureWork"):
            node_counts[ntype] = sum(
                1 for _, d in self.graph.nodes(data=True)
                if d.get("node_type") == ntype
            )
        edge_counts: dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_counts[et] = edge_counts.get(et, 0) + 1
        return {"node_counts": node_counts, "edge_counts": edge_counts}


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Knowledge Graph Population v2.1.0"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to Phase-1 claims_output.json files")
    parser.add_argument("--db",   default="shared_memory/research.db")
    parser.add_argument("--gexf", default="shared_memory/knowledge_graph.gexf")
    parser.add_argument("--ollama-host",
                        default=os.environ.get("OLLAMA_HOST", ""),
                        help="Ollama host URL for entity clustering (optional)")
    args = parser.parse_args()

    ollama_host = args.ollama_host or os.environ.get("OLLAMA_HOST", "")
    if not ollama_host:
        print("[Warning] No --ollama-host provided. Using Tier-1 exact match only.")
    else:
        print(f"[KGBuilder] Ollama host: {ollama_host}")

    # Step 1: Build merge maps via LLM clustering
    merge_maps = build_merge_maps(args.inputs, ollama_host)

    # Step 2: Ingest all papers
    builder = KnowledgeGraphBuilder(
        db_path    = args.db,
        gexf_path  = args.gexf,
        merge_maps = merge_maps,
    )

    for json_file in args.inputs:
        builder.ingest_paper(json_file)

    summary = builder.get_graph_summary()
    print("\n── Node counts ──────────────────")
    for k, v in summary["node_counts"].items():
        print(f"  {k:25s}: {v}")
    print("\n── Edge counts ──────────────────")
    for k, v in summary["edge_counts"].items():
        print(f"  {k:25s}: {v}")

    builder.close()
    print(f"\n  DB   → {args.db}")
    print(f"  GEXF → {args.gexf}")


if __name__ == "__main__":
    main()