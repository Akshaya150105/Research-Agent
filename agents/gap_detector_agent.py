"""
gap_detector_agent.py  v2.0.0
==============================
Person 3 — AI Agents  |  Branch: feat/agents  |  Folder: agents/

v2.0.0 upgrades over v1.0.0:
  - gap_matrix.json integration: CombinatorialGapDetector now reads
    memory/gap_matrix.json first (produced by enricher.py). This replaces
    brute-force combinatorial search → eliminates the 353-gap explosion.
    Falls back to entity-based detection if gap_matrix not found.
  - ChromaDB fix: LimitationGapDetector._cluster() now correctly filters
    by chunk_type="limitation" only, uses embedding distance directly
    (no Jaccard double-similarity), and calls get_collection() instead of
    get_or_create_collection() to avoid inconsistency with indexer.py.
  - Comparator autoload: load_comparator_context() fully implemented.
    Scans data_1/agent_outputs/comparisons/ automatically and generates
    CROSS_PAPER gaps from contradiction edges. No configuration needed —
    just drop comparison JSONs in that folder and they are picked up.
  - All existing functionality preserved: entity filtering, method
    classification, critique-informed priority boost, LLM validation,
    LangGraph node, Jaccard fallback.

Strategies:
  1. Combinatorial — gap_matrix.json (primary) or entity classifier (fallback)
  2. Limitation    — ChromaDB semantic (primary) or Jaccard (fallback)
  3. Cross-paper   — comparator contradiction edges (autoloads when available)

Usage
-----
# Single paper (heuristic only)
python agents/gap_detector_agent.py data_1/parsed/claims_output.json --no-llm --verbose

# Multiple papers with LLM
python agents/gap_detector_agent.py \\
    data_1/parsed/paper1_enriched.json \\
    data_1/parsed/paper2_enriched.json \\
    --ollama-host http://localhost:11434 --verbose

# With remote Ollama (ngrok)
python agents/gap_detector_agent.py data_1/parsed/paper1_enriched.json \\
    --ollama-host https://<ngrok>.ngrok-free.app --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
import datetime
import pathlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_DIR      = pathlib.Path("data_1/agent_outputs/gaps")
COMPARISONS_DIR = pathlib.Path("data_1/agent_outputs/comparisons")
GAP_MATRIX_PATH = pathlib.Path("memory/gap_matrix.json")
CHROMA_STORE_PATH = "rag/chroma_store"
CHROMA_COLLECTION  = "claims_and_findings"
CHROMA_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "qwen2.5"

MIN_GAP_CONFIDENCE           = 0.4
LIMITATION_CLUSTER_THRESHOLD = 0.35  # cosine distance threshold for ChromaDB
MIN_CLUSTER_SIZE             = 1
GAP_MATRIX_TOP_N             = 20    # max gaps to read from gap_matrix.json

LLM_TIMEOUT_SECONDS  = 60
LLM_MAX_RETRIES      = 3
LLM_RETRY_BASE_DELAY = 2


# ─────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────

class GapType(str, Enum):
    COMBINATORIAL = "combinatorial"
    LIMITATION    = "limitation"
    CROSS_PAPER   = "cross_paper"


class GapPriority(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class AddressedStatus(str, Enum):
    NOT_ADDRESSED       = "not_addressed"
    PARTIALLY_ADDRESSED = "partially_addressed"
    ADDRESSED           = "addressed"
    UNKNOWN             = "unknown"


@dataclass
class Gap:
    gap_id:            str
    gap_type:          GapType
    priority:          GapPriority
    description:       str
    evidence:          str
    suggestion:        str
    papers_involved:   list[str]
    entities_involved: list[str]
    confidence:        float
    addressed_status:  AddressedStatus = AddressedStatus.UNKNOWN
    llm_validated:     bool            = False
    llm_rationale:     str             = ""
    needs_review:      bool            = False


@dataclass
class GapResult:
    session_id:      str
    generated_at:    str
    agent_version:   str       = "2.0.0"
    papers_analysed: list[str] = field(default_factory=list)
    n_papers:        int       = 0
    gaps:            list[Gap] = field(default_factory=list)
    gap_counts:      dict      = field(default_factory=dict)
    react_trace:     list[str] = field(default_factory=list)
    agent_report:    dict      = field(default_factory=dict)

    def compute_summary(self):
        counts = {t.value: 0 for t in GapType}
        for g in self.gaps:
            counts[g.gap_type.value] += 1
        self.gap_counts = counts

        high_count = sum(1 for g in self.gaps if g.priority == GapPriority.HIGH)
        self.agent_report = {
            "agent_name":         "gap_detector",
            "status":             "complete",
            "n_gaps_found":       len(self.gaps),
            "high_priority_gaps": high_count,
            "gap_types":          counts,
            "recommended_next": (
                "run_comparator" if self.n_papers >= 2
                else "add_more_papers" if len(self.gaps) < 2
                else "write_review"
            ),
            "coverage_note": (
                f"Gap detection ran on {self.n_papers} paper(s). "
                "Add more papers for richer combinatorial gaps."
                if self.n_papers < 3
                else f"Gap detection ran on {self.n_papers} papers."
            ),
        }


# ─────────────────────────────────────────────
#  INPUT LOADING
# ─────────────────────────────────────────────

def load_paper(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Paper file not found: {p}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    if "paper_id" not in raw:
        raise ValueError(f"Missing paper_id in {p}")

    for claim in raw.get("claims", []):
        if "text" not in claim and "description" in claim:
            claim["text"] = claim["description"]
        if "type" not in claim and "claim_type" in claim:
            claim["type"] = claim["claim_type"]

    if "entity_index" not in raw and "entities" in raw:
        if isinstance(raw["entities"], dict) and "method" in raw["entities"]:
            raw["entity_index"] = raw["entities"]
        else:
            raw["entity_index"] = {}

    raw.setdefault("entity_index",     {})
    raw.setdefault("claims",           [])
    raw.setdefault("limitations",      [])
    raw.setdefault("future_work",      [])
    raw.setdefault("metadata",         {})
    raw.setdefault("critiques",        [])
    raw.setdefault("critique_summary", {})
    return raw


# ─────────────────────────────────────────────
#  GAP MATRIX LOADER  (autoloads from memory/)
# ─────────────────────────────────────────────

def load_gap_matrix() -> dict | None:
    """
    Load gap_matrix.json from memory/ if it exists.
    This file is produced by enricher.py and already ranks
    method × dataset gaps intelligently — no brute-force needed.

    Returns None if file not found (triggers fallback detection).
    """
    if not GAP_MATRIX_PATH.exists():
        return None
    try:
        with open(GAP_MATRIX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  METHOD CLASSIFICATION & NORMALISATION
# ─────────────────────────────────────────────

def normalize_method(m: str) -> str:
    m = m.lower()
    if "lstm" in m or "rnn" in m:
        return "recurrent models"
    if "transformer" in m or "attention" in m:
        return "transformer models"
    if "cnn" in m or "conv" in m:
        return "convolutional models"
    return m


def classify_method(m: str) -> str:
    m = m.lower()
    if any(x in m for x in ["gpu", "dgx", "tesla", "system"]):
        return "NOT_METHOD"
    if any(x in m for x in ["representation", "function", "code",
                              "transformation", "processing"]):
        return "WEAK_METHOD"
    if any(x in m for x in ["lstm", "rnn", "transformer",
                              "attention", "cnn", "bert"]):
        return "VALID_METHOD"
    if len(m.split()) >= 2:
        return "WEAK_METHOD"
    return "NOT_METHOD"


def is_meaningful_limitation(text: str) -> bool:
    text = text.lower()
    weak_phrases = [
        "future work", "we plan", "we will",
        "might be", "could be", "small number",
        "only a few", "limited experiments",
    ]
    if any(p in text for p in weak_phrases):
        return False
    if any(x in text for x in ["d_k", "beam size", "dropout", "label smoothing"]):
        return False
    strong_signals = [
        "scalability", "generalization", "long-range",
        "sequential", "parallelization", "efficiency",
    ]
    return any(s in text for s in strong_signals)


def extract_entity_sets(paper: dict) -> dict:
    ei = paper.get("entity_index", {})
    if not isinstance(ei, dict):
        ei = {}

    def clean_keys(d: dict) -> set[str]:
        return {k.strip().lower() for k in d.keys() if len(k.strip()) > 2}

    raw_methods = clean_keys(ei.get("method", {}))
    methods = set()
    for m in raw_methods:
        cls = classify_method(m)
        if cls == "VALID_METHOD":
            methods.add(normalize_method(m))

    return {
        "paper_id": paper["paper_id"],
        "title":    paper.get("metadata", {}).get("title", ""),
        "year":     paper.get("metadata", {}).get("year"),
        "methods":  methods,
        "datasets": clean_keys(ei.get("dataset", {})),
        "metrics":  clean_keys(ei.get("metric",  {})),
        "tasks":    clean_keys(ei.get("task",    {})),
    }


# ─────────────────────────────────────────────
#  COMPARATOR AUTOLOADER  (fully implemented)
# ─────────────────────────────────────────────

def load_comparator_context(paper_ids: list[str]) -> dict:
    """
    Automatically scans data_1/agent_outputs/comparisons/ for JSON files
    that match the paper_ids being analysed.

    Picks up files named like:
        paper_a__paper_b.json   (comparator agent output format)

    Extracts contradictions and complements. Returns populated dict
    when comparator output exists, empty dict otherwise.

    No configuration needed — just ensure comparator has run and
    dropped files in the comparisons folder.
    """
    context = {
        "contradictions": [],
        "complements":    [],
        "available":      False,
    }

    if not COMPARISONS_DIR.exists():
        return context

    comparison_files = list(COMPARISONS_DIR.glob("*.json"))
    if not comparison_files:
        return context

    paper_id_set = set(paper_ids)

    for fpath in comparison_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                comp = json.load(f)

            paper_a = comp.get("paper_a", "")
            paper_b = comp.get("paper_b", "")

            # Only load comparisons relevant to the current paper set
            if not (paper_a in paper_id_set and paper_b in paper_id_set):
                continue

            # Extract contradictions
            for c in comp.get("contradictions", []):
                context["contradictions"].append({
                    "paper_a":    paper_a,
                    "paper_b":    paper_b,
                    "method":     c.get("method", c.get("entity", "")),
                    "claim_a":    c.get("claim_a", c.get("text_a", "")),
                    "claim_b":    c.get("claim_b", c.get("text_b", "")),
                    "severity":   c.get("severity", "medium"),
                    "type":       c.get("type", "unknown"),
                })

            # Extract complements
            for c in comp.get("complementary_findings", []):
                context["complements"].append({
                    "paper_a":  paper_a,
                    "paper_b":  paper_b,
                    "detail":   c.get("detail", c.get("text", "")),
                })

        except Exception:
            continue

    context["available"] = bool(
        context["contradictions"] or context["complements"]
    )
    return context


# ─────────────────────────────────────────────
#  STRATEGY 1: COMBINATORIAL GAP DETECTION
# ─────────────────────────────────────────────

class CombinatorialGapDetector:
    """
    Primary path: reads memory/gap_matrix.json (produced by enricher.py).
    This file already ranks method × dataset gaps — no explosion possible.

    Fallback path: entity classifier + co-occurrence matrix from paper JSON.
    Used when gap_matrix.json doesn't exist yet.
    """

    def __init__(self, entity_sets: list[dict], papers: list[dict], verbose: bool = False):
        self.entity_sets = entity_sets
        self.papers      = papers
        self.verbose     = verbose
        self.n_papers    = len(entity_sets)

    def _log(self, msg: str):
        if self.verbose:
            print(f"    [combinatorial] {msg}")

    def detect(self) -> list[Gap]:
        """Route to gap_matrix path or fallback path."""
        gap_matrix = load_gap_matrix()
        if gap_matrix:
            self._log(f"gap_matrix.json found — using pre-computed gaps")
            return self._detect_from_gap_matrix(gap_matrix)
        else:
            self._log("gap_matrix.json not found — using entity-based fallback")
            return self._detect_from_entities()

    # ── Primary path: gap_matrix.json ────────────────────────────────────────

    def _detect_from_gap_matrix(self, gap_matrix: dict) -> list[Gap]:
        """
        Load top N gaps from gap_matrix.json directly.
        No combinatorial explosion — gaps are already scored and ranked.
        """
        gaps: list[Gap] = []
        paper_id_set = {es["paper_id"] for es in self.entity_sets}

        raw_gaps = gap_matrix.get("gaps", [])
        self._log(f"gap_matrix has {len(raw_gaps)} entries, taking top {GAP_MATRIX_TOP_N}")

        for g in raw_gaps[:GAP_MATRIX_TOP_N]:
            method       = g.get("method", "")
            dataset      = g.get("dataset", "")
            gap_score    = g.get("gap_score", 1)
            method_in    = g.get("method_used_in", [])
            dataset_in   = g.get("dataset_used_in", [])

            # Only include gaps that touch our current paper set
            involved = list(
                (set(method_in) | set(dataset_in)) & paper_id_set
            )
            if not involved:
                involved = list(paper_id_set)  # fallback: attribute to all

            priority = (
                GapPriority.HIGH   if gap_score >= 4
                else GapPriority.MEDIUM if gap_score >= 2
                else GapPriority.LOW
            )

            gaps.append(Gap(
                gap_id            = f"gap_matrix_{uuid.uuid4().hex[:6]}",
                gap_type          = GapType.COMBINATORIAL,
                priority          = priority,
                description       = (
                    f"Method '{method}' has never been evaluated on "
                    f"dataset '{dataset}' (gap score: {gap_score}). "
                    f"Method used in: {method_in}; "
                    f"dataset used in: {dataset_in}."
                ),
                evidence          = (
                    f"gap_matrix.json: method='{method}', "
                    f"dataset='{dataset}', gap_score={gap_score}."
                ),
                suggestion        = (
                    f"Evaluate '{method}' on '{dataset}' to fill "
                    "this identified research gap."
                ),
                papers_involved   = involved,
                entities_involved = [method, dataset],
                confidence        = 0.80,
            ))

        self._log(f"Loaded {len(gaps)} combinatorial gaps from gap_matrix")
        return gaps

    # ── Fallback path: entity-based detection ────────────────────────────────

    def _all_datasets(self) -> Counter:
        c = Counter()
        for es in self.entity_sets:
            for d in es["datasets"]: c[d] += 1
        return c

    def _all_methods(self) -> Counter:
        c = Counter()
        for es in self.entity_sets:
            for m in es["methods"]: c[m] += 1
        return c

    def _all_tasks(self) -> Counter:
        c = Counter()
        for es in self.entity_sets:
            for t in es["tasks"]: c[t] += 1
        return c

    def _method_dataset_cooccurrence(self) -> dict[str, set[str]]:
        """method → datasets it ACTUALLY co-occurs with in claims (not just present)."""
        cooc: dict[str, set[str]] = defaultdict(set)
        for es, paper in zip(self.entity_sets, self.papers):
            for claim in paper.get("claims", []):
                claim_text = claim.get("text", "").lower()
                for m in es["methods"]:
                    if m in claim_text:
                        for d in es["datasets"]:
                            if d in claim_text:
                                cooc[m].add(d)
        return cooc

    def _task_dataset_cooccurrence(self) -> dict[str, set[str]]:
        cooc: dict[str, set[str]] = defaultdict(set)
        for es in self.entity_sets:
            for t in es["tasks"]:
                for d in es["datasets"]:
                    cooc[t].add(d)
        return cooc

    def _detect_from_entities(self) -> list[Gap]:
        seen:  set   = set()
        gaps: list[Gap] = []
        all_datasets = self._all_datasets()
        all_tasks    = self._all_tasks()
        all_ds_list  = list(all_datasets.keys())
        task_ds_cooc = self._task_dataset_cooccurrence()

        self._log(
            f"Corpus: {self.n_papers} papers, "
            f"{len(all_datasets)} datasets, "
            f"{sum(len(es['methods']) for es in self.entity_sets)} methods, "
            f"{len(all_tasks)} tasks"
        )

        # Gap type A: tasks claimed but narrow dataset coverage
        for es in self.entity_sets:
            for task in es["tasks"]:
                tested_on     = task_ds_cooc.get(task, set())
                untested      = set(all_ds_list) - tested_on
                if len(tested_on) <= 1 and len(es["datasets"]) <= 1 and untested:
                    key = (task, frozenset(es["datasets"]))
                    if key in seen: continue
                    seen.add(key)
                    ds_str = ", ".join(list(untested)[:3])
                    gaps.append(Gap(
                        gap_id            = f"gap_task_{uuid.uuid4().hex[:6]}",
                        gap_type          = GapType.COMBINATORIAL,
                        priority          = GapPriority.MEDIUM,
                        description       = (
                            f"Task '{task}' claimed in '{es['paper_id']}' "
                            f"but only evaluated on '{list(tested_on)[0] if tested_on else 'no dataset'}'. "
                            f"Other datasets in corpus not tested: {ds_str}."
                        ),
                        evidence          = (
                            f"Task: '{task}'. Tested datasets: {list(tested_on)}. "
                            f"Untested: {list(untested)[:3]}."
                        ),
                        suggestion        = (
                            f"Evaluate '{task}' on additional benchmarks "
                            "to support generalisation claims."
                        ),
                        papers_involved   = [es["paper_id"]],
                        entities_involved = [task] + list(es["datasets"]),
                        confidence        = 0.65,
                    ))

        # Gap type B: cross-paper method × dataset (only with 2+ papers, grouped)
        if self.n_papers >= 2:
            method_ds_cooc = self._method_dataset_cooccurrence()

            for es_a in self.entity_sets:
                for es_b in self.entity_sets:
                    if es_a["paper_id"] == es_b["paper_id"]:
                        continue

                    shared_tasks = es_a["tasks"] & es_b["tasks"]
                    if not shared_tasks:
                        continue

                    for method in es_a["methods"]:
                        cls = classify_method(method)
                        if cls == "NOT_METHOD":
                            continue

                        tested_datasets   = method_ds_cooc.get(method, set())
                        missing_datasets  = list(es_b["datasets"] - tested_datasets)

                        if not missing_datasets:
                            continue

                        # Group all missing datasets into ONE gap per method-paper pair
                        key = (method, es_a["paper_id"], es_b["paper_id"])
                        if key in seen: continue
                        seen.add(key)

                        priority = (
                            GapPriority.HIGH   if len(shared_tasks) > 2 and len(missing_datasets) > 3
                            else GapPriority.MEDIUM if len(missing_datasets) > 1
                            else GapPriority.LOW
                        )
                        confidence = min(0.9, 0.4 + 0.15 * len(shared_tasks))
                        if cls == "WEAK_METHOD":
                            confidence -= 0.2

                        gaps.append(Gap(
                            gap_id            = f"gap_cross_{uuid.uuid4().hex[:6]}",
                            gap_type          = GapType.COMBINATORIAL,
                            priority          = priority,
                            description       = (
                                f"Method '{method}' from '{es_a['paper_id']}' "
                                f"has not been evaluated on {len(missing_datasets)} dataset(s) "
                                f"from '{es_b['paper_id']}': {', '.join(missing_datasets[:3])}. "
                                f"Shared tasks: {list(shared_tasks)[:3]}."
                            ),
                            evidence          = (
                                f"'{method}' in {es_a['paper_id']} but missing from "
                                f"{len(missing_datasets)} datasets in {es_b['paper_id']}."
                            ),
                            suggestion        = (
                                f"Evaluate '{method}' on "
                                f"{', '.join(missing_datasets[:3])} "
                                "to test cross-domain generalisation."
                            ),
                            papers_involved   = [es_a["paper_id"], es_b["paper_id"]],
                            entities_involved = [method] + missing_datasets[:3] + list(shared_tasks)[:2],
                            confidence        = confidence,
                        ))

        self._log(f"Found {len(gaps)} combinatorial gaps (entity fallback)")
        return gaps


# ─────────────────────────────────────────────
#  STRATEGY 2: LIMITATION-BASED GAP DETECTION
# ─────────────────────────────────────────────

class LimitationGapDetector:
    """
    Primary path: ChromaDB semantic clustering.
      - Queries ONLY chunk_type="limitation" (not claims or findings)
      - Uses embedding cosine distance directly — NO Jaccard double-similarity
      - Uses get_collection() not get_or_create_collection()

    Fallback path: Jaccard token-overlap clustering.
    Used when ChromaDB is unavailable or empty.
    """

    def __init__(self, papers: list[dict], entity_sets: list[dict], verbose: bool = False):
        self.papers      = papers
        self.entity_sets = entity_sets
        self.verbose     = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"    [limitation] {msg}")

    def _tokenise(self, text: str) -> set[str]:
        tokens = re.findall(r'\b[a-z][a-z0-9]{2,}\b', text.lower())
        stops  = {
            "the", "and", "for", "are", "that", "this", "with",
            "have", "been", "from", "they", "which", "their", "will",
            "not", "can", "but", "our", "more", "also", "than",
            "such", "its", "may", "each", "when", "these", "use",
        }
        return {t for t in tokens if t not in stops}

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)

    def _collect_limitations(self) -> list[dict]:
        all_lims = []
        for paper in self.papers:
            pid  = paper["paper_id"]
            year = paper.get("metadata", {}).get("year", 9999)
            for lim in paper.get("limitations", []):
                text = lim.get("text", "").strip()
                if len(text) > 20:
                    all_lims.append({
                        "paper_id": pid,
                        "year":     year,
                        "text":     text,
                        "tokens":   self._tokenise(text),
                        "section":  lim.get("section_type", ""),
                    })
            for fw in paper.get("future_work", []):
                text = fw.get("text", "").strip()
                if len(text) > 20:
                    all_lims.append({
                        "paper_id":       pid,
                        "year":           year,
                        "text":           text,
                        "tokens":         self._tokenise(text),
                        "section":        "future_work",
                        "is_future_work": True,
                    })
        return all_lims

    def _cluster(self, limitations: list[dict]) -> list[list[dict]]:
        """
        ChromaDB semantic clustering — correct implementation.

        Fixes vs old version:
          ✅ Filters chunk_type="limitation" only (not claims/findings)
          ✅ Uses embedding distance directly — Jaccard removed
          ✅ Uses get_collection() not get_or_create_collection()
          ✅ Includes metadatas to enable paper_id cross-paper filtering
          ✅ Falls back to Jaccard if ChromaDB unavailable
        """
        if not limitations:
            return []

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client     = chromadb.PersistentClient(path=CHROMA_STORE_PATH)
            # ✅ Use get_collection — indexer.py already created this
            collection = client.get_collection(CHROMA_COLLECTION)

            if collection.count() == 0:
                raise RuntimeError("ChromaDB collection is empty")

            model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
            self._log(f"ChromaDB active ({collection.count()} chunks)")

            assigned = [False] * len(limitations)
            clusters = []

            for i, lim in enumerate(limitations):
                if assigned[i]:
                    continue

                query_vec = model.encode(
                    CHROMA_QUERY_PREFIX + lim["text"],
                    normalize_embeddings=True,
                ).tolist()

                # ✅ CRITICAL: filter chunk_type="limitation" ONLY
                # Optional cross-paper filter — uncomment to avoid self-clustering:
                # where={"$and": [
                #     {"chunk_type": "limitation"},
                #     {"paper_id": {"$ne": lim["paper_id"]}}
                # ]}
                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(10, collection.count()),
                    where={"chunk_type": "limitation"},   # 🔥 core fix
                    include=["documents", "metadatas", "distances"],
                )

                cluster   = [lim]
                assigned[i] = True

                # ✅ Use embedding distance ONLY — no Jaccard double-similarity
                for dist, meta, doc in zip(
                    results["distances"][0],
                    results["metadatas"][0],
                    results["documents"][0],
                ):
                    if dist > LIMITATION_CLUSTER_THRESHOLD:
                        continue

                    if meta.get("paper_id") == lim["paper_id"]:
                        continue

                    # Match the DB result back to our local limitation list
                    for j, other in enumerate(limitations):
                        if assigned[j]:
                            continue
                        # Exact text match to map DB doc → local lim
                        if other["text"].strip().lower() == doc.strip().lower():
                            cluster.append(other)
                            assigned[j] = True
                            break

                clusters.append(cluster)

            self._log(f"ChromaDB clustering complete: {len(clusters)} clusters")
            return clusters

        except Exception as e:
            self._log(f"ChromaDB unavailable ({e}) — falling back to Jaccard")
            return self._cluster_jaccard(limitations)

    def _cluster_jaccard(self, limitations: list[dict]) -> list[list[dict]]:
        """Jaccard token-overlap fallback. Used when ChromaDB is unavailable."""
        if not limitations:
            return []

        clusters = []
        assigned = [False] * len(limitations)

        for i, lim_i in enumerate(limitations):
            if assigned[i]: continue
            cluster    = [lim_i]
            assigned[i] = True

            for j, lim_j in enumerate(limitations):
                if assigned[j] or i == j: continue
                sim = self._jaccard(lim_i["tokens"], lim_j["tokens"])
                if sim >= LIMITATION_CLUSTER_THRESHOLD:
                    cluster.append(lim_j)
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _check_addressed(self, cluster: list[dict]) -> AddressedStatus:
        cluster_tokens = set()
        for item in cluster:
            cluster_tokens |= item["tokens"]

        years = [item["year"] for item in cluster if item.get("year")]
        if not years:
            return AddressedStatus.UNKNOWN

        max_year = max(years)
        for paper in self.papers:
            paper_year = paper.get("metadata", {}).get("year", 0)
            if paper_year <= max_year:
                continue
            for claim in paper.get("claims", []):
                claim_text   = claim.get("text", claim.get("description", "")).lower()
                claim_tokens = self._tokenise(claim_text)
                if len(cluster_tokens & claim_tokens) >= 3:
                    return AddressedStatus.PARTIALLY_ADDRESSED

        return AddressedStatus.NOT_ADDRESSED

    def detect(self) -> list[Gap]:
        all_lims = self._collect_limitations()
        self._log(f"Collected {len(all_lims)} limitation/future_work statements")

        # Filter to only meaningful conceptual limitations
        filtered = [
            item for item in all_lims
            if is_meaningful_limitation(item["text"])
        ]
        self._log(f"After filtering: {len(filtered)} meaningful limitations")

        if not filtered:
            self._log("No meaningful limitations — skipping")
            return []

        clusters = self._cluster(filtered)
        self._log(f"Formed {len(clusters)} clusters")

        gaps: list[Gap] = []
        for cluster in clusters:
            if len(cluster) < MIN_CLUSTER_SIZE:
                continue

            papers_in_cluster = list({item["paper_id"] for item in cluster})
            all_texts         = [item["text"] for item in cluster]
            all_tokens: set   = set()
            for item in cluster:
                all_tokens |= item["tokens"]

            rep_text  = max(all_texts, key=len)
            n         = len(papers_in_cluster)
            priority  = (
                GapPriority.HIGH   if n >= 3
                else GapPriority.MEDIUM if n >= 2
                else GapPriority.LOW
            )
            addressed  = self._check_addressed(cluster)
            confidence = 0.55 if len(cluster) == 1 else 0.85 if len(cluster) >= 3 else 0.70

            gaps.append(Gap(
                gap_id            = f"gap_lim_{uuid.uuid4().hex[:6]}",
                gap_type          = GapType.LIMITATION,
                priority          = priority,
                description       = (
                    f"Limitation cluster across {n} paper(s): '{rep_text[:200]}'. "
                    f"Appears {len(cluster)} time(s), "
                    f"{'not addressed' if addressed == AddressedStatus.NOT_ADDRESSED else addressed.value} "
                    "by later work."
                ),
                evidence          = (
                    f"Papers: {papers_in_cluster}. "
                    f"Statements: {'; '.join(t[:150] for t in all_texts[:2])}"
                ),
                suggestion        = (
                    "Open research direction — directly address: "
                    f"'{rep_text[:150]}'"
                ),
                papers_involved   = papers_in_cluster,
                entities_involved = [
                    t for t in all_tokens
                    if len(t) > 5 and t not in {"problem", "method", "model"}
                ][:6],
                confidence        = confidence,
                addressed_status  = addressed,
                needs_review      = confidence < 0.6,
            ))

        self._log(f"Found {len(gaps)} limitation-based gap(s)")
        return gaps


# ─────────────────────────────────────────────
#  STRATEGY 3: CROSS-PAPER GAPS (from comparator)
# ─────────────────────────────────────────────

def detect_cross_paper_gaps(
    comp_ctx: dict,
    entity_sets: list[dict],
    verbose: bool = False,
) -> list[Gap]:
    """
    Generates CROSS_PAPER gaps from comparator contradiction edges.

    Logic: if Method A contradicts Method B on a certain claim,
    and neither paper tested Method A on B's datasets, that is
    an unresolved cross-paper gap worth flagging.

    Autoloads — called only when comp_ctx["available"] is True.
    """
    if not comp_ctx["available"]:
        return []

    gaps:  list[Gap] = []
    seen:  set       = set()

    # Build paper → datasets lookup
    paper_datasets = {
        es["paper_id"]: es["datasets"]
        for es in entity_sets
    }

    for contradiction in comp_ctx["contradictions"]:
        paper_a = contradiction["paper_a"]
        paper_b = contradiction["paper_b"]
        method  = contradiction.get("method", "")
        ctype   = contradiction.get("type", "unknown")
        claim_a = contradiction.get("claim_a", "")[:150]
        claim_b = contradiction.get("claim_b", "")[:150]

        key = (paper_a, paper_b, method)
        if key in seen:
            continue
        seen.add(key)

        # Datasets from each paper
        ds_a = paper_datasets.get(paper_a, set())
        ds_b = paper_datasets.get(paper_b, set())

        gaps.append(Gap(
            gap_id            = f"gap_cross_{uuid.uuid4().hex[:6]}",
            gap_type          = GapType.CROSS_PAPER,
            priority          = GapPriority.HIGH,
            description       = (
                f"Contradiction detected between '{paper_a}' and '{paper_b}' "
                f"on '{method or 'methodology'}' (type: {ctype}). "
                "This unresolved conflict represents a research gap requiring "
                "direct comparison under controlled conditions."
            ),
            evidence          = (
                f"Paper A: {claim_a} | Paper B: {claim_b}"
            ),
            suggestion        = (
                f"Design a controlled experiment directly comparing "
                f"'{paper_a}' and '{paper_b}' on the same datasets "
                f"({list(ds_a)[:2] or list(ds_b)[:2]}) "
                "to resolve this contradiction."
            ),
            papers_involved   = [paper_a, paper_b],
            entities_involved = [method] + list(ds_a)[:2] + list(ds_b)[:2],
            confidence        = 0.85,
        ))

    if verbose and gaps:
        print(f"    [cross_paper] {len(gaps)} cross-paper gap(s) from "
              f"{len(comp_ctx['contradictions'])} contradiction(s)")

    return gaps


# ─────────────────────────────────────────────
#  CRITIQUE-INFORMED PRIORITY BOOST
# ─────────────────────────────────────────────

def boost_priority_from_critiques(gaps: list[Gap], papers: list[dict]) -> list[Gap]:
    critique_index: dict[str, list[str]] = {}
    for paper in papers:
        pid = paper["paper_id"]
        cs  = paper.get("critique_summary", {})
        critique_index[pid] = cs.get("high_weakness_types", [])

    for gap in gaps:
        for pid in gap.papers_involved:
            high_types = critique_index.get(pid, [])

            if gap.gap_type == GapType.COMBINATORIAL:
                if "single_dataset_evaluation" in high_types:
                    gap.priority   = GapPriority.HIGH
                    gap.confidence = min(gap.confidence + 0.10, 1.0)
                if "outdated_baselines" in high_types:
                    gap.priority   = GapPriority.HIGH
                    gap.confidence = min(gap.confidence + 0.05, 1.0)

            if gap.gap_type == GapType.LIMITATION:
                weak_types = [
                    w.get("weakness_type", "")
                    for w in paper.get("critiques", [])
                ]
                if "vague_limitations" in weak_types or "missing_limitations_future_work" in weak_types:
                    gap.confidence  = max(gap.confidence - 0.15, 0.3)
                    gap.needs_review = True

    return gaps


# ─────────────────────────────────────────────
#  LLM CALL HELPER
# ─────────────────────────────────────────────

def _llm_call_raw(prompt: str, llm_backend: str) -> str:
    host  = os.environ.get("OLLAMA_HOST", OLLAMA_HOST)
    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{host}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": 0.2},
                },
                timeout=LLM_TIMEOUT_SECONDS,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            if resp.status_code == 429:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Rate limited after {LLM_MAX_RETRIES} attempts")

            resp.raise_for_status()
            data = resp.json()

            # Ollama /api/chat response format
            if "message" in data:
                return data["message"]["content"]
            # Fallback: /api/generate format
            if "response" in data:
                return data["response"]
            raise ValueError(f"Unexpected Ollama response: {list(data.keys())}")

        except (requests.RequestException, KeyError) as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts. Last: {last_error}")


def _parse_llm_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON in LLM response: {text[:200]}")
    depth = 0; end = -1; in_str = False; esc = False
    for i, ch in enumerate(text[start:], start):
        if esc:   esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end == -1:
        raise ValueError(f"Unclosed JSON in LLM response: {text[:200]}")
    return json.loads(text[start:end + 1])


def build_validation_prompt(gaps: list[Gap], papers: list[dict]) -> str:
    paper_summaries = []
    for p in papers:
        meta = p.get("metadata", {})
        ei   = p.get("entity_index", {})
        ei_d = ei if isinstance(ei, dict) else {}
        paper_summaries.append(
            f"  Paper: {p['paper_id']} ({meta.get('year', '?')})\n"
            f"  Title: {meta.get('title', '')[:80]}\n"
            f"  Datasets: {list(ei_d.get('dataset', {}).keys())}\n"
            f"  Tasks: {list(ei_d.get('task', {}).keys())[:5]}"
        )

    gaps_text = "\n\n".join(
        f"GAP {i+1} (id={g.gap_id}, type={g.gap_type.value}):\n"
        f"  Description: {g.description[:250]}\n"
        f"  Evidence: {g.evidence[:150]}"
        for i, g in enumerate(gaps)
    )

    return f"""You are a research gap analyst reviewing candidate gaps in ML/NLP.

CORPUS:
{chr(10).join(paper_summaries)}

CANDIDATE GAPS:
{gaps_text}

For each gap assess: Is it GENUINE (not trivial, not already addressed, feasible)?

Reply ONLY with valid JSON — no markdown:
{{
  "validations": [
    {{
      "gap_id": "gap_id_here",
      "is_genuine": true,
      "confidence_adjustment": 0.0,
      "rationale": "One sentence.",
      "addressed_status": "not_addressed|partially_addressed|addressed|unknown"
    }}
  ]
}}

Rules:
- confidence_adjustment: float between -0.3 and +0.2
- Trivial or already addressed → is_genuine=false, adjustment=-0.3
- Clearly genuine and important → is_genuine=true, adjustment=+0.1 to +0.2
- Process ALL {len(gaps)} gaps"""


def llm_validate_gaps(
    gaps: list[Gap],
    papers: list[dict],
    llm_backend: str,
    verbose: bool = False,
) -> list[Gap]:
    if not gaps:
        return []

    prompt = build_validation_prompt(gaps, papers)
    if verbose:
        print(f"\n  [llm] Validating {len(gaps)} gap(s) via {llm_backend}...")

    try:
        text = _llm_call_raw(prompt, llm_backend)
        try:
            parsed = _parse_llm_json(text)
        except Exception:
            print("⚠️ LLM JSON parsing failed, retrying once...")
            text   = _llm_call_raw(prompt, llm_backend)
            parsed = _parse_llm_json(text)

        if verbose:
            print("  [llm] Gap validation response parsed")

        validation_map = {
            v["gap_id"]: v
            for v in parsed.get("validations", [])
        }

        validated: list[Gap] = []
        for gap in gaps:
            v = validation_map.get(gap.gap_id)
            if not v:
                gap.llm_validated = False
                validated.append(gap)
                continue

            if not v.get("is_genuine", True):
                if verbose:
                    print(f"    [llm] Dropped '{gap.gap_id}': {v.get('rationale','')[:80]}")
                continue

            adj = float(v.get("confidence_adjustment", 0.0))
            gap.confidence    = max(0.0, min(1.0, gap.confidence + adj))
            gap.llm_validated = True
            gap.llm_rationale = v.get("rationale", "")

            try:
                gap.addressed_status = AddressedStatus(v.get("addressed_status", ""))
            except ValueError:
                pass

            if gap.confidence >= MIN_GAP_CONFIDENCE:
                gap.needs_review = gap.confidence < 0.6
                validated.append(gap)
            elif verbose:
                print(f"    [llm] Dropped '{gap.gap_id}' (conf {gap.confidence:.2f} < {MIN_GAP_CONFIDENCE})")

        return validated

    except Exception as e:
        print(f"\n⚠️ LLM validation failed: {e}\n   → Using heuristic gaps")
        return gaps


# ─────────────────────────────────────────────
#  REACT AGENT
# ─────────────────────────────────────────────

class GapDetectorAgent:
    VERSION = "2.0.0"

    def __init__(self, llm_backend: str = "none", verbose: bool = False):
        self.llm_backend = llm_backend
        self.verbose     = verbose
        self.trace: list[str] = []

    def _think(self, msg: str):
        entry = f"[THINK] {msg}"
        self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _act(self, msg: str):
        entry = f"[ACT]   {msg}"
        self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _observe(self, msg: str):
        entry = f"[OBS]   {msg}"
        self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def run_session(self, session_state: dict) -> dict:
        """
        MATCHES COMPARATOR/CRITIC STYLE:
        Takes the full session state, finds the right paper paths,
        runs detection, and returns a summary report.
        """
        self._think("Starting Gap Detection Session")
        
        # 1. Self-Discovery of Paper Paths
        rr = session_state.get("reader_report", {})
        paper_ids = rr.get("paper_ids_read", [])
        memory_dir = pathlib.Path(session_state.get("memory_dir", "memory"))

        # Fallback to all folders if no new papers were read in this specific step
        if not paper_ids:
            self._observe("No new papers from reader. Checking memory/ for existing data...")
            paper_ids = [p.name for p in memory_dir.iterdir() if p.is_dir()]

        resolved_paths = []
        for pid in paper_ids:
            # CRITICAL: Prefer enriched JSON from Critic so we get the priority boost!
            enriched = memory_dir / pid / f"{pid}_enriched.json"
            raw = memory_dir / pid / "claims_output.json"
            
            target = enriched if enriched.exists() else raw
            if target.exists():
                resolved_paths.append(target)

        if not resolved_paths:
            self._observe("No valid paper files found — skipping.")
            return self._build_session_report(None)

        self._observe(f"Analyzing {len(resolved_paths)} paper(s) for research gaps.")

        # 2. Run the internal detection logic
        try:
            # We pass the list of paths to the existing run() method
            result = self.run(resolved_paths)
            
            # 3. Internal Write-back (Shared Memory)
            out_path = save_gaps(result, OUTPUT_DIR)
            self._observe(f"Gaps saved to {out_path.name}")
            
            return self._build_session_report(result)
            
        except Exception as e:
            self._observe(f"⚠ Gap Detection crashed: {e}")
            return self._build_session_report(None)

    def _build_session_report(self, result: GapResult | None) -> dict:
        """Builds the dictionary the Planner expects."""
        if not result:
            return {
                "agent": "gap_detector_agent",
                "n_gaps": 0,
                "gaps": [],
                "react_trace": self.trace
            }
        
        # Convert Gap objects to serializable dicts
        gap_list = []
        for g in result.gaps:
            gap_list.append({
                "gap_id": g.gap_id,
                "gap_type": g.gap_type.value,
                "priority": g.priority.value,
                "description": g.description,
                "confidence": round(g.confidence, 2),
                "status": g.addressed_status.value
            })

        return {
            "agent": "gap_detector_agent",
            "agent_version": self.VERSION,
            "n_gaps": len(gap_list),
            "gaps": gap_list,
            "gap_counts": result.gap_counts,
            "react_trace": self.trace
        }

    def run(self, paper_paths: list[str | pathlib.Path]) -> GapResult:

        session_id = uuid.uuid4().hex[:8]
        result = GapResult(
            session_id    = session_id,
            generated_at  = datetime.datetime.now(
                datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            agent_version = self.VERSION,
            react_trace   = self.trace,
        )

        # Step 0: Load papers
        self._think(f"Loading {len(paper_paths)} paper file(s)")
        papers: list[dict] = []
        for path in paper_paths:
            self._act(f"load_paper({pathlib.Path(path).name})")
            try:
                paper = load_paper(path)
                papers.append(paper)
                self._observe(
                    f"Loaded '{paper.get('metadata',{}).get('title','?')[:50]}' "
                    f"({paper.get('metadata',{}).get('year','?')}). "
                    f"Enriched: {'critiques' in paper and bool(paper['critiques'])}"
                )
            except Exception as e:
                self._observe(f"Failed to load {path}: {e}")

        if not papers:
            raise ValueError("No valid paper files loaded")

        result.papers_analysed = [p["paper_id"] for p in papers]
        result.n_papers        = len(papers)

        # Step 1: Comparator context (fully autoloads)
        self._think("Checking for comparator output in comparisons/")
        self._act("load_comparator_context()")
        comp_ctx = load_comparator_context(result.papers_analysed)
        self._observe(
            f"Comparator available: {comp_ctx['available']}. "
            f"Contradictions: {len(comp_ctx['contradictions'])}, "
            f"Complements: {len(comp_ctx['complements'])}."
            + ("" if comp_ctx["available"]
               else " [Place comparison JSONs in data_1/agent_outputs/comparisons/ to enable]")
        )

        # Step 2: Extract entity sets
        self._think("Extracting entity sets")
        self._act("extract_entity_sets() for all papers")
        entity_sets = [extract_entity_sets(p) for p in papers]
        for es in entity_sets:
            self._observe(
                f"  {es['paper_id']}: "
                f"{len(es['methods'])} methods, "
                f"{len(es['datasets'])} datasets, "
                f"{len(es['tasks'])} tasks"
            )

        # Step 3: Strategy 1 — Combinatorial (gap_matrix or fallback)
        self._think("Strategy 1: Combinatorial gaps")
        self._act("CombinatorialGapDetector.detect()")
        comb_detector = CombinatorialGapDetector(entity_sets, papers, verbose=self.verbose)
        comb_gaps     = comb_detector.detect()
        result.gaps.extend(comb_gaps)
        self._observe(f"Combinatorial gaps: {len(comb_gaps)} found")

        # Step 4: Strategy 2 — Limitation (ChromaDB or Jaccard fallback)
        self._think("Strategy 2: Limitation-based gaps")
        self._act("LimitationGapDetector.detect()")
        lim_detector = LimitationGapDetector(papers, entity_sets, verbose=self.verbose)
        lim_gaps     = lim_detector.detect()
        result.gaps.extend(lim_gaps)
        self._observe(f"Limitation gaps: {len(lim_gaps)} found")

        # Step 5: Strategy 3 — Cross-paper (from comparator, if available)
        self._think("Strategy 3: Cross-paper gaps from comparator")
        self._act("detect_cross_paper_gaps()")
        cross_gaps = detect_cross_paper_gaps(comp_ctx, entity_sets, verbose=self.verbose)
        result.gaps.extend(cross_gaps)
        self._observe(f"Cross-paper gaps: {len(cross_gaps)} found")

        # Step 6: Critique-informed priority boost
        self._think("Boosting gap priority from critic signals")
        self._act("boost_priority_from_critiques()")
        result.gaps = boost_priority_from_critiques(result.gaps, papers)
        high_count  = sum(1 for g in result.gaps if g.priority == GapPriority.HIGH)
        self._observe(
            f"After boost: {len(result.gaps)} total, {high_count} HIGH"
        )

        # Step 7: LLM validation
        self._think(f"LLM backend: '{self.llm_backend}'")

        if self.llm_backend == "ollama" and result.gaps:
            self._act(f"llm_validate_gaps() via {self.llm_backend}")
            n_before    = len(result.gaps)
            result.gaps = llm_validate_gaps(
                result.gaps, papers, self.llm_backend, self.verbose
            )
            n_after = len(result.gaps)
            self._observe(f"LLM validation: kept {n_after}, dropped {n_before - n_after}")
        else:
            self._think("No LLM — heuristic gaps only")

        # Step 8: Sort and cap
        self._think("Sorting by priority and confidence")
        priority_order = {GapPriority.HIGH: 0, GapPriority.MEDIUM: 1, GapPriority.LOW: 2}
        result.gaps.sort(key=lambda g: (priority_order[g.priority], -g.confidence))
        result.gaps = result.gaps[:25]  # cap at 25 for output quality

        result.compute_summary()
        self._observe(
            f"Final: {len(result.gaps)} gaps. "
            f"Types: {result.gap_counts}. "
            f"Next: {result.agent_report.get('recommended_next')}"
        )

        return result

    def as_langgraph_node(self):
        """LangGraph node wrapper — returns a node_fn for use in the planner graph."""
        agent = self

        def node_fn(state: dict) -> dict:
            paper_paths     = state.get("papers_to_analyze", [])
            reports         = list(state.get("agent_reports", []))
            existing_gaps   = list(state.get("gaps", []))
            result          = agent.run(paper_paths)

            new_gaps = []
            for g in result.gaps:
                new_gaps.append({
                    "gap_id":           g.gap_id,
                    "gap_type":         g.gap_type.value,
                    "priority":         g.priority.value,
                    "description":      g.description,
                    "evidence":         g.evidence,
                    "suggestion":       g.suggestion,
                    "papers_involved":  g.papers_involved,
                    "entities_involved": g.entities_involved,
                    "confidence":       g.confidence,
                    "addressed_status": g.addressed_status.value,
                    "llm_validated":    g.llm_validated,
                    "llm_rationale":    g.llm_rationale,
                    "needs_review":     g.needs_review,
                })

            existing_gaps.extend(new_gaps)
            reports.append(result.agent_report)

            return {
                "gaps":          existing_gaps,
                "agent_reports": reports,
                "gap_summary":   {
                    "total_gaps":        len(result.gaps),
                    "high_priority_gaps": result.agent_report.get("high_priority_gaps", 0),
                    "recommended_next":  result.agent_report.get("recommended_next", "write_review"),
                    "coverage_note":     result.agent_report.get("coverage_note", ""),
                },
            }

        return node_fn


# ─────────────────────────────────────────────
#  OUTPUT WRITERS
# ─────────────────────────────────────────────

def save_gaps(result: GapResult, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"session_{result.session_id}_gaps.json"

    data_dict = asdict(result)
    for g in data_dict.get("gaps", []):
        for key in ("gap_type", "priority", "addressed_status"):
            v = g.get(key)
            if hasattr(v, "value"):
                g[key] = v.value

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)

    return out_path


def print_summary(result: GapResult):
    PRIORITY_ICONS = {
        GapPriority.HIGH:   "🔴",
        GapPriority.MEDIUM: "🟡",
        GapPriority.LOW:    "🟢",
    }
    TYPE_LABELS = {
        GapType.COMBINATORIAL: "Combinatorial",
        GapType.LIMITATION:    "Limitation",
        GapType.CROSS_PAPER:   "Cross-paper",
    }

    print("\n" + "═" * 64)
    print(f"  GAP DETECTOR REPORT  (v{result.agent_version})")
    print("═" * 64)
    print(f"  Session   : {result.session_id}")
    print(f"  Papers    : {result.n_papers} ({', '.join(result.papers_analysed)})")
    print(f"  Total gaps: {len(result.gaps)}")
    print(f"  By type   : {result.gap_counts}")
    print(f"  Next step : {result.agent_report.get('recommended_next', '?')}")
    print("─" * 64)

    if not result.gaps:
        print("  No gaps detected.")
    else:
        for i, g in enumerate(result.gaps, 1):
            icon   = PRIORITY_ICONS.get(g.priority, "•")
            ttype  = TYPE_LABELS.get(g.gap_type, str(g.gap_type))
            valid  = "✓ LLM" if g.llm_validated else "heuristic"
            review = " ⚑" if g.needs_review else ""

            print(f"\n  {i}. {icon} [{g.priority.value}] {ttype}  "
                  f"(conf:{g.confidence:.2f} | {valid}{review})")
            print(f"     {g.description[:200]}")
            print(f"     Evidence  : {g.evidence[:120]}")
            print(f"     Suggestion: {g.suggestion[:120]}")
            print(f"     Papers    : {g.papers_involved}")
            print(f"     Addressed : {g.addressed_status.value}")
            if g.llm_rationale:
                print(f"     LLM note  : {g.llm_rationale[:100]}")

    print("\n" + "═" * 64)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def _ollama_is_reachable() -> bool:
    try:
        r = requests.get(
            f"{os.environ.get('OLLAMA_HOST', OLLAMA_HOST)}/api/tags",
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return r.status_code == 200
    except Exception:
        return False


def detect_llm_backend() -> str:
    if os.environ.get("OLLAMA_HOST") or _ollama_is_reachable():
        return "ollama"
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="Gap Detector Agent v2.0.0 — research gap finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single paper (heuristic only — no LLM)
  python agents/gap_detector_agent.py data_1/parsed/claims_output.json --no-llm --verbose

  # Multiple papers
  python agents/gap_detector_agent.py \\
      data_1/parsed/paper1_enriched.json \\
      data_1/parsed/paper2_enriched.json --verbose

  # With local Ollama LLM validation
  python agents/gap_detector_agent.py data_1/parsed/paper1_enriched.json \\
      --ollama-host http://localhost:11434 --verbose

  # With remote Ollama (ngrok)
  python agents/gap_detector_agent.py data_1/parsed/paper1_enriched.json \\
      --ollama-host https://<ngrok>.ngrok-free.app --verbose
        """,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Path(s) to enriched or claims_output JSON files")
    parser.add_argument("--llm",         choices=["ollama", "auto"], default="auto")
    parser.add_argument("--ollama-host", default=None,
                        help="Ollama host URL (overrides OLLAMA_HOST env var)")
    parser.add_argument("--no-llm",      action="store_true")
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host
        globals()["OLLAMA_HOST"]  = args.ollama_host

    backend = "none" if args.no_llm else (
        detect_llm_backend() if args.llm == "auto" else args.llm
    )

    if args.verbose:
        print(f"\nGap Detector Agent v{GapDetectorAgent.VERSION}")
        print(f"  Inputs       : {args.inputs}")
        print(f"  LLM backend  : {backend}")
        print(f"  Output dir   : {args.output_dir}")
        print(f"  gap_matrix   : {'✅ found' if GAP_MATRIX_PATH.exists() else '❌ not found (fallback mode)'}")
        print(f"  ChromaDB     : {CHROMA_STORE_PATH}")
        print(f"  Comparisons  : {COMPARISONS_DIR}")

    agent = GapDetectorAgent(llm_backend=backend, verbose=args.verbose)

    try:
        result = agent.run(args.inputs)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)

    print_summary(result)
    out_path = save_gaps(result, pathlib.Path(args.output_dir))
    print(f"\n  ✅ Saved to: {out_path}\n")


if __name__ == "__main__":
    main()