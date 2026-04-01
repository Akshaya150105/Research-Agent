"""
gap_detector_agent.py  v1.0.0
==============================
Person 3 — AI Agents  |  Branch: feat/agents  |  Folder: agents/
 
Detects two types of research gaps from enriched paper JSONs:
 
  Strategy 1 — Combinatorial gaps
    Builds a method × dataset × task co-occurrence matrix across all input
    papers. Finds high-frequency methods/tasks that were never tested on
    certain datasets. These are "untried combinations" — the core gap type.
 
  Strategy 2 — Limitation-based gaps
    Clusters limitation statements across papers by semantic similarity
    (token overlap, no embeddings needed — FAISS stub ready for Phase 4).
    Finds repeated unsolved problems. Checks if later papers address them.
 
  LLM validation
    Every candidate gap is validated by the LLM before being accepted.
    The LLM checks: is this a genuine gap? is it trivial? why has nobody
    tried it? Confidence is set based on LLM response.
 
Comparator stub
    load_comparator_context() checks for comparator output in
    data_1/agent_outputs/comparisons/ and loads contradiction edges if
    present. Returns empty dict if comparator hasn't run yet.
    Wire this in when comparator_agent.py is built (Phase 4).
 
LangGraph compatibility
    run() returns a CritiqueResult-style GapResult plus an agent_report
    dict that planner_agent.py reads to decide next routing step.
 
Input
    One or more enriched JSON files (output of critic_agent.py write-back):
      data_1/parsed/{paper_id}_enriched.json
    OR plain claims_output.json files if critic hasn't run yet.
 
Output
    data_1/agent_outputs/gaps/session_{id}_gaps.json
 
Usage
-----
# Single paper
python agents/gap_detector_agent.py data_1/parsed/claims_output.json
 
# Multiple papers (pass all enriched JSONs)
python agents/gap_detector_agent.py \\
    data_1/parsed/paper1_enriched.json \\
    data_1/parsed/paper2_enriched.json
 
# With LLM validation
GEMINI_API_KEY=... python agents/gap_detector_agent.py data_1/parsed/claims_output.json
 
# Verbose ReAct trace
python agents/gap_detector_agent.py data_1/parsed/claims_output.json --verbose
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
 
OUTPUT_DIR       = pathlib.Path("data_1/agent_outputs/gaps")
COMPARISONS_DIR  = pathlib.Path("data_1/agent_outputs/comparisons")
 
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
 
# Minimum frequency for an entity to be considered "common enough" to flag
MIN_ENTITY_FREQUENCY   = 1   # with 1 paper, every entity counts; rises with more papers
MIN_GAP_CONFIDENCE     = 0.4  # below this, gap is dropped even if LLM validated
 
# Similarity threshold for limitation clustering (token-overlap Jaccard)
LIMITATION_CLUSTER_THRESHOLD = 0.35
 
# How many tokens must overlap for two limitations to be "about the same problem"
MIN_CLUSTER_SIZE = 1   # even a single repeated limitation is worth noting
 
LLM_TIMEOUT_SECONDS  = 60
LLM_MAX_RETRIES      = 3
LLM_RETRY_BASE_DELAY = 2
 
 
# ─────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────
 
class GapType(str, Enum):
    COMBINATORIAL  = "combinatorial"   # method/task never tested on certain dataset
    LIMITATION     = "limitation"      # repeated unsolved problem across papers
    CROSS_PAPER    = "cross_paper"     # contradiction-based gap (needs comparator)
 
 
class GapPriority(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"
 
 
class AddressedStatus(str, Enum):
    NOT_ADDRESSED      = "not_addressed"
    PARTIALLY_ADDRESSED = "partially_addressed"
    ADDRESSED          = "addressed"
    UNKNOWN            = "unknown"
 
 
@dataclass
class Gap:
    gap_id:            str
    gap_type:          GapType
    priority:          GapPriority
    description:       str          # human-readable gap description
    evidence:          str          # what in the data supports this gap
    suggestion:        str          # what research would address it
    papers_involved:   list[str]    # paper_ids that contribute to this gap
    entities_involved: list[str]    # methods/datasets/tasks in the gap
    confidence:        float        # 0.0–1.0
    addressed_status:  AddressedStatus = AddressedStatus.UNKNOWN
    llm_validated:     bool          = False
    llm_rationale:     str           = ""
    needs_review:      bool          = False  # True if confidence < 0.6
 
 
@dataclass
class GapResult:
    session_id:      str
    generated_at:    str
    agent_version:   str = "1.0.0"
    papers_analysed: list[str] = field(default_factory=list)
    n_papers:        int       = 0
    gaps:            list[Gap] = field(default_factory=list)
    gap_counts:      dict      = field(default_factory=dict)
    react_trace:     list[str] = field(default_factory=list)
    # LangGraph planner signal
    agent_report:    dict      = field(default_factory=dict)
 
    def compute_summary(self):
        counts = {t.value: 0 for t in GapType}
        for g in self.gaps:
            counts[g.gap_type.value] += 1
        self.gap_counts = counts
 
        high_count = sum(1 for g in self.gaps if g.priority == GapPriority.HIGH)
        self.agent_report = {
            "agent_name":        "gap_detector",
            "status":            "complete",
            "n_gaps_found":      len(self.gaps),
            "high_priority_gaps": high_count,
            "gap_types":         counts,
            # Planner routing hint
            "recommended_next":  (
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
    """
    Load a paper JSON — either enriched (has critiques key) or plain claims_output.
    Normalises field names for consistency.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Paper file not found: {p}")
 
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
 
    if "paper_id" not in raw:
        raise ValueError(f"Missing paper_id in {p}")
 
    # Normalise claim fields
    for claim in raw.get("claims", []):
        if "text" not in claim and "description" in claim:
            claim["text"] = claim["description"]
        if "type" not in claim and "claim_type" in claim:
            claim["type"] = claim["claim_type"]
 
    # Normalise entity_index location
    if "entity_index" not in raw and "entities" in raw:
        if isinstance(raw["entities"], dict) and "method" in raw["entities"]:
            raw["entity_index"] = raw["entities"]
        else:
            raw["entity_index"] = {}
 
    raw.setdefault("entity_index", {})
    raw.setdefault("claims",       [])
    raw.setdefault("limitations",  [])
    raw.setdefault("future_work",  [])
    raw.setdefault("metadata",     {})
    raw.setdefault("critiques",    [])        # may not exist if critic hasn't run
    raw.setdefault("critique_summary", {})
 
    return raw

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

    # ❌ hardware / infra
    if any(x in m for x in ["gpu", "dgx", "tesla", "system"]):
        return "NOT_METHOD"

    # ❌ generic / meaningless
    if any(x in m for x in [
        "representation", "function", "code",
        "transformation", "processing"
    ]):
        return "WEAK_METHOD"

    # ✅ real ML methods
    if any(x in m for x in [
        "lstm", "rnn", "transformer",
        "attention", "cnn", "bert"
    ]):
        return "VALID_METHOD"

    # fallback
    if len(m.split()) >= 2:
        return "WEAK_METHOD"

    return "NOT_METHOD"

def is_meaningful_limitation(text: str) -> bool:
    text = text.lower()

    # ❌ remove weak / obvious / hyper-specific
    weak_phrases = [
        "future work", "we plan", "we will",
        "might be", "could be", "small number",
        "only a few", "limited experiments"
    ]

    if any(p in text for p in weak_phrases):
        return False

    # ❌ remove hyper-specific technical details
    if any(x in text for x in ["d_k", "beam size", "dropout", "label smoothing"]):
        return False

    # ✅ keep only conceptual limitations
    strong_signals = [
        "scalability", "generalization",
        "long-range", "sequential",
        "parallelization", "efficiency"
    ]

    return any(s in text for s in strong_signals)

def extract_entity_sets(paper: dict) -> dict:
    """
    Extract clean entity sets from a paper for gap analysis.
    Returns dict with methods, datasets, metrics, tasks as sets.
    """
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

        # 🔴 IMPORTANT: drop weak methods completely for now
        elif cls == "WEAK_METHOD":
            continue

    return {
        "paper_id": paper["paper_id"],
        "title": paper.get("metadata", {}).get("title", ""),
        "year": paper.get("metadata", {}).get("year"),

        "methods": methods,
        "datasets": clean_keys(ei.get("dataset", {})),
        "metrics": clean_keys(ei.get("metric", {})),
        "tasks": clean_keys(ei.get("task", {})),
    }
 
# ─────────────────────────────────────────────
#  COMPARATOR STUB
#  Wire this in when comparator_agent.py is built (Phase 4)
# ─────────────────────────────────────────────
 
def load_comparator_context(paper_ids: list[str]) -> dict:
    """
    TODO (Phase 4 — comparator_agent.py):
    Load contradiction and complement edges from comparator output.
    These enable cross-paper gap detection based on contradictions:
      "Method A contradicts Method B on Dataset X — has anyone tried
       Method A on Dataset Y?"
 
    When comparator_agent.py is built, implement this function to:
      1. Scan COMPARISONS_DIR for files matching paper_ids
      2. Load contradiction edges: {paper_a, paper_b, method, dataset, severity}
      3. Load complement edges: {paper_a, paper_b, shared_approach}
      4. Return structured dict for gap detector to use
 
    For now: returns empty context so gap detector runs without it.
    """
    if COMPARISONS_DIR.exists():
        comparison_files = list(COMPARISONS_DIR.glob("*.json"))
        if comparison_files:
            # TODO: parse and return comparison data
            pass
 
    return {
        "contradictions":  [],   # list of {paper_a, paper_b, method, dataset, severity}
        "complements":     [],   # list of {paper_a, paper_b, shared_approach}
        "available":       False,
    }
 
 
# ─────────────────────────────────────────────
#  STRATEGY 1: COMBINATORIAL GAP DETECTION
# ─────────────────────────────────────────────
 
class CombinatorialGapDetector:
    """
    Finds method × dataset and task × dataset combinations that
    appear frequently in the corpus but were never tested together.
 
    With 1 paper:  finds tasks claimed but not evaluated on available datasets
    With 2+ papers: finds cross-paper untried combinations
    """
 
    def __init__(self, entity_sets: list[dict], papers: list[dict], verbose: bool = False):
        self.entity_sets = entity_sets
        self.papers = papers
        self.verbose = verbose
        self.n_papers = len(entity_sets)
 
    def _log(self, msg: str):
        if self.verbose:
            print(f"    [combinatorial] {msg}")
 
    def _all_datasets(self) -> Counter:
        c = Counter()
        for es in self.entity_sets:
            for d in es["datasets"]:
                c[d] += 1
        return c
 
    def _all_methods(self) -> Counter:
        c = Counter()
        for es in self.entity_sets:
            for m in es["methods"]:
                c[m] += 1
        return c
 
    def _all_tasks(self) -> Counter:
        c = Counter()
        for es in self.entity_sets:
            for t in es["tasks"]:
                c[t] += 1
        return c
 
    def _method_dataset_cooccurrence(self) -> dict[str, set[str]]:
        """method → datasets it ACTUALLY co-occurs with (via claims)"""
        cooc = defaultdict(set)

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
        """task → set of datasets it was evaluated on"""
        cooc: dict[str, set[str]] = defaultdict(set)
        for es in self.entity_sets:
            for t in es["tasks"]:
                for d in es["datasets"]:
                    cooc[t].add(d)
        return cooc
 
    def detect(self) -> list[Gap]:
        seen = set()
        gaps: list[Gap] = []
        all_datasets = self._all_datasets()
        all_methods  = self._all_methods()
        all_tasks    = self._all_tasks()
 
        self._log(
            f"Corpus: {self.n_papers} papers, "
            f"{len(all_datasets)} datasets, "
            f"{len(all_methods)} methods, "
            f"{len(all_tasks)} tasks"
        )
 
        # ── Gap type A: Tasks claimed but only one dataset tested ────────────
        # Paper claims to address task X (speech recognition, MT etc.)
        # but only reports results on one dataset
        all_ds_list = list(all_datasets.keys())
        task_ds_cooc = self._task_dataset_cooccurrence()
 
        for es in self.entity_sets:
            paper_tasks    = es["tasks"]
            paper_datasets = es["datasets"]
 
            # Tasks claimed but with narrow dataset coverage
            for task in paper_tasks:
                tested_on = task_ds_cooc.get(task, set())
                if len(tested_on) <= 1 and len(paper_datasets) <= 1:
                    # Task is claimed but only tested on ≤1 dataset
                    untested_datasets = set(all_ds_list) - tested_on
                    if untested_datasets:
                        ds_str = ", ".join(list(untested_datasets)[:3])
                        gaps.append(Gap(
                            gap_id           = f"gap_comb_{uuid.uuid4().hex[:6]}",
                            gap_type         = GapType.COMBINATORIAL,
                            priority         = GapPriority.MEDIUM,
                            description      = (
                                f"Task '{task}' is addressed in "
                                f"'{es['paper_id']}' but only evaluated on "
                                f"'{list(tested_on)[0] if tested_on else 'no dataset'}'. "
                                f"No results are provided for: {ds_str}."
                            ),
                            evidence         = (
                                f"Paper claims task: '{task}'. "
                                f"Datasets used: {list(paper_datasets)}. "
                                f"Other datasets in corpus: {list(untested_datasets)[:3]}."
                            ),
                            suggestion       = (
                                f"Evaluate '{task}' on additional benchmarks "
                                f"to support generalisability claims."
                            ),
                            papers_involved  = [es["paper_id"]],
                            entities_involved = [task] + list(paper_datasets),
                            confidence       = 0.70,
                        ))
 
        # ── Gap type B: Cross-paper method × dataset gaps ────────────────────
        # Only meaningful with 2+ papers
        if self.n_papers >= 2:
            method_ds_cooc = self._method_dataset_cooccurrence()
 
            for es_a in self.entity_sets:
                for es_b in self.entity_sets:
                    if es_a["paper_id"] == es_b["paper_id"]:
                        continue
 
                    # Methods from paper A not tested on datasets from paper B
                    for method in es_a["methods"]:

                        cls = classify_method(method)

                        if cls == "NOT_METHOD":
                            continue

                        missing_datasets = []

                        for dataset in es_b["datasets"]:
                            if dataset not in es_a["datasets"]:
                                missing_datasets.append(dataset)

                        if not missing_datasets:
                            continue

                        shared_tasks = list(es_a["tasks"].intersection(es_b["tasks"]))
                        if not shared_tasks:
                            continue

                        key = (
                            normalize_method(method),
                            tuple(sorted(shared_tasks))
                        )

                        if key in seen:
                            continue
                        seen.add(key)

                        if len(shared_tasks) > 2 and len(missing_datasets) > 5:
                            priority = GapPriority.HIGH
                        elif len(missing_datasets) > 3:
                            priority = GapPriority.MEDIUM
                        else:
                            priority = GapPriority.LOW

                        confidence = min(0.9, 0.4 + 0.15 * len(shared_tasks))

                        if cls == "WEAK_METHOD":
                            confidence -= 0.2

                        gaps.append(Gap(
                            gap_id=f"gap_cross_{uuid.uuid4().hex[:6]}",
                            gap_type=GapType.COMBINATORIAL,

                            
                            priority=priority,

                            description=(
                                f"Method '{method}' from '{es_a['paper_id']}' "
                                f"has not been evaluated on multiple datasets from '{es_b['paper_id']}', "
                                f"including {', '.join(missing_datasets[:3])}. "
                                f"Both papers address shared tasks: {shared_tasks}."
                            ),

                            evidence=(
                                f"'{method}' appears in {es_a['paper_id']} but not applied to "
                                f"{len(missing_datasets)} datasets from {es_b['paper_id']}."
                            ),

                            suggestion=(
                                f"Evaluate '{method}' on broader datasets like "
                                f"{', '.join(missing_datasets[:3])} to test generalisation."
                            ),

                            papers_involved=[es_a["paper_id"], es_b["paper_id"]],
                            entities_involved=[method] + missing_datasets[:3] + shared_tasks,

                            confidence=confidence,
                        ))
 
        # Deduplicate gaps with identical entity sets
        gaps = self._deduplicate(gaps)
        self._log(f"Found {len(gaps)} combinatorial gap(s)")
        return gaps
 
    def _deduplicate(self, gaps: list[Gap]) -> list[Gap]:
        seen = set()
        unique = []

        def normalize(text):
            text = text.lower()
            text = re.sub(r"(dataset|task|evaluation|benchmark)", "", text)
            return text.strip()

        for g in gaps:
            key = tuple(sorted(normalize(e) for e in g.entities_involved))
            if key not in seen:
                seen.add(key)
                unique.append(g)

        return unique
 
 
# ─────────────────────────────────────────────
#  STRATEGY 2: LIMITATION-BASED GAP DETECTION
# ─────────────────────────────────────────────
 
class LimitationGapDetector:
    """
    Clusters limitation statements across papers using token-overlap
    Jaccard similarity. No embeddings needed — FAISS stub ready for
    Phase 4 when embeddings are available.
 
    Finds repeated unsolved problems and checks if later papers address them.
    """
 
    def __init__(
        self,
        papers: list[dict],
        entity_sets: list[dict],
        verbose: bool = False
    ):
        self.papers      = papers
        self.entity_sets = entity_sets
        self.verbose     = verbose
 
    def _log(self, msg: str):
        if self.verbose:
            print(f"    [limitation] {msg}")
 
    def _tokenise(self, text: str) -> set[str]:
        """Lowercase, strip punctuation, return meaningful tokens."""
        tokens = re.findall(r'\b[a-z][a-z0-9]{2,}\b', text.lower())
        # Remove very common stop words
        stops = {
            "the", "and", "for", "are", "that", "this", "with",
            "have", "been", "from", "they", "which", "their", "will",
            "not", "can", "but", "our", "more", "also", "than",
            "such", "its", "may", "each", "when", "these", "use",
        }
        return {t for t in tokens if t not in stops}
 
    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
 
    def _collect_limitations(self) -> list[dict]:
        """Collect all limitations across all papers with provenance."""
        all_lims = []
        for paper in self.papers:
            pid = paper["paper_id"]
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
            # Also include future_work as potential gap seeds
            for fw in paper.get("future_work", []):
                text = fw.get("text", "").strip()
                if len(text) > 20:
                    all_lims.append({
                        "paper_id": pid,
                        "year":     year,
                        "text":     text,
                        "tokens":   self._tokenise(text),
                        "section":  "future_work",
                        "is_future_work": True,
                    })
        return all_lims
 

    def _cluster(self, limitations: list[dict]) -> list[list[dict]]:
        """
        Clusters limitation statements using ChromaDB semantic search.
        Falls back to Jaccard token-overlap if ChromaDB is unavailable.
        """

        if not limitations:
            return []

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            CHROMA_STORE_PATH = "rag/chroma_store"
            QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

            client = chromadb.PersistentClient(path=CHROMA_STORE_PATH)
            collection = client.get_or_create_collection(
                name="claims_and_findings",
                metadata={"hnsw:space": "cosine"},
            )

            if collection.count() == 0:
                raise RuntimeError("ChromaDB empty")

            model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

            self._log(f"Using ChromaDB ({collection.count()} chunks)")

            assigned = [False] * len(limitations)
            clusters = []

            for i, lim in enumerate(limitations):
                if assigned[i]:
                    continue

                query_vec = model.encode(
                    QUERY_PREFIX + lim["text"],
                    normalize_embeddings=True,
                ).tolist()

                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(10, collection.count()),
                    include=["documents", "distances"],
                )

                similar_texts = set()
                for dist, doc in zip(results["distances"][0], results["documents"][0]):
                    if dist < 0.35:
                        similar_texts.add(doc.lower())

                cluster = [lim]
                assigned[i] = True

                for j, other in enumerate(limitations):
                    if assigned[j] or i == j:
                        continue

                    for sim_text in similar_texts:
                        overlap = self._jaccard(
                            self._tokenise(other["text"]),
                            self._tokenise(sim_text),
                        )
                        if overlap >= 0.25:
                            cluster.append(other)
                            assigned[j] = True
                            break

                clusters.append(cluster)

            return clusters

        except Exception as e:
            self._log(f"Fallback to Jaccard: {e}")
            return self._cluster_jaccard(limitations)


    def _cluster_jaccard(self, limitations: list[dict]) -> list[list[dict]]:
        if not limitations:
            return []

        clusters = []
        assigned = [False] * len(limitations)

        for i, lim_i in enumerate(limitations):
            if assigned[i]:
                continue

            cluster = [lim_i]
            assigned[i] = True

            for j, lim_j in enumerate(limitations):
                if assigned[j] or i == j:
                    continue

                sim = self._jaccard(lim_i["tokens"], lim_j["tokens"])
                if sim >= LIMITATION_CLUSTER_THRESHOLD:
                    cluster.append(lim_j)
                    assigned[j] = True

            clusters.append(cluster)

        return clusters
 
    def _check_addressed(self, cluster: list[dict]) -> AddressedStatus:
        """
        Check if a limitation cluster is addressed by a later paper.
        Simple heuristic: if future_work items appear in the cluster
        alongside limitations, and a later paper's claims mention the
        same concepts, consider it partially addressed.
 
        TODO (Phase 4): Use semantic search against all paper claims
        via FAISS when embeddings are available.
        """
        cluster_tokens = set()
        for item in cluster:
            cluster_tokens |= item["tokens"]
 
        years = [item["year"] for item in cluster if item.get("year")]
        if not years:
            return AddressedStatus.UNKNOWN
 
        max_year = max(years)
 
        # Check if any later paper's claims address this limitation
        for paper in self.papers:
            paper_year = paper.get("metadata", {}).get("year", 0)
            if paper_year <= max_year:
                continue  # Only check later papers
 
            # Check claims from this later paper
            for claim in paper.get("claims", []):
                claim_text   = claim.get("text", claim.get("description", "")).lower()
                claim_tokens = self._tokenise(claim_text)
                overlap      = cluster_tokens & claim_tokens
 
                if len(overlap) >= 3:  # Substantial overlap
                    return AddressedStatus.PARTIALLY_ADDRESSED
 
        return AddressedStatus.NOT_ADDRESSED
 
    def detect(self) -> list[Gap]:

        all_lims = self._collect_limitations()
        self._log(f"Collected {len(all_lims)} limitation/future_work statements")

        # ✅ ADD FILTER HERE
        filtered_lims = []

        for item in all_lims:
            text = item["text"]

            if not is_meaningful_limitation(text):
                continue

            filtered_lims.append(item)

        self._log(f"After filtering: {len(filtered_lims)} meaningful limitations")

        if not filtered_lims:
            self._log("No meaningful limitations found — skipping limitation gap detection")
            return []
 
        clusters = self._cluster(filtered_lims)
        self._log(f"Formed {len(clusters)} clusters (threshold={LIMITATION_CLUSTER_THRESHOLD})")
 
        gaps: list[Gap] = []
 
        for cluster in clusters:
            if len(cluster) < MIN_CLUSTER_SIZE:
                continue
 
            # Build cluster summary
            papers_in_cluster = list({item["paper_id"] for item in cluster})
            all_texts         = [item["text"] for item in cluster]
            all_tokens        = set()
            for item in cluster:
                all_tokens |= item["tokens"]
 
            # Representative text (longest, most informative)
            rep_text = max(all_texts, key=len)
 
            # Priority: more papers mentioning = higher priority
            n = len(papers_in_cluster)
            priority = GapPriority.HIGH if n >= 3 else GapPriority.MEDIUM if n >= 2 else GapPriority.LOW
 
            # Check if addressed by later work
            addressed = self._check_addressed(cluster)
 
            # Confidence based on cluster coherence
            if len(cluster) == 1:
                confidence = 0.55  # single paper limitation — needs review
            elif len(cluster) >= 3:
                confidence = 0.85
            else:
                confidence = 0.70
 
            gap = Gap(
                gap_id           = f"gap_lim_{uuid.uuid4().hex[:6]}",
                gap_type         = GapType.LIMITATION,
                priority         = priority,
                description      = (
                    f"Limitation cluster across {n} paper(s): '{rep_text[:200]}'. "
                    f"This problem appears {len(cluster)} time(s) and is "
                    f"{'not addressed' if addressed == AddressedStatus.NOT_ADDRESSED else addressed.value} "
                    "by later work in the corpus."
                ),
                evidence         = (
                    f"Papers: {papers_in_cluster}. "
                    f"Sample statements: {'; '.join([t[:150] for t in all_texts[:2]])}"
                ),
                suggestion       = (
                    f"This limitation cluster represents an open research direction. "
                    f"Future work should directly address: '{rep_text[:150]}'"
                ),
                papers_involved  = papers_in_cluster,
                entities_involved = [
                    t for t in all_tokens
                    if len(t) > 5 and t not in {"problem", "method", "model"}
                ][:6],

                confidence       = confidence,
                addressed_status = addressed,
                needs_review     = confidence < 0.6,
            )
            gaps.append(gap)
 
        self._log(f"Found {len(gaps)} limitation-based gap(s)")
        return gaps
 
 
# ─────────────────────────────────────────────
#  CRITIQUE-INFORMED PRIORITY BOOST
# ─────────────────────────────────────────────
 
def boost_priority_from_critiques(gaps: list[Gap], papers: list[dict]) -> list[Gap]:
    """
    Use critique_summary from enriched JSONs to boost gap priority.
 
    Logic:
    - If a paper has HIGH 'single_dataset_evaluation' weakness →
      boost all combinatorial gaps involving that paper to HIGH
    - If a paper has 'vague_limitations' → reduce confidence of
      limitation gaps from that paper (unreliable source)
    - If a paper has 'outdated_baselines' → boost combinatorial
      gaps (the field has moved on, lots of untried combos)
    """
    # Build lookup: paper_id → high_weakness_types
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
                    gap.confidence = min(gap.confidence + 0.1, 1.0)
 
                if "outdated_baselines" in high_types:
                    gap.priority   = GapPriority.HIGH
                    gap.confidence = min(gap.confidence + 0.05, 1.0)
 
            if gap.gap_type == GapType.LIMITATION:
                # Critic said limitations were vague — reduce confidence
                weak_types = [
                    w.get("weakness_type", "")
                    for w in paper.get("critiques", [])
                ]
                if "vague_limitations" in weak_types or "missing_limitations_future_work" in weak_types:
                    gap.confidence = max(gap.confidence - 0.15, 0.3)
                    gap.needs_review = True
 
    return gaps
 
 
# ─────────────────────────────────────────────
#  LLM VALIDATION
# ─────────────────────────────────────────────
 
def _llm_call_raw(prompt: str, llm_backend: str) -> str:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
 
    if llm_backend == "openai" and not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    if llm_backend == "gemini" and not gemini_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
 
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            if llm_backend == "gemini":
                resp = requests.post(
                    f"{GEMINI_URL}?key={gemini_key}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
                    },
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            else:
                resp = requests.post(
                    OPENAI_URL,
                    headers={"Authorization": f"Bearer {openai_key}",
                             "Content-Type": "application/json"},
                    json={
                        "model":    "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens":  2048,
                    },
                    timeout=LLM_TIMEOUT_SECONDS,
                )
 
            if resp.status_code == 429:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Rate limited after {LLM_MAX_RETRIES} attempts")
 
            resp.raise_for_status()
            raw = resp.json()
 
            if llm_backend == "gemini":
                return raw["candidates"][0]["content"]["parts"][0]["text"]
            return raw["choices"][0]["message"]["content"]
 
        except (requests.RequestException, KeyError, IndexError) as e:
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
    """Build a single batch validation prompt for all gaps."""
    # Paper context
    paper_summaries = []
    for p in papers:
        meta = p.get("metadata", {})
        ei   = p.get("entity_index", {})
        ei_d = ei if isinstance(ei, dict) else {}
        paper_summaries.append(
            f"  Paper: {p['paper_id']} ({meta.get('year', '?')})\n"
            f"  Title: {meta.get('title', '')[:80]}\n"
            f"  Datasets: {list(ei_d.get('dataset', {}).keys())}\n"
            f"  Tasks: {list(ei_d.get('task', {}).keys())[:5]}\n"
            f"  Methods: {list(ei_d.get('method', {}).keys())[:5]}"
        )
 
    gaps_text = "\n\n".join(
        f"GAP {i+1} (id={g.gap_id}, type={g.gap_type.value}):\n"
        f"  Description: {g.description[:300]}\n"
        f"  Evidence: {g.evidence[:200]}"
        for i, g in enumerate(gaps)
    )
 
    return f"""You are a research gap analyst reviewing candidate research gaps in ML/NLP.
 
CORPUS OF PAPERS:
{chr(10).join(paper_summaries)}
 
CANDIDATE GAPS TO VALIDATE:
{gaps_text}
 
For each gap, assess:
1. Is this a GENUINE gap? (not trivial, not already addressed, actually feasible)
2. Why has nobody tried this combination?
3. How valuable would it be to address this gap?
 
Reply with ONLY valid JSON — no markdown fences:
 
{{
  "validations": [
    {{
      "gap_id": "gap_id_from_above",
      "is_genuine": true,
      "confidence_adjustment": 0.0,
      "rationale": "One sentence explaining if genuine and why.",
      "addressed_status": "not_addressed|partially_addressed|addressed|unknown"
    }}
  ]
}}
 
Rules:
- confidence_adjustment: float between -0.3 and +0.2
- If trivial or already addressed: is_genuine=false, adjustment=-0.3
- If clearly genuine and important: is_genuine=true, adjustment=+0.1 to +0.2
- Keep rationale under 100 words
- Process ALL {len(gaps)} gaps"""
 
 
def llm_validate_gaps(
    gaps: list[Gap],
    papers: list[dict],
    llm_backend: str,
    verbose: bool = False,
) -> list[Gap]:
    """
    Batch validate all gaps with a single LLM call.
    Updates gap confidence, llm_validated, llm_rationale, addressed_status.
    Removes gaps where is_genuine=False.
    """
    if not gaps:
        return []
 
    prompt = build_validation_prompt(gaps, papers)
 
    if verbose:
        print(f"\n  [llm] Validating {len(gaps)} gap(s) via {llm_backend}...")
 
    try:
        text   = _llm_call_raw(prompt, llm_backend)
        try:
            parsed = _parse_llm_json(text)
        except:
            print("⚠️ LLM JSON parsing failed, retrying once...")
            text = _llm_call_raw(prompt, llm_backend)
            parsed = _parse_llm_json(text)
 
        if verbose:
            print("  [llm] Gap validation response parsed")
 
        # Build lookup by gap_id
        validation_map = {
            v["gap_id"]: v
            for v in parsed.get("validations", [])
        }
 
        validated: list[Gap] = []
        for gap in gaps:
            v = validation_map.get(gap.gap_id)
            if not v:
                # LLM didn't address this gap — keep with original confidence
                gap.llm_validated = False
                validated.append(gap)
                continue
 
            is_genuine = v.get("is_genuine", True)
            if not is_genuine:
                if verbose:
                    print(f"    [llm] Gap '{gap.gap_id}' rejected: {v.get('rationale','')[:80]}")
                continue  # Drop non-genuine gaps
 
            # Apply confidence adjustment
            adj = float(v.get("confidence_adjustment", 0.0))
            gap.confidence = max(0.0, min(1.0, gap.confidence + adj))
            gap.llm_validated  = True
            gap.llm_rationale  = v.get("rationale", "")
 
            # Update addressed status if LLM has better info
            status_str = v.get("addressed_status", "")
            try:
                gap.addressed_status = AddressedStatus(status_str)
            except ValueError:
                pass
 
            # Apply minimum confidence filter
            if gap.confidence >= MIN_GAP_CONFIDENCE:
                gap.needs_review = gap.confidence < 0.6
                validated.append(gap)
            else:
                if verbose:
                    print(f"    [llm] Gap '{gap.gap_id}' dropped (confidence {gap.confidence:.2f} < {MIN_GAP_CONFIDENCE})")
 
        return validated
 
    except Exception as e:
        print("\n⚠️ LLM JSON broken or invalid response")
        print(f"   Error: {e}")
        print("   → Skipping LLM validation, using heuristic gaps\n")
        return gaps
    
 
# ─────────────────────────────────────────────
#  REACT AGENT
# ─────────────────────────────────────────────
 
class GapDetectorAgent:
    VERSION = "1.0.0"
 
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
 
    def run(self, paper_paths: list[str | pathlib.Path]) -> GapResult:
 
        session_id = uuid.uuid4().hex[:8]
        result = GapResult(
            session_id   = session_id,
            generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            agent_version = self.VERSION,
            react_trace   = self.trace,
        )
 
        # ── Step 0: Load papers ──────────────────────────────────────────────
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
 
        # ── Step 1: Load comparator context (stub) ───────────────────────────
        self._think("Checking for comparator output (Phase 4 stub)")
        self._act("load_comparator_context()")
        comp_ctx = load_comparator_context(result.papers_analysed)
        self._observe(
            f"Comparator context available: {comp_ctx['available']}. "
            f"Contradictions: {len(comp_ctx['contradictions'])}, "
            f"Complements: {len(comp_ctx['complements'])}."
            + (" [TODO: wire in when comparator_agent.py is built]"
               if not comp_ctx["available"] else "")
        )
 
        # ── Step 2: Extract entity sets ──────────────────────────────────────
        self._think("Extracting entity sets for gap matrix analysis")
        self._act("extract_entity_sets() for all papers")
        entity_sets = [extract_entity_sets(p) for p in papers]
        for es in entity_sets:
            self._observe(
                f"  {es['paper_id']}: "
                f"{len(es['methods'])} methods, "
                f"{len(es['datasets'])} datasets, "
                f"{len(es['tasks'])} tasks"
            )
 
        # ── Step 3: Strategy 1 — Combinatorial gaps ──────────────────────────
        self._think("Running Strategy 1: Combinatorial gap detection")
        self._act("CombinatorialGapDetector.detect()")
        comb_detector = CombinatorialGapDetector(entity_sets, papers, verbose=self.verbose)
        comb_gaps     = comb_detector.detect()
        result.gaps.extend(comb_gaps)
        self._observe(
            f"Combinatorial gaps: {len(comb_gaps)} found"
        )
 
        # ── Step 4: Strategy 2 — Limitation-based gaps ───────────────────────
        self._think("Running Strategy 2: Limitation-based gap detection")
        self._act("LimitationGapDetector.detect()")
        lim_detector = LimitationGapDetector(papers, entity_sets, verbose=self.verbose)
        lim_gaps     = lim_detector.detect()
        result.gaps.extend(lim_gaps)
        self._observe(
            f"Limitation gaps: {len(lim_gaps)} found"
        )
 
        # ── Step 5: Boost priority from critique signals ─────────────────────
        self._think("Boosting gap priority using critic agent signals")
        self._act("boost_priority_from_critiques()")
        result.gaps = boost_priority_from_critiques(result.gaps, papers)
        high_count = sum(1 for g in result.gaps if g.priority == GapPriority.HIGH)
        self._observe(
            f"After critique boost: {len(result.gaps)} gaps total, "
            f"{high_count} HIGH priority"
        )
 
        # ── Step 6: LLM validation ───────────────────────────────────────────
        self._think(f"LLM backend: '{self.llm_backend}'. Running gap validation.")
 
        if self.llm_backend in ("openai", "gemini") and result.gaps:
            self._act(f"llm_validate_gaps() via {self.llm_backend}")
            n_before = len(result.gaps)
            time.sleep(10)  # rate limit breathing room
            result.gaps = llm_validate_gaps(
                result.gaps, papers, self.llm_backend, self.verbose
            )
            n_after = len(result.gaps)
            self._observe(
                f"LLM validation complete. "
                f"Kept: {n_after}, Dropped: {n_before - n_after}"
            )
        else:
            self._think(
                "No LLM backend — skipping validation. "
                "All gaps kept with heuristic confidence scores."
            )
 
        # ── Step 7: Sort and finalise ─────────────────────────────────────────
        self._think("Sorting gaps by priority and confidence")
        priority_order = {GapPriority.HIGH: 0, GapPriority.MEDIUM: 1, GapPriority.LOW: 2}
        MAX_GAPS = 25
        result.gaps = result.gaps[:MAX_GAPS]
        result.gaps.sort(
            key=lambda g: (priority_order[g.priority], -g.confidence)
        )
        result.compute_summary()
 
        self._observe(
            f"Final: {len(result.gaps)} gaps. "
            f"Types: {result.gap_counts}. "
            f"Recommended next: {result.agent_report.get('recommended_next')}"
        )
 
        return result

    # ✅ NOW AT CLASS LEVEL (same level as run)
    def as_langgraph_node(self):
        agent = self

        def node_fn(state: dict) -> dict:
            paper_paths = state.get("papers_to_analyze", [])
            reports = list(state.get("agent_reports", []))
            existing_gaps = list(state.get("gaps", []))

            result = agent.run(paper_paths)

            new_gaps = []
            for g in result.gaps:
                new_gaps.append({
                    "gap_id": g.gap_id,
                    "gap_type": g.gap_type.value,
                    "priority": g.priority.value,
                    "description": g.description,
                    "evidence": g.evidence,
                    "suggestion": g.suggestion,
                    "papers_involved": g.papers_involved,
                    "entities_involved": g.entities_involved,
                    "confidence": g.confidence,
                    "addressed_status": g.addressed_status.value,
                    "llm_validated": g.llm_validated,
                    "llm_rationale": g.llm_rationale,
                    "needs_review": g.needs_review,
                })

            existing_gaps.extend(new_gaps)
            reports.append(result.agent_report)

            return {
                "gaps": existing_gaps,
                "agent_reports": reports,
                "gap_summary": {
                    "total_gaps": len(result.gaps),
                    "high_priority_gaps": result.agent_report.get("high_priority_gaps", 0),
                    "recommended_next": result.agent_report.get("recommended_next", "write_review"),
                    "coverage_note": result.agent_report.get("coverage_note", ""),
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
    # Fix enum serialisation
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
    print("  GAP DETECTOR REPORT  (v1.0.0)")
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
            icon  = PRIORITY_ICONS.get(g.priority, "•")
            ttype = TYPE_LABELS.get(g.gap_type, g.gap_type)
            valid = "✓ LLM" if g.llm_validated else "heuristic"
            conf  = f"{g.confidence:.2f}"
            review = " ⚑" if g.needs_review else ""
 
            print(f"\n  {i}. {icon} [{g.priority.value}] {ttype}  "
                  f"(conf:{conf} | {valid}{review})")
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
 
def detect_llm_backend() -> str:
    if os.environ.get("OPENAI_API_KEY"): return "openai"
    if os.environ.get("GEMINI_API_KEY"): return "gemini"
    return "none"
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Gap Detector Agent v1.0.0 — research gap finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single paper (heuristic only)
  python agents/gap_detector_agent.py data_1/parsed/claims_output.json
 
  # Multiple papers with LLM validation
  GEMINI_API_KEY=... python agents/gap_detector_agent.py \\
      data_1/parsed/paper1_enriched.json \\
      data_1/parsed/paper2_enriched.json
 
  # Verbose ReAct trace
  python agents/gap_detector_agent.py data_1/parsed/claims_output.json --verbose
        """,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Path(s) to enriched JSON or claims_output.json files"
    )
    parser.add_argument("--llm", choices=["openai", "gemini", "auto"], default="auto")
    parser.add_argument("--no-llm",    action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--verbose", "-v", action="store_true")
 
    args = parser.parse_args()
 
    backend = "none" if args.no_llm else (
        detect_llm_backend() if args.llm == "auto" else args.llm
    )
 
    if args.verbose:
        print(f"\nGap Detector Agent v{GapDetectorAgent.VERSION}")
        print(f"  Inputs     : {args.inputs}")
        print(f"  LLM backend: {backend}")
        print(f"  Output dir : {args.output_dir}")
 
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