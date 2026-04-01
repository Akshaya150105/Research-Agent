"""
entity_resolver.py  v2.0.0
===========================
Folder: agents/

v2.0.0 fixes three bugs from v1.0.0:

BUG 1 — Fixed waterfall type-trying
  Old: try metric → if fail, try dataset → if fail, try method
  Problem: "BLEU" fails metric at sim=0.657 → gets tried as dataset
           → accidentally resolves to a dataset → breaks contradiction
  Fix: try ALL types, return highest similarity winner

BUG 2 — Missing domain knowledge
  "BLEU" should ALWAYS be tried as a metric first.
  "WMT" should ALWAYS be tried as a dataset first.
  Fix: KNOWN_METRIC_PATTERNS and KNOWN_DATASET_PATTERNS as fast-path guards.

BUG 3 — generalize() function in comparator
  The old `generalize()` stripped "score" from "bleu score" producing "bleu",
  then matched it against a different "bleu" key from another paper.
  The dataset key "bleuscore" appeared because normalization collapsed
  metric and dataset names together — producing 12 nonsense contradictions.
  Fix: remove generalize() entirely. Use resolver + exact canonical key matching.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Optional
from canonical_normalizer import normalize_resolved

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MATCH_THRESHOLD = 0.75
N_CANDIDATES    = 3

# ── Domain knowledge guards ───────────────────────────────────
# These patterns force the correct entity_type BEFORE trying ChromaDB.
# Prevents "BLEU" from being tried as dataset/method at all.

KNOWN_METRIC_PATTERNS = re.compile(
    r"^(bleu|rouge[\-\d]*|f[\-\s]?1|f\d|accuracy|perplexity|ppl|"
    r"recall|precision|auc|map|mrr|ndcg|wer|cer|ter|meteor|"
    r"sacrebleu|cider|bits per char|bpc|top[\-\s]?\d|"
    r"training loss|test loss|val loss|cross.entropy|"
    r"bleu score|edit distance|exact match)$",
    re.IGNORECASE,
)

KNOWN_DATASET_PATTERNS = re.compile(
    r"^(wmt|wmt[\d\-]+|newstest[\d]*|squad|glue|superglue|conll|"
    r"penn treebank|wsj|ptb|imdb|sst[\-\d]*|mnli|snli|qnli|qqp|"
    r"cola|sts[\-\d]*|mrpc|rte|wnli|iwslt|europarl|"
    r"ntst[\d]*|ntst14|newstest2013|newstest2014|"
    r"wmt[\'\u2019]?14|wmt 2014)",
    re.IGNORECASE,
)


class EmbeddingEntityResolver:
    """
    Resolves raw entity strings from claim["entities_involved"] to
    canonical forms in ChromaDB entities_global.

    Key design decisions in v2.0.0:
      - Domain knowledge patterns guard the most common metric/dataset names
      - All entity types compete by similarity; highest winner is chosen
      - String fallback uses exact then substring matching
      - Results cached per (raw, type, paper_id) for session efficiency
    """

    def __init__(self, verbose: bool = False):
        self.verbose      = verbose
        self._cache:  dict[tuple, tuple[Optional[str], float]] = {}
        self._chroma: Optional[object] = None
        self._embedder_fn              = None
        self._available: Optional[bool] = None

    # ── RAG init ──────────────────────────────────────────────

    def _init_rag(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from rag.indexer  import get_collections
            from rag.embedder import embed_query
            collections       = get_collections()
            self._chroma      = collections.get("entities_global")
            self._embedder_fn = embed_query
            if self._chroma is None:
                raise RuntimeError("entities_global collection not found")
            self._vprint(f"entities_global has {self._chroma.count()} docs")
            self._available = True
        except Exception as e:
            self._vprint(f"RAG unavailable ({e}) — string fallback active")
            self._available = False
        return self._available

    def _vprint(self, msg: str):
        if self.verbose:
            print(f"  [resolver] {msg}")

    # ── Public API ────────────────────────────────────────────

    def resolve_claim(
        self,
        claim:    dict,
        paper_id: str,
        typed:    dict[str, set],
    ) -> dict:
        """
        Resolve all strings in claim["entities_involved"].
        Returns {metric: str|None, dataset: str|None, methods: list}.

        Algorithm per entity string:
          1. KNOWN_METRIC_PATTERNS match → try metric only
          2. KNOWN_DATASET_PATTERNS match → try dataset only
          3. Otherwise → compete all types, take highest sim winner
        """
        resolved: dict = {"metric": None, "dataset": None, "methods": []}

        for raw in claim.get("entities_involved", []):
            raw = raw.strip()
            if not raw:
                continue

            # Guard 1: known metric
            if KNOWN_METRIC_PATTERNS.match(raw):
                if resolved["metric"] is None:
                    canon, _ = self._best_match(raw, ["metric"], paper_id, typed)
                    if canon:
                        resolved["metric"] = canon
                        continue
                # Metric slot full — ignore, it's not a dataset or method
                continue

            # Guard 2: known dataset
            if KNOWN_DATASET_PATTERNS.match(raw):
                if resolved["dataset"] is None:
                    canon, _ = self._best_match(raw, ["dataset"], paper_id, typed)
                    if canon:
                        resolved["dataset"] = canon
                        continue
                continue

            # General: compete all unfilled types
            types_to_try = []
            if resolved["metric"]  is None: types_to_try.append("metric")
            if resolved["dataset"] is None: types_to_try.append("dataset")
            types_to_try.extend(["method", "task"])

            canon, sim = self._best_match(raw, types_to_try, paper_id, typed)
            if not canon:
                continue

            # Assign to the right slot
            # We need to know which type won — re-check
            winning_type = self._winning_type(raw, types_to_try, paper_id, typed)
            if winning_type == "metric"  and resolved["metric"]  is None:
                resolved["metric"]  = canon
            elif winning_type == "dataset" and resolved["dataset"] is None:
                resolved["dataset"] = canon
            elif winning_type in ("method", "task"):
                resolved["methods"].append(canon)

        resolved = normalize_resolved(resolved)
        return resolved

    # ── Internal resolution ───────────────────────────────────

    def _best_match(
        self,
        raw:        str,
        types:      list[str],
        paper_id:   str,
        typed:      dict[str, set],
    ) -> tuple[Optional[str], float]:
        """Try given types, return (best_canonical, best_similarity)."""
        best_canon = None
        best_sim   = 0.0

        for etype in types:
            fallback = typed.get(etype + "s", set())
            canon, sim = self._resolve_one(raw, etype, paper_id, fallback)
            if canon and sim > best_sim:
                best_sim   = sim
                best_canon = canon

        return best_canon, best_sim

    def _winning_type(
        self,
        raw:      str,
        types:    list[str],
        paper_id: str,
        typed:    dict[str, set],
    ) -> Optional[str]:
        """Return which entity_type gave the highest similarity for raw."""
        best_type = None
        best_sim  = 0.0

        for etype in types:
            fallback = typed.get(etype + "s", set())
            canon, sim = self._resolve_one(raw, etype, paper_id, fallback)
            if canon and sim > best_sim:
                best_sim  = sim
                best_type = etype

        return best_type

    def _resolve_one(
        self,
        raw:          str,
        entity_type:  str,
        paper_id:     str,
        fallback_keys: set,
    ) -> tuple[Optional[str], float]:
        """
        Resolve raw against one entity_type.
        Returns (canonical, similarity). Caches result.
        """
        cache_key = (raw.lower().strip(), entity_type, paper_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # ChromaDB semantic search
        if self._init_rag():
            try:
                q_vec   = self._embedder_fn(raw)
                results = self._chroma.query(
                    query_embeddings=[q_vec],
                    n_results=N_CANDIDATES,
                    where={"$and": [
                        {"paper_id":    {"$eq": paper_id}},
                        {"entity_type": {"$eq": entity_type}},
                    ]},
                    include=["metadatas", "distances"],
                )
                if results["ids"] and results["ids"][0]:
                    dist      = results["distances"][0][0]
                    sim       = round(1.0 - dist, 4)
                    canonical = results["metadatas"][0][0].get(
                        "entity_text_normalized", ""
                    )
                    self._vprint(
                        f"'{raw}' →[{entity_type}] '{canonical}' sim={sim:.3f}"
                    )
                    if sim >= MATCH_THRESHOLD and canonical:
                        self._cache[cache_key] = (canonical, sim)
                        return canonical, sim
            except Exception as e:
                self._vprint(f"ChromaDB error '{raw}': {e}")

        # String fallback
        norm = raw.lower().strip()
        if norm in fallback_keys:
            self._cache[cache_key] = (norm, 1.0)
            return norm, 1.0
        for key in fallback_keys:
            if norm in key or key in norm:
                self._cache[cache_key] = (key, 0.85)
                return key, 0.85

        self._cache[cache_key] = (None, 0.0)
        return None, 0.0

    def cache_stats(self) -> dict:
        hits   = sum(1 for v, _ in self._cache.values() if v is not None)
        misses = sum(1 for v, _ in self._cache.values() if v is None)
        return {
            "total_queries": len(self._cache),
            "resolved":      hits,
            "unresolved":    misses,
            "hit_rate":      hits / len(self._cache) if self._cache else 0.0,
        }


def get_typed_entities(paper: dict) -> dict[str, set]:
    """
    Extract typed entity sets from claims_output.json.
    Returns {methods, datasets, metrics, tasks} — all lowercase.
    Used as string-fallback keys by EmbeddingEntityResolver.
    """
    for key in ("entity_index", "llm_entity_index"):
        ei = paper.get(key)
        if ei and isinstance(ei, dict):
            return {
                "methods":  set(ei.get("method",  {}).keys()),
                "datasets": set(ei.get("dataset", {}).keys()),
                "metrics":  set(ei.get("metric",  {}).keys()),
                "tasks":    set(ei.get("task",    {}).keys()),
            }
    typed: dict[str, set] = {
        "methods": set(), "datasets": set(),
        "metrics": set(), "tasks":    set(),
    }
    for ent in paper.get("entities", []) + paper.get("llm_entities", []):
        etype = ent.get("entity_type", "")
        text  = ent.get("text", "").lower().strip()
        if not text:
            continue
        if etype == "method":    typed["methods"].add(text)
        elif etype == "dataset": typed["datasets"].add(text)
        elif etype == "metric":  typed["metrics"].add(text)
        elif etype == "task":    typed["tasks"].add(text)
    return typed