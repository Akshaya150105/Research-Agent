"""
retriever.py
------------
Takes a query plan from query_handler.py, searches ChromaDB,
re-ranks results with a cross-encoder, and returns top-k chunks
with full provenance.

Two-stage retrieval:
    Stage 1 — ChromaDB ANN search
        Embeds the query, applies metadata filter, fetches n_results
        candidates from each target collection.

    Stage 2 — Cross-encoder re-ranking
        Scores each (query, document) pair more precisely.
        Multiplies cross-encoder score by chunk confidence.
        Returns top_k from the re-ranked list.

Why two stages?
    ChromaDB's ANN search is fast but approximate.
    The cross-encoder is slower but much more precise.
    Fetching 12 candidates and re-ranking to top 5 gives
    the best of both: speed + precision.

Usage:
    from rag.retriever import retrieve

    results = retrieve("What are the limitations of STGCN?", top_k=5)
    for r in results:
        print(r["score"], r["document"], r["paper_id"])

Install:
    pip install sentence-transformers  (already installed)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR     = Path("memory")
CROSS_ENCODER  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# module-level cross-encoder cache
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError("sentence-transformers not installed.\nRun: pip install sentence-transformers")

    print(f"  Loading cross-encoder: {CROSS_ENCODER}")
    print(f"  First run downloads ~85MB to ~/.cache/huggingface/\n")
    _cross_encoder = CrossEncoder(CROSS_ENCODER)
    return _cross_encoder


# ── public entry point ────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 5, known_methods: list = None) -> list[dict]:
    """
    Full retrieval pipeline for one question.

    Args:
        question:      Natural language question from the user or agent.
        top_k:         Number of results to return after re-ranking.
        known_methods: Optional list of known method names for query enrichment.

    Returns:
        List of result dicts, sorted by final score descending.
        Each result has:
            document      - the chunk text (display_text)
            score         - final re-ranked score (0.0 to 1.0)
            paper_id      - which paper this came from
            chunk_type    - claim / limitation / section / entity / etc.
            chunk_id      - stable ID for this chunk
            collection    - which ChromaDB collection it came from
            metadata      - full metadata dict from ChromaDB
            intent        - detected intent from query_handler
    """
    from rag.query_handler import parse_query

    plan = parse_query(question, known_methods=known_methods)

    # gap queries don't use vector search — read gap_matrix.json directly
    if plan.get("special") == "gap_matrix":
        return _retrieve_gaps(question)

    candidates = _stage1_ann_search(plan)

    if not candidates:
        logger.warning(f"No candidates found for: {question}")
        return []

    results = _stage2_rerank(question, candidates, top_k)
    return results


# ── Stage 1: ANN search ───────────────────────────────────────────────────

def _stage1_ann_search(plan: dict) -> list[dict]:
    """
    Embeds the query and searches each target collection.
    Applies metadata filter if present.
    Returns a flat list of candidate dicts.
    """
    from rag.embedder import embed_query
    from rag.indexer  import get_collections

    if plan["n_results"] == 0:
        return []

    q_vec       = embed_query(plan["query_text"])
    collections = get_collections()
    candidates  = []

    for coll_name in plan["collections"]:
        coll = collections.get(coll_name)
        if coll is None or coll.count() == 0:
            continue

        # build query kwargs — only add where if filter is non-empty
        kwargs = {
            "query_embeddings": [q_vec],
            "n_results":        min(plan["n_results"], coll.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if plan["where"]:
            kwargs["where"] = plan["where"]

        try:
            result = coll.query(**kwargs)
        except Exception as e:
            logger.warning(f"Query failed on '{coll_name}': {e}")
            continue

        ids       = result["ids"][0]
        docs      = result["documents"][0]
        metas     = result["metadatas"][0]
        distances = result["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            candidates.append({
                "chunk_id":   cid,
                "document":   doc,
                "metadata":   meta,
                "collection": coll_name,
                # cosine distance → similarity (ChromaDB returns distance, not similarity)
                "ann_score":  round(1 - dist, 4),
                # provenance fields lifted to top level for convenience
                "paper_id":   meta.get("paper_id", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "confidence": meta.get("confidence", 1.0),
            })

    return candidates


# ── Stage 2: Cross-encoder re-ranking ────────────────────────────────────

def _stage2_rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Scores each (question, document) pair with a cross-encoder.
    Multiplies by chunk confidence to downweight low-confidence extractions.
    Returns top_k results sorted by final score.
    """
    ce = _get_cross_encoder()

    pairs = [(question, c["document"]) for c in candidates]
    ce_scores = ce.predict(pairs)

    # normalize cross-encoder scores to 0-1 range using sigmoid
    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for candidate, ce_score in zip(candidates, ce_scores):
        norm_score  = sigmoid(float(ce_score))
        confidence  = float(candidate.get("confidence", 1.0))
        # if confidence is 0.0 (missing from source data), treat as 1.0
        # so we don't zero out valid results
        if confidence == 0.0:
            confidence = 1.0
        final_score = round(norm_score * confidence, 4)

        candidate["ce_score"]    = round(norm_score, 4)
        candidate["score"]       = final_score

    # sort by final score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # add rank and intent, remove internal fields
    from rag.query_handler import parse_query
    plan = parse_query(question)

    results = []
    for rank, c in enumerate(candidates[:top_k], 1):
        results.append({
            "rank":       rank,
            "score":      c["score"],
            "ce_score":   c["ce_score"],
            "ann_score":  c["ann_score"],
            "document":   c["document"],
            "paper_id":   c["paper_id"],
            "chunk_type": c["chunk_type"],
            "chunk_id":   c["chunk_id"],
            "collection": c["collection"],
            "metadata":   c["metadata"],
            "intent":     plan["intent"],
        })

    return results


# ── Gap retrieval ─────────────────────────────────────────────────────────

def _retrieve_gaps(question: str) -> list[dict]:
    """
    For gap queries: reads gap_matrix.json directly.
    Returns top gaps as result dicts.
    """
    gap_path = MEMORY_DIR / "gap_matrix.json"
    if not gap_path.exists():
        logger.warning("gap_matrix.json not found — run enricher first")
        return []

    with open(gap_path, encoding="utf-8") as f:
        data = json.load(f)

    gaps = data.get("gaps", [])
    if not gaps:
        return [{
            "rank":       1,
            "score":      1.0,
            "document":   "No research gaps found yet. Add more papers and re-run the enricher.",
            "paper_id":   "",
            "chunk_type": "gap",
            "chunk_id":   "",
            "collection": "gap_matrix",
            "metadata":   {},
            "intent":     "gap",
        }]

    results = []
    for rank, gap in enumerate(gaps[:10], 1):
        doc = (
            f"Research gap: '{gap['method']}' has never been tested on "
            f"'{gap['dataset']}'. "
            f"Method appears in: {gap['method_used_in']}. "
            f"Dataset appears in: {gap['dataset_used_in']}."
        )
        results.append({
            "rank":       rank,
            "score":      round(gap["gap_score"] / 10, 4),
            "document":   doc,
            "paper_id":   "",
            "chunk_type": "gap",
            "chunk_id":   f"gap_{rank}",
            "collection": "gap_matrix",
            "metadata":   gap,
            "intent":     "gap",
        })

    return results


# ── convenience display ───────────────────────────────────────────────────

def print_results(results: list[dict], question: str = "") -> None:
    """Prints retrieval results in a readable format."""
    if question:
        print(f"\n  Query: '{question}'")
    print(f"  {'─'*56}")
    if not results:
        print("  No results found.")
        return
    for r in results:
        print(f"  #{r['rank']}  score={r['score']:.4f}  "
              f"[{r['chunk_type']}]  paper={r['paper_id']}")
        print(f"       {r['document'][:90]}...")
        print()