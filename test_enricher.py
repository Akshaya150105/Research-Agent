"""
enricher.py
-----------
Runs after ALL papers are indexed. Performs three passes that
transform isolated per-paper chunks into a connected cross-paper
knowledge base.

This file does nothing useful with just one paper.
It becomes valuable once you have 2+ papers indexed.

Three passes:
    Pass 1 — Entity linking
        Finds entities that appear in multiple papers.
        Updates their also_in_papers and appears_in_n_papers fields.
        Powers: gap detection, "which papers use method X?"

    Pass 2 — Contradiction candidate flagging
        Finds comparative claims with numeric values from different
        papers that mention the same methods.
        Writes: memory/contradiction_candidates.json
        Powers: comparison agent

    Pass 3 — Gap matrix computation
        Builds method × dataset co-occurrence matrix.
        Finds (method, dataset) pairs no paper has tried.
        Writes: memory/gap_matrix.json
        Powers: gap detection agent

Usage:
    from rag.enricher import run_all_passes
    run_all_passes()

Run this after indexing all papers:
    pipeline.index_paper("memory/stgcn_yu_2018",  "stgcn_yu_2018")
    pipeline.index_paper("memory/dcrnn_li_2018",  "dcrnn_li_2018")
    pipeline.index_paper("memory/gwnet_wu_2019",  "gwnet_wu_2019")
    run_all_passes()
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("memory")


# ── normalization helper ──────────────────────────────────────────────────

def _normalize_entity_text(text: str) -> str:
    """
    Light normalization for entity matching across papers.
    Lowercase + strip — enough for exact dedup.
    Fuzzy matching (RapidFuzz) is your friend's job in the KG layer.
    """
    return text.strip().lower()


# ── public entry point ────────────────────────────────────────────────────

def run_all_passes() -> dict:
    """
    Runs all three enrichment passes in order.
    Returns a summary dict with counts from each pass.

    Call this after all papers are indexed.
    Safe to call multiple times — each pass overwrites its outputs.
    """
    from rag.indexer import get_collections, collection_counts

    counts = collection_counts()
    total  = sum(counts.values())
    if total == 0:
        print("  No documents in ChromaDB. Index some papers first.")
        return {}

    print(f"\n  Starting enrichment across {total} total chunks...")
    print(f"  Collections: {counts}\n")

    results = {}
    results["pass1"] = _pass1_entity_linking()
    results["pass2"] = _pass2_contradiction_candidates()
    results["pass3"] = _pass3_gap_matrix()

    _print_summary(results)
    return results


# ── Pass 1: Entity linking ────────────────────────────────────────────────

def _pass1_entity_linking() -> dict:
    """
    Groups entities by normalized text + type across all papers.
    Updates also_in_papers and appears_in_n_papers for entities
    that appear in more than one paper.

    Example result:
        entity "STGCN" appears in stgcn_yu_2018 + dcrnn_li_2018
        → both chunks get also_in_papers = "dcrnn_li_2018" / "stgcn_yu_2018"
        → both get appears_in_n_papers = 2
    """
    print("  Pass 1: Entity linking...")
    from rag.indexer import get_collections
    collections = get_collections()
    eg = collections["entities_global"]

    # fetch all entity chunks
    result = eg.get(include=["metadatas"])
    if not result["ids"]:
        print("  Pass 1: No entities found, skipping.")
        return {"linked_entities": 0, "updated_chunks": 0}

    # group chunk_ids by (normalized_text, entity_type)
    # key → {paper_id → [chunk_id, ...]}
    entity_map: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for chunk_id, meta in zip(result["ids"], result["metadatas"]):
        text  = _normalize_entity_text(meta.get("entity_text", ""))
        etype = meta.get("entity_type", "")
        pid   = meta.get("paper_id", "")
        if text and etype and pid:
            entity_map[(text, etype)][pid].append(chunk_id)

    # find entities that appear in 2+ papers
    cross_paper = {
        key: paper_dict
        for key, paper_dict in entity_map.items()
        if len(paper_dict) > 1
    }

    print(f"     {len(entity_map)} unique entities found")
    print(f"     {len(cross_paper)} appear in 2+ papers")

    if not cross_paper:
        print("     (Only one paper indexed — re-run after adding more papers)")
        return {"linked_entities": 0, "updated_chunks": 0}

    # update each affected chunk
    updated_chunks = 0
    for (text, etype), paper_dict in cross_paper.items():
        all_paper_ids = list(paper_dict.keys())
        n_papers      = len(all_paper_ids)

        for paper_id, chunk_ids in paper_dict.items():
            # other papers this entity appears in
            others = [p for p in all_paper_ids if p != paper_id]
            also_in_papers = ",".join(others)

            for chunk_id in chunk_ids:
                eg.update(
                    ids=[chunk_id],
                    metadatas=[{
                        "also_in_papers":      also_in_papers,
                        "appears_in_n_papers": n_papers,
                    }]
                )
                updated_chunks += 1

    print(f"     Updated {updated_chunks} entity chunks with cross-paper links")
    return {
        "linked_entities": len(cross_paper),
        "updated_chunks":  updated_chunks,
    }


# ── Pass 2: Contradiction candidate flagging ──────────────────────────────

def _pass2_contradiction_candidates() -> dict:
    """
    Finds pairs of comparative claims from different papers that:
      - Both have numeric values
      - Mention at least one common method

    These are CANDIDATES — not confirmed contradictions.
    The comparison agent uses LLM reasoning to confirm.

    Writes: memory/contradiction_candidates.json
    """
    print("\n  Pass 2: Contradiction candidate flagging...")
    from rag.indexer import get_collections
    collections = get_collections()
    caf = collections["claims_and_findings"]

    # get all comparative claims with numeric values
    result = caf.get(
        where={"$and": [
            {"claim_type":       {"$eq": "comparative"}},
            {"has_numeric_value": {"$eq": True}},
        ]},
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        print("     No comparative numeric claims found.")
        _write_json("contradiction_candidates.json", [])
        return {"candidates": 0}

    # build list of candidate objects
    claims = []
    for cid, doc, meta in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        claims.append({
            "chunk_id":    cid,
            "paper_id":    meta.get("paper_id", ""),
            "text":        doc,
            "numeric_value": meta.get("numeric_value", 0.0),
            "entities":    meta.get("entities_mentioned", "").split(","),
            "methods":     meta.get("methods_mentioned", "").split(","),
        })

    print(f"     {len(claims)} comparative numeric claims found")

    # find pairs from different papers with overlapping method mentions
    candidates = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a = claims[i]
            b = claims[j]

            # must be from different papers
            if a["paper_id"] == b["paper_id"]:
                continue

            # must share at least one non-empty method mention
            methods_a = set(m.strip().lower() for m in a["methods"] if m.strip())
            methods_b = set(m.strip().lower() for m in b["methods"] if m.strip())
            shared    = methods_a & methods_b

            # also check entity overlap if methods are empty
            if not shared:
                ents_a = set(e.strip().lower() for e in a["entities"] if e.strip())
                ents_b = set(e.strip().lower() for e in b["entities"] if e.strip())
                shared = ents_a & ents_b

            if shared:
                candidates.append({
                    "claim_a": {
                        "chunk_id":      a["chunk_id"],
                        "paper_id":      a["paper_id"],
                        "text":          a["text"],
                        "numeric_value": a["numeric_value"],
                    },
                    "claim_b": {
                        "chunk_id":      b["chunk_id"],
                        "paper_id":      b["paper_id"],
                        "text":          b["text"],
                        "numeric_value": b["numeric_value"],
                    },
                    "shared_entities": list(shared),
                    "confirmed":       False,   # set to True by comparison agent
                })

    print(f"     {len(candidates)} contradiction candidates found")
    _write_json("contradiction_candidates.json", candidates)
    print(f"     Written to memory/contradiction_candidates.json")

    return {"candidates": len(candidates)}


# ── Pass 3: Gap matrix ────────────────────────────────────────────────────

def _pass3_gap_matrix() -> dict:
    """
    Builds a method × dataset co-occurrence matrix.
    An empty cell = research gap (method never tested on that dataset).

    Structure of gap_matrix.json:
    {
        "methods":  {"STGCN": ["stgcn_yu_2018"], "DCRNN": ["dcrnn_li_2018"]},
        "datasets": {"PeMSD7": ["stgcn_yu_2018"], "METR-LA": ["dcrnn_li_2018"]},
        "matrix":   {"STGCN": {"PeMSD7": ["stgcn_yu_2018"], "METR-LA": []}},
        "gaps":     [{"method": "STGCN", "dataset": "METR-LA", "gap_score": 2}]
    }

    gap_score = (papers using method) + (papers using dataset)
    Higher score = more significant gap (both sides are well-studied)

    Writes: memory/gap_matrix.json
    """
    print("\n  Pass 3: Gap matrix computation...")
    from rag.indexer import get_collections
    collections = get_collections()
    eg = collections["entities_global"]

    # get all method entities
    method_result = eg.get(
        where={"entity_type": {"$eq": "method"}},
        include=["metadatas"],
    )

    # get all dataset entities
    dataset_result = eg.get(
        where={"entity_type": {"$eq": "dataset"}},
        include=["metadatas"],
    )

    if not method_result["ids"] or not dataset_result["ids"]:
        print("     Not enough entities for gap matrix.")
        _write_json("gap_matrix.json", {})
        return {"methods": 0, "datasets": 0, "gaps": 0}

    # build method → [paper_ids] map
    methods: dict[str, list[str]] = defaultdict(list)
    for meta in method_result["metadatas"]:
        text = _normalize_entity_text(meta.get("entity_text", ""))
        pid  = meta.get("paper_id", "")
        if text and pid and pid not in methods[text]:
            methods[text].append(pid)

    # build dataset → [paper_ids] map
    datasets: dict[str, list[str]] = defaultdict(list)
    for meta in dataset_result["metadatas"]:
        text = _normalize_entity_text(meta.get("entity_text", ""))
        pid  = meta.get("paper_id", "")
        if text and pid and pid not in datasets[text]:
            datasets[text].append(pid)

    print(f"     {len(methods)} unique methods, {len(datasets)} unique datasets")

    # build co-occurrence matrix
    matrix: dict[str, dict[str, list]] = {}
    for method in methods:
        matrix[method] = {}
        for dataset in datasets:
            # papers that use BOTH this method AND this dataset
            shared = list(
                set(methods[method]) & set(datasets[dataset])
            )
            matrix[method][dataset] = shared

    # find gaps (empty cells) and score them
    gaps = []
    for method, dataset_row in matrix.items():
        for dataset, shared_papers in dataset_row.items():
            if not shared_papers:
                gap_score = len(methods[method]) + len(datasets[dataset])
                gaps.append({
                    "method":       method,
                    "dataset":      dataset,
                    "gap_score":    gap_score,
                    "method_used_in":  methods[method],
                    "dataset_used_in": datasets[dataset],
                })

    # sort by gap_score descending — highest priority gaps first
    gaps.sort(key=lambda x: x["gap_score"], reverse=True)

    gap_data = {
        "methods":  dict(methods),
        "datasets": dict(datasets),
        "matrix":   matrix,
        "gaps":     gaps,
    }

    _write_json("gap_matrix.json", gap_data)

    print(f"     {len(gaps)} research gaps found")
    print(f"     Written to memory/gap_matrix.json")

    # show top 5 gaps
    if gaps:
        print(f"\n     Top gaps (method never tested on dataset):")
        for g in gaps[:5]:
            print(f"       score={g['gap_score']} | "
                  f"{g['method']} × {g['dataset']}")

    return {
        "methods":  len(methods),
        "datasets": len(datasets),
        "gaps":     len(gaps),
    }


# ── file writer ───────────────────────────────────────────────────────────

def _write_json(filename: str, data) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    path = MEMORY_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── summary printer ───────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    print("\n" + "="*60)
    print("  ENRICHMENT COMPLETE")
    print("="*60)
    p1 = results.get("pass1", {})
    p2 = results.get("pass2", {})
    p3 = results.get("pass3", {})
    print(f"  Pass 1 — Entity linking:")
    print(f"    {p1.get('linked_entities', 0)} cross-paper entities")
    print(f"    {p1.get('updated_chunks', 0)} chunks updated")
    print(f"  Pass 2 — Contradiction candidates:")
    print(f"    {p2.get('candidates', 0)} candidates written to memory/")
    print(f"  Pass 3 — Gap matrix:")
    print(f"    {p3.get('methods', 0)} methods × {p3.get('datasets', 0)} datasets")
    print(f"    {p3.get('gaps', 0)} research gaps found")
    print("="*60)
if __name__ == "__main__":
    run_all_passes()
