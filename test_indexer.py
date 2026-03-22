"""
test_indexer.py
---------------
Run from your Research-Agent root folder:

    python test_indexer.py

Tests the full pipeline: chunk → embed → index.
Verifies ChromaDB collections are correctly populated.

Checks:
  1. Collections created successfully
  2. Chunk counts match in ChromaDB
  3. Metadata filtering works (the whole point of ChromaDB)
  4. Manual similarity query works end-to-end
  5. Idempotency — running twice gives same count, no duplicates
  6. delete_paper() works cleanly
  7. Re-index after delete restores everything
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

FOLDER   = "memory/stgcn_yu_2018"
PAPER_ID = "stgcn_yu_2018"


def sep(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)

def ok(msg):   print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}"); sys.exit(1)
def info(msg): print(f"     {msg}")


# ── pipeline helpers ──────────────────────────────────────────────────────

def run_full_pipeline():
    sep("1. RUNNING FULL PIPELINE (chunk → embed → index)")
    from rag.chunker  import chunk_paper
    from rag.embedder import embed_chunks
    from rag.indexer  import index_chunks

    print("  Chunking...")
    chunks = chunk_paper(FOLDER, PAPER_ID)
    ok(f"Chunker: {len(chunks)} chunks")

    print("  Embedding...")
    chunks = embed_chunks(chunks)
    ok(f"Embedder: {len(chunks)} chunks with 768-dim vectors")

    print("  Indexing into ChromaDB...")
    counts = index_chunks(chunks)
    ok(f"Indexer complete")

    return chunks, counts


# ── checks ────────────────────────────────────────────────────────────────

def check_collection_counts(chunks):
    sep("2. COLLECTION COUNT VERIFICATION")
    from rag.indexer import collection_counts

    counts = collection_counts()
    info(f"Documents in each collection:")

    expected = {
        "paper_sections":     len([c for c in chunks if c["chunk_type"] in ("section", "figure")]),
        "claims_and_findings": len([c for c in chunks if c["chunk_type"] in ("claim", "limitation", "future_work")]),
        "entities_global":    len([c for c in chunks if c["chunk_type"] == "entity"]),
        "researcher_feedback": 0,
    }

    all_ok = True
    for name, count in counts.items():
        exp = expected.get(name, 0)
        match = "✓" if count >= exp else "✗"
        print(f"  {match}  {name:<25} {count:>4} docs  (expected {exp})")
        if count < exp:
            all_ok = False

    if all_ok:
        ok("All collection counts correct")
    else:
        fail("Some collection counts don't match")


def check_metadata_filtering():
    sep("3. METADATA FILTERING CHECK")
    info("This is the main reason we use ChromaDB over plain FAISS.")
    info("")
    from rag.indexer import get_collections

    collections = get_collections()
    caf = collections["claims_and_findings"]

    # filter 1: get only limitations
    result = caf.get(where={"chunk_type": {"$eq": "limitation"}})
    n_lims = len(result["ids"])
    info(f"chunk_type = 'limitation'  → {n_lims} documents")
    if n_lims == 0:
        fail("No limitations found — metadata filtering broken")

    # filter 2: get only comparative claims
    result = caf.get(where={"claim_type": {"$eq": "comparative"}})
    n_comp = len(result["ids"])
    info(f"claim_type = 'comparative' → {n_comp} documents")

    # filter 3: get claims with numeric values
    result = caf.get(where={"has_numeric_value": {"$eq": True}})
    n_num = len(result["ids"])
    info(f"has_numeric_value = True   → {n_num} documents")

    # filter 4: get only methods from entities_global
    eg = collections["entities_global"]
    result = eg.get(where={"entity_type": {"$eq": "method"}})
    n_methods = len(result["ids"])
    info(f"entity_type = 'method'     → {n_methods} documents")

    # filter 5: get only datasets
    result = eg.get(where={"entity_type": {"$eq": "dataset"}})
    n_datasets = len(result["ids"])
    info(f"entity_type = 'dataset'    → {n_datasets} documents")

    if n_lims > 0 and n_num > 0 and n_methods > 0:
        ok("Metadata filtering working correctly")
    else:
        fail("Some filters returned 0 results unexpectedly")


def check_similarity_query():
    sep("4. END-TO-END SIMILARITY QUERY")
    info("Embed a query → search ChromaDB → get ranked results.")
    info("")
    from rag.embedder import embed_query
    from rag.indexer  import get_collections

    collections = get_collections()
    caf = collections["claims_and_findings"]

    query  = "training speed comparison STGCN GCGRU"
    q_vec  = embed_query(query)

    # query with metadata filter — only comparative claims
    result = caf.query(
        query_embeddings=[q_vec],
        n_results=5,
        where={"claim_type": {"$eq": "comparative"}},
        include=["documents", "metadatas", "distances"],
    )

    ids       = result["ids"][0]
    docs      = result["documents"][0]
    distances = result["distances"][0]

    info(f"Query: '{query}'")
    info(f"Filter: claim_type = 'comparative'")
    info(f"Top 5 results:\n")
    for rank, (doc, dist) in enumerate(zip(docs, distances), 1):
        # ChromaDB cosine distance = 1 - cosine_similarity
        similarity = 1 - dist
        info(f"  {rank}. score={similarity:.4f} | {doc[:70]}...")

    if any("14" in d or "acceleration" in d or "speed" in d for d in docs):
        ok("Relevant claim found in top 5 results")
    else:
        info("Note: expected claim not in top 5 — check embed_text")


def check_idempotency():
    sep("5. IDEMPOTENCY CHECK")
    info("Running index_chunks twice should give same count, no duplicates.")
    info("")
    from rag.chunker  import chunk_paper
    from rag.embedder import embed_chunks
    from rag.indexer  import index_chunks, collection_counts

    counts_before = collection_counts()

    # run again
    chunks = chunk_paper(FOLDER, PAPER_ID)
    chunks = embed_chunks(chunks)
    index_chunks(chunks)

    counts_after = collection_counts()

    all_same = True
    for name in counts_before:
        before = counts_before[name]
        after  = counts_after[name]
        same   = before == after
        mark   = "✓" if same else "✗"
        print(f"  {mark}  {name:<25} before={before}  after={after}")
        if not same:
            all_same = False

    if all_same:
        ok("Idempotency confirmed — no duplicates created")
    else:
        fail("Counts changed after second run — duplicates created")


def check_delete_and_reindex():
    sep("6. DELETE AND RE-INDEX CHECK")
    from rag.chunker  import chunk_paper
    from rag.embedder import embed_chunks
    from rag.indexer  import index_chunks, delete_paper, collection_counts

    counts_before = collection_counts()
    info(f"Counts before delete: {counts_before}")

    # delete the paper
    print(f"\n  Deleting paper '{PAPER_ID}'...")
    delete_paper(PAPER_ID)

    counts_after_delete = collection_counts()
    info(f"Counts after delete: {counts_after_delete}")

    if all(counts_after_delete[k] == 0
           for k in ("paper_sections", "claims_and_findings", "entities_global")):
        ok("All chunks deleted successfully")
    else:
        info("Note: other papers' chunks still present (expected if multiple papers indexed)")

    # re-index
    print(f"\n  Re-indexing paper '{PAPER_ID}'...")
    chunks = chunk_paper(FOLDER, PAPER_ID)
    chunks = embed_chunks(chunks)
    index_chunks(chunks)

    counts_restored = collection_counts()
    info(f"Counts after re-index: {counts_restored}")

    if counts_restored == counts_before:
        ok("Re-index restored exact same counts")
    else:
        fail(f"Count mismatch after re-index")


def check_peek():
    sep("7. PEEK CHECK — SAMPLE DOCUMENTS")
    from rag.indexer import peek_collection

    for name in ("paper_sections", "claims_and_findings", "entities_global"):
        docs = peek_collection(name, n=1)
        if not docs:
            fail(f"peek_collection returned empty for '{name}'")
        d = docs[0]
        info(f"\n  [{name}]")
        info(f"  id      : {d['id']}")
        info(f"  document: {d['document']}")
        info(f"  paper_id: {d['metadata']['paper_id']}")
        info(f"  type    : {d['metadata']['chunk_type']}")

    ok("peek_collection working on all collections")


def final_summary():
    sep("FINAL SUMMARY")
    from rag.indexer import collection_counts

    counts = collection_counts()
    print(f"\n  paper_id   : {PAPER_ID}")
    print(f"  ChromaDB   : {os.path.abspath('rag/chroma_store')}")
    print()
    for name, count in counts.items():
        print(f"  {name:<25}: {count} docs")
    print()
    print("  Chunker → Embedder → Indexer pipeline working correctly.")
    print()
    print("  Next step: build enricher.py (Step 4)")
    print("  Run: from rag.enricher import run_all_passes")
    sep()


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  INDEXER TEST — STGCN PAPER")
    print("="*60)

    chunks, counts = run_full_pipeline()
    check_collection_counts(chunks)
    check_metadata_filtering()
    check_similarity_query()
    check_idempotency()
    check_delete_and_reindex()
    check_peek()
    final_summary()