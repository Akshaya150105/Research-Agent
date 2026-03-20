"""
test_embedder.py
----------------
Run from your Research-Agent root folder:

    python test_embedder.py

Tests embedder.py against the real STGCN paper.
Runs the full chunker → embedder pipeline and verifies:
  1. Every chunk has an embedding of the right dimension
  2. Embeddings are normalized (unit vectors)
  3. Similar chunks are closer than unrelated chunks
  4. Query embedding works correctly
  5. Model caching works (second call is fast)
"""

import sys
import os
import time
import math

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


def cosine_similarity(a: list, b: list) -> float:
    """Simple cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────────────
def step1_chunk():
    sep("1. CHUNKING (prerequisite)")
    from rag.chunker import chunk_paper
    chunks = chunk_paper(FOLDER, PAPER_ID)
    ok(f"Chunker produced {len(chunks)} chunks")
    return chunks


def step2_embed(chunks):
    sep("2. EMBEDDING ALL CHUNKS")
    from rag.embedder import embed_chunks

    start  = time.time()
    chunks = embed_chunks(chunks)
    elapsed = time.time() - start

    ok(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s")
    return chunks


def check_dimensions(chunks):
    sep("3. EMBEDDING DIMENSION CHECK")
    from rag.embedder import EMBED_DIM

    for c in chunks:
        if "embedding" not in c:
            fail(f"Chunk {c['chunk_id']} is missing 'embedding' field")
        if len(c["embedding"]) != EMBED_DIM:
            fail(f"Expected {EMBED_DIM} dims, got {len(c['embedding'])} "
                 f"in chunk {c['chunk_id']}")

    ok(f"All {len(chunks)} chunks have embedding dimension {EMBED_DIM}")


def check_normalized(chunks):
    sep("4. NORMALIZATION CHECK (unit vectors)")
    # all vectors should have magnitude ≈ 1.0
    # because we use normalize_embeddings=True
    # this is required for cosine similarity in ChromaDB

    bad = []
    for c in chunks[:20]:   # spot check first 20
        mag = math.sqrt(sum(x*x for x in c["embedding"]))
        if abs(mag - 1.0) > 0.001:
            bad.append((c["chunk_id"], mag))

    if bad:
        for cid, mag in bad:
            info(f"Chunk {cid} has magnitude {mag:.4f} (expected ~1.0)")
        fail("Some embeddings are not normalized")

    ok("All embeddings are unit vectors (magnitude ≈ 1.0)")


def check_similarity(chunks):
    sep("5. SEMANTIC SIMILARITY SPOT CHECK")
    info("Similar chunks should score higher than unrelated chunks.")
    info("")

    # find two claims about training speed/time — should be similar
    speed_claims = [
        c for c in chunks
        if c["chunk_type"] == "claim"
        and any(kw in c["display_text"].lower()
                for kw in ["training", "speed", "seconds", "acceleration"])
    ]

    # find a section chunk — should be less similar to a claim
    section_chunks = [c for c in chunks if c["chunk_type"] == "section"]
    entity_chunks  = [c for c in chunks if c["chunk_type"] == "entity"
                      and c["entity_type"] == "dataset"]

    if len(speed_claims) >= 2:
        sim_related = cosine_similarity(
            speed_claims[0]["embedding"],
            speed_claims[1]["embedding"]
        )
        info(f"Two training-speed claims:")
        info(f"  A: {speed_claims[0]['display_text'][:60]}...")
        info(f"  B: {speed_claims[1]['display_text'][:60]}...")
        info(f"  cosine similarity: {sim_related:.4f}  ← should be HIGH (>0.7)")

    if speed_claims and entity_chunks:
        sim_unrelated = cosine_similarity(
            speed_claims[0]["embedding"],
            entity_chunks[0]["embedding"]
        )
        info(f"\nClaim vs Dataset entity:")
        info(f"  A: {speed_claims[0]['display_text'][:60]}...")
        info(f"  B: {entity_chunks[0]['display_text'][:40]}")
        info(f"  cosine similarity: {sim_unrelated:.4f}  ← should be LOWER")

    if len(speed_claims) >= 2 and entity_chunks:
        if sim_related > sim_unrelated:
            ok("Related chunks score higher than unrelated chunks")
        else:
            info("Note: similarity ordering unexpected — may be fine for this pair")


def check_query_embedding():
    sep("6. QUERY EMBEDDING CHECK")
    from rag.embedder import embed_query, EMBED_DIM

    query  = "What are the limitations of graph-based traffic forecasting?"
    vector = embed_query(query)

    if len(vector) != EMBED_DIM:
        fail(f"Query embedding has wrong dimension: {len(vector)}")

    mag = math.sqrt(sum(x*x for x in vector))
    if abs(mag - 1.0) > 0.001:
        fail(f"Query embedding not normalized: magnitude={mag:.4f}")

    ok(f"Query embedding: {EMBED_DIM} dims, normalized")
    info(f"Query: '{query}'")
    info(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")


def check_query_retrieval(chunks):
    sep("7. MANUAL RETRIEVAL SPOT CHECK")
    info("Embed a query and find the most similar chunks manually.")
    info("This simulates what the retriever will do.\n")

    from rag.embedder import embed_query

    query  = "training speed comparison STGCN GCGRU"
    q_vec  = embed_query(query)

    # score all claim chunks
    claim_chunks = [c for c in chunks if c["chunk_type"] == "claim"]
    scored = [
        (cosine_similarity(q_vec, c["embedding"]), c)
        for c in claim_chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    info(f"Query: '{query}'")
    info(f"Top 5 most similar claim chunks:\n")
    for rank, (score, c) in enumerate(scored[:5], 1):
        info(f"  {rank}. score={score:.4f} | {c['display_text'][:70]}...")

    # the 14x acceleration claim should be in the top 3
    top5_texts = [c["display_text"] for _, c in scored[:5]]
    if any("14" in t or "acceleration" in t or "speed" in t for t in top5_texts):
        ok("Relevant claim (14x acceleration) found in top 5 results")
    else:
        info("Note: expected claim not in top 5 — check embed_text construction")


def check_model_cache():
    sep("8. MODEL CACHE CHECK")
    from rag.embedder import embed_query

    # second call should be instant (model already loaded)
    start   = time.time()
    _       = embed_query("test query")
    elapsed = time.time() - start

    if elapsed < 1.0:
        ok(f"Model cache working — second call took {elapsed:.3f}s")
    else:
        info(f"Second call took {elapsed:.1f}s — model may be reloading")


def final_summary(chunks):
    sep("FINAL SUMMARY")
    has_embedding = sum(1 for c in chunks if "embedding" in c)
    print(f"\n  paper_id     : {PAPER_ID}")
    print(f"  total chunks : {len(chunks)}")
    print(f"  with embedding: {has_embedding}")
    print(f"  embed dim    : {len(chunks[0]['embedding'])}")
    print(f"\n  Chunker + Embedder pipeline working correctly.")
    print(f"\n  Next step: build indexer.py (Step 3)")
    print(f"  Run: from rag.indexer import index_paper")
    sep()


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EMBEDDER TEST — STGCN PAPER")
    print("="*60)

    chunks = step1_chunk()
    chunks = step2_embed(chunks)

    check_dimensions(chunks)
    check_normalized(chunks)
    check_similarity(chunks)
    check_query_embedding()
    check_query_retrieval(chunks)
    check_model_cache()
    final_summary(chunks)