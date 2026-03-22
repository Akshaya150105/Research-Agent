"""
embedder.py
-----------
Takes the chunk list from chunker.py and adds an 'embedding'
field to each chunk. Nothing is stored here — this step only
adds vectors to the existing chunk dicts.

Model: BAAI/bge-base-en-v1.5
  - 768 dimensions
  - Runs locally, no API key needed
  - First run downloads ~440MB to ~/.cache/huggingface/
  - CPU speed: ~2-4 seconds per batch of 32 chunks
  - For 379 chunks → roughly 30-60 seconds total on CPU

Usage:
    from rag.embedder import embed_chunks

    chunks   = chunk_paper("memory/stgcn_yu_2018", "stgcn_yu_2018")
    chunks   = embed_chunks(chunks)

    # now every chunk has an 'embedding' field
    print(len(chunks[0]["embedding"]))   # → 768

Install:
    pip install sentence-transformers
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── model config ──────────────────────────────────────────────────────────
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
EMBED_DIM    = 768
BATCH_SIZE   = 32     # safe for CPU RAM — increase to 64 if you have 16GB+

# BGE models need this prefix on the QUERY side only (not documents).
# Since we're embedding documents here, no prefix needed.
# (The retriever adds the prefix when embedding the user's question.)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── module-level model cache ───────────────────────────────────────────────
# Model is loaded once and reused across multiple embed_chunks() calls.
# Loading takes ~4 seconds — we don't want to repeat that per paper.
_model = None

def _get_model():
    """
    Loads the model on first call, returns cached model on subsequent calls.
    Prints a clear message on first download so the user knows what's happening.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Run: pip install sentence-transformers"
        )

    print(f"\n  Loading embedding model: {MODEL_NAME}")
    print(f"  First run will download ~440MB to ~/.cache/huggingface/")
    print(f"  Subsequent runs load from cache (a few seconds).\n")

    start  = time.time()
    _model = SentenceTransformer(MODEL_NAME, device="cpu")
    elapsed = time.time() - start

    print(f"  Model loaded in {elapsed:.1f}s\n")
    return _model


# ── public entry point ────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds an 'embedding' field (list of 768 floats) to every chunk.
    Modifies chunks in-place AND returns the list.

    Args:
        chunks: List of chunk dicts from chunker.chunk_paper()

    Returns:
        Same list with 'embedding' field added to every chunk.

    Example:
        chunks = embed_chunks(chunks)
        print(len(chunks[0]["embedding"]))   # 768
    """
    if not chunks:
        logger.warning("embed_chunks called with empty list")
        return chunks

    model = _get_model()

    # extract embed_text in the same order as chunks
    texts = [c["embed_text"] for c in chunks]

    print(f"  Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    print(f"  This takes ~30-60 seconds on CPU. Progress below:\n")

    start      = time.time()
    embeddings = _embed_in_batches(model, texts)
    elapsed    = time.time() - start

    # attach embeddings back to chunks
    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    print(f"\n  Done. {len(chunks)} chunks embedded in {elapsed:.1f}s")
    print(f"  Embedding dimension: {len(embeddings[0])}\n")

    return chunks


def embed_query(query_text: str) -> list[float]:
    """
    Embeds a single query string for retrieval.
    Uses the BGE query prefix — this is important for retrieval quality.
    Called by retriever.py, not by the indexing pipeline.

    Args:
        query_text: The user's question or search string.

    Returns:
        List of 768 floats.
    """
    model  = _get_model()
    text   = QUERY_PREFIX + query_text
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ── internal batching ─────────────────────────────────────────────────────

def _embed_in_batches(model, texts: list[str]) -> list[list[float]]:
    """
    Embeds texts in batches with a simple progress display.
    Returns list of embedding vectors (as plain Python lists, not numpy).

    normalize_embeddings=True means all vectors have length 1.
    This makes cosine similarity equivalent to dot product,
    which is what ChromaDB uses internally.
    """
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start_idx in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start_idx : start_idx + BATCH_SIZE]

        # encode the batch
        vectors = model.encode(
            batch,
            normalize_embeddings=True,   # unit vectors — required for cosine similarity
            show_progress_bar=False,     # we handle our own progress display
            batch_size=BATCH_SIZE,
        )

        # convert numpy arrays to plain Python lists
        # ChromaDB requires plain Python lists, not numpy arrays
        all_embeddings.extend([v.tolist() for v in vectors])

        # progress display
        done     = batch_num + 1
        bar_len  = 30
        filled   = int(bar_len * done / total_batches)
        bar      = "█" * filled + "░" * (bar_len - filled)
        n_done   = min(start_idx + BATCH_SIZE, len(texts))
        print(f"  [{bar}] {n_done}/{len(texts)} chunks", end="\r")

    return all_embeddings