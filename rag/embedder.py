"""
embedder.py
-----------
Takes the chunk list from chunker.py and adds an 'embedding'
field to each chunk. 

Model: BAAI/bge-base-en-v1.5
  - 768 dimensions
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── model config ──────────────────────────────────────────────────────────
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
EMBED_DIM    = 768
BATCH_SIZE   = 32     
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "



_model = None

def _get_model():
    """
    Loads the model on first call, returns cached model on subsequent calls.
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




def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds an 'embedding' field (list of 768 floats) to every chunk.
    Modifies chunks in-place AND returns the list.
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
    Args:
        query_text: The user's question or search string.

    Returns:
        List of 768 floats.
    """
    model  = _get_model()
    text   = QUERY_PREFIX + query_text
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()



def _embed_in_batches(model, texts: list[str]) -> list[list[float]]:
    """
    Embeds texts in batches with a simple progress display.
    Returns list of embedding vectors

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
            normalize_embeddings=True,   
            show_progress_bar=False,     
            batch_size=BATCH_SIZE,
        )

       
        all_embeddings.extend([v.tolist() for v in vectors])

        # progress display
        done     = batch_num + 1
        bar_len  = 30
        filled   = int(bar_len * done / total_batches)
        bar      = "█" * filled + "░" * (bar_len - filled)
        n_done   = min(start_idx + BATCH_SIZE, len(texts))
        print(f"  [{bar}] {n_done}/{len(texts)} chunks", end="\r")

    return all_embeddings
