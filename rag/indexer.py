"""
indexer.py
----------
Takes embedded chunks from embedder.py and upserts them into
the correct ChromaDB collections.

This step is idempotent — running it twice on the same paper
produces the same result. No duplicates, no errors.

Collections populated here:
    paper_sections      ← section and figure chunks
    claims_and_findings ← claim, limitation, future_work chunks
    entities_global     ← entity chunks
    researcher_feedback ← written at runtime by agents, NOT here

Usage:
    from rag.indexer import index_chunks, get_collections

    chunks = chunk_paper(...)
    chunks = embed_chunks(chunks)
    index_chunks(chunks)

Install:
    pip install chromadb
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── ChromaDB store location ───────────────────────────────────────────────
# All 4 collections live in this single folder.
# Add rag/chroma_store/ to your .gitignore — it can get large.
CHROMA_STORE_PATH = "rag/chroma_store"

# ── collection routing ────────────────────────────────────────────────────
# Which chunk_type goes into which collection.
COLLECTION_MAP = {
    "section":     "paper_sections",
    "figure":      "paper_sections",
    "claim":       "claims_and_findings",
    "limitation":  "claims_and_findings",
    "future_work": "claims_and_findings",
    "entity":      "entities_global",
}

# ── metadata fields stored in ChromaDB ───────────────────────────────────
# ChromaDB stores metadata as a flat dict alongside each document.
# These are the fields that become filterable via where= clauses.
# 'embedding' and 'embed_text' are NOT included — embedding is stored
# as a vector, embed_text is not needed after indexing.
METADATA_FIELDS = [
    "paper_id",
    "paper_title",
    "chunk_type",
    "claim_type",
    "section_type",
    "section_heading",
    "entity_type",
    "entity_text",
    "entity_text_normalized",
    "confidence",
    "source",
    "has_numeric_value",
    "numeric_value",
    "entities_mentioned",
    "methods_mentioned",
    "datasets_mentioned",
    "metrics_mentioned",
    "also_in_papers",
    "appears_in_n_papers",
    "embed_model",
    "embed_model_version",
]

# ── module-level client and collection cache ──────────────────────────────
_client      = None
_collections = {}


def _get_client():
    """
    Returns a persistent ChromaDB client.
    Creates the chroma_store folder if it doesn't exist.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb not installed.\n"
            "Run: pip install chromadb"
        )

    store_path = Path(CHROMA_STORE_PATH)
    store_path.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(path=str(store_path))
    logger.info(f"ChromaDB client initialised at: {store_path.resolve()}")
    return _client


def get_collections() -> dict:
    """
    Returns all 4 ChromaDB collections, creating them if they don't exist.
    Uses cosine similarity space — matches our normalized embeddings.

    Returns dict with keys:
        paper_sections, claims_and_findings, entities_global, researcher_feedback
    """
    global _collections
    if _collections:
        return _collections

    client = _get_client()

    collection_names = [
        "paper_sections",
        "claims_and_findings",
        "entities_global",
        "researcher_feedback",
    ]

    for name in collection_names:
        _collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    return _collections


# ── metadata extraction ───────────────────────────────────────────────────

def _extract_metadata(chunk: dict) -> dict:
    """
    Extracts only the filterable metadata fields from a chunk.
    Skips embed_text, display_text, embedding — those are stored separately.

    ChromaDB metadata values must be: str, int, float, or bool.
    No None values — we default to "" or 0.0 if a field is missing.
    """
    metadata = {}
    for field in METADATA_FIELDS:
        value = chunk.get(field)

        # replace None with safe defaults — ChromaDB rejects None
        if value is None:
            if field in ("numeric_value", "confidence", "appears_in_n_papers"):
                value = 0.0
            elif field == "has_numeric_value":
                value = False
            else:
                value = ""

        metadata[field] = value

    return metadata


# ── core upsert function ──────────────────────────────────────────────────

def index_chunks(chunks: list[dict]) -> dict:
    """
    Upserts all chunks into the correct ChromaDB collections.
    Idempotent — safe to run multiple times on the same paper.

    Args:
        chunks: List of chunk dicts with 'embedding' field populated.
                (output of embedder.embed_chunks)

    Returns:
        Dict with counts of chunks inserted per collection.
        e.g. {"paper_sections": 14, "claims_and_findings": 163, "entities_global": 202}
    """
    if not chunks:
        logger.warning("index_chunks called with empty list")
        return {}

    # verify embeddings are present
    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        raise ValueError(
            f"{len(missing)} chunks are missing 'embedding' field. "
            f"Run embed_chunks() before index_chunks()."
        )

    collections = get_collections()

    # group chunks by target collection
    batches: dict[str, list[dict]] = {name: [] for name in collections}

    for chunk in chunks:
        ctype      = chunk.get("chunk_type", "")
        collection = COLLECTION_MAP.get(ctype)
        if collection is None:
            logger.warning(f"Unknown chunk_type '{ctype}', skipping chunk {chunk.get('chunk_id')}")
            continue
        batches[collection].append(chunk)

    # upsert each collection
    counts = {}
    for collection_name, batch in batches.items():
        if not batch:
            continue

        collection = collections[collection_name]

        ids         = [c["chunk_id"]    for c in batch]
        embeddings  = [c["embedding"]   for c in batch]
        documents   = [c["display_text"] for c in batch]
        metadatas   = [_extract_metadata(c) for c in batch]

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = documents,
            metadatas  = metadatas,
        )

        counts[collection_name] = len(batch)
        print(f"  ✓  {collection_name:<25} {len(batch):>4} chunks upserted")

    return counts


# ── inspection helpers ────────────────────────────────────────────────────

def collection_counts() -> dict:
    """
    Returns the number of documents in each collection.
    Useful for verifying after indexing.
    """
    collections = get_collections()
    return {name: col.count() for name, col in collections.items()}


def peek_collection(collection_name: str, n: int = 3) -> list[dict]:
    """
    Returns the first n documents from a collection for inspection.
    Shows metadata and a snippet of the document text.

    Args:
        collection_name: one of paper_sections, claims_and_findings,
                         entities_global, researcher_feedback
        n: number of documents to return (default 3)
    """
    collections = get_collections()
    if collection_name not in collections:
        raise ValueError(f"Unknown collection: {collection_name}")

    result = collections[collection_name].peek(limit=n)

    docs = []
    for i in range(len(result["ids"])):
        docs.append({
            "id":       result["ids"][i],
            "document": result["documents"][i][:100] + "...",
            "metadata": result["metadatas"][i],
        })
    return docs


def delete_paper(paper_id: str) -> dict:
    """
    Removes all chunks for a given paper from all collections.
    Useful when you want to re-index a paper after updating its JSONs.

    Args:
        paper_id: e.g. "stgcn_yu_2018"

    Returns:
        Dict with number of documents deleted per collection.
    """
    collections = get_collections()
    deleted = {}

    for name, col in collections.items():
        results = col.get(where={"paper_id": {"$eq": paper_id}})
        ids     = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
            deleted[name] = len(ids)
            print(f"  Deleted {len(ids)} chunks from '{name}' for paper '{paper_id}'")

    return deleted