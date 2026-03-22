"""
pipeline.py
-----------
The single entry point for the entire RAG system.
Ties together chunker → embedder → indexer → enricher → retriever.

Two public functions:

    index_paper(folder_path)
        Indexes one paper's output_folder into ChromaDB.
        Call this for each paper, then call enrich() once.

    enrich()
        Runs cross-paper enrichment after all papers are indexed.
        Populates also_in_papers, contradiction candidates, gap matrix.

    query(question, top_k=5)
        Answers a question against the indexed papers.
        Returns ranked results with full provenance.

Usage — indexing:
    from rag.pipeline import index_paper, enrich

    index_paper("memory/stgcn_yu_2018")
    index_paper("memory/dcrnn_li_2018")
    index_paper("memory/gwnet_wu_2019")
    enrich()

Usage — querying:
    from rag.pipeline import query

    results = query("What are the limitations of STGCN?")
    for r in results:
        print(f"[{r['chunk_type']}] {r['document']}")
        print(f"  paper: {r['paper_id']}  score: {r['score']}")
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── indexing ──────────────────────────────────────────────────────────────

def index_paper(folder_path: str) -> dict:
    """
    Full indexing pipeline for one paper.
    Reads paper_id from claims_output.json directly.

    Steps:
        1. chunk_paper()   — reads 3 JSON files, produces chunk dicts
        2. embed_chunks()  — adds 768-dim embedding to each chunk
        3. index_chunks()  — upserts into ChromaDB collections

    Does NOT run enricher — call enrich() after all papers are indexed.

    Args:
        folder_path: Path to the paper's output_folder.
                     e.g. "memory/stgcn_yu_2018"

    Returns:
        Dict with paper_id and chunk counts per collection.
    """
    from rag.chunker  import chunk_paper
    from rag.embedder import embed_chunks
    from rag.indexer  import index_chunks
    from rag.utils.paper_id import paper_id_from_folder

    folder   = Path(folder_path)
    paper_id = paper_id_from_folder(str(folder))

    print(f"\n{'='*60}")
    print(f"  Indexing: {paper_id}")
    print(f"  Folder  : {folder.resolve()}")
    print(f"{'='*60}\n")

    # step 1 — chunk
    print("  Step 1/3 — Chunking...")
    chunks = chunk_paper(str(folder), paper_id)

    # step 2 — embed
    print("  Step 2/3 — Embedding...")
    chunks = embed_chunks(chunks)

    # step 3 — index
    print("  Step 3/3 — Indexing into ChromaDB...")
    counts = index_chunks(chunks)

    print(f"\n  Done. '{paper_id}' indexed successfully.\n")

    return {
        "paper_id":   paper_id,
        "total":      len(chunks),
        "collections": counts,
    }


def index_all(memory_dir: str = "memory") -> list[dict]:
    """
    Indexes all paper folders found in memory_dir.
    Skips folders that don't contain claims_output.json.
    Does NOT run enricher — call enrich() after.

    Args:
        memory_dir: Path to the memory folder containing paper subfolders.

    Returns:
        List of result dicts, one per paper indexed.
    """
    memory = Path(memory_dir)
    if not memory.exists():
        raise FileNotFoundError(f"memory_dir not found: {memory}")

    folders = [
        f for f in memory.iterdir()
        if f.is_dir() and (f / "claims_output.json").exists()
    ]

    if not folders:
        print(f"No paper folders found in {memory}")
        return []

    print(f"\nFound {len(folders)} papers to index: "
          f"{[f.name for f in folders]}")

    results = []
    for folder in sorted(folders):
        result = index_paper(str(folder))
        results.append(result)

    return results


# ── enrichment ────────────────────────────────────────────────────────────

def enrich() -> dict:
    """
    Runs cross-paper enrichment. Call this once after all papers are indexed.

    Passes:
        1. Entity linking — populates also_in_papers across papers
        2. Contradiction candidates — writes memory/contradiction_candidates.json
        3. Gap matrix — writes memory/gap_matrix.json

    Returns:
        Summary dict from enricher.
    """
    from rag.enricher import run_all_passes

    print(f"\n{'='*60}")
    print(f"  Running post-index enrichment...")
    print(f"{'='*60}\n")

    results = run_all_passes()
    return results


# ── querying ──────────────────────────────────────────────────────────────

def query(question: str, top_k: int = 5,
          known_methods: list = None) -> list[dict]:
    """
    Answers a question against the indexed papers.

    Steps:
        1. parse_query()  — detects intent, builds filter
        2. embed_query()  — embeds the question
        3. ChromaDB ANN   — fetches candidates with filter
        4. cross-encoder  — re-ranks candidates
        5. returns top_k  — with full provenance

    Args:
        question:      Natural language question.
        top_k:         Number of results to return (default 5).
        known_methods: Optional list of method names for query enrichment.

    Returns:
        List of result dicts sorted by score descending.
    """
    from rag.retriever import retrieve
    return retrieve(question, top_k=top_k, known_methods=known_methods)


# ── status helpers ────────────────────────────────────────────────────────

def status() -> None:
    """Prints current state of ChromaDB and memory files."""
    from rag.indexer import collection_counts

    counts = collection_counts()
    total  = sum(counts.values())

    print(f"\n{'='*60}")
    print(f"  RAG SYSTEM STATUS")
    print(f"{'='*60}")
    print(f"\n  ChromaDB collections:")
    for name, count in counts.items():
        print(f"    {name:<25}: {count} docs")

    print(f"\n  Memory files:")
    for fname in ("contradiction_candidates.json", "gap_matrix.json"):
        path = Path("memory") / fname
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"    {fname:<35}: {size:.1f} KB")
        else:
            print(f"    {fname:<35}: not found (run enrich())")

    print(f"\n  Total chunks indexed: {total}")
    print(f"{'='*60}\n")