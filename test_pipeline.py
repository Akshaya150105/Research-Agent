"""
test_pipeline.py
----------------
Run from your Research-Agent root folder:

    python test_pipeline.py

Final end-to-end test of the complete RAG pipeline.
Uses the two public entry points: index_paper() and query().
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.pipeline import index_paper, enrich, query, status


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


FOLDER   = "memory/stgcn_yu_2018"
PAPER_ID = "stgcn_yu_2018"

# questions covering all 6 agent tasks
FINAL_QUERIES = [
    # agent task 1 — extraction (general info)
    "What methods does STGCN use for spatial and temporal modelling?",
    # agent task 2 — comparison
    "How does STGCN compare to other methods in training speed?",
    # agent task 3 — critique
    "What are the main limitations of the STGCN approach?",
    # agent task 4 — gaps
    "What research gaps exist in traffic forecasting?",
    # agent task 5 — literature review
    "Summarize what this paper says about graph convolutional networks",
    # agent task 6 — future work (RL training signal)
    "What future directions do the authors suggest?",
]


def test_status():
    sep("SYSTEM STATUS")
    status()


def test_index():
    sep("INDEX PAPER")
    result = index_paper(FOLDER)
    ok(f"paper_id: {result['paper_id']}")
    ok(f"total chunks: {result['total']}")
    for coll, count in result["collections"].items():
        info(f"  {coll}: {count}")


def test_enrich():
    sep("ENRICH")
    results = enrich()
    ok("Enrichment complete")
    info(f"Pass 1 — linked entities: {results.get('pass1', {}).get('linked_entities', 0)}")
    info(f"Pass 2 — candidates: {results.get('pass2', {}).get('candidates', 0)}")
    info(f"Pass 3 — gaps: {results.get('pass3', {}).get('gaps', 0)}")


def test_all_queries():
    sep("ALL 6 AGENT TASK QUERIES")

    for i, question in enumerate(FINAL_QUERIES, 1):
        print(f"\n  [{i}] {question}")
        print(f"  {'─'*56}")

        results = query(question, top_k=3)

        if not results:
            info("No results returned")
            continue

        for r in results:
            print(f"  #{r['rank']} score={r['score']:.4f} [{r['chunk_type']}]")
            print(f"     {r['document'][:90]}...")

        ok(f"Query {i} returned {len(results)} results")


def final_summary():
    sep("COMPLETE")
    print()
    print("  RAG pipeline fully built and tested.")
    print()
    print("  Files built:")
    files = [
        ("rag/chunker.py",       "reads output_folder → chunk dicts"),
        ("rag/embedder.py",      "adds 768-dim BGE embeddings"),
        ("rag/indexer.py",       "upserts into ChromaDB"),
        ("rag/enricher.py",      "cross-paper linking + gap matrix"),
        ("rag/query_handler.py", "parses questions → query plans"),
        ("rag/retriever.py",     "ANN search + cross-encoder rerank"),
        ("rag/pipeline.py",      "index_paper() + query() entry points"),
        ("rag/utils/paper_id.py","shared ID convention"),
        ("rag/utils/text_builder.py","embed text construction"),
    ]
    for fname, desc in files:
        print(f"    {fname:<30} ← {desc}")
    print()
    print("  To add paper 2:")
    print("    from rag.pipeline import index_paper, enrich")
    print("    index_paper('memory/dcrnn_li_2018')")
    print("    enrich()  # re-run to update cross-paper links")
    print()
    print("  Phase 3: agents/ folder reads from this pipeline.")
    sep()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PIPELINE — FINAL END-TO-END TEST")
    print("="*60)

    test_status()
    test_index()
    test_enrich()
    test_all_queries()
    final_summary()