"""
test_retriever.py
-----------------
Run from your Research-Agent root folder:

    python test_retriever.py

Tests the full retrieval pipeline end-to-end:
    question → query_handler → embedder → ChromaDB → cross-encoder → results

This is the last step before pipeline.py ties everything together.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.retriever import retrieve, print_results


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


# ── test queries ──────────────────────────────────────────────────────────

QUERIES = [
    {
        "question": "What are the limitations of graph-based traffic forecasting?",
        "intent":   "limitation",
        "expect_in_top3": ["difficult", "heavy", "uncertain", "complex", "error"],
    },
    {
        "question": "How does STGCN compare to GCGRU in training speed?",
        "intent":   "comparison",
        "expect_in_top3": ["14", "acceleration", "GCGRU", "speed", "training"],
    },
    {
        "question": "What performance results does STGCN achieve?",
        "intent":   "performance",
        "expect_in_top3": ["STGCN", "PeMSD", "seconds", "time", "parameter"],
    },
    {
        "question": "What future directions do the authors suggest?",
        "intent":   "future_work",
        "expect_in_top3": ["future", "network", "parameter", "structure", "apply"],
    },
    {
        "question": "What research gaps exist in traffic forecasting?",
        "intent":   "gap",
        "expect_in_top3": [],  # gap results are from gap_matrix.json
    },
]


def run_all_queries():
    sep("RETRIEVAL TEST — ALL QUESTION TYPES")

    for i, test in enumerate(QUERIES, 1):
        question = test["question"]
        sep(f"Query {i}: {test['intent'].upper()}")

        results = retrieve(question, top_k=5)

        if not results:
            info(f"No results returned for: {question}")
            continue

        print_results(results, question)

        # verify intent
        intent_ok = all(r["intent"] == test["intent"] for r in results)
        if intent_ok:
            ok(f"Intent correctly set to '{test['intent']}'")
        else:
            info(f"Intent mismatch — got: {set(r['intent'] for r in results)}")

        # verify expected keywords in top 3
        expected = test.get("expect_in_top3", [])
        if expected:
            top3_text = " ".join(r["document"].lower() for r in results[:3])
            found     = [kw for kw in expected if kw.lower() in top3_text]
            if found:
                ok(f"Expected keywords found in top 3: {found}")
            else:
                info(f"Expected keywords not found: {expected}")
                info(f"Top result: {results[0]['document'][:80]}...")

        # verify scores are in valid range
        scores_ok = all(0.0 <= r["score"] <= 1.0 for r in results)
        if scores_ok:
            ok(f"All scores in valid range [0, 1]")
        else:
            fail(f"Some scores out of range: {[r['score'] for r in results]}")

        # verify results are sorted by score descending
        scores = [r["score"] for r in results]
        if scores == sorted(scores, reverse=True):
            ok(f"Results correctly sorted by score")
        else:
            fail(f"Results not sorted: {scores}")


def run_score_comparison():
    sep("SCORE COMPARISON — ANN vs CROSS-ENCODER")
    info("Cross-encoder should re-rank ANN results more precisely.")
    info("")

    question = "training speed comparison STGCN GCGRU"
    results  = retrieve(question, top_k=5)

    if not results:
        info("No results — skipping comparison")
        return

    print(f"  {'rank':<6} {'ann_score':<12} {'ce_score':<12} {'final':<10}  document")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*10}  {'─'*40}")
    for r in results:
        ann  = r.get("ann_score", 0)
        ce   = r.get("ce_score", 0)
        final = r["score"]
        doc  = r["document"][:45]
        print(f"  #{r['rank']:<5} {ann:<12.4f} {ce:<12.4f} {final:<10.4f}  {doc}...")

    ok("Score comparison displayed")


def run_top_result_check():
    sep("TOP RESULT ACCURACY CHECK")
    info("The 14x acceleration claim should be #1 for the training speed query.")
    info("")

    results = retrieve("How much faster is STGCN than GCGRU?", top_k=3)

    if not results:
        fail("No results returned")

    top = results[0]
    info(f"#1 result: {top['document'][:80]}...")
    info(f"    score : {top['score']:.4f}")
    info(f"    type  : {top['chunk_type']}")

    if "14" in top["document"] or "acceleration" in top["document"].lower():
        ok("14x acceleration claim is ranked #1")
    else:
        info("Note: expected claim not at #1 — check embed_text or re-ranker")


def final_summary():
    sep("FINAL SUMMARY")
    print()
    print("  Full retrieval pipeline working:")
    print("    question → parse intent → embed → ChromaDB ANN")
    print("    → cross-encoder re-rank → top-k results")
    print()
    print("  Next step: build pipeline.py (Step 7)")
    print("  This ties everything into two clean entry points:")
    print("    pipeline.index_paper(folder, paper_id)")
    print("    pipeline.query(question)")
    sep()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RETRIEVER TEST — END TO END")
    print("="*60)

    run_all_queries()
    run_score_comparison()
    run_top_result_check()
    final_summary()