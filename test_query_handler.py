"""
test_query_handler.py
---------------------
Run from your Research-Agent root folder:

    python test_query_handler.py

Tests that every question type produces the correct query plan.
No ChromaDB access needed — this is pure logic testing.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.query_handler import parse_query, explain_plan


def sep(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)

def ok(msg):   print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}"); sys.exit(1)


# ── test cases ────────────────────────────────────────────────────────────
# (question, expected_intent, expected_collection, expected_filter_key)

TEST_CASES = [
    (
        "What are the limitations of graph-based traffic forecasting?",
        "limitation",
        "claims_and_findings",
        "chunk_type",
    ),
    (
        "What are the weaknesses of this approach?",
        "limitation",
        "claims_and_findings",
        "chunk_type",
    ),
    (
        "What future work did the authors suggest?",
        "future_work",
        "claims_and_findings",
        "chunk_type",
    ),
    (
        "What are the next steps mentioned in the paper?",
        "future_work",
        "claims_and_findings",
        "chunk_type",
    ),
    (
        "How does STGCN compare to GCGRU in training speed?",
        "comparison",
        "claims_and_findings",
        "claim_type",
    ),
    (
        "Which model outperforms the others on PeMSD7?",
        "comparison",
        "claims_and_findings",
        "claim_type",
    ),
    (
        "What is the RMSE score of STGCN on the benchmark?",
        "performance",
        "claims_and_findings",
        "claim_type",
    ),
    (
        "What accuracy does the model achieve?",
        "performance",
        "claims_and_findings",
        "claim_type",
    ),
    (
        "What research gaps exist in traffic forecasting?",
        "gap",
        "entities_global",
        None,   # gap uses special handler, no filter key
    ),
    (
        "Which method has not been tested on METR-LA?",
        "gap",
        "entities_global",
        None,
    ),
    (
        "Show me figure 2 from the paper",
        "figure",
        "paper_sections",
        "chunk_type",
    ),
    (
        "What does the architecture diagram show?",
        "figure",
        "paper_sections",
        "chunk_type",
    ),
    (
        "Write a literature review on spatiotemporal forecasting",
        "literature_review",
        "paper_sections",
        None,
    ),
    (
        "Summarize what the papers say about graph convolution",
        "literature_review",
        "paper_sections",
        None,
    ),
    (
        "What is the STGCN method?",
        "entity_lookup",
        "entities_global",
        None,
    ),
    (
        "Explain how the graph convolution works",
        "entity_lookup",
        "entities_global",
        None,
    ),
    (
        "Tell me everything about this paper",
        "general",            # no specific intent → default
        "paper_sections",
        None,
    ),
]


def run_intent_tests():
    sep("1. INTENT DETECTION — ALL QUESTION TYPES")
    passed = 0
    failed = 0

    for question, expected_intent, expected_coll, expected_filter_key in TEST_CASES:
        plan = parse_query(question)

        intent_ok = plan["intent"] == expected_intent
        coll_ok   = expected_coll in plan["collections"]
        filter_ok = (
            expected_filter_key is None
            or expected_filter_key in str(plan["where"])
        )

        if intent_ok and coll_ok and filter_ok:
            passed += 1
        else:
            failed += 1
            print(f"\n  ✗ FAILED: {question[:60]}")
            print(f"    intent    : got={plan['intent']} expected={expected_intent}")
            print(f"    collection: got={plan['collections']} expected contains {expected_coll}")
            print(f"    filter    : got={plan['where']}")

    print(f"\n  {passed}/{len(TEST_CASES)} intent tests passed")
    if failed > 0:
        fail(f"{failed} intent tests failed")
    else:
        ok("All intent tests passed")


def run_paper_filter_test():
    sep("2. PAPER ID FILTER DETECTION")

    q = "What are the limitations of stgcn_yu_2018?"
    plan = parse_query(q)
    has_paper_filter = "paper_id" in str(plan["where"])

    if has_paper_filter:
        ok(f"Paper ID filter detected: {plan['where']}")
    else:
        fail(f"Paper ID filter not detected in: {plan['where']}")


def run_combined_filter_test():
    sep("3. COMBINED FILTER — INTENT + PAPER ID")

    q = "What are the limitations of stgcn_yu_2018?"
    plan = parse_query(q)

    has_chunk_type = "chunk_type" in str(plan["where"])
    has_paper_id   = "paper_id"   in str(plan["where"])

    if has_chunk_type and has_paper_id:
        ok(f"Combined filter works: {plan['where']}")
    else:
        fail(
            f"Combined filter missing fields.\n"
            f"    chunk_type present: {has_chunk_type}\n"
            f"    paper_id present  : {has_paper_id}\n"
            f"    where = {plan['where']}"
        )


def run_entity_enrichment_test():
    sep("4. ENTITY ENRICHMENT IN QUERY TEXT")

    known_methods = ["STGCN", "GCGRU", "DCRNN"]
    q = "How well does this method perform on the dataset?"

    plan_without = parse_query(q)
    plan_with    = parse_query(q, known_methods=known_methods)

    ok(f"Without entity hint: '{plan_without['query_text']}'")
    ok(f"With entity hint   : '{plan_with['query_text']}'")


def run_gap_special_test():
    sep("5. GAP QUERY — SPECIAL FLAG")

    q = "What research gaps exist in traffic forecasting?"
    plan = parse_query(q)

    if plan.get("special") == "gap_matrix":
        ok(f"Gap query has special='gap_matrix' flag")
    else:
        fail(f"Gap query missing special flag: {plan}")

    if plan["n_results"] == 0:
        ok("n_results=0 for gap query (reads gap_matrix.json directly)")
    else:
        fail(f"Gap query should have n_results=0, got {plan['n_results']}")


def show_all_plans():
    sep("6. FULL PLAN DISPLAY — SAMPLE QUESTIONS")

    samples = [
        "What are the limitations of STGCN?",
        "How does STGCN compare to GCGRU?",
        "What research gaps exist?",
        "Write a literature review on traffic forecasting",
    ]

    for q in samples:
        plan = parse_query(q)
        explain_plan(plan)


def final_summary():
    sep("FINAL SUMMARY")
    print()
    print("  Query handler correctly parses all question types.")
    print()
    print("  Intent → Collection → Filter mapping:")
    print("    limitation    → claims_and_findings → chunk_type=limitation")
    print("    future_work   → claims_and_findings → chunk_type=future_work")
    print("    comparison    → claims_and_findings → claim_type=comparative")
    print("    performance   → claims_and_findings → claim_type=performance")
    print("    gap           → entities_global     → reads gap_matrix.json")
    print("    figure        → paper_sections      → chunk_type=figure")
    print("    lit_review    → paper_sections      → (semantic only)")
    print("    entity_lookup → entities_global     → (semantic only)")
    print("    general       → paper_sections      → (semantic only)")
    print()
    print("  Next step: build retriever.py (Step 6)")
    sep()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  QUERY HANDLER TEST")
    print("="*60)

    run_intent_tests()
    run_paper_filter_test()
    run_combined_filter_test()
    run_entity_enrichment_test()
    run_gap_special_test()
    show_all_plans()
    final_summary()