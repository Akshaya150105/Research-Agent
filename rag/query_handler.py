"""
query_handler.py
----------------
Parses a natural language question into a structured query object
that the retriever uses to search ChromaDB.

The output of this step is NOT a result — it's a plan:
    - what text to embed
    - which collections to search
    - which metadata filters to apply

This separation matters because:
    - Metadata filters run BEFORE vector search (fast, exact)
    - Vector search runs AFTER filtering (semantic, approximate)
    - Getting the filter right improves precision dramatically

Usage:
    from rag.query_handler import parse_query

    plan = parse_query("What are the limitations of STGCN?")
    print(plan)
    # {
    #   "query_text": "What are the limitations of STGCN?",
    #   "collections": ["claims_and_findings"],
    #   "where": {"chunk_type": {"$eq": "limitation"}},
    #   "n_results": 8,
    #   "intent": "limitation"
    # }
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── intent rules ──────────────────────────────────────────────────────────
# Each rule has:
#   patterns   - regex patterns matched against the question (case-insensitive)
#   collections - which ChromaDB collections to search
#   where       - metadata filter to apply
#   intent      - label for logging and debugging
#   n_results   - how many candidates to fetch before re-ranking

INTENT_RULES = [

    # --- limitations ---
    {
        "intent":      "limitation",
        "patterns":    [
            r"\blimitation", r"\bweakness", r"\bshortcoming",
            r"\bproblem with", r"\bissue with", r"\bdrawback",
            r"\bwhat.{0,20}wrong", r"\bwhat.{0,20}bad",
        ],
        "collections": ["claims_and_findings"],
        "where":       {"chunk_type": {"$eq": "limitation"}},
        "n_results":   10,
    },

    # --- future work ---
    {
        "intent":      "future_work",
        "patterns":    [
            r"\bfuture work", r"\bfuture direction", r"\bnext step",
            r"\bopen problem", r"\bopen question", r"\bto be explored",
            r"\bwhat.{0,20}next", r"\bwhat.{0,20}remain",
        ],
        "collections": ["claims_and_findings"],
        "where":       {"chunk_type": {"$eq": "future_work"}},
        "n_results":   8,
    },

    # --- comparison / contradiction ---
    {
        "intent":      "comparison",
        "patterns":    [
            r"\bcompar", r"\bvs\b", r"\bversus", r"\boutperform",
            r"\bfaster than", r"\bbetter than", r"\bworse than",
            r"\bcontradiction", r"\bconflict", r"\bdisagree",
            r"\bdifference between", r"\bhow does .{0,30} differ",
        ],
        "collections": ["claims_and_findings"],
        "where":       {"claim_type": {"$eq": "comparative"}},
        "n_results":   12,
    },

    # --- performance / results ---
    {
        "intent":      "performance",
        "patterns":    [
            r"\bperformance", r"\baccuracy", r"\bresult",
            r"\bscore", r"\brmse\b", r"\bmae\b", r"\bf1\b",
            r"\bbleu\b", r"\bbenchmark", r"\bevaluat",
            r"\bhow well", r"\bhow (good|accurate|fast)",
        ],
        "collections": ["claims_and_findings"],
        "where":       {"claim_type": {"$eq": "performance"}},
        "n_results":   10,
    },

    # --- research gaps ---
    {
        "intent":      "gap",
        "patterns":    [
            r"\bgap", r"\bnobody", r"\bno one", r"\bnot (yet )?tried",
            r"\buntried", r"\bmissing combination",
            r"\bwhich.{0,30}(hasn.t|have not|not been) (been )?tested",
            r"\bwhich.{0,30}(hasn.t|have not) (been )?explored",
            r"\bwhat.{0,30}(hasn.t|have not|not been) (been )?tested",
            r"\bwhat.{0,30}(hasn.t|have not) (been )?explored",
        ],
        "collections": ["entities_global"],
        "where":       {},   # gap agent reads gap_matrix.json directly
        "n_results":   0,    # no vector search needed — pure matrix lookup
        "special":     "gap_matrix",
    },

    # --- entity / method / dataset lookup ---
    {
        "intent":      "entity_lookup",
        "patterns":    [
            r"\bwhat is\b", r"\bwhat are\b", r"\bwhich (method|model|dataset|metric)",
            r"\btell me about\b", r"\bexplain\b",
            r"\bhow does .{0,30} work",
        ],
        "collections": ["entities_global", "paper_sections"],
        "where":       {},
        "n_results":   8,
    },

    # --- figure / diagram ---
    {
        "intent":      "figure",
        "patterns":    [
            r"\bfigure\b", r"\bdiagram\b", r"\barchitecture diagram",
            r"\bshow.{0,20}figure", r"\bfig\.?\s*\d",
        ],
        "collections": ["paper_sections"],
        "where":       {"chunk_type": {"$eq": "figure"}},
        "n_results":   5,
    },

    # --- literature review / full text ---
    {
        "intent":      "literature_review",
        "patterns":    [
            r"\bliterature review", r"\bsurvey", r"\bsummar",
            r"\boverview", r"\bwrite.{0,20}review",
            r"\bwhat do (the )?papers say",
        ],
        "collections": ["paper_sections", "claims_and_findings"],
        "where":       {},
        "n_results":   15,
    },
]

# default when no intent matches
DEFAULT_RULE = {
    "intent":      "general",
    "collections": ["paper_sections", "claims_and_findings"],
    "where":       {},
    "n_results":   10,
}


# ── named entity detection ────────────────────────────────────────────────

def _detect_paper_filter(question: str) -> Optional[dict]:
    """
    Detects if the question asks about a specific paper by paper_id.
    Returns a paper_id filter if found, None otherwise.

    Example: "What are STGCN's limitations?" doesn't filter by paper —
    it filters by chunk_type=limitation.

    But: "What does stgcn_yu_2018 say about training?" adds a paper filter.
    Paper IDs look like lowercase_underscore strings.
    """
    # match paper_id pattern: word_word_year or similar
    matches = re.findall(r'\b([a-z][a-z0-9]+_[a-z0-9_]+\d{4})\b', question.lower())
    if matches:
        return {"paper_id": {"$eq": matches[0]}}
    return None


def _detect_entity_filter(question: str, known_methods: list[str] = None) -> Optional[str]:
    """
    Detects if the question mentions a known method or entity name.
    Returns the entity text if found, None otherwise.

    This is used to add an entity hint to the query_text so the
    embedding focuses on the right method.

    known_methods: list of method names from entities_global.
    If not provided, skips entity detection.
    """
    if not known_methods:
        return None

    q_lower = question.lower()
    for method in known_methods:
        if method.lower() in q_lower:
            return method
    return None


# ── main parser ───────────────────────────────────────────────────────────

def parse_query(question: str, known_methods: list[str] = None) -> dict:
    """
    Parses a natural language question into a ChromaDB query plan.

    Args:
        question:      The user's question string.
        known_methods: Optional list of known method names from entities_global.
                       Used to enrich the query_text with entity hints.

    Returns:
        A query plan dict with keys:
            query_text   - text to embed (may be enriched with entity hints)
            collections  - list of collection names to search
            where        - ChromaDB metadata filter dict (may be empty)
            n_results    - number of candidates to fetch
            intent       - detected intent label
            original     - original question (preserved for display)
            special      - optional special handling flag (e.g. "gap_matrix")
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    q_lower = question.lower()

    # match against intent rules in order
    matched_rule = None
    for rule in INTENT_RULES:
        for pattern in rule["patterns"]:
            if re.search(pattern, q_lower):
                matched_rule = rule
                break
        if matched_rule:
            break

    # fall back to default if no intent matched
    rule = matched_rule or DEFAULT_RULE

    # check for paper_id filter in the question
    paper_filter = _detect_paper_filter(question)

    # build the where clause
    where = dict(rule.get("where", {}))

    if paper_filter and where:
        # combine paper filter with intent filter using $and
        where = {"$and": [where, paper_filter]}
    elif paper_filter:
        where = paper_filter

    # detect named entity to enrich query text
    detected_entity = _detect_entity_filter(question, known_methods)
    query_text = question
    if detected_entity and detected_entity.lower() not in q_lower[:20]:
        # append entity hint only if not already prominent in the question
        query_text = f"{question} {detected_entity}"

    plan = {
        "original":    question,
        "query_text":  query_text,
        "collections": rule["collections"],
        "where":       where,
        "n_results":   rule.get("n_results", 10),
        "intent":      rule["intent"],
    }

    if "special" in rule:
        plan["special"] = rule["special"]

    logger.info(
        f"Query parsed | intent={plan['intent']} | "
        f"collections={plan['collections']} | "
        f"where={plan['where']}"
    )

    return plan


# ── convenience helpers ───────────────────────────────────────────────────

def explain_plan(plan: dict) -> None:
    """Prints a human-readable explanation of a query plan."""
    print(f"\n  Question  : {plan['original']}")
    print(f"  Intent    : {plan['intent']}")
    print(f"  Collections: {plan['collections']}")
    print(f"  Filter    : {plan['where'] or '(none — pure semantic search)'}")
    print(f"  n_results : {plan['n_results']}")
    if plan.get("special"):
        print(f"  Special   : {plan['special']}")