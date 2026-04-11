"""
arxiv_utils.py — ArXiv search/download + paper registry (SHA-256 dedup)
═══════════════════════════════════════════════════════════════════════
Pure logic, no Streamlit dependency.
"""

import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
ARXIV_API         = "https://export.arxiv.org/api/query"
ARXIV_NS          = "http://www.w3.org/2005/Atom"
ARXIV_MAX_RESULTS = 10


# ─────────────────────────────────────────────────────────────────────
# Registry helpers — SHA-256 deduplication
# ─────────────────────────────────────────────────────────────────────
def _registry_path(output_root: str) -> Path:
    return Path(output_root) / "paper_registry.json"


def load_registry(output_root: str) -> dict:
    """
    Returns a dict with two sub-dicts:
      "hashes"    : {sha256_hex → paper_id}
      "arxiv_ids" : {arxiv_id  → paper_id}
    Creates the file if missing.
    """
    p = _registry_path(output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "hashes" not in data:
                return {"hashes": data, "arxiv_ids": {}}
            return data
        except json.JSONDecodeError:
            pass
    return {"hashes": {}, "arxiv_ids": {}}


def save_registry(output_root: str, registry: dict) -> None:
    _registry_path(output_root).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_duplicate_hash(pdf_bytes: bytes, registry: dict) -> Optional[str]:
    """Return existing paper_id if SHA-256 hash already known, else None."""
    return registry["hashes"].get(sha256_of(pdf_bytes))


def check_duplicate_arxiv(arxiv_id: str, registry: dict) -> Optional[str]:
    """Return existing paper_id if ArXiv ID already known, else None."""
    return registry["arxiv_ids"].get(arxiv_id)


def register_paper(
    pdf_bytes: bytes,
    paper_id: str,
    registry: dict,
    arxiv_id: Optional[str] = None,
) -> None:
    """Record both hash and (optionally) arxiv_id → paper_id."""
    registry["hashes"][sha256_of(pdf_bytes)] = paper_id
    if arxiv_id:
        registry["arxiv_ids"][arxiv_id] = paper_id


# ─────────────────────────────────────────────────────────────────────
# LLM Query Expansion — converts natural language to ArXiv search terms
#                        Uses Groq (separate from Ollama/claim extraction)
# ─────────────────────────────────────────────────────────────────────

GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"  # fast + free

_EXPANSION_SYSTEM_PROMPT = """You are an expert academic search assistant specializing in
computer science and machine learning research. Your job is to convert a user's natural
language research request into structured ArXiv search terms.

You MUST respond with a single valid JSON object and nothing else — no markdown fences,
no preamble. Use this exact schema:

{
  "primary_query": "<2-3 core technical keywords that MUST appear in the abstract>",
  "intent_summary": "<one sentence describing what the user is looking for>",
  "topic_tags": ["<tag1>", "<tag2>", ...],
  "related_terms": ["<term1>", "<term2>", ...]
}

Rules:
- primary_query: Use a tight 2-3 word technical phrase that will appear in paper abstracts.
  Wrap the core concept in quotes for exact matching when appropriate.
  GOOD: '"transformer architecture" attention'
  GOOD: '"reinforcement learning" multi-agent'
  BAD: 'papers about transformers deep learning models survey'  (too long, too vague)
- topic_tags: 3-6 concise topic labels (e.g. "attention mechanism", "BERT", "NLP").
- related_terms: 3-5 closely related concepts the user likely also wants.
- Keep everything focused on academic CS/ML research.
- primary_query must be SHORT and PRECISE — it searches paper abstracts directly.
"""

_FILLER_RE = re.compile(
    r"(?i)\b(i want|i need|find me|show me|give me|"
    r"research papers? on|papers? about|articles? on|"
    r"papers? related to|look for|search for)\b"
)


def _fallback_expansion(natural_language_query: str, reason: str) -> dict:
    """Strip conversational filler and return a plain-text fallback."""
    fallback = _FILLER_RE.sub("", natural_language_query).strip()
    return {
        "primary_query":  fallback or natural_language_query,
        "intent_summary": f"(Groq expansion failed: {reason}) — using raw query.",
        "topic_tags":     [],
        "related_terms":  [],
        "raw_nl_query":   natural_language_query,
        "expansion_ok":   False,
    }


def expand_query_with_groq(
    natural_language_query: str,
    groq_api_key: str,
    model: str = GROQ_DEFAULT_MODEL,
    timeout: int = 20,
) -> dict:
    """
    Send the user's natural language query to Groq and get back structured
    ArXiv search terms.

    Returns a dict with keys:
        primary_query    — the optimised ArXiv search string
        intent_summary   — human-readable description of what was understood
        topic_tags       — list of topic labels
        related_terms    — list of related concepts
        raw_nl_query     — the original user input (passed through)
        expansion_ok     — True if Groq succeeded, False if fallback used
    """
    if not groq_api_key or not groq_api_key.strip():
        return _fallback_expansion(natural_language_query, "no Groq API key provided")

    headers = {
        "Authorization": f"Bearer {groq_api_key.strip()}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXPANSION_SYSTEM_PROMPT},
            {"role": "user",   "content": natural_language_query},
        ],
        "temperature":     0.2,
        "max_tokens":      400,
        "response_format": {"type": "json_object"},  # enforces JSON output
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()

        raw_text = resp.json()["choices"][0]["message"]["content"]
        # Strip accidental fences just in case
        clean  = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
        parsed = json.loads(clean)

        return {
            "primary_query":  parsed.get("primary_query",  natural_language_query),
            "intent_summary": parsed.get("intent_summary", ""),
            "topic_tags":     parsed.get("topic_tags",     []),
            "related_terms":  parsed.get("related_terms",  []),
            "raw_nl_query":   natural_language_query,
            "expansion_ok":   True,
        }

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        reason = "rate limited, try again in a moment" if status == 429 else "check your API key"
        return _fallback_expansion(natural_language_query, f"HTTP {status} — {reason}")
    except (KeyError, json.JSONDecodeError) as exc:
        return _fallback_expansion(natural_language_query, f"unexpected response: {exc}")
    except Exception as exc:
        return _fallback_expansion(natural_language_query, str(exc))


# ─────────────────────────────────────────────────────────────────────
# ArXiv helpers
# ─────────────────────────────────────────────────────────────────────
def _build_arxiv_query(query: str) -> str:
    """
    Convert a Groq-generated query string into a precise ArXiv search query.

    Strategy:
      - Search title (ti:) AND abstract (abs:) rather than all: fields
      - This eliminates papers that only mention the topic in passing
        (e.g. in references or boilerplate text)
      - Quoted phrases from Groq are preserved for exact matching
    """
    query = query.strip()
    if not query:
        return "abs:transformer"

    # If Groq already produced a quoted phrase, use it directly in abs+ti
    if '"' in query:
        # e.g. '"transformer architecture" attention'
        # Search: abstract contains the whole thing
        return f"abs:({query})"

    # Otherwise split into words and build a combined ti+abs query
    # First word/phrase → must appear in title (high precision)
    # Remaining words   → must appear in abstract
    words = query.split()
    if len(words) == 1:
        return f"ti:{words[0]} OR abs:{words[0]}"

    # Core topic in title, supporting terms in abstract
    core = words[0]
    support = " AND abs:".join(words[1:])
    return f"(ti:{core} OR abs:{core}) AND abs:{support}"


def arxiv_search(query: str, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
    """
    Query ArXiv Atom API with a precision-focused search strategy.

    Returns list of dicts:
      id, title, authors, abstract, published, pdf_url, arxiv_id
    """
    arxiv_query = _build_arxiv_query(query)

    params = urllib.parse.urlencode({
        "search_query": arxiv_query,
        "start":        0,
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    })

    resp = requests.get(f"{ARXIV_API}?{params}", timeout=15)
    resp.raise_for_status()

    root   = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        raw_id   = entry.findtext(f"{{{ARXIV_NS}}}id", default="")
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rstrip("/").split("/")[-1])

        title     = (entry.findtext(f"{{{ARXIV_NS}}}title")   or "").replace("\n", " ").strip()
        abstract  = (entry.findtext(f"{{{ARXIV_NS}}}summary") or "").replace("\n", " ").strip()
        published = entry.findtext(f"{{{ARXIV_NS}}}published", default="")[:10]
        authors   = [
            (a.findtext(f"{{{ARXIV_NS}}}name") or "")
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
        ]

        pdf_url = ""
        for link in entry.findall(f"{{{ARXIV_NS}}}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append({
            "id":        arxiv_id,
            "title":     title,
            "authors":   authors,
            "abstract":  abstract,
            "published": published,
            "pdf_url":   pdf_url,
            "arxiv_id":  arxiv_id,
        })

    return papers


def arxiv_search_with_expansion(
    natural_language_query: str,
    groq_api_key: str,
    max_results: int = ARXIV_MAX_RESULTS,
    model: str = GROQ_DEFAULT_MODEL,
) -> tuple[list[dict], dict]:
    """
    Full pipeline: NL query → Groq expansion → ArXiv search.

    Returns:
        (papers, expansion_info)
        expansion_info has keys: primary_query, intent_summary,
                                  topic_tags, related_terms,
                                  raw_nl_query, expansion_ok
    """
    expansion = expand_query_with_groq(natural_language_query, groq_api_key, model=model)
    papers    = arxiv_search(expansion["primary_query"], max_results=max_results)
    return papers, expansion


def download_arxiv_pdf(pdf_url: str, timeout: int = 60) -> bytes:
    resp = requests.get(
        pdf_url, timeout=timeout,
        headers={"User-Agent": "ResearchPipeline/2.0"},
    )
    resp.raise_for_status()
    return resp.content


def arxiv_id_to_filename(arxiv_id: str) -> str:
    safe_id = re.sub(r'[^\w\-.]', '_', arxiv_id)
    return f"arxiv_{safe_id}.pdf"