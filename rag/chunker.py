"""
chunker.py
----------
Reads one paper's output_folder and produces a flat list of
typed chunk dictionaries ready for embedding and indexing.

Nothing is embedded or stored here.
This step only transforms raw JSON into a clean intermediate format.

Usage:
    from rag.chunker import chunk_paper

    chunks = chunk_paper(
        folder_path="memory/stgcn_yu_2018",
        paper_id="stgcn_yu_2018"
    )
    print(f"Total chunks: {len(chunks)}")

Source files consumed:
    claims_output.json   → claims, limitations, future_work, llm_entities
    sections.json        → full section text
    figures.json         → figure captions

Output chunk types:
    "claim"        → goes to claims_and_findings collection
    "limitation"   → goes to claims_and_findings collection
    "future_work"  → goes to claims_and_findings collection
    "entity"       → goes to entities_global collection
    "section"      → goes to paper_sections collection
    "figure"       → goes to paper_sections collection
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any

from rag.utils.paper_id import validate_paper_id
from rag.utils.text_builder import (
    build_claim_text,
    build_entity_text,
    build_section_text,
    build_figure_text,
)

logger = logging.getLogger(__name__)

# embed model name stored with every chunk so we know which chunks
# need re-embedding if we upgrade the model later
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_MODEL_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_paper(folder_path: str, paper_id: str) -> list[dict]:
    """
    Main entry point. Reads the output_folder for one paper and returns
    a flat list of chunk dicts.

    Args:
        folder_path: Path to the paper's output_folder.
                     e.g. "memory/stgcn_yu_2018"
        paper_id:    Canonical paper identifier.
                     e.g. "stgcn_yu_2018"
                     Must match the folder name and your friend's KG node ID.

    Returns:
        List of chunk dicts. Each chunk has:
            chunk_id, paper_id, chunk_type, embed_text,
            display_text, and type-specific metadata fields.

    Raises:
        FileNotFoundError: if claims_output.json or sections.json is missing.
        ValueError: if paper_id is invalid.
    """
    paper_id  = validate_paper_id(paper_id)
    folder    = Path(folder_path)

    _verify_folder(folder, paper_id)

    # load source files
    claims_data  = _load_json(folder / "claims_output.json")
    sections_data = _load_json(folder / "sections.json")
    figures_data  = _load_json(folder / "figures.json", required=False)

    # extract paper title for embed text context
    paper_title = _get_paper_title(claims_data, paper_id)

    logger.info(f"[{paper_id}] Chunking paper: '{paper_title}'")

    # --- produce chunks from each source ---
    chunks = []

    claim_chunks      = _chunk_claims(claims_data, paper_id, paper_title)
    limitation_chunks = _chunk_limitations(claims_data, paper_id, paper_title)
    future_chunks     = _chunk_future_work(claims_data, paper_id, paper_title)
    entity_chunks     = _chunk_entities(claims_data, paper_id, paper_title)
    section_chunks    = _chunk_sections(sections_data, paper_id, paper_title)
    figure_chunks     = _chunk_figures(figures_data or [], paper_id, paper_title)

    chunks.extend(claim_chunks)
    chunks.extend(limitation_chunks)
    chunks.extend(future_chunks)
    chunks.extend(entity_chunks)
    chunks.extend(section_chunks)
    chunks.extend(figure_chunks)

    # log summary
    _log_summary(paper_id, chunks)

    return chunks


# ---------------------------------------------------------------------------
# Source file loaders
# ---------------------------------------------------------------------------

def _verify_folder(folder: Path, paper_id: str) -> None:
    if not folder.exists():
        raise FileNotFoundError(
            f"output_folder not found: {folder}\n"
            f"Expected folder for paper_id '{paper_id}'."
        )

def _load_json(path: Path, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Make sure Phase 1 completed successfully for this paper."
            )
        logger.warning(f"Optional file not found, skipping: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_paper_title(claims_data: dict, paper_id: str) -> str:
    """Extracts paper title from claims_output.json metadata."""
    try:
        return claims_data["metadata"]["title"]
    except (KeyError, TypeError):
        logger.warning(f"[{paper_id}] Could not find title in metadata, using paper_id.")
        return paper_id


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(paper_id: str, content: str) -> str:
    """
    Generates a stable, unique chunk ID.
    Deterministic: same paper + same content always produces same ID.
    This makes the indexer idempotent — upserting the same chunk twice
    is safe and produces no duplicates.

    Returns first 12 hex chars of SHA-256.
    """
    raw = f"{paper_id}::{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Numeric value extraction
# ---------------------------------------------------------------------------

def _extract_numeric(claim: dict) -> tuple[bool, float | None]:
    """
    Checks if a claim has a numeric value.
    Returns (has_numeric_value, numeric_value).

    The 'value' field in claims_output.json holds floats like 14.0, 272.34, 95.0.
    These are critical for contradiction detection — two papers claiming
    different numeric results about the same method is a contradiction signal.
    """
    value = claim.get("value")
    if value is not None:
        try:
            return True, float(value)
        except (TypeError, ValueError):
            pass
    return False, None


# ---------------------------------------------------------------------------
# Entity field extraction
# ---------------------------------------------------------------------------

def _extract_entity_fields(claim: dict) -> dict:
    """
    Extracts method/dataset/metric mentions from entities_involved.
    These become filterable metadata fields in ChromaDB.

    Strategy: we don't have entity_type labels on entities_involved items
    in claims — just names. We map them against the entity_index to find type.
    If not found, include in a generic 'entities_mentioned' field.

    This is populated more accurately in the enricher after full indexing,
    but we do a best-effort pass here.
    """
    entities = claim.get("entities_involved", [])
    if not entities:
        return {
            "entities_mentioned": "",
            "methods_mentioned": "",
            "datasets_mentioned": "",
            "metrics_mentioned": "",
        }

    return {
        "entities_mentioned": ",".join(entities),
        # methods/datasets/metrics are filled properly by enricher.py
        # we store empty strings here as placeholders — ChromaDB requires
        # consistent metadata schema across all documents in a collection
        "methods_mentioned": "",
        "datasets_mentioned": "",
        "metrics_mentioned": "",
    }


# ---------------------------------------------------------------------------
# Claim chunks (from claims_output.json → "claims" list)
# ---------------------------------------------------------------------------

def _chunk_claims(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    """
    Produces one chunk per claim.
    Claims include: methodological, performance, comparative.
    All go to claims_and_findings collection.
    """
    claims = claims_data.get("claims", [])
    if not claims:
        logger.warning(f"[{paper_id}] No claims found in claims_output.json")
        return []

    chunks = []
    for claim in claims:
        # inject chunk_type so text_builder knows what prefix to use
        claim["_chunk_type"] = "claim"

        embed_text   = build_claim_text(claim, paper_title)
        display_text = claim.get("description", "")

        has_numeric, numeric_value = _extract_numeric(claim)
        entity_fields = _extract_entity_fields(claim)

        chunk = {
            # identity
            "chunk_id":    _make_chunk_id(paper_id, display_text),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "claim",

            # text
            "embed_text":   embed_text,
            "display_text": display_text,

            # claim-specific metadata
            "claim_type":    claim.get("claim_type", ""),
            "section_type":  claim.get("section_type", ""),
            "section_heading": claim.get("section_heading", ""),
            "confidence":    float(claim.get("confidence", 0.0)),
            "source":        claim.get("source", "llm"),

            # numeric value — used for contradiction detection
            "has_numeric_value": has_numeric,
            "numeric_value":     numeric_value if has_numeric else 0.0,

            # entity fields — enricher fills methods/datasets/metrics later
            **entity_fields,

            # provenance
            "embed_model":         EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,

            # cross-paper fields — populated by enricher.py
            "also_in_papers":      "",
            "appears_in_n_papers": 1,
        }
        chunks.append(chunk)

    logger.info(f"[{paper_id}] Produced {len(chunks)} claim chunks")
    return chunks


# ---------------------------------------------------------------------------
# Limitation chunks (from claims_output.json → "limitations" list)
# ---------------------------------------------------------------------------

def _chunk_limitations(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    """
    Produces one chunk per limitation statement.
    Limitations are sentence-level, extracted from across the paper.
    Goes to claims_and_findings collection with chunk_type='limitation'.
    """
    limitations = claims_data.get("limitations", [])
    if not limitations:
        logger.warning(f"[{paper_id}] No limitations found")
        return []

    chunks = []
    for lim in limitations:
        # limitations use the 'text' field, not 'description'
        lim["_chunk_type"] = "limitation"
        lim["claim_type"]  = ""  # limitations don't have claim_type

        embed_text   = build_claim_text(lim, paper_title)
        display_text = lim.get("text", "")

        entity_fields = _extract_entity_fields(lim)

        chunk = {
            "chunk_id":    _make_chunk_id(paper_id, display_text),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "limitation",

            "embed_text":   embed_text,
            "display_text": display_text,

            "claim_type":      "limitation",
            "section_type":    lim.get("section_type", ""),
            "section_heading": lim.get("section_heading", ""),
            "confidence":      float(lim.get("confidence", 0.0)),
            "source":          lim.get("source", "llm"),

            "has_numeric_value": False,
            "numeric_value":     0.0,

            **entity_fields,

            "embed_model":         EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
            "also_in_papers":      "",
            "appears_in_n_papers": 1,
        }
        chunks.append(chunk)

    logger.info(f"[{paper_id}] Produced {len(chunks)} limitation chunks")
    return chunks


# ---------------------------------------------------------------------------
# Future work chunks (from claims_output.json → "future_work" list)
# ---------------------------------------------------------------------------

def _chunk_future_work(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    """
    Produces one chunk per future work item.
    Future work items point toward research gaps — important for the
    gap detection agent to cross-reference with what actually exists.
    Goes to claims_and_findings collection with chunk_type='future_work'.
    """
    future_work = claims_data.get("future_work", [])
    if not future_work:
        logger.warning(f"[{paper_id}] No future_work items found")
        return []

    chunks = []
    for fw in future_work:
        fw["_chunk_type"] = "future_work"
        fw["claim_type"]  = ""

        embed_text   = build_claim_text(fw, paper_title)
        display_text = fw.get("text", "")

        entity_fields = _extract_entity_fields(fw)

        chunk = {
            "chunk_id":    _make_chunk_id(paper_id, display_text),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "future_work",

            "embed_text":   embed_text,
            "display_text": display_text,

            "claim_type":      "future_work",
            "section_type":    fw.get("section_type", ""),
            "section_heading": fw.get("section_heading", ""),
            "confidence":      float(fw.get("confidence", 0.0)),
            "source":          fw.get("source", "llm"),

            "has_numeric_value": False,
            "numeric_value":     0.0,

            **entity_fields,

            "embed_model":         EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
            "also_in_papers":      "",
            "appears_in_n_papers": 1,
        }
        chunks.append(chunk)

    logger.info(f"[{paper_id}] Produced {len(chunks)} future_work chunks")
    return chunks


# ---------------------------------------------------------------------------
# Entity chunks (from claims_output.json → "llm_entities" list)
# ---------------------------------------------------------------------------

def _chunk_entities(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    """
    Produces one chunk per unique entity mention.
    Uses llm_entities (higher quality, LLM-validated) rather than
    the raw NER entities — the LLM entities have cleaner text and
    better type classification.

    Deduplicates by (entity_text_lower, entity_type) so the same
    entity mentioned in multiple sections produces ONE chunk with
    the also_in_sections field populated.

    Goes to entities_global collection.
    """
    entities = claims_data.get("llm_entities", [])
    if not entities:
        # fall back to the top-level entities list if llm_entities not present
        entities = claims_data.get("entities", [])

    if not entities:
        logger.warning(f"[{paper_id}] No entities found")
        return []

    # deduplicate: keep the highest-confidence occurrence as the primary,
    # collect all section appearances
    seen: dict[tuple, dict] = {}   # (text_lower, type) → best entity dict

    for ent in entities:
        text  = ent.get("text", "").strip()
        etype = ent.get("entity_type", "")
        if not text or not etype:
            continue

        key = (text.lower(), etype)
        if key not in seen:
            seen[key] = dict(ent)
            seen[key]["_all_sections"] = [ent.get("section_type", "")]
        else:
            # keep higher confidence
            if ent.get("confidence", 0) > seen[key].get("confidence", 0):
                section_backup = seen[key]["_all_sections"]
                seen[key] = dict(ent)
                seen[key]["_all_sections"] = section_backup
            # accumulate section appearances
            sec = ent.get("section_type", "")
            if sec and sec not in seen[key]["_all_sections"]:
                seen[key]["_all_sections"].append(sec)

    chunks = []
    for (text_lower, etype), ent in seen.items():
        embed_text   = build_entity_text(ent, paper_title)
        display_text = ent.get("text", "")

        # sections where this entity appears
        primary_section = ent.get("section_type", "")
        also_in_sections = ent.get("also_in_sections", []) or []
        all_sections = ent.get("_all_sections", [primary_section])
        all_sections_str = ",".join(dict.fromkeys(all_sections))

        chunk = {
            "chunk_id":    _make_chunk_id(paper_id, f"entity::{etype}::{text_lower}"),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "entity",

            "embed_text":   embed_text,
            "display_text": display_text,

            # entity-specific metadata
            "entity_type":    etype,
            "entity_text":    display_text,
            "entity_text_normalized": text_lower,
            "section_type":   primary_section,
            "all_sections":   all_sections_str,
            "confidence":     float(ent.get("confidence", 0.0)),
            "source":         ent.get("source", "llm"),

            # claim_type not applicable to entities — empty string keeps schema consistent
            "claim_type":      "",
            "section_heading": "",

            # numeric — not applicable
            "has_numeric_value": False,
            "numeric_value":     0.0,

            # entity mention fields
            "entities_mentioned":  display_text,
            "methods_mentioned":   display_text if etype == "method" else "",
            "datasets_mentioned":  display_text if etype == "dataset" else "",
            "metrics_mentioned":   display_text if etype == "metric" else "",

            # cross-paper — enricher fills these after all papers indexed
            "also_in_papers":      "",
            "appears_in_n_papers": 1,

            "embed_model":         EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
        }
        chunks.append(chunk)

    logger.info(f"[{paper_id}] Produced {len(chunks)} entity chunks "
                f"(deduplicated from {len(entities)} raw entity mentions)")
    return chunks


# ---------------------------------------------------------------------------
# Section chunks (from sections.json)
# ---------------------------------------------------------------------------

def _chunk_sections(sections_data: list, paper_id: str, paper_title: str) -> list[dict]:
    """
    Produces one chunk per section.
    Sections carry the full raw text — no truncation.
    The BGE model handles long texts via mean pooling.

    These are the primary source for the literature review agent
    and any general semantic search over full paper content.
    Goes to paper_sections collection.
    """
    if not sections_data:
        logger.warning(f"[{paper_id}] No sections found in sections.json")
        return []

    chunks = []
    for section in sections_data:
        section_type = section.get("section_type", "")
        heading      = section.get("heading", "")
        text         = section.get("text", "").strip()

        if not text:
            logger.warning(f"[{paper_id}] Empty text in section '{section_type}', skipping")
            continue

        embed_text   = build_section_text(section, paper_title)
        display_text = text

        chunk = {
            "chunk_id":    _make_chunk_id(paper_id, f"section::{section_type}::{heading}"),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "section",

            "embed_text":   embed_text,
            "display_text": display_text,

            "section_type":    section_type,
            "section_heading": heading,
            "text_length":     len(text),

            # not applicable to sections
            "claim_type":        "",
            "entity_type":       "",
            "entity_text":       "",
            "confidence":        1.0,
            "has_numeric_value": False,
            "numeric_value":     0.0,
            "entities_mentioned":  "",
            "methods_mentioned":   "",
            "datasets_mentioned":  "",
            "metrics_mentioned":   "",
            "also_in_papers":      "",
            "appears_in_n_papers": 1,

            "embed_model":         EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
        }
        chunks.append(chunk)

    logger.info(f"[{paper_id}] Produced {len(chunks)} section chunks")
    return chunks


# ---------------------------------------------------------------------------
# Figure chunks (from figures.json)
# ---------------------------------------------------------------------------

def _chunk_figures(figures_data: list, paper_id: str, paper_title: str) -> list[dict]:
    """
    Produces one chunk per figure caption.
    Figure captions often contain rich architectural descriptions
    that don't appear verbatim in the section text.

    Goes to paper_sections collection (same as sections,
    distinguished by chunk_type='figure').
    """
    if not figures_data:
        logger.info(f"[{paper_id}] No figures found, skipping figure chunks")
        return []

    chunks = []
    for fig in figures_data:
        caption  = fig.get("caption", "").strip()
        label    = fig.get("label", "")
        fig_id   = fig.get("figure_id", "")

        if not caption:
            continue

        embed_text   = build_figure_text(fig, paper_title)
        display_text = caption

        chunk = {
            "chunk_id":    _make_chunk_id(paper_id, f"figure::{fig_id}::{label}"),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "figure",

            "embed_text":   embed_text,
            "display_text": display_text,

            "figure_id":       fig_id,
            "figure_label":    label,
            "section_type":    "Figure",
            "section_heading": f"Figure {label}",

            # not applicable
            "claim_type":        "",
            "entity_type":       "",
            "entity_text":       "",
            "confidence":        1.0,
            "has_numeric_value": False,
            "numeric_value":     0.0,
            "entities_mentioned":  "",
            "methods_mentioned":   "",
            "datasets_mentioned":  "",
            "metrics_mentioned":   "",
            "also_in_papers":      "",
            "appears_in_n_papers": 1,
            "text_length":         len(caption),

            "embed_model":         EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
        }
        chunks.append(chunk)

    logger.info(f"[{paper_id}] Produced {len(chunks)} figure chunks")
    return chunks


# ---------------------------------------------------------------------------
# Summary logger
# ---------------------------------------------------------------------------

def _log_summary(paper_id: str, chunks: list[dict]) -> None:
    from collections import Counter
    counts = Counter(c["chunk_type"] for c in chunks)
    logger.info(
        f"[{paper_id}] Chunking complete. Total: {len(chunks)} chunks | "
        + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    )
    print(
        f"\n{'='*60}\n"
        f"  paper_id : {paper_id}\n"
        f"  total    : {len(chunks)} chunks\n"
        + "\n".join(f"  {k:<14}: {v}" for k, v in sorted(counts.items()))
        + f"\n{'='*60}\n"
    )