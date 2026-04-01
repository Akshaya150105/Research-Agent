"""
chunker.py
----------
Integrated version: 
1. Uses claims_output.json["paper_id"] as the single source of truth (Friend's logic).
2. Uses enumerate() indexing to ensure unique ChromaDB IDs (Your logic).
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

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_MODEL_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_paper(folder_path: str) -> list[dict]:
    """
    Main entry point. Reads the output_folder for one paper.
    paper_id is read directly from claims_output.json (Friend's logic).
    """
    folder = Path(folder_path)
    
    # Load required source files to find the ID first
    claims_data   = _load_json(folder / "claims_output.json")
    sections_data = _load_json(folder / "sections.json")
    figures_data  = _load_json(folder / "figures.json", required=False)

    # READ paper_id directly from JSON (Friend's requirement)
    paper_id = _get_paper_id(claims_data, folder)
    
    # Verify folder now that we have the proper ID
    _verify_folder(folder, paper_id)

    paper_title = _get_paper_title(claims_data, paper_id)

    logger.info(f"[{paper_id}] Chunking paper: '{paper_title}'")

    chunks = []

    # Your indexing logic (enumerate) preserved for uniqueness
    chunks.extend(_chunk_claims(claims_data, paper_id, paper_title))
    chunks.extend(_chunk_limitations(claims_data, paper_id, paper_title))
    chunks.extend(_chunk_future_work(claims_data, paper_id, paper_title))
    chunks.extend(_chunk_entities(claims_data, paper_id, paper_title))
    chunks.extend(_chunk_sections(sections_data, paper_id, paper_title))
    chunks.extend(_chunk_figures(figures_data or [], paper_id, paper_title))

    _log_summary(paper_id, chunks)

    return chunks

# ---------------------------------------------------------------------------
# Source file loaders
# ---------------------------------------------------------------------------

def _get_paper_id(claims_data: dict, folder: Path) -> str:
    """
    Friend's Logic: Reads paper_id from claims_output.json['paper_id'].
    Falls back to folder name if field is missing.
    """
    paper_id = claims_data.get("paper_id", "")
    if paper_id:
        return validate_paper_id(paper_id)

    logger.warning(
        f"No 'paper_id' field found in claims_output.json. "
        f"Falling back to folder name: '{folder.name}'."
    )
    return validate_paper_id(folder.name)

def _verify_folder(folder: Path, paper_id: str) -> None:
    if not folder.exists():
        raise FileNotFoundError(f"output_folder not found for paper_id '{paper_id}': {folder}")

def _load_json(path: Path, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_paper_title(claims_data: dict, paper_id: str) -> str:
    try:
        return claims_data["metadata"]["title"]
    except (KeyError, TypeError):
        return paper_id

# ---------------------------------------------------------------------------
# Chunk ID generation - YOUR UNIQUE INDEXING LOGIC
# ---------------------------------------------------------------------------

def _make_chunk_id(paper_id: str, content: str, index: int = 0) -> str:
    """
    Your Logic: Uses an occurrence index to ensure unique IDs 
    even if the same sentence appears twice.
    """
    raw = f"{paper_id}::{content}::{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

# ---------------------------------------------------------------------------
# Numeric/Entity helpers
# ---------------------------------------------------------------------------

def _extract_numeric(claim: dict) -> tuple[bool, float | None]:
    value = claim.get("value")
    if value is not None:
        try:
            return True, float(value)
        except (TypeError, ValueError):
            pass
    return False, None

def _extract_entity_fields(claim: dict) -> dict:
    entities = claim.get("entities_involved", [])
    return {
        "entities_mentioned": ",".join(entities) if entities else "",
        "methods_mentioned": "",
        "datasets_mentioned": "",
        "metrics_mentioned": "",
    }

# ---------------------------------------------------------------------------
# Specialized Chunking Functions
# ---------------------------------------------------------------------------

def _chunk_claims(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    claims = claims_data.get("claims", [])
    chunks = []
    for i, claim in enumerate(claims):
        claim["_chunk_type"] = "claim"
        embed_text   = build_claim_text(claim, paper_title)
        display_text = claim.get("description", "")
        has_numeric, numeric_value = _extract_numeric(claim)
        
        chunks.append({
            "chunk_id":    _make_chunk_id(paper_id, display_text, i),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "claim",
            "embed_text":   embed_text,
            "display_text": display_text,
            "claim_type":   claim.get("claim_type", ""),
            "section_type": claim.get("section_type", ""),
            "section_heading": claim.get("section_heading", ""),
            "confidence":   float(claim.get("confidence", 0.0)),
            "source":       claim.get("source", "llm"),
            "has_numeric_value": has_numeric,
            "numeric_value":     numeric_value if has_numeric else 0.0,
            **_extract_entity_fields(claim),
            "embed_model": EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
            "also_in_papers": "",
            "appears_in_n_papers": 1,
        })
    return chunks

def _chunk_limitations(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    limitations = claims_data.get("limitations", [])
    chunks = []
    for i, lim in enumerate(limitations):
        lim["_chunk_type"] = "limitation"
        display_text = lim.get("text", "")
        chunks.append({
            "chunk_id":    _make_chunk_id(paper_id, display_text, i),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "limitation",
            "embed_text":   build_claim_text(lim, paper_title),
            "display_text": display_text,
            "claim_type":   "limitation",
            "section_type": lim.get("section_type", ""),
            "section_heading": lim.get("section_heading", ""),
            "confidence":   float(lim.get("confidence", 0.0)),
            "source":       lim.get("source", "llm"),
            "has_numeric_value": False,
            "numeric_value": 0.0,
            **_extract_entity_fields(lim),
            "embed_model": EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
            "also_in_papers": "",
            "appears_in_n_papers": 1,
        })
    return chunks

def _chunk_future_work(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    future_work = claims_data.get("future_work", [])
    chunks = []
    for i, fw in enumerate(future_work):
        fw["_chunk_type"] = "future_work"
        display_text = fw.get("text", "")
        chunks.append({
            "chunk_id":    _make_chunk_id(paper_id, display_text, i),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "future_work",
            "embed_text":   build_claim_text(fw, paper_title),
            "display_text": display_text,
            "claim_type":   "future_work",
            "section_type": fw.get("section_type", ""),
            "section_heading": fw.get("section_heading", ""),
            "confidence":   float(fw.get("confidence", 0.0)),
            "source":       fw.get("source", "llm"),
            "has_numeric_value": False,
            "numeric_value": 0.0,
            **_extract_entity_fields(fw),
            "embed_model": EMBED_MODEL,
            "embed_model_version": EMBED_MODEL_VERSION,
            "also_in_papers": "",
            "appears_in_n_papers": 1,
        })
    return chunks

def _chunk_entities(claims_data: dict, paper_id: str, paper_title: str) -> list[dict]:
    entities = claims_data.get("llm_entities", []) or claims_data.get("entities", [])
    if not entities: return []

    seen: dict[tuple, dict] = {}
    for ent in entities:
        text, etype = ent.get("text", "").strip(), ent.get("entity_type", "")
        if not text or not etype: continue
        key = (text.lower(), etype)
        if key not in seen:
            seen[key] = dict(ent)
            seen[key]["_all_sections"] = [ent.get("section_type", "")]
        else:
            if ent.get("confidence", 0) > seen[key].get("confidence", 0):
                backup = seen[key]["_all_sections"]
                seen[key] = dict(ent); seen[key]["_all_sections"] = backup
            sec = ent.get("section_type", "")
            if sec and sec not in seen[key]["_all_sections"]: seen[key]["_all_sections"].append(sec)

    chunks = []
    for i, ((text_lower, etype), ent) in enumerate(seen.items()):
        display_text = ent.get("text", "")
        chunks.append({
            "chunk_id":    _make_chunk_id(paper_id, f"entity::{etype}::{text_lower}", i),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "entity",
            "embed_text":   build_entity_text(ent, paper_title),
            "display_text": display_text,
            "entity_type": etype,
            "entity_text": display_text,
            "entity_text_normalized": text_lower,
            "section_type": ent.get("section_type", ""),
            "all_sections": ",".join(dict.fromkeys(ent["_all_sections"])),
            "confidence": float(ent.get("confidence", 0.0)),
            "source": ent.get("source", "llm"),
            "claim_type": "", "section_heading": "",
            "has_numeric_value": False, "numeric_value": 0.0,
            "entities_mentioned": display_text,
            "methods_mentioned": display_text if etype == "method" else "",
            "datasets_mentioned": display_text if etype == "dataset" else "",
            "metrics_mentioned": display_text if etype == "metric" else "",
            "also_in_papers": "", "appears_in_n_papers": 1,
            "embed_model": EMBED_MODEL, "embed_model_version": EMBED_MODEL_VERSION,
        })
    return chunks

def _chunk_sections(sections_data: list, paper_id: str, paper_title: str) -> list[dict]:
    chunks = []
    for i, section in enumerate(sections_data):
        text = section.get("text", "").strip()
        if not text: continue
        
        chunks.append({
            "chunk_id":    _make_chunk_id(paper_id, f"section::{section.get('section_type')}::{section.get('heading')}", i),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "section",
            "embed_text":   build_section_text(section, paper_title),
            "display_text": text,
            "section_type": section.get("section_type", ""),
            "section_heading": section.get("heading", ""),
            "text_length": len(text),
            "claim_type": "", "entity_type": "", "entity_text": "",
            "confidence": 1.0, "has_numeric_value": False, "numeric_value": 0.0,
            "entities_mentioned": "", "methods_mentioned": "",
            "datasets_mentioned": "", "metrics_mentioned": "",
            "also_in_papers": "", "appears_in_n_papers": 1,
            "embed_model": EMBED_MODEL, "embed_model_version": EMBED_MODEL_VERSION,
        })
    return chunks

def _chunk_figures(figures_data: list, paper_id: str, paper_title: str) -> list[dict]:
    chunks = []
    for i, fig in enumerate(figures_data):
        caption = fig.get("caption", "").strip()
        if not caption: continue
        
        chunks.append({
            "chunk_id":    _make_chunk_id(paper_id, f"figure::{fig.get('figure_id')}::{fig.get('label')}", i),
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_type":  "figure",
            "embed_text":   build_figure_text(fig, paper_title),
            "display_text": caption,
            "figure_id": fig.get("figure_id", ""),
            "figure_label": fig.get("label", ""),
            "section_type": "Figure",
            "section_heading": f"Figure {fig.get('label')}",
            "claim_type": "", "entity_type": "", "entity_text": "",
            "confidence": 1.0, "has_numeric_value": False, "numeric_value": 0.0,
            "entities_mentioned": "", "methods_mentioned": "",
            "datasets_mentioned": "", "metrics_mentioned": "",
            "also_in_papers": "", "appears_in_n_papers": 1,
            "text_length": len(caption),
            "embed_model": EMBED_MODEL, "embed_model_version": EMBED_MODEL_VERSION,
        })
    return chunks

# ---------------------------------------------------------------------------
# Summary logger
# ---------------------------------------------------------------------------

def _log_summary(paper_id: str, chunks: list[dict]) -> None:
    from collections import Counter
    counts = Counter(c["chunk_type"] for c in chunks)
    summary = "\n".join(f"  {k:<14}: {v}" for k, v in sorted(counts.items()))
    print(f"\n{'='*60}\n  paper_id : {paper_id}\n  total    : {len(chunks)} chunks\n{summary}\n{'='*60}\n")