import json
import time
import argparse
from pathlib import Path
from typing import Optional

from .ner_extractor import SciBERTNERExtractor, extract_entities_from_sections


def load_sections(grobid_output_dir: Path) -> dict:
    """Load sections.json and metadata.json from grobid_parser output folder."""
    sections_path = grobid_output_dir / "sections.json"
    metadata_path = grobid_output_dir / "metadata.json"

    if not sections_path.exists():
        raise FileNotFoundError(
            f"sections.json not found in {grobid_output_dir}\n"
            f"Make sure you've run grobid_parser first."
        )

    with open(sections_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    tables_path = grobid_output_dir / "tables.json"
    if tables_path.exists():
        with open(tables_path, "r", encoding="utf-8") as f:
            for t in json.load(f):
                content = t.get("caption", "").strip()
                rows = t.get("rows", [])
                if rows:
                    content += "\n" + "\n".join(" | ".join(str(c) for c in r) for r in rows)
                sections.append({
                    "section_type": "Table",
                    "heading": f"Table {t.get('label', '')}",
                    "text": content.strip()
                })

    figures_path = grobid_output_dir / "figures.json"
    if figures_path.exists():
        with open(figures_path, "r", encoding="utf-8") as f:
            for fig in json.load(f):
                sections.append({
                    "section_type": "Figure",
                    "heading": f"Figure {fig.get('label', '')}",
                    "text": fig.get("caption", "").strip()
                })

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return sections, metadata


def run_pipeline(
    grobid_output_dir: str,
    model_name: Optional[str] = None,
    device: int = -1,
    priority_only: bool = False,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Full NER pipeline for one paper.

    Args:
        grobid_output_dir: Path to folder containing sections.json (grobid_parser output)
        model_name:        HuggingFace model ID or local path
        device:            -1 = CPU, 0 = first GPU
        priority_only:     Only run NER on high-priority sections (faster)
        output_dir:        Where to save output. Defaults to grobid_output_dir.

    Returns:
        Path to the written enriched_entities.json
    """
    grobid_output_dir = Path(grobid_output_dir)
    output_dir = Path(output_dir) if output_dir else grobid_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Pipeline] Input:  {grobid_output_dir}")
    print(f"[Pipeline] Output: {output_dir}")
    print(f"{'='*60}\n")

    # 1. Load sections
    print("[Pipeline] Loading sections.json ...")
    sections, metadata = load_sections(grobid_output_dir)
    print(f"[Pipeline] Found {len(sections)} sections, paper: {metadata.get('title', 'Unknown')[:80]}")

    # 2. Load model
    extractor = SciBERTNERExtractor(model_name=model_name, device=device)

    # 3. Run NER
    print(f"\n[Pipeline] Running NER on {'priority' if priority_only else 'all'} sections ...")
    t0 = time.time()
    result = extract_entities_from_sections(sections, extractor, priority_only=priority_only)
    elapsed = time.time() - t0
    print(f"[Pipeline] NER complete in {elapsed:.1f}s")

    # 4. Build enriched output (the paper's Phase 1 structured JSON)
    paper_id = _make_paper_id(metadata)
    enriched = {
        "paper_id":       paper_id,
        "metadata":       metadata,
        "ner_summary": {
            "total_entities":     result["total_entities"],
            "entity_type_counts": result["entity_type_counts"],
            "section_coverage":   result["section_coverage"],
            "elapsed_seconds":    round(elapsed, 2),
            "model_used":         extractor.model_name,
            "priority_only":      priority_only,
        },
        # Flat list — used by Phase 2 LLM claim extraction
        "entities":        result["entities_flat"],
        # Indexed by type — used by Phase 2, Phase 3 Reader Agent, Phase 6 Gap Detector
        "entity_index":    result["entity_index"],
        # Per-section — preserves provenance for Critic Agent
        "entities_by_section": result["entities_by_section"],
    }

    # 5. Write output
    output_path = output_dir / "enriched_entities.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"\n[Pipeline] Saved: {output_path}")
    _print_summary(result, metadata)

    return output_path


def _make_paper_id(metadata: dict) -> str:
    """Generate a stable paper ID from metadata."""
    doi = metadata.get("doi", "")
    if doi:
        return "doi:" + doi.replace("/", "_")
    title = metadata.get("title", "unknown")
    year = metadata.get("year", "0000")
    # Simple slug from first 5 words of title + year
    slug = "_".join(title.lower().split()[:5])
    slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
    return f"{slug}_{year}"


def _print_summary(result: dict, metadata: dict):
    """Print a human-readable summary to terminal."""
    print(f"\n{'='*60}")
    print(f"ENTITY EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Paper:   {metadata.get('title', 'Unknown')[:70]}")
    print(f"Total unique entities: {result['total_entities']}")
    print()
    for etype, items in result["entity_index"].items():
        if items:
            print(f"  {etype.upper()} ({len(items)} unique):")
            # Show top 10 by confidence
            flat = [e for elist in items.values() for e in elist]
            top = sorted(flat, key=lambda x: x["confidence"], reverse=True)[:10]
            for ent in top:
                sections_str = ent["section_type"]
                if ent.get("also_in_sections"):
                    sections_str += ", " + ", ".join(ent["also_in_sections"])
                print(f"    • {ent['text']:<35} conf={ent['confidence']:.3f}  [{sections_str}]")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Batch runner — process multiple grobid output folders at once
# ---------------------------------------------------------------------------

def run_batch(
    root_dir: str,
    model_name: Optional[str] = None,
    device: int = -1,
    priority_only: bool = False,
):
    """
    Process all subfolders under root_dir that contain a sections.json.
    Useful when you've parsed many papers and want to run NER in one shot.
    """
    root = Path(root_dir)
    folders = [f for f in root.iterdir() if f.is_dir() and (f / "sections.json").exists()]

    if not folders:
        print(f"[Batch] No folders with sections.json found under {root}")
        return

    print(f"[Batch] Found {len(folders)} papers to process")

    # Load model once and reuse
    extractor = SciBERTNERExtractor(model_name=model_name, device=device)

    results = []
    for i, folder in enumerate(folders):
        print(f"\n[Batch] Paper {i+1}/{len(folders)}: {folder.name}")
        try:
            sections, metadata = load_sections(folder)
            result = extract_entities_from_sections(sections, extractor, priority_only=priority_only)

            paper_id = _make_paper_id(metadata)
            enriched = {
                "paper_id":            paper_id,
                "metadata":            metadata,
                "entities":            result["entities_flat"],
                "entity_index":        result["entity_index"],
                "entities_by_section": result["entities_by_section"],
                "ner_summary": {
                    "total_entities":     result["total_entities"],
                    "entity_type_counts": result["entity_type_counts"],
                    "section_coverage":   result["section_coverage"],
                    "model_used":         extractor.model_name,
                },
            }

            out_path = folder / "enriched_entities.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)

            results.append({"folder": str(folder), "paper_id": paper_id, "status": "ok",
                            "entities": result["total_entities"]})
            print(f"[Batch] ✓ {paper_id} — {result['total_entities']} entities")

        except Exception as e:
            print(f"[Batch] ✗ {folder.name} — ERROR: {e}")
            results.append({"folder": str(folder), "status": "error", "error": str(e)})

    # Write batch summary
    summary_path = root / "ner_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Batch] Summary written to {summary_path}")