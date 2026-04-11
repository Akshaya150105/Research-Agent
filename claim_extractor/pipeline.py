import json
import os
import time
from pathlib import Path
from typing import Optional

from .claim_extractor import LLMClaimExtractor


def run_pipeline(
    grobid_output_dir: str,
    ner_results_dir: Optional[str] = None,
    ollama_host: Optional[str] = None,
    requests_per_minute: int = 12,
) -> Path:
  
    grobid_folder = Path(grobid_output_dir)
    ner_folder    = Path(ner_results_dir) if ner_results_dir else grobid_folder

    host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    sections_path = grobid_folder / "sections.json"
    entities_path = ner_folder    / "enriched_entities.json"
    claims_path   = ner_folder    / "claims_output.json"

    if not sections_path.exists():
        raise FileNotFoundError(f"sections.json not found in {grobid_folder}")
    if not entities_path.exists():
        raise FileNotFoundError(
            f"enriched_entities.json not found in {ner_folder}\n"
            f"Run Stage 1 (scibert_ner) first."
        )

    with open(sections_path, encoding="utf-8") as f:
        sections = json.load(f)

    # Append tables
    tables_path = grobid_folder / "tables.json"
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

    # Append figures
    figures_path = grobid_folder / "figures.json"
    if figures_path.exists():
        with open(figures_path, "r", encoding="utf-8") as f:
            for fig in json.load(f):
                sections.append({
                    "section_type": "Figure",
                    "heading": f"Figure {fig.get('label', '')}",
                    "text": fig.get("caption", "").strip()
                })

    with open(entities_path, encoding="utf-8") as f:
        existing = json.load(f)

    paper_title = existing.get("metadata", {}).get("title", "Unknown")
    print(f"\n{'='*60}")
    print(f"[Pipeline] Paper:    {paper_title[:70]}")
    print(f"[Pipeline] Sections: {len(sections)}")
    print(f"[Pipeline] Ollama host            : {host}")
    print(f"[Pipeline] sections.json           : {sections_path}")
    print(f"[Pipeline] enriched_entities       : {entities_path}")
    print(f"[Pipeline] claims_output           : {claims_path}")
    print(f"{'='*60}\n")

    ner_by_section = existing.get("entities_by_section", {})
    extractor = LLMClaimExtractor(ollama_host=host, requests_per_minute=requests_per_minute)

    all_llm_entities   = []
    all_claims         = []
    all_limitations    = []
    all_future_work    = []
    results_by_section = {}

    t0 = time.time()
    for section in sections:
        sec_type  = section.get("section_type", "Unknown")
        ner_hints = ner_by_section.get(sec_type, [])
        result    = extractor.extract_from_section(section, ner_hints)

        results_by_section[sec_type] = result
        all_llm_entities.extend(result["entities"])
        all_claims.extend(result["claims"])
        all_limitations.extend(result["limitations"])
        all_future_work.extend(result["future_work"])

    elapsed = time.time() - t0

    deduped_entities = _deduplicate_entities(all_llm_entities)
    llm_entity_index = _build_index(deduped_entities)

    summary = {
        "total_llm_entities": len(deduped_entities),
        "llm_entity_type_counts": {
            etype: len(idx) for etype, idx in llm_entity_index.items()
        },
        "total_claims": len(all_claims),
        "claim_type_counts": {
            ct: sum(1 for c in all_claims if c["claim_type"] == ct)
            for ct in ("performance", "comparative", "methodological")
        },
        "total_limitations":  len(all_limitations),
        "total_future_work":  len(all_future_work),
        "elapsed_seconds":    round(elapsed, 2),
        "model_used":         LLMClaimExtractor.MODEL_NAME,
        "sections_processed": len(sections),
    }

  
    existing["llm_entities"]           = deduped_entities
    existing["llm_entity_index"]       = llm_entity_index
    existing["claims"]                 = all_claims
    existing["limitations"]            = all_limitations
    existing["future_work"]            = all_future_work
    existing["llm_results_by_section"] = results_by_section
    existing["llm_extraction_summary"] = summary

    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[Pipeline] enriched_entities.json updated: {entities_path}")


    claims_output = {
        "paper_id":     existing.get("paper_id", ""),
        "metadata":     existing.get("metadata", {}),
        "summary":      summary,
        "entities":     deduped_entities,
        "entity_index": llm_entity_index,
        "claims":       all_claims,
        "limitations":  all_limitations,
        "future_work":  all_future_work,
    }

    with open(claims_path, "w", encoding="utf-8") as f:
        json.dump(claims_output, f, indent=2, ensure_ascii=False)
    print(f"[Pipeline] claims_output.json saved      : {claims_path}")

    _print_summary(claims_output)
    return claims_path


def _build_index(entities):
    index = {"method": {}, "dataset": {}, "metric": {}, "task": {}}
    for ent in entities:
        etype   = ent["entity_type"]
        key_str = ent["text"].lower()
        if etype in index:
            index[etype].setdefault(key_str, []).append(ent)
    return index


def _deduplicate_entities(entities):
    seen = {}
    for ent in entities:
        key = (ent["entity_type"], ent["text"].lower())
        if key not in seen:
            seen[key] = ent
        else:
            ex  = seen[key]
            sec = ent["section_type"]
            if sec != ex["section_type"]:
                ex.setdefault("also_in_sections", [])
                if sec not in ex["also_in_sections"]:
                    ex["also_in_sections"].append(sec)
            if ent["confidence"] > ex["confidence"]:
                ex["confidence"] = ent["confidence"]
    return list(seen.values())


def _print_summary(data: dict):
    s    = data.get("summary", {})
    meta = data.get("metadata", {})

    print(f"\n{'='*60}")
    print(f"PHASE 1 STAGE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Paper : {meta.get('title','Unknown')[:70]}")
    print(f"Model : {s.get('model_used','?')}")
    print()
    print(f"LLM Entities ({s.get('total_llm_entities',0)} unique):")
    for etype, count in s.get("llm_entity_type_counts", {}).items():
        print(f"  {etype:10} {count}")
    print()
    print(f"Claims ({s.get('total_claims',0)} total):")
    for ctype, count in s.get("claim_type_counts", {}).items():
        print(f"  {ctype:15} {count}")
    print()
    print(f"Limitations : {s.get('total_limitations',0)}")
    print(f"Future work : {s.get('total_future_work',0)}")
    print(f"Time        : {s.get('elapsed_seconds',0)}s")
    print(f"{'='*60}")

    for c in data.get("claims", [])[:5]:
        print(f"\n  [{c['claim_type']:15}] {c['description'][:85]}")
        print(f"    entities : {c['entities_involved']}")
        print(f"    value    : {c['value']}   conf={c['confidence']}")

    for lim in data.get("limitations", []):
        print(f"\n  [LIMITATION] {lim['text'][:100]}")

    for fw in data.get("future_work", []):
        print(f"\n  [FUTURE WORK] {fw['text'][:100]}")