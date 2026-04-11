from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .tei_parser import TEIParseResult


def save_grobid_output(
    result: TEIParseResult,
    tei_xml: str,
    output_dir: str | Path,
) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. metadata.json
    metadata_path = out / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result.metadata), f, indent=2, ensure_ascii=False)

    # 2. sections.json
    sections_path = out / "sections.json"
    sections_data = [asdict(s) for s in result.sections]
    with open(sections_path, "w", encoding="utf-8") as f:
        json.dump(sections_data, f, indent=2, ensure_ascii=False)

    # 2b. figures.json
    figures_path = out / "figures.json"
    with open(figures_path, "w", encoding="utf-8") as f:
        json.dump([asdict(f) for f in result.figures], f, indent=2, ensure_ascii=False)

    # 2c. tables.json
    tables_path = out / "tables.json"
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in result.tables], f, indent=2, ensure_ascii=False)

    # 3. tei_raw.xml
    tei_path = out / "tei_raw.xml"
    with open(tei_path, "w", encoding="utf-8") as f:
        f.write(tei_xml)

    # 4. summary.txt 
    summary_path = out / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        m = result.metadata
        f.write("=" * 50 + "\n")
        f.write("GROBID Parse Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"[META] Title:   {m.title or '(not found)'}\n")
        f.write(f"[META] Authors: {', '.join(m.authors) or '(not found)'}\n")
        f.write(f"[META] Year:    {m.year or '(not found)'}\n")
        f.write(f"[META] Venue:   {m.venue or '(not found)'}\n")
        f.write(f"[META] DOI:     {m.doi or '(not found)'}\n\n")

        f.write(f"[SECTIONS] {len(result.sections)} sections extracted:\n\n")
        for i, s in enumerate(result.sections, 1):
            f.write(f"  {i:2}. [{s.section_type:20s}] {s.heading}  "
                    f"({len(s.text)} chars)\n")

        if m.abstract:
            f.write(f"\n[ABSTRACT]\n{m.abstract[:500]}"
                    f"{'...' if len(m.abstract) > 500 else ''}\n")