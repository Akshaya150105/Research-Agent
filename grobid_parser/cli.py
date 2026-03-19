import argparse
import json
import os
import sys
from pathlib import Path

from .client import GROBIDClient
from .tei_parser import parse_tei
from .utils import save_grobid_output


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "pdf_file",
        nargs="?",         
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for parsed results "
             "(metadata.json, sections.json, tei_raw.xml, summary.txt)",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="GROBID service URL (default: http://localhost:8070)",
    )
    parser.add_argument(
        "--tei-cache",
        help="Path to a cached tei_raw.xml file. "
             "If provided, skips the HTTP call to GROBID and parses this file instead. "
             "Useful for re-running the parser without re-calling GROBID.",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip CrossRef metadata enrichment (faster, but title/year/venue "
             "may be less accurate for some papers)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the GROBID service is running and exit",
    )

    args = parser.parse_args()


    if args.check:
        client = GROBIDClient(base_url=args.grobid_url)
        if client.is_alive():
            print(f"[OK] GROBID is running at {args.grobid_url}")
            sys.exit(0)
        else:
            print(f"[ERROR] GROBID is NOT running at {args.grobid_url}")
            print("Start it with: cd grobid && ./gradlew run")
            sys.exit(1)


    if not args.pdf_file:
        parser.error("pdf_file is required (or use --check to test the service)")

    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)

    print(f"Processing: {args.pdf_file}")

 
    tei_xml: str | None = None

    if args.tei_cache:
        # Load from cached file (skip GROBID HTTP call)
        if not os.path.exists(args.tei_cache):
            print(f"Error: TEI cache file not found: {args.tei_cache}")
            sys.exit(1)
        print(f"[CACHE] Loading TEI XML from: {args.tei_cache}")
        with open(args.tei_cache, "r", encoding="utf-8") as f:
            tei_xml = f.read()

    else:
        # Call GROBID
        client = GROBIDClient(base_url=args.grobid_url)

        if not client.is_alive():
            print(f"[ERROR] GROBID is not running at {args.grobid_url}")
            print("Start it with: docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0")
            print("Verify with:   curl http://localhost:8070/api/isalive")
            sys.exit(1)

        print(f"[GROBID] Sending to service at {args.grobid_url}...")
        tei_xml = client.process_fulltext(
            pdf_path=args.pdf_file,
            consolidate_header=not args.no_consolidate,
        )

        if not tei_xml:
            print("[ERROR] GROBID returned no output.")
            sys.exit(1)

        print("[GROBID] TEI XML received.")

    print("[PARSE] Parsing TEI XML...")
    result = parse_tei(tei_xml, pdf_path=args.pdf_file)

    if not result.success:
        print(f"[ERROR] Parsing failed: {result.error}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"PDF: {args.pdf_file}")
    print(f"{'='*50}")

    m = result.metadata
    print(f"\n[META] Title:   {m.title or '(not found)'}")
    print(f"[META] Authors: {', '.join(m.authors) or '(not found)'}")
    print(f"[META] Year:    {m.year or '(not found)'}")
    print(f"[META] Venue:   {m.venue or '(not found)'}")
    print(f"[META] DOI:     {m.doi or '(not found)'}")

    print(f"\n[SECTIONS] {len(result.sections)} sections extracted:")
    for i, s in enumerate(result.sections, 1):
        print(f"  {i:2}. [{s.section_type:20s}] \"{s.heading}\"  ({len(s.text)} chars)")


    if args.output:
        save_grobid_output(result, tei_xml, args.output)
        print(f"\n[SUCCESS] Output saved to: {args.output}")
        print(f"          metadata.json  — title, authors, year, venue, doi")
        print(f"          sections.json  — {len(result.sections)} sections with text")
        print(f"          tei_raw.xml    — raw GROBID output (cache for re-runs)")
        print(f"          summary.txt    — human-readable summary")


if __name__ == "__main__":
    main()