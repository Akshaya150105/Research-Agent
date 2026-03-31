import argparse
import os
import sys
from pathlib import Path
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Stage 2: LLM Claim Extraction using Ollama (phi).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single paper, default local Ollama
  python -m yourpkg.cli -g ./paper/grobid_output

  # Custom remote Ollama host (e.g. via Kaggle SSH tunnel)
  python -m yourpkg.cli -g ./paper/grobid_output --ollama-host https://612be70f6a31cf.lhr.life

  # Batch mode
  python -m yourpkg.cli --batch -g ./papers_root --ollama-host http://localhost:11434
        """
    )

    parser.add_argument(
        "--grobid-dir", "-g",
        required=True,
        help="Folder containing sections.json (grobid_parser output)."
    )
    parser.add_argument(
        "--ner-dir", "-n",
        default=None,
        help="Folder containing enriched_entities.json (scibert_ner output). "
             "Defaults to same as --grobid-dir if not specified."
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama base URL (e.g. http://localhost:11434 or a remote tunnel URL). "
             "If not set, reads OLLAMA_HOST env var, then falls back to http://localhost:11434."
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=12,
        help="Requests per minute throttle (default: 12)."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: under --grobid-dir, find all subfolders with grobid_output/ "
             "and match them to corresponding ner_results/ under --ner-dir."
    )

    args = parser.parse_args()

    # Resolve Ollama host: CLI arg > env var > default
    ollama_host = (
        args.ollama_host
        or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    print(f"[CLI] Using Ollama host: {ollama_host}")

    if args.batch:
        root = Path(args.grobid_dir)
        grobid_folders = sorted(root.rglob("grobid_output/sections.json"))

        if not grobid_folders:
            print(f"No grobid_output/sections.json found under {root}")
            sys.exit(1)

        print(f"[Batch] Found {len(grobid_folders)} papers")
        for i, sections_path in enumerate(grobid_folders):
            grobid_dir = sections_path.parent
            paper_root = grobid_dir.parent
            ner_dir    = paper_root / "ner_results"

            if not ner_dir.exists():
                print(f"[Batch] SKIP {paper_root.name} — no ner_results folder found")
                continue
            if not (ner_dir / "enriched_entities.json").exists():
                print(f"[Batch] SKIP {paper_root.name} — enriched_entities.json missing")
                continue

            print(f"\n[Batch] {i+1}/{len(grobid_folders)}: {paper_root.name}")
            try:
                run_pipeline(
                    str(grobid_dir),
                    ner_results_dir=str(ner_dir),
                    ollama_host=ollama_host,
                    requests_per_minute=args.rpm,
                )
            except Exception as e:
                print(f"[Batch] ERROR on {paper_root.name}: {e}")
    else:
        run_pipeline(
            args.grobid_dir,
            ner_results_dir=args.ner_dir,
            ollama_host=ollama_host,
            requests_per_minute=args.rpm,
        )


if __name__ == "__main__":
    main()