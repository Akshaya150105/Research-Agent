import argparse
import os
import sys
from pathlib import Path
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Stage 2: LLM Claim Extraction using Gemini 1.5 Flash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--api-key", "-k",
        default=None,
        help="Gemini API key. If not set, reads GEMINI_API_KEY env var."
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=12,
        help="Requests per minute (default: 12, free tier limit is 15)."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: under --grobid-dir, find all subfolders with grobid_output/ "
             "and match them to corresponding ner_results/ under --ner-dir."
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: No Gemini API key found.")
        print("Either pass --api-key or set the GEMINI_API_KEY environment variable.")
        print("Get a free key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    if args.batch:
        # Batch: walk root, find paper folders containing grobid_output/sections.json
        root = Path(args.grobid_dir)
        grobid_folders = sorted(root.rglob("grobid_output/sections.json"))

        if not grobid_folders:
            print(f"No grobid_output/sections.json found under {root}")
            sys.exit(1)

        print(f"[Batch] Found {len(grobid_folders)} papers")
        for i, sections_path in enumerate(grobid_folders):
            grobid_dir = sections_path.parent

            # Find corresponding ner_results folder (sibling of grobid_output)
            paper_root = grobid_dir.parent
            ner_dir = paper_root / "ner_results"

            if not ner_dir.exists():
                print(f"[Batch] SKIP {paper_root.name} — no ner_results folder found")
                continue
            if not (ner_dir / "enriched_entities.json").exists():
                print(f"[Batch] SKIP {paper_root.name} — enriched_entities.json not in ner_results/")
                continue

            print(f"\n[Batch] {i+1}/{len(grobid_folders)}: {paper_root.name}")
            try:
                run_pipeline(
                    str(grobid_dir),
                    ner_results_dir=str(ner_dir),
                    api_key=api_key,
                    requests_per_minute=args.rpm,
                )
            except Exception as e:
                print(f"[Batch] ERROR on {paper_root.name}: {e}")
    else:
        run_pipeline(
            args.grobid_dir,
            ner_results_dir=args.ner_dir,
            api_key=api_key,
            requests_per_minute=args.rpm,
        )


if __name__ == "__main__":
    main()