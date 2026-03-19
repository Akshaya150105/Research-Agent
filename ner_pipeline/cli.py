import argparse
import sys
from .pipeline import run_pipeline, run_batch


def main():
    parser = argparse.ArgumentParser(
        description="SciBERT NER: Extract method/dataset/metric/task entities from parsed papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        help="Path to grobid_parser output folder (must contain sections.json). "
             "In --batch mode: root folder containing multiple such subfolders."
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="HuggingFace model ID or local path. "
             "Default: Jason9693/SciERC-scibert-cased"
    )
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=-1,
        metavar="DEVICE_ID",
        help="GPU device ID (0, 1, ...). Default: -1 (CPU)."
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only process high-priority sections (Abstract, Methods, Results, etc.). "
             "Faster but may miss entities in minor sections."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for enriched_entities.json. "
             "Defaults to same folder as input_dir."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all subfolders under input_dir that contain sections.json."
    )

    args = parser.parse_args()

    if args.batch:
        run_batch(
            root_dir=args.input_dir,
            model_name=args.model,
            device=args.gpu,
            priority_only=args.priority_only,
        )
    else:
        output_path = run_pipeline(
            grobid_output_dir=args.input_dir,
            model_name=args.model,
            device=args.gpu,
            priority_only=args.priority_only,
            output_dir=args.output,
        )
        print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()