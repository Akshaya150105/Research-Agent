import argparse
import os
import sys
import json
from pathlib import Path

from modular_parser.extractor import extract_all
from modular_parser.utils import save_extracted_content

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Universal PDF Extractor')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output directory for extracted content')
    parser.add_argument('-t', '--text-only', action='store_true', help='Extract only text')
    parser.add_argument('-m', '--metadata-only', action='store_true', help='Extract only metadata')
    parser.add_argument('-i', '--images', action='store_true', help='Extract images')
    parser.add_argument('-l', '--links', action='store_true', help='Extract links')
    parser.add_argument('-f', '--figure-bboxes', help='JSON file containing mapping of page_num (int) to list of bounding boxes [x0, y0, x1, y1] for synthetic figures')
    parser.add_argument('-a', '--auto-detect-figures', action='store_true', help='Automatically detect synthetic figures based on vector paths')
    parser.add_argument('--no-structured', action='store_true', help='Disable structured layout reading order JSON extraction')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)
        
    figure_bboxes = None
    if args.figure_bboxes:
        if os.path.exists(args.figure_bboxes):
            with open(args.figure_bboxes, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert string keys to int page numbers
                figure_bboxes = {int(k): v for k, v in data.items()}
        else:
            print(f"Warning: Figure bboxes file not found: {args.figure_bboxes}")
    
    print(f"Processing: {args.pdf_file}")
    
    # Extract content
    extract_structured = not args.no_structured
    results = extract_all(args.pdf_file, args.output, figure_bboxes, args.auto_detect_figures, extract_structured)
    
    # Display results
    if args.verbose or not args.text_only and not args.metadata_only:
        print(f"\n{'='*50}")
        print(f"PDF: {args.pdf_file}")
        print(f"{'='*50}")
        
        if not args.metadata_only:
            print(f"\n[PAGE] Page Count: {results['page_count']}")
            print(f"[TEXT] Text Length: {len(results['text'])} characters")
        
        if not args.text_only:
            print(f"\n[META] Metadata:")
            for key, value in results['metadata'].items():
                print(f"   {key}: {value}")
            
            if results['images']:
                print(f"\n[IMAGE] Images: {len(results['images'])} found")
                
            if results.get('synthetic_figures'):
                print(f"\n[FIGURE] Synthetic Figures: {len(results['synthetic_figures'])} rendered")
            
            if results['links']:
                print(f"\n[LINK] Links: {len(results['links'])} found")
            
            if results['tables']:
                print(f"\n[TABLE] Tables: {len(results['tables'])} found")
                
            if results.get('structured_pages'):
                print(f"\n[JSON] Structured read-order JSON successfully compiled")
        
        if results['errors']:
            print(f"\n[ERROR] Errors:")
            for error in results['errors']:
                print(f"   {error}")
    
    # Save to files if output directory specified
    if args.output:
        save_extracted_content(results, args.output)
        print(f"\n[SUCCESS] Content saved to: {args.output}")
    
    # Save text only if requested
    if args.text_only:
        output_file = Path(args.output) / 'extracted_text.txt' if args.output else 'extracted_text.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results['text'])
        print(f"\n[SUCCESS] Text saved to: {output_file}")
    
    # Save metadata only if requested
    if args.metadata_only:
        output_file = Path(args.output) / 'metadata.json' if args.output else 'metadata.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results['metadata'], f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Metadata saved to: {output_file}")

if __name__ == '__main__':
    main()
