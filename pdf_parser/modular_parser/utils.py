import json
from pathlib import Path
from typing import Dict

def save_extracted_content(results: Dict, output_dir: str):
    """Save extracted content to files with proper formatting"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(Path(output_dir) / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(results['metadata'], f, indent=2, ensure_ascii=False)
    
    # Save text
    with open(Path(output_dir) / 'text.txt', 'w', encoding='utf-8') as f:
        f.write(results['text'])
        
    # Save structured json layout
    if results.get('structured_pages'):
        with open(Path(output_dir) / 'structured_content.json', 'w', encoding='utf-8') as f:
            json.dump(results['structured_pages'], f, indent=2, ensure_ascii=False)
            
    # Save page count info
    with open(Path(output_dir) / 'page_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"Total Pages: {results['page_count']}\n")
        f.write(f"Total Text Length: {len(results['text'])} characters\n")
    
    # Save links
    with open(Path(output_dir) / 'links.txt', 'w', encoding='utf-8') as f:
        for link in results['links']:
            f.write(f"Page {link['page']}: {link.get('uri', 'N/A')}\n")
    
    # Save tables
    with open(Path(output_dir) / 'tables.txt', 'w', encoding='utf-8') as f:
        for i, table in enumerate(results['tables']):
            f.write(f"Table {i + 1}:\n")
            if table:
                # Calculate column widths and clean cells
                num_cols = max((len(r) for r in table), default=0)
                col_widths = [0] * num_cols
                clean_table = []
                
                for row in table:
                    clean_row = []
                    for c_idx, cell in enumerate(row):
                        val = "" if cell is None else str(cell).replace('\n', ' ').strip()
                        clean_row.append(val)
                        if c_idx < len(col_widths) and len(val) > col_widths[c_idx]:
                            col_widths[c_idx] = len(val)
                    clean_table.append(clean_row)
                    
                for row in clean_table:
                    formatted_row = " | ".join(val.ljust(col_widths[idx]) for idx, val in enumerate(row))
                    f.write(f"  {formatted_row}\n")
            f.write("\n")
    
    # Save errors
    if results['errors']:
        with open(Path(output_dir) / 'errors.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(results['errors']))
