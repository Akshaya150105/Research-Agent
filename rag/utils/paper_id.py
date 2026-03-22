"""
paper_id.py
-----------
Single source of truth for paper ID generation.
Your friend's knowledge graph uses this same function so IDs
are guaranteed to match across ChromaDB and SQLite.
 
Convention: {first_author_lastname}_{first_meaningful_title_word}_{year}
Example:    stgcn_yu_2018   (but derived from folder name, not computed)
 
In practice, the paper_id IS the output_folder name.
The researcher names the folder when they run Phase 1.
This module just validates and normalises that name.
"""
 
import re
 
 
def validate_paper_id(paper_id: str) -> str:
    """
    Validates and normalises a paper_id string.
    Rules:
      - lowercase only
      - only letters, digits, underscores
      - no spaces, hyphens, or special chars
      - minimum 3 characters
 
    Raises ValueError if the id cannot be normalised.
    Returns the cleaned paper_id.
    """
    cleaned = paper_id.strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)       # hyphens/spaces → underscore
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)    # remove everything else
    cleaned = re.sub(r"_+", "_", cleaned)            # collapse multiple underscores
    cleaned = cleaned.strip("_")                     # strip leading/trailing underscores
 
    if len(cleaned) < 3:
        raise ValueError(
            f"paper_id '{paper_id}' normalises to '{cleaned}' which is too short. "
            f"Use a descriptive folder name like 'stgcn_yu_2018'."
        )
 
    return cleaned
 
 
def paper_id_from_folder(folder_path: str) -> str:
    """
    Derives paper_id from the output_folder path.
    The folder name IS the paper_id by convention.
 
    Example:
        paper_id_from_folder("memory/stgcn_yu_2018") → "stgcn_yu_2018"
        paper_id_from_folder("memory/stgcn_yu_2018/") → "stgcn_yu_2018"
    """
    from pathlib import Path
    folder_name = Path(folder_path).name
    return validate_paper_id(folder_name)