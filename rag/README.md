# Agentic AI Research Assistant — RAG Pipeline (Phase 2)

A local vector database pipeline that turns structured paper JSONs from Phase 1 into a searchable, cross-paper research memory. Built for 5–6 research papers, designed to power six downstream AI agents.

## 1. Folder structure and file map

```

│
├── pipeline.py              ← THE ENTRY POINT. Start here.
│                               index_paper(), enrich(), query(), status()
│                               agents and tests only call this file
│
├── chunker.py               ← Step 1. JSON files → chunk dicts
│                               reads 3 source files, produces typed list
│                               nothing embedded or stored here
│
├── embedder.py              ← Step 2. chunk dicts → vectors
│                               adds "embedding": [768 floats] to each chunk
│                               also embeds queries at retrieval time
│
├── indexer.py               ← Step 3. chunks → ChromaDB
│                               routes each chunk type to the right collection
│                               idempotent upsert using SHA-256 chunk IDs
│
├── enricher.py              ← Step 4. cross-paper linking (runs once after all papers)
│                               Pass 1: entity linking across papers
│                               Pass 2: contradiction candidate detection
│                               Pass 3: gap matrix computation
│
├── query_handler.py         ← Step 5. question → query plan
│                               pure logic, no database access
│                               detects intent, builds ChromaDB filter
│
├── retriever.py             ← Step 6. query plan → ranked results
│                               Stage 1: ChromaDB ANN search
│                               Stage 2: cross-encoder re-ranking
│
├── chroma_store/            ← ChromaDB data on disk (auto-created)
│   ├── chroma.sqlite3       ← ChromaDB's internal metadata and index
│   └── [uuid folders]       ← HNSW index files per collection
│                               ADD THIS TO .gitignore
│
└── utils/
    ├── __init__.py          ← empty, makes utils/ a Python sub-package
    ├── paper_id.py          ← shared paper ID convention
    │                           used by chunker, pipeline, and friend's KG
    └── text_builder.py      ← builds embed_text strings per chunk type
                                isolated so embed text can be tuned independently
```

---

## 2. How the complete pipeline works — data flow start to finish

### Indexing flow (adding a paper)

```
memory/stgcn_yu_2018/
├── claims_output.json   178 KB
├── sections.json         27 KB
└── figures.json           2 KB
         │
         ▼
  pipeline.index_paper("memory/stgcn_yu_2018")
         │
         ├── Step 1: chunker.chunk_paper()
         │       reads claims_output.json → 141 claims, 18 limitations,
         │                                   4 future_work, 202 entities
         │       reads sections.json      → 8 section chunks
         │       reads figures.json       → 6 figure chunks
         │       deduplicates entities by (text.lower(), type)
         │       builds embed_text for each chunk via text_builder.py
         │       assigns SHA-256 chunk IDs
         │       returns: list of 379 chunk dicts (no embeddings yet)
         │
         ├── Step 2: embedder.embed_chunks()
         │       loads BAAI/bge-base-en-v1.5
         │       encodes all 379 embed_text strings in batches of 32
         │       normalize_embeddings=True → all vectors have magnitude 1.0
         │       adds "embedding": [768 floats] to every chunk dict
         │       returns: same 379 chunks, now with embedding field
         │
         └── Step 3: indexer.index_chunks()
                 routes chunks by chunk_type to correct collection:
                   section + figure     → paper_sections       (14 docs)
                   claim + limitation
                   + future_work        → claims_and_findings   (163 docs)
                   entity               → entities_global       (202 docs)
                 calls collection.upsert() per collection
                 stores: id, embedding vector, display_text, metadata
                 does NOT store: embed_text (discarded after encoding)

  pipeline.enrich()  ← call once after ALL papers indexed
         │
         ├── Pass 1: enricher._pass1_entity_linking()
         │       reads all entity chunks from entities_global
         │       groups by (normalized_text, entity_type)
         │       finds groups with 2+ different paper_ids
         │       updates also_in_papers and appears_in_n_papers
         │       on every affected entity chunk
         │
         ├── Pass 2: enricher._pass2_contradiction_candidates()
         │       queries claims_and_findings for comparative+numeric claims
         │       finds pairs from different papers with shared method mentions
         │       writes memory/contradiction_candidates.json
         │
         └── Pass 3: enricher._pass3_gap_matrix()
                 queries entities_global for all methods and datasets
                 builds method × dataset co-occurrence matrix
                 finds empty cells = research gaps
                 scores and ranks gaps
                 writes memory/gap_matrix.json
```

### Query flow (answering a question)

```
pipeline.query("How does STGCN compare to GCGRU?")
         │
         ├── Step 5: query_handler.parse_query()
         │       lowercases question
         │       iterates through 8 INTENT_RULES in order
         │       matches "compare" → intent="comparison"
         │       checks for paper_id pattern in question
         │       builds where filter: {claim_type: {$eq: "comparative"}}
         │       returns query plan dict
         │
         ├── Step 6a: retriever._stage1_ann_search()
         │       calls embedder.embed_query() with BGE query prefix
         │       queries ChromaDB claims_and_findings collection
         │       applies where filter BEFORE vector search
         │       fetches top 12 candidates by cosine distance
         │       converts distance to similarity: ann_score = 1 - distance
         │
         └── Step 6b: retriever._stage2_rerank()
                 loads cross-encoder/ms-marco-MiniLM-L-6-v2 (once, cached)
                 scores all 12 (question, document) pairs together
                 applies sigmoid() to normalize raw scores to 0-1
                 multiplies by chunk confidence
                 if confidence == 0.0, uses 1.0 instead
                 sorts by final_score descending
                 returns top 5 result dicts with full provenance
```

---

## 3. The universal chunk schema — every field explained

Every chunk produced by `chunker.py` is a Python dict. Here is every single
field with a detailed explanation of what it is, where it comes from, and
how it is used downstream.

```python
{
    # ── IDENTITY FIELDS ───────────────────────────────────────────────────

    "chunk_id": "86ac1a06d297",
    # 12-character hex string.
    # Generated by: sha256(f"{paper_id}::{content}").hexdigest()[:12]
    # For claims/limitations/future_work: content = display_text
    # For entities: content = f"entity::{entity_type}::{text.lower()}"
    # For sections: content = f"section::{section_type}::{heading}"
    # For figures:  content = f"figure::{figure_id}::{label}"
    # WHY: Same input always → same output. This makes indexer.upsert()
    # idempotent. Running indexer twice = no duplicates, just updates.
    # Used as: ChromaDB document ID.

    "paper_id": "stgcn_yu_2018",
    # The folder name of the paper's output_folder.
    # Derived by: paper_id_from_folder(folder_path)
    # CRITICAL: Must be identical in ChromaDB and in friend's KG SQLite.
    # Used for: per-paper filtering, cross-paper linking, delete_paper().
    # ChromaDB filter: {"paper_id": {"$eq": "stgcn_yu_2018"}}

    "paper_title": "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting",
    # Full title from claims_output.json["metadata"]["title"].
    # Falls back to paper_id if title not found.
    # Used in: embed_text construction by text_builder.py.
    # Stored in: ChromaDB metadata.

    "chunk_type": "claim",
    # One of: "claim" | "limitation" | "future_work" | "entity" | "section" | "figure"
    # Determines which ChromaDB collection this chunk goes to.
    # Also used as a metadata filter: {"chunk_type": {"$eq": "limitation"}}
    # ROUTING:
    #   claim + limitation + future_work → claims_and_findings
    #   entity                           → entities_global
    #   section + figure                 → paper_sections

    # ── TEXT FIELDS ───────────────────────────────────────────────────────

    "embed_text": "comparative claim in Results | Training Efficiency:\nSTGCN achieves 14x...\nentities: STGCN, GCGRU",
    # The text string fed to the embedding model (BAAI/bge-base-en-v1.5).
    # Built by text_builder.py — one builder function per chunk type.
    # Contains: type prefix + location context + content + entity hints.
    # WHY TYPE PREFIX: A claim about STGCN training and a limitation about
    # STGCN training use similar words but should not cluster together in
    # vector space. The prefix "comparative claim" vs "limitation" pushes
    # their vectors apart, improving retrieval precision.
    # NOT stored in ChromaDB — discarded after the embedding is generated.
    # Only used during Step 2 (embedding).

    "display_text": "STGCN achieves 14 times acceleration in training speed compared to GCGRU.",
    # The raw content text shown to users and agents.
    # For claims: claim["description"]
    # For limitations/future_work: item["text"]
    # For entities: the entity name string
    # For sections: full raw section text (no truncation)
    # For figures: the figure caption text
    # Stored as: ChromaDB "document" field (not in metadata).
    # Returned in: all retrieval results as result["document"].

    # ── CLAIM-SPECIFIC FIELDS ─────────────────────────────────────────────

    "claim_type": "comparative",
    # Sub-type of claim. One of: "performance" | "comparative" | "methodological"
    # For limitations: always "limitation"
    # For future_work: always "future_work"
    # For entities/sections/figures: always ""
    # WHY: Agents filter by claim_type to get exactly the kind of claims needed.
    # Comparison agent: {"claim_type": {"$eq": "comparative"}}
    # Performance agent: {"claim_type": {"$eq": "performance"}}

    "section_type": "Results",
    # Which section of the paper this chunk came from.
    # Examples: "Abstract", "Introduction", "Methods", "Results", "Conclusion"
    # For figures: hardcoded to "Figure"
    # For entities: the section where this entity first appeared
    # Used for context display and for building embed_text location prefix.

    "section_heading": "Training Efficiency and Generalization",
    # The specific subsection heading (more granular than section_type).
    # May equal section_type if no subsection heading exists.
    # For entities: always ""
    # Used in: embed_text to give more precise location context.

    # ── QUALITY FIELDS ────────────────────────────────────────────────────

    "confidence": 0.95,
    # LLM extraction confidence score. Float 0.0 to 1.0.
    # Set by Phase 1 LLM extractor.
    # For sections and figures: hardcoded to 1.0 (not extracted by LLM).
    # IMPORTANT: Some future_work items have confidence=0.0 because
    # the Phase 1 LLM didn't fill this field. The retriever detects
    # confidence==0.0 and treats it as 1.0 to avoid zeroing out valid results.
    # Used in: retriever final_score = sigmoid(ce_score) × confidence

    "source": "llm",
    # Where this chunk was extracted from. "llm" or "scibert".
    # Stored for debugging and provenance tracking.

    # ── NUMERIC VALUE FIELDS ──────────────────────────────────────────────

    "has_numeric_value": True,
    # Boolean. True if this claim contains a measurable number.
    # Extracted from claim["value"] field in claims_output.json.
    # Examples: 14.0 (14x acceleration), 272.34 (seconds), 95.0 (% savings)
    # For limitations, future_work, entities, sections, figures: always False.
    # Used in: enricher Pass 2 filters for contradiction candidates:
    #   {"$and": [{"claim_type": {"$eq": "comparative"}},
    #             {"has_numeric_value": {"$eq": True}}]}

    "numeric_value": 14.0,
    # The extracted float value. 0.0 if has_numeric_value is False.
    # WHY: Two papers claiming different numbers about the same method
    # (e.g., paper A says STGCN takes 272s, paper B says 310s) is a
    # contradiction signal. enricher.py uses this field to find such pairs.

    # ── ENTITY MENTION FIELDS ─────────────────────────────────────────────

    "entities_mentioned": "STGCN,GCGRU,training_speed",
    # Comma-joined list of all entities mentioned in this chunk.
    # Read from claim["entities_involved"] list in claims_output.json.
    # For sections and figures: always ""
    # Used in: enricher Pass 2 to find entity overlap between claim pairs.

    "methods_mentioned": "",
    # Comma-joined list of method-type entities mentioned.
    # For claim/limitation/future_work: left as "" by chunker.
    #   enricher.py fills this properly after indexing (planned enhancement).
    # For entity chunks where entity_type == "method":
    #   set to the entity_text value itself.
    # Used in: contradiction detection filtering.

    "datasets_mentioned": "",
    # Same as methods_mentioned but for dataset-type entities.
    # For entity chunks where entity_type == "dataset":
    #   set to the entity_text value.

    "metrics_mentioned": "",
    # Same pattern for metric-type entities.
    # For entity chunks where entity_type == "metric":
    #   set to the entity_text value.

    # ── ENTITY-SPECIFIC FIELDS (only meaningful for entity chunks) ────────

    "entity_type": "method",
    # One of: "method" | "dataset" | "metric" | "task"
    # For all non-entity chunk types: always ""
    # Used in: gap matrix computation (filters by entity_type)
    # ChromaDB filter: {"entity_type": {"$eq": "method"}}

    "entity_text": "STGCN",
    # The canonical entity name as extracted by the LLM.
    # For all non-entity chunk types: always ""
    # Used in: enricher Pass 1 to group entities across papers.

    "entity_text_normalized": "stgcn",
    # Lowercase version of entity_text. Used for cross-paper matching.
    # The enricher normalizes before comparing to handle case differences.

    # ── CROSS-PAPER FIELDS (start empty, enricher fills them) ─────────────

    "also_in_papers": "",
    # Comma-joined list of OTHER paper_ids where this entity also appears.
    # Starts as "" for all chunks.
    # Enricher Pass 1 updates this for entity chunks that appear in 2+ papers.
    # Example after enrichment: "dcrnn_li_2018,gwnet_wu_2019"
    # Powers: "which papers use method X?" queries
    # ChromaDB filter: {"also_in_papers": {"$ne": ""}}

    "appears_in_n_papers": 1,
    # Integer count of papers this entity appears in.
    # Starts as 1 for all chunks.
    # Enricher Pass 1 updates this: 2, 3, etc. for shared entities.
    # Used in: gap scoring (more papers using a method = more significant gap)

    # ── MODEL PROVENANCE ──────────────────────────────────────────────────

    "embed_model": "BAAI/bge-base-en-v1.5",
    # Name of the embedding model used.
    # Stored with every chunk so if you upgrade models later,
    # you can identify which chunks need re-embedding.
    # ChromaDB filter: {"embed_model": {"$eq": "BAAI/bge-base-en-v1.5"}}

    "embed_model_version": "1.0",
    # Version string for the embedding configuration.
    # Change this when you change BATCH_SIZE, QUERY_PREFIX, or normalization.

    # ── ADDED BY EMBEDDER (not in ChromaDB metadata) ──────────────────────

    "embedding": [0.12, -0.34, 0.07, 0.55, ...],
    # List of 768 floats. Added by embedder.embed_chunks().
    # Present in the chunk dict ONLY between Steps 2 and 3.
    # Stored in ChromaDB as a vector, NOT in the metadata dict.
    # All vectors have magnitude exactly 1.0 (normalized).
    # Discarded from the Python dict after indexer.index_chunks() runs.
}
```

---

## 4. What ChromaDB actually stores

ChromaDB stores exactly 4 things per document. NOT the full chunk dict.

```
ChromaDB document storage:
┌─────────────────┬────────────────────────────────────────────────────────────┐
│ ids             │ ["86ac1a06d297"]                                            │
│                 │ The chunk_id. Used for upsert idempotency and delete.       │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ embeddings      │ [[0.12, -0.34, 0.07, ...]]   768 floats                    │
│                 │ Stored in HNSW index for approximate nearest neighbor search│
│                 │ Not in metadata. Retrieved separately.                      │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ documents       │ ["STGCN achieves 14 times acceleration..."]                 │
│                 │ The display_text. Returned in query results.                │
│                 │ This is what users and agents read.                         │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ metadatas       │ { paper_id, paper_title, chunk_type, claim_type,           │
│                 │   section_type, section_heading, entity_type, entity_text,  │
│                 │   entity_text_normalized, confidence, source,               │
│                 │   has_numeric_value, numeric_value, entities_mentioned,     │
│                 │   methods_mentioned, datasets_mentioned, metrics_mentioned, │
│                 │   also_in_papers, appears_in_n_papers,                      │
│                 │   embed_model, embed_model_version }                        │
│                 │ These are all filterable via where= clauses.                │
│                 │ Values must be str, int, float, or bool. NO None values.    │
└─────────────────┴────────────────────────────────────────────────────────────┘
```

**What is NOT stored in ChromaDB:**
- `embed_text` — discarded after embedding, not needed again
- `embedding` — stored as a vector in the HNSW index, not in metadata
- `figure_label`, `figure_id`, `text_length` — supplementary fields
  only present on certain chunk types, not in METADATA_FIELDS list

**HNSW index:** ChromaDB uses Hierarchical Navigable Small World graphs
for approximate nearest neighbor search. Setting `hnsw:space=cosine`
means all distance calculations use cosine distance (1 - cosine_similarity).
All our vectors are normalized to magnitude 1.0, which makes cosine
distance equivalent to Euclidean distance and faster to compute.

---

---

## 5. `utils/paper_id.py` — complete breakdown

**Purpose:** Single source of truth for paper ID generation. Both the RAG
pipeline and the collaborator's knowledge graph call this same file so IDs
always match between ChromaDB and SQLite without any translation layer.

**The fundamental rule:** The folder name = the paper ID. The researcher
names the folder when running Phase 1. This module validates and normalises
that name into a consistent format.

**Imports:** Only `re` (Python standard library). No external dependencies.

### Constants and module-level state
None. Pure functions only.

### `validate_paper_id(paper_id: str) → str`

Takes any string and normalises it into a valid paper ID.

**Normalisation steps (applied in order):**
1. `.strip().lower()` — remove whitespace, convert to lowercase
2. `re.sub(r"[-\s]+", "_", cleaned)` — replace hyphens and spaces with underscores
3. `re.sub(r"[^a-z0-9_]", "", cleaned)` — remove all characters that are not
   letters, digits, or underscores
4. `re.sub(r"_+", "_", cleaned)` — collapse consecutive underscores into one
5. `.strip("_")` — remove leading or trailing underscores

**Validation:** If result is shorter than 3 characters, raises `ValueError`
with a message explaining the problem and suggesting a better name.

**Examples:**
```python
validate_paper_id("stgcn_yu_2018")     → "stgcn_yu_2018"   (unchanged)
validate_paper_id("STGCN Yu 2018")     → "stgcn_yu_2018"   (spaces → underscore, lowercase)
validate_paper_id("stgcn-yu-2018")     → "stgcn_yu_2018"   (hyphens → underscore)
validate_paper_id("STGCN (Yu, 2018)")  → "stgcn_yu_2018"   (parens and comma removed)
validate_paper_id("ab")               → ValueError          (too short)
```

**Used by:** `chunker.chunk_paper()`, `pipeline.index_paper()`,
collaborator's KG node insertion.

### `paper_id_from_folder(folder_path: str) → str`

Extracts the paper ID from a folder path. Uses `Path(folder_path).name`
to get just the folder name (last component), then passes it through
`validate_paper_id()`.

**Examples:**
```python
paper_id_from_folder("memory/stgcn_yu_2018")    → "stgcn_yu_2018"
paper_id_from_folder("memory/stgcn_yu_2018/")   → "stgcn_yu_2018"
paper_id_from_folder("C:\\Research\\stgcn_yu_2018") → "stgcn_yu_2018"
```

**Delayed import:** `from pathlib import Path` is inside the function body,
not at the top of the file. This avoids importing pathlib at module load time
when the function may not be called.

---

## 6. `utils/text_builder.py` — complete breakdown

**Purpose:** Builds the `embed_text` string for each chunk type. Completely
separated from `chunker.py` so that embed text can be tuned without touching
the chunking logic.

**Why this separation matters:** The embed_text is what gets encoded into a
768-dimensional vector. Small changes to wording, prefixes, or structure here
can significantly change which documents come back for a given query. Having
this isolated means you can experiment and test embed text quality independently.

**Design principle:**
```
embed_text = type_context + location_context + content + entity_hints
display_text = just the content
```

**Why type prefixes work:** The embedding model is trained to cluster similar
texts close together in vector space. By prefixing claims with
`"comparative claim in Results"` and limitations with `"limitation in Introduction"`,
we push their vectors apart even when the content words overlap. This means
a query for "limitations" returns limitation chunks, not comparison chunks,
even if both mention the same method names.

**Imports:** None. No imports at all. Pure Python string manipulation.

### `build_claim_text(claim: dict, paper_title: str) → str`

Used for: claim, limitation, and future_work chunks.
The function distinguishes between them using `claim["_chunk_type"]`
which is injected by chunker.py before calling this function.

**Fields read from claim dict:**
- `claim_type` — "comparative", "performance", "methodological", or ""
- `_chunk_type` — "claim", "limitation", or "future_work" (injected by chunker)
- `section_type` — e.g. "Results", "Introduction"
- `section_heading` — e.g. "Training Efficiency and Generalization"
- `description` — for claim chunks (or falls back to `text`)
- `entities_involved` — list of entity names mentioned

**Type label construction:**
```python
if chunk_type == "limitation":
    type_label = "limitation"
elif chunk_type == "future_work":
    type_label = "future work"
else:  # regular claim
    type_label = f"{claim_type} claim"  # e.g. "comparative claim"
    # if claim_type is empty: just "claim"
```

**Location construction:**
```python
location = section_type  # e.g. "Results"
if heading and heading != section_type:
    location += f" | {heading}"  # e.g. "Results | Training Efficiency"
```

**Entity hint construction:**
```python
if entities:
    entity_hint = f"\nentities: {', '.join(entities)}"
# adds: "\nentities: STGCN, GCGRU, training_speed"
```

**Final output format:**
```
{type_label} in {location}:
{description}
entities: {e1}, {e2}, ...
```

**Real example output:**
```
comparative claim in Results | Training Efficiency and Generalization:
STGCN achieves 14 times acceleration in training speed compared to GCGRU.
entities: STGCN, GCGRU, training_speed
```

### `build_entity_text(entity: dict, paper_title: str) → str`

**Fields read:**
- `entity_type` — "method", "dataset", "metric", "task"
- `text` — the entity name
- `section_type` — primary section where entity appeared
- `also_in_sections` — list of other sections

**Section deduplication:** Combines `section_type` and `also_in_sections`
into a single list, then deduplicates while preserving order using
`dict.fromkeys(sections_list)`.

**Final output format:**
```
{entity_type}: {entity_text}
appears in: {section1}, {section2}, {section3}
paper: {paper_title}
```

**Real example output:**
```
method: STGCN
appears in: Abstract, Experiments, Results, Future Work
paper: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework...
```

### `build_section_text(section: dict, paper_title: str) → str`

**Fields read:**
- `section_type` — e.g. "Methods"
- `heading` — e.g. "Network Architecture"
- `text` — full raw section text

**Location construction:**
```python
location = f"{section_type} section"
if heading and heading != section_type:
    location += f" — {heading}"
# → "Methods section — Network Architecture"
```

**Final output format:**
```
{section_type} section — {heading} | paper: {paper_title}

{full raw text}
```

Note the double newline before the text — this visually separates the
header from the content, and also signals to the embedding model that
the header and body are related but distinct.

Full section text is included without truncation. The BGE model handles
long texts by mean-pooling token embeddings across the entire sequence.

### `build_figure_text(figure: dict, paper_title: str) → str`

**Fields read:**
- `label` — figure number, e.g. "2"
- `caption` — the figure caption text

**Label construction:**
```python
label_str = f"figure {label}" if label else "figure"
# → "figure 2" if label exists, "figure" if not
```

**Final output format:**
```
figure {label} caption | paper: {paper_title}:
{caption}
```

**Real example output:**
```
figure 2 caption | paper: Spatio-Temporal Graph Convolutional Networks...:
Figure 2: Architecture of spatio-temporal graph convolutional networks.
The framework STGCN consists of two ST-Conv blocks and a fully-connected output layer.
```

---

## 7. `chunker.py` — complete breakdown

**Purpose:** Reads one paper's output folder (3 JSON files) and transforms
them into a flat list of typed chunk dicts ready for embedding and indexing.
No embedding or storage happens here — pure data transformation.

**Input:** A folder path + paper_id string
**Output:** A Python list of chunk dicts (see schema in Section 3)

**Source files consumed:**
- `claims_output.json` → claims, limitations, future_work, llm_entities
- `sections.json` → section text chunks
- `figures.json` → figure caption chunks (optional)

**Imports:**
```python
import json          # loading JSON files
import hashlib       # SHA-256 for chunk IDs
import logging       # logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any
from rag.utils.paper_id import validate_paper_id
from rag.utils.text_builder import (
    build_claim_text, build_entity_text,
    build_section_text, build_figure_text,
)
```

### Module-level constants

```python
EMBED_MODEL         = "BAAI/bge-base-en-v1.5"
EMBED_MODEL_VERSION = "1.0"
```
Stored in every chunk dict so you can identify which chunks need
re-embedding if you upgrade the model in the future.

### `chunk_paper(folder_path: str, paper_id: str) → list[dict]`

The only public function. Everything else is private (prefixed with `_`).

**Step by step:**
1. `validate_paper_id(paper_id)` — normalises and validates the ID
2. `_verify_folder(folder, paper_id)` — confirms folder exists
3. `_load_json(folder / "claims_output.json")` — required
4. `_load_json(folder / "sections.json")` — required
5. `_load_json(folder / "figures.json", required=False)` — optional
6. `_get_paper_title(claims_data, paper_id)` — reads title from metadata
7. Calls each private chunk function and extends the main list
8. `_log_summary(paper_id, chunks)` — prints formatted count table

**Raises:**
- `FileNotFoundError` if claims_output.json or sections.json is missing
- `ValueError` if paper_id is invalid (from validate_paper_id)

### `_verify_folder(folder: Path, paper_id: str) → None`

Checks `folder.exists()`. Raises `FileNotFoundError` with a clear message
showing the full path and the paper_id being looked for.

### `_load_json(path: Path, required: bool = True) → Any`

Opens and parses a JSON file. Two behaviours:
- `required=True` (default): raises `FileNotFoundError` with helpful message if missing
- `required=False`: logs a WARNING and returns `None` if missing

Returns the parsed Python object (dict or list depending on the JSON structure).

### `_get_paper_title(claims_data: dict, paper_id: str) → str`

Reads `claims_data["metadata"]["title"]`. Uses try/except for both
`KeyError` (key doesn't exist) and `TypeError` (metadata is None).
Falls back to returning `paper_id` as the title if anything goes wrong.

### `_make_chunk_id(paper_id: str, content: str) → str`

**Formula:** `sha256(f"{paper_id}::{content}".encode("utf-8")).hexdigest()[:12]`

The `::` separator prevents collisions where paper_id and content could
accidentally combine to the same string in two different ways.

**Content string per chunk type:**
- claim/limitation/future_work: `display_text` (the description/text field)
- entity: `f"entity::{entity_type}::{text.lower()}"` — uses normalized text
  so the same entity (regardless of capitalisation) always gets the same ID
- section: `f"section::{section_type}::{heading}"`
- figure: `f"figure::{figure_id}::{label}"`

**Why SHA-256 and not UUID:**
UUID is random — two runs on the same data produce different IDs, causing
duplicate documents in ChromaDB. SHA-256 is deterministic — identical input
always produces the same output, so upsert correctly updates in place.

### `_extract_numeric(claim: dict) → tuple[bool, float | None]`

Reads `claim.get("value")`. If the value field exists and is not None,
attempts `float(value)`. Returns `(True, 14.0)` on success.
Returns `(False, None)` if value is None or conversion fails.

**Why this field matters:** Claims like "STGCN achieves 14x acceleration"
have `value=14.0`. The enricher uses `has_numeric_value=True` to find
claims worth comparing numerically between papers. A claim like
"STGCN is faster" without a number cannot be used for contradiction detection.

### `_extract_entity_fields(claim: dict) → dict`

Reads `claim.get("entities_involved", [])`. Joins the list to a
comma-separated string for `entities_mentioned`.

Returns four fields:
```python
{
    "entities_mentioned": "STGCN,GCGRU,training_speed",
    "methods_mentioned":  "",   # enricher fills later
    "datasets_mentioned": "",   # enricher fills later
    "metrics_mentioned":  "",   # enricher fills later
}
```

The empty strings for methods/datasets/metrics are intentional placeholders.
ChromaDB requires consistent metadata schema across all documents in a
collection — if one document has these fields, all must have them.
The enricher is supposed to fill them properly, but this hasn't been
fully implemented yet.

### `_chunk_claims(claims_data, paper_id, paper_title) → list[dict]`

Reads `claims_data.get("claims", [])`. Returns empty list with warning if none found.

For each claim:
- Injects `claim["_chunk_type"] = "claim"` so `build_claim_text` knows the type
- Calls `build_claim_text(claim, paper_title)` for embed_text
- Uses `claim.get("description", "")` for display_text
- Calls `_extract_numeric(claim)` for has_numeric_value / numeric_value
- Calls `_extract_entity_fields(claim)` and unpacks with `**entity_fields`
- Sets `claim_type` from `claim.get("claim_type", "")`
- Sets `confidence` as `float(claim.get("confidence", 0.0))`
- Sets `also_in_papers = ""` and `appears_in_n_papers = 1` as empty defaults

### `_chunk_limitations(claims_data, paper_id, paper_title) → list[dict]`

Reads `claims_data.get("limitations", [])`.

Key difference from claims:
- `chunk_type = "limitation"` (not "claim")
- `claim_type = "limitation"` (hardcoded — limitations have no sub-type)
- Uses `lim.get("text", "")` for display_text (NOT "description")
- `has_numeric_value = False` always (limitations don't have numeric comparisons)
- `numeric_value = 0.0` always

The text vs description difference is important: in claims_output.json,
claims use a `description` field while limitations use a `text` field.

### `_chunk_future_work(claims_data, paper_id, paper_title) → list[dict]`

Reads `claims_data.get("future_work", [])`.

Identical structure to limitations:
- `chunk_type = "future_work"`
- `claim_type = "future_work"` (hardcoded)
- Uses `fw.get("text", "")` for display_text
- `has_numeric_value = False`, `numeric_value = 0.0` always

### `_chunk_entities(claims_data, paper_id, paper_title) → list[dict]`

This is the most complex chunking function due to deduplication.

**Source priority:**
```python
entities = claims_data.get("llm_entities", [])
if not entities:
    entities = claims_data.get("entities", [])
```
Prefers `llm_entities` because the LLM-extracted entities have better
quality, cleaner text, and more accurate type classification than the
raw SciBERT NER entities.

**Deduplication algorithm:**
```python
seen: dict[tuple, dict] = {}   # key = (text.lower(), entity_type)

for ent in entities:
    text  = ent.get("text", "").strip()
    etype = ent.get("entity_type", "")
    if not text or not etype:
        continue   # skip invalid entities

    key = (text.lower(), etype)   # normalise for comparison
    if key not in seen:
        # first occurrence — store it and start section list
        seen[key] = dict(ent)
        seen[key]["_all_sections"] = [ent.get("section_type", "")]
    else:
        # duplicate — keep higher confidence version
        if ent.get("confidence", 0) > seen[key].get("confidence", 0):
            section_backup = seen[key]["_all_sections"]
            seen[key] = dict(ent)        # replace with higher confidence
            seen[key]["_all_sections"] = section_backup  # restore sections
        # add new section to the list
        sec = ent.get("section_type", "")
        if sec and sec not in seen[key]["_all_sections"]:
            seen[key]["_all_sections"].append(sec)
```

**Result:** STGCN mentioned in Abstract, Methods, Results, and Future Work
produces ONE entity chunk with `all_sections = "Abstract,Methods,Results,Future Work"`.

**Entity chunk special fields:**
- `entity_type`, `entity_text`, `entity_text_normalized`
- `all_sections` — comma-joined string of all sections where entity appeared
- `methods_mentioned = entity_text if entity_type == "method" else ""`
- `datasets_mentioned = entity_text if entity_type == "dataset" else ""`
- `metrics_mentioned = entity_text if entity_type == "metric" else ""`
- `claim_type = ""` — not applicable
- `section_heading = ""` — not applicable
- `has_numeric_value = False`, `numeric_value = 0.0` — not applicable

**Chunk ID for entities:**
Uses `f"entity::{etype}::{text_lower}"` as content for SHA-256.
This means "STGCN" as a method always gets the same chunk ID regardless
of which paper section it first appeared in.

### `_chunk_sections(sections_data, paper_id, paper_title) → list[dict]`

Reads the list from `sections.json` directly (top-level is a list, not a dict).

For each section:
- Skips sections with empty or whitespace-only text (logs warning)
- Stores FULL raw text as display_text — no truncation whatsoever
- Adds `text_length = len(text)` for debugging

Chunk ID uses `f"section::{section_type}::{heading}"` — based on position
in paper, not content. This handles the case where two sections might have
identical text.

Non-applicable fields set to safe defaults:
- `claim_type = ""`, `entity_type = ""`, `entity_text = ""`
- `confidence = 1.0` (not LLM-extracted, so assume full confidence)
- `has_numeric_value = False`, `numeric_value = 0.0`
- All entity mention fields = ""

### `_chunk_figures(figures_data, paper_id, paper_title) → list[dict]`

Reads the list from `figures.json`.

If `figures_data` is `None` (file not found) or empty list, logs INFO
and returns empty list (not a warning because figures.json is optional).

For each figure:
- Skips figures with empty captions
- `chunk_type = "figure"` but goes to `paper_sections` collection
  (same collection as sections, distinguished by chunk_type field)
- `section_type = "Figure"` (hardcoded)
- `section_heading = f"Figure {label}"`
- Adds `figure_id` and `figure_label` fields

### `_log_summary(paper_id: str, chunks: list[dict]) → None`

Uses `collections.Counter` to count by chunk_type. Logs at INFO level
and prints a formatted table to stdout:
```
============================================================
  paper_id : stgcn_yu_2018
  total    : 379 chunks
  claim         : 141
  entity        : 202
  ...
============================================================
```

---

## 8. `embedder.py` — complete breakdown

**Purpose:** Adds a 768-float `"embedding"` field to every chunk dict by
encoding `embed_text` through a local sentence-transformer model. The model
runs entirely locally — no API calls, no internet required after first download.

**Input:** List of chunk dicts from chunker.py (no embedding field yet)
**Output:** Same list, each dict now has `"embedding": [768 floats]`

**Imports:**
```python
import logging
import time
from typing import Optional
```
sentence_transformers is imported lazily inside `_get_model()` so the
import error message is helpful and the file can be imported even if
sentence-transformers isn't installed yet.

### Module-level constants

```python
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
EMBED_DIM    = 768           # embedding vector size
BATCH_SIZE   = 32            # chunks processed per encode() call
                              # 32 is safe for 8GB RAM on CPU
                              # increase to 64 if you have 16GB+
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```

**Why BAAI/bge-base-en-v1.5:**
- 768 dimensions matches most cross-encoders and is a good balance
- Outperforms SPECTER2 on retrieval tasks (not classification)
- Runs fully locally, ~440MB download once
- ~40ms per batch of 32 on modern CPU

**Why BATCH_SIZE=32:**
Each batch is loaded into RAM as a tensor. 32 chunks × average 200 tokens
× 4 bytes = manageable. Going higher risks OOM on 8GB systems.

**Why a query prefix for queries but NOT for documents:**
BGE models are trained asymmetrically. Documents are encoded without any prefix.
Queries are encoded with the prefix to tell the model "this is a search query,
find the most relevant document for it." This asymmetry is intentional and
improves retrieval quality by 10-15% vs encoding both sides the same way.
The prefix is ONLY added in `embed_query()`, never in `embed_chunks()`.

**Why normalize_embeddings=True:**
ChromaDB uses cosine similarity internally. For unit vectors (magnitude=1.0),
cosine similarity equals dot product, which is computed faster. More importantly,
without normalization, vectors from differently-worded texts can have very
different magnitudes, making distance comparisons unreliable.

### Module-level cache

```python
_model = None
```

The SentenceTransformer object is stored at module level. Loading the model
takes ~4 seconds. If we loaded it per call, indexing 5 papers would waste 20
seconds on repeated model loading. The cache means it loads once per session.

### `embed_chunks(chunks: list[dict]) → list[dict]`

Main entry point for the indexing pipeline.

**Step by step:**
1. Returns empty list if chunks is empty (with warning)
2. Calls `_get_model()` to load or retrieve cached model
3. Extracts all `embed_text` values in the same order as chunks:
   `texts = [c["embed_text"] for c in chunks]`
4. Calls `_embed_in_batches(model, texts)` — returns list of 768-float lists
5. Zips embeddings back onto chunks: `chunk["embedding"] = vector`
6. Modifies in-place AND returns the same list

**Note:** The function modifies the list in-place. The `return chunks`
at the end returns the same list object, not a copy. This means:
```python
chunks = embed_chunks(chunks)  # works
embed_chunks(chunks)           # also works (in-place modification)
```

### `embed_query(query_text: str) → list[float]`

Called by `retriever._stage1_ann_search()` only. NOT called during indexing.

Prepends `QUERY_PREFIX` to the query text before encoding:
```python
text = QUERY_PREFIX + query_text
# → "Represent this sentence for searching relevant passages: What are the limitations..."
```

Encodes the single string (no batching needed for one query).
Returns a plain Python list of 768 floats (not numpy array).

### `_get_model() → SentenceTransformer`

Checks `_model is not None` first — returns cached instance immediately.

On first call:
1. Tries to import `sentence_transformers`
2. Raises `ImportError` with install command if not installed
3. Prints download notice for first-run users
4. Times the load: `start = time.time()` → prints elapsed
5. `SentenceTransformer(MODEL_NAME, device="cpu")` — forces CPU even if
   CUDA is available (prevents unexpected GPU memory usage)
6. Stores in `_model` and returns

### `_embed_in_batches(model, texts: list[str]) → list[list[float]]`

Processes texts in groups of `BATCH_SIZE`:
```python
for batch_num, start_idx in enumerate(range(0, len(texts), BATCH_SIZE)):
    batch = texts[start_idx : start_idx + BATCH_SIZE]
    vectors = model.encode(
        batch,
        normalize_embeddings=True,  # unit vectors
        show_progress_bar=False,    # we handle our own display
        batch_size=BATCH_SIZE,
    )
    all_embeddings.extend([v.tolist() for v in vectors])
```

**numpy → list conversion:**
`model.encode()` returns numpy arrays. ChromaDB's Python client requires
plain Python lists. `.tolist()` converts each numpy array to a Python list.

**Progress bar:**
```python
bar_len  = 30
filled   = int(bar_len * done / total_batches)
bar      = "█" * filled + "░" * (bar_len - filled)
print(f"  [{bar}] {n_done}/{len(texts)} chunks", end="\r")
```
Uses `end="\r"` to overwrite the same line rather than printing new lines.

---

## 9. `indexer.py` — complete breakdown

**Purpose:** Takes embedded chunk dicts and permanently stores them in
ChromaDB. Idempotent — running twice on the same paper produces exactly
the same database state. No duplicates are created.

**Input:** List of chunk dicts with `embedding` field populated
**Output:** ChromaDB database updated; returns count dict

**Imports:**
```python
import logging
from pathlib import Path
```
`chromadb` is imported lazily inside `_get_client()`.

### Module-level constants

```python
CHROMA_STORE_PATH = "rag/chroma_store"
```
The folder where ChromaDB writes its SQLite database and HNSW index files.
This path is relative to wherever you run Python from (your project root).
Add `rag/chroma_store/` to `.gitignore` — it can grow to hundreds of MB.

```python
COLLECTION_MAP = {
    "section":     "paper_sections",
    "figure":      "paper_sections",      # same collection as sections
    "claim":       "claims_and_findings",
    "limitation":  "claims_and_findings",
    "future_work": "claims_and_findings",
    "entity":      "entities_global",
}
```
Maps each `chunk_type` to its target ChromaDB collection name.
`researcher_feedback` is NOT in this map — it's written at runtime by agents.

```python
METADATA_FIELDS = [
    "paper_id", "paper_title", "chunk_type", "claim_type",
    "section_type", "section_heading", "entity_type", "entity_text",
    "entity_text_normalized", "confidence", "source",
    "has_numeric_value", "numeric_value", "entities_mentioned",
    "methods_mentioned", "datasets_mentioned", "metrics_mentioned",
    "also_in_papers", "appears_in_n_papers",
    "embed_model", "embed_model_version",
]
```
The exact fields extracted from the chunk dict and stored in ChromaDB metadata.
Fields NOT in this list are ignored: `embed_text`, `embedding`, `figure_id`,
`figure_label`, `text_length`, `all_sections`.

### Module-level cache

```python
_client      = None   # chromadb.PersistentClient instance
_collections = {}     # dict: collection_name → collection object
```

### `_get_client() → chromadb.PersistentClient`

Creates `rag/chroma_store/` folder if it doesn't exist using
`store_path.mkdir(parents=True, exist_ok=True)`.
Creates a `PersistentClient` — this opens (or creates) the SQLite database
at that path. Cached in `_client` after first call.

**PersistentClient vs EphemeralClient:**
`PersistentClient` writes to disk. `EphemeralClient` keeps everything in RAM
and loses data when Python exits. We always use PersistentClient.

### `get_collections() → dict`

Called by `enricher.py` and `retriever.py` as well as `index_chunks()`.
Creates all 4 collections with `get_or_create_collection()` — creates if
they don't exist, opens if they do. All use `{"hnsw:space": "cosine"}`.

**Why cosine similarity space:**
Our vectors are all normalized to magnitude 1.0 (from `normalize_embeddings=True`
in embedder.py). For unit vectors, cosine distance = Euclidean distance ÷ 2.
ChromaDB can compute cosine distance faster than Euclidean for unit vectors.

Returns the same cached dict on subsequent calls — no repeated DB queries.

### `_extract_metadata(chunk: dict) → dict`

Iterates through `METADATA_FIELDS` and pulls each value from the chunk dict.

**None protection:**
```python
if value is None:
    if field in ("numeric_value", "confidence", "appears_in_n_papers"):
        value = 0.0
    elif field == "has_numeric_value":
        value = False
    else:
        value = ""
```
ChromaDB's metadata values must be `str`, `int`, `float`, or `bool`.
Passing `None` raises a `ValueError` from ChromaDB. This protection
ensures no None ever reaches ChromaDB regardless of what chunker.py produced.

### `index_chunks(chunks: list[dict]) → dict`

**Pre-flight check:**
```python
missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
if missing:
    raise ValueError(f"{len(missing)} chunks are missing 'embedding' field...")
```
Prevents the silent failure of storing chunks without vectors.

**Grouping by collection:**
```python
batches = {name: [] for name in collections}
for chunk in chunks:
    ctype = chunk.get("chunk_type", "")
    collection = COLLECTION_MAP.get(ctype)
    if collection is None:
        logger.warning(f"Unknown chunk_type '{ctype}', skipping...")
        continue
    batches[collection].append(chunk)
```

**The upsert call:**
```python
collection.upsert(
    ids        = [c["chunk_id"]     for c in batch],   # list of strings
    embeddings = [c["embedding"]    for c in batch],   # list of 768-float lists
    documents  = [c["display_text"] for c in batch],   # list of strings
    metadatas  = [_extract_metadata(c) for c in batch],# list of dicts
)
```
ChromaDB's `upsert` is "insert or update" — if a document with that ID exists,
it's updated. If not, it's inserted. This is what makes re-running the indexer
safe: same SHA-256 ID = update in place = no duplicates.

Returns: `{"paper_sections": 14, "claims_and_findings": 163, "entities_global": 202}`

### `collection_counts() → dict`

Calls `col.count()` on each collection. Returns
`{"paper_sections": 14, "claims_and_findings": 163, ...}`.
Used by tests and `pipeline.status()`.

### `peek_collection(collection_name: str, n: int = 3) → list[dict]`

Calls `collection.peek(limit=n)`. Returns first n documents as a list of dicts
with keys `id`, `document` (first 100 chars + "..."), and `metadata`.
Used for quick inspection without querying.

### `delete_paper(paper_id: str) → dict`

For each collection:
1. `col.get(where={"paper_id": {"$eq": paper_id}})` — get all IDs for that paper
2. If any IDs found: `col.delete(ids=ids)` — bulk delete

Useful when you update a paper's Phase 1 output and need to re-index from scratch.
Returns dict of `{collection_name: n_deleted}`.

---

## 10. `enricher.py` — complete breakdown

**Purpose:** Runs three passes over the already-indexed ChromaDB data to
create cross-paper connections. Must run AFTER all papers are indexed.
With only 1 paper: Pass 1 and 2 produce 0 results (correct and expected).
With 2+ papers: all three passes produce meaningful results.

**Imports:**
```python
import json
import logging
from pathlib import Path
from collections import defaultdict
```

### Module-level constants

```python
MEMORY_DIR = Path("memory")
```
Where the output JSON files are written.

### `_normalize_entity_text(text: str) → str`

Simple normalisation: `text.strip().lower()`. Used for cross-paper matching.
Only exact matches after normalisation are linked — fuzzy matching
(handling "BERT" vs "bert-base" vs "Google BERT") is left to the
knowledge graph layer (friend's responsibility using RapidFuzz).

### `run_all_passes() → dict`

Entry point. Checks `collection_counts()` to ensure data exists.
If no data found, prints message and returns `{}` immediately.
Otherwise calls all three passes in order and calls `_print_summary()`.

Returns: `{"pass1": {...}, "pass2": {...}, "pass3": {...}}`
Each inner dict contains counts from that pass.

### `_pass1_entity_linking() → dict`

**Goal:** Find which entities appear in multiple papers and update their
`also_in_papers` and `appears_in_n_papers` fields in ChromaDB.

**Algorithm step by step:**

1. Fetch all entity chunks: `eg.get(include=["metadatas"])`
   This returns ALL metadata without filtering — we need to scan everything.

2. Build entity_map grouped by (normalized_text, entity_type):
   ```python
   entity_map: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
   # structure: {("stgcn", "method"): {"stgcn_yu_2018": ["chunk_id_1"], "dcrnn_li_2018": ["chunk_id_2"]}}
   ```

3. Find cross-paper entities:
   ```python
   cross_paper = {
       key: paper_dict
       for key, paper_dict in entity_map.items()
       if len(paper_dict) > 1   # appears in 2+ different papers
   }
   ```

4. Update each affected chunk:
   ```python
   for (text, etype), paper_dict in cross_paper.items():
       all_paper_ids = list(paper_dict.keys())
       for paper_id, chunk_ids in paper_dict.items():
           others = [p for p in all_paper_ids if p != paper_id]
           eg.update(
               ids=[chunk_id],
               metadatas=[{
                   "also_in_papers":      ",".join(others),
                   "appears_in_n_papers": len(all_paper_ids),
               }]
           )
   ```

**After this pass** with 3 papers where STGCN appears in all three:
- Each STGCN entity chunk gets `also_in_papers = "dcrnn_li_2018,gwnet_wu_2019"`
  (the OTHER papers, not its own)
- Each gets `appears_in_n_papers = 3`

Returns: `{"linked_entities": N, "updated_chunks": M}`

### `_pass2_contradiction_candidates() → dict`

**Goal:** Pre-compute pairs of claims that might contradict each other.
These are CANDIDATES, not confirmed contradictions. The comparison agent
confirms them using LLM reasoning.

**Query to get candidates:**
```python
result = caf.get(
    where={"$and": [
        {"claim_type":       {"$eq": "comparative"}},
        {"has_numeric_value": {"$eq": True}},
    ]},
    include=["documents", "metadatas"],
)
```
Only comparative claims with numeric values can contradict.
A limitation or methodological claim doesn't produce a numeric contradiction.

**Pairing algorithm:**
Nested loop O(n²) over all candidate claims — acceptable at 5-6 papers scale.
For each pair (a, b):
1. Skip if same paper
2. Check method overlap: normalize and intersect `methods_mentioned` sets
3. If no method overlap, check entity overlap: intersect `entities_mentioned` sets
4. If any overlap: record as candidate pair

**Output structure per candidate:**
```json
{
    "claim_a": {
        "chunk_id": "...",
        "paper_id": "stgcn_yu_2018",
        "text": "STGCN achieves 14x acceleration...",
        "numeric_value": 14.0
    },
    "claim_b": {
        "chunk_id": "...",
        "paper_id": "dcrnn_li_2018",
        "text": "DCRNN achieves 10x speedup...",
        "numeric_value": 10.0
    },
    "shared_entities": ["stgcn"],
    "confirmed": false
}
```

`confirmed` starts as `false`. The comparison agent sets it to `true`
after reasoning with an LLM that the two claims genuinely contradict.

Always writes the file (empty list if no candidates) via `_write_json()`.

### `_pass3_gap_matrix() → dict`

**Goal:** Build a method × dataset co-occurrence matrix to find research gaps.

**Two separate queries:**
```python
method_result  = eg.get(where={"entity_type": {"$eq": "method"}},  include=["metadatas"])
dataset_result = eg.get(where={"entity_type": {"$eq": "dataset"}}, include=["metadatas"])
```

**Build lookup maps:**
```python
methods:  {"stgcn": ["stgcn_yu_2018"], "dcrnn": ["dcrnn_li_2018", "stgcn_yu_2018"]}
datasets: {"pemsd7": ["stgcn_yu_2018"], "metr-la": ["dcrnn_li_2018"]}
```
Uses `_normalize_entity_text()` as keys. Prevents duplicates with
`if pid not in methods[text]` check.

**Matrix computation:**
```python
matrix = {}
for method in methods:
    matrix[method] = {}
    for dataset in datasets:
        shared = list(set(methods[method]) & set(datasets[dataset]))
        matrix[method][dataset] = shared
```
`shared` is a list of paper_ids that use BOTH this method AND this dataset.
An empty list means no paper has tried this combination — a research gap.

**Gap scoring:**
```python
gap_score = len(methods[method]) + len(datasets[dataset])
```
If 3 papers use method X and 3 papers use dataset Y but none use both,
gap_score = 6. This is a highly significant gap — both sides are
well-studied but the combination is unexplored.

**Sorted:** `gaps.sort(key=lambda x: x["gap_score"], reverse=True)`
Highest priority gaps first.

**Output file structure:**
```json
{
    "methods": {"stgcn": ["stgcn_yu_2018"]},
    "datasets": {"pemsd7": ["stgcn_yu_2018"]},
    "matrix": {"stgcn": {"pemsd7": ["stgcn_yu_2018"], "metr-la": []}},
    "gaps": [
        {
            "method": "stgcn",
            "dataset": "metr-la",
            "gap_score": 2,
            "method_used_in": ["stgcn_yu_2018"],
            "dataset_used_in": ["dcrnn_li_2018"]
        }
    ]
}
```

### `_write_json(filename: str, data) → None`

Creates `memory/` folder if it doesn't exist (`MEMORY_DIR.mkdir(exist_ok=True)`).
Writes with `indent=2` for human-readable JSON and `ensure_ascii=False`
so non-ASCII characters (accented names, etc.) are preserved not escaped.

### `_print_summary(results: dict) → None`

Prints a formatted summary box to stdout showing counts from all 3 passes.

---

## 11. `query_handler.py` — complete breakdown

**Purpose:** Parses a natural language question into a structured query plan
dict. No database access. Pure Python logic with regex matching.

**Why this is separated from retriever.py:**
- Metadata filters run BEFORE vector search inside ChromaDB (fast, exact matching)
- Vector search runs AFTER filtering (semantic, approximate)
- Separating query planning from execution makes both easier to test and tune
- You can unit test all 17 query patterns without a running ChromaDB instance

**Imports:**
```python
import re
import logging
from typing import Optional
```

### `INTENT_RULES` — the core of this file

A list of 8 rule dicts. Evaluated in order — first match wins.
Order matters: gap is before entity_lookup to prevent "which method hasn't
been tested?" from matching entity_lookup's "which" patterns.

**Each rule dict contains:**
- `intent` — string label for the matched intent
- `patterns` — list of regex strings, any one matching triggers this rule
- `collections` — list of collection names to search
- `where` — ChromaDB filter dict (can be empty `{}` for no filter)
- `n_results` — how many candidates to fetch before re-ranking
- `special` — optional. Only "gap_matrix" currently, skips vector search

**All 8 rules in detail:**

**Rule 1 — limitation** (n_results=10)
```python
patterns: [r"\blimitation", r"\bweakness", r"\bshortcoming",
           r"\bproblem with", r"\bissue with", r"\bdrawback",
           r"\bwhat.{0,20}wrong", r"\bwhat.{0,20}bad"]
where: {"chunk_type": {"$eq": "limitation"}}
collections: ["claims_and_findings"]
```
`\b` is a word boundary anchor — prevents "elimination" from matching "limitation".
`.{0,20}` allows up to 20 characters between words — handles "what is wrong with this?"

**Rule 2 — future_work** (n_results=8)
```python
patterns: [r"\bfuture work", r"\bfuture direction", r"\bnext step",
           r"\bopen problem", r"\bopen question", r"\bto be explored",
           r"\bwhat.{0,20}next", r"\bwhat.{0,20}remain"]
where: {"chunk_type": {"$eq": "future_work"}}
collections: ["claims_and_findings"]
```

**Rule 3 — comparison** (n_results=12, most candidates — comparisons are complex)
```python
patterns: [r"\bcompar", r"\bvs\b", r"\bversus", r"\boutperform",
           r"\bfaster than", r"\bbetter than", r"\bworse than",
           r"\bcontradiction", r"\bconflict", r"\bdisagree",
           r"\bdifference between", r"\bhow does .{0,30} differ"]
where: {"claim_type": {"$eq": "comparative"}}
collections: ["claims_and_findings"]
```
`\bvs\b` — `\b` on both sides prevents "obvious" from matching "vs".
`r"\bcompar"` — matches "compare", "comparison", "comparative" etc.

**Rule 4 — performance** (n_results=10)
```python
patterns: [r"\bperformance", r"\baccuracy", r"\bresult",
           r"\bscore", r"\brmse\b", r"\bmae\b", r"\bf1\b",
           r"\bbleu\b", r"\bbenchmark", r"\bevaluat",
           r"\bhow well", r"\bhow (good|accurate|fast)"]
where: {"claim_type": {"$eq": "performance"}}
collections: ["claims_and_findings"]
```

**Rule 5 — gap** (n_results=0, special="gap_matrix")
```python
patterns: [r"\bgap", r"\bnobody", r"\bno one", r"\bnot (yet )?tried",
           r"\buntried", r"\bmissing combination",
           r"\bwhich.{0,30}(hasn.t|have not|not been) (been )?tested",
           r"\bwhich.{0,30}(hasn.t|have not) (been )?explored",
           r"\bwhat.{0,30}(hasn.t|have not|not been) (been )?tested",
           r"\bwhat.{0,30}(hasn.t|have not) (been )?explored"]
where: {}  # no vector search at all
collections: ["entities_global"]
special: "gap_matrix"
n_results: 0
```
When `n_results=0` and `special="gap_matrix"`, the retriever skips ALL
vector search and reads `memory/gap_matrix.json` directly instead.
"hasn.t" uses `.` instead of `'` because regex `.` matches any character,
making it handle "hasn't" and "hasnt" and other variants.

**Rule 6 — entity_lookup** (n_results=8)
```python
patterns: [r"\bwhat is\b", r"\bwhat are\b", r"\bwhich (method|model|dataset|metric)",
           r"\btell me about\b", r"\bexplain\b",
           r"\bhow does .{0,30} work"]
where: {}  # pure semantic search
collections: ["entities_global", "paper_sections"]
```
Searches both entity index and section text. No filter — cast wide.

**Rule 7 — figure** (n_results=5)
```python
patterns: [r"\bfigure\b", r"\bdiagram\b", r"\barchitecture diagram",
           r"\bshow.{0,20}figure", r"\bfig\.?\s*\d"]
where: {"chunk_type": {"$eq": "figure"}}
collections: ["paper_sections"]
```
`r"\bfig\.?\s*\d"` matches "fig. 2", "fig2", "fig.2" with the optional dot
and optional whitespace.

**Rule 8 — literature_review** (n_results=15, most results — synthesis needs wide coverage)
```python
patterns: [r"\bliterature review", r"\bsurvey", r"\bsummar",
           r"\boverview", r"\bwrite.{0,20}review",
           r"\bwhat do (the )?papers say"]
where: {}  # semantic search only
collections: ["paper_sections", "claims_and_findings"]
```

### `DEFAULT_RULE`

Used when no INTENT_RULES match:
```python
DEFAULT_RULE = {
    "intent": "general",
    "collections": ["paper_sections", "claims_and_findings"],
    "where": {},
    "n_results": 10,
}
```
Broadest possible search — both full text and structured claims.

### `_detect_paper_filter(question: str) → dict | None`

Looks for paper_id patterns in the question:
```python
matches = re.findall(r'\b([a-z][a-z0-9]+_[a-z0-9_]+\d{4})\b', question.lower())
```
Pattern explanation:
- `[a-z]` — starts with a lowercase letter
- `[a-z0-9]+` — one or more alphanumeric chars
- `_` — underscore
- `[a-z0-9_]+` — more alphanumeric or underscore
- `\d{4}` — exactly 4 digits (the year)
- `\b` — word boundary

This matches "stgcn_yu_2018" but not "STGCN" or general words.
Returns `{"paper_id": {"$eq": "stgcn_yu_2018"}}` if found, None if not.

### `_detect_entity_filter(question: str, known_methods: list) → str | None`

If `known_methods` is provided (a list of method names from entities_global),
iterates through them checking if any appears in the lowercased question.
Returns the first matching method name, or None.

Used to enrich `query_text` for the embedding: if the question mentions STGCN
but it's not prominent in the first 20 characters, append it to the query text
so the embedding focuses on the right method.

### `parse_query(question: str, known_methods: list = None) → dict`

**Full algorithm:**

1. Strip whitespace. Raise ValueError if empty.
2. Lowercase: `q_lower = question.lower()`
3. Iterate INTENT_RULES. For each rule, iterate its patterns.
   `re.search(pattern, q_lower)` — matches anywhere in the string.
   Break at first matching pattern. Break at first matching rule.
4. Fall back to DEFAULT_RULE if no match.
5. Call `_detect_paper_filter(question)`.
6. Build where clause:
   ```python
   where = dict(rule.get("where", {}))  # copy to avoid mutating the rule
   if paper_filter and where:
       where = {"$and": [where, paper_filter]}  # combine both filters
   elif paper_filter:
       where = paper_filter  # paper filter only
   # if no paper_filter: where stays as rule's filter
   ```
7. Detect entity enrichment (only if known_methods provided):
   ```python
   if detected_entity and detected_entity.lower() not in q_lower[:20]:
       query_text = f"{question} {detected_entity}"
   ```
   Only appends if not already prominent in first 20 chars of question.
8. Build and return the plan dict.

**Output plan dict:**
```python
{
    "original":    "What are the limitations of stgcn_yu_2018?",
    "query_text":  "What are the limitations of stgcn_yu_2018?",
    "collections": ["claims_and_findings"],
    "where":       {"$and": [
                       {"chunk_type": {"$eq": "limitation"}},
                       {"paper_id":   {"$eq": "stgcn_yu_2018"}}
                   ]},
    "n_results":   10,
    "intent":      "limitation",
    # "special" key only present for gap queries
}
```

### `explain_plan(plan: dict) → None`

Convenience function. Prints the plan in a human-readable format.
Used in tests and debugging. Not used in production pipeline.

---

## 12. `retriever.py` — complete breakdown

**Purpose:** Takes a question, runs the full two-stage retrieval pipeline
(ChromaDB ANN → cross-encoder re-ranking), and returns top-k results with
complete provenance information.

**Imports:**
```python
import json
import logging
from pathlib import Path
```
`sentence_transformers.CrossEncoder` and `math` are imported lazily inside
their respective functions.

### Module-level constants

```python
MEMORY_DIR    = Path("memory")
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Why ms-marco-MiniLM-L-6-v2:**
- Trained on MS MARCO, a large passage retrieval dataset
- "MiniLM" = distilled small model — fast on CPU
- "L-6" = 6 layers (vs 12 for full models)
- Downloads ~85MB (much smaller than the bi-encoder)
- Good precision for scientific text despite being trained on web queries

### Module-level cache

```python
_cross_encoder = None
```

### `_get_cross_encoder() → CrossEncoder`

Same pattern as `_get_model()` in embedder.py. Loads once, caches at module level.
Raises helpful ImportError if sentence-transformers not installed.

### `retrieve(question: str, top_k: int = 5, known_methods: list = None) → list[dict]`

**Decision point for gap queries:**
```python
if plan.get("special") == "gap_matrix":
    return _retrieve_gaps(question)
```
Gap queries bypass all vector search and go directly to `_retrieve_gaps()`.

**For all other queries:**
1. `_stage1_ann_search(plan)` — get candidates
2. If empty: return `[]` with warning
3. `_stage2_rerank(question, candidates, top_k)` — re-rank and return

### `_stage1_ann_search(plan: dict) → list[dict]`

**Immediate return for gap queries:**
```python
if plan["n_results"] == 0:
    return []
```

**For each collection in plan["collections"]:**
```python
kwargs = {
    "query_embeddings": [q_vec],
    "n_results": min(plan["n_results"], coll.count()),  # can't fetch more than exist
    "include": ["documents", "metadatas", "distances"],
}
if plan["where"]:      # only add where if filter is non-empty
    kwargs["where"] = plan["where"]
```
The `if plan["where"]` check is critical — passing an empty `{}` as `where`
to ChromaDB causes an error. Only pass it when non-empty.

**Wrapped in try/except:** If a collection query fails (e.g. filter on a field
that doesn't exist), logs warning and continues to next collection rather
than crashing.

**Distance to similarity conversion:**
ChromaDB returns cosine DISTANCE (0=identical, 2=opposite).
We need cosine SIMILARITY (1=identical, -1=opposite).
Conversion: `ann_score = round(1 - dist, 4)`
For our unit vectors, cosine distance = 1 - cosine_similarity,
so ann_score = 1 - distance = cosine_similarity. Correct.

**Result structure per candidate:**
```python
{
    "chunk_id":   "86ac1a06d297",
    "document":   "STGCN achieves 14x...",    # display_text
    "metadata":   { all metadata fields },
    "collection": "claims_and_findings",
    "ann_score":  0.8753,                      # 1 - cosine_distance
    "paper_id":   "stgcn_yu_2018",            # lifted from metadata
    "chunk_type": "claim",                     # lifted from metadata
    "confidence": 0.95,                        # lifted from metadata
}
```

### `_stage2_rerank(question: str, candidates: list, top_k: int) → list[dict]`

**Batch scoring (more efficient than one-by-one):**
```python
pairs = [(question, c["document"]) for c in candidates]
ce_scores = ce.predict(pairs)  # scores all pairs at once
```

The cross-encoder reads question AND document together (concatenated with
a separator token). Unlike the bi-encoder which encodes them independently,
the cross-encoder can model interactions between question words and document
words. This is why it's more accurate but slower.

**Sigmoid normalisation:**
```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
norm_score = sigmoid(float(ce_score))
```
Cross-encoder raw scores are arbitrary floats (can be negative or >1).
Sigmoid maps any real number to (0, 1). This makes scores interpretable.

**Confidence weighting:**
```python
confidence = float(candidate.get("confidence", 1.0))
if confidence == 0.0:
    confidence = 1.0   # treat missing as full confidence
final_score = round(norm_score * confidence, 4)
```
A claim with confidence=0.95 scoring 0.90 from cross-encoder gets
`final_score = 0.90 × 0.95 = 0.855`. A claim with confidence=0.5 scoring 0.92
gets `final_score = 0.92 × 0.5 = 0.46`. This downweights uncertain extractions.

**The confidence=0.0 special case:** Some future_work items from Phase 1
have confidence=0.0 because the LLM extractor didn't fill that field.
Without this fix, all future_work results would have score=0.0, making them
invisible. The fix treats 0.0 as "unset" rather than "not confident".

**Final result dict per result:**
```python
{
    "rank":       1,
    "score":      0.9998,      # sigmoid(ce_score) × confidence — final ranking score
    "ce_score":   0.9998,      # sigmoid(ce_score) alone — cross-encoder quality
    "ann_score":  0.8753,      # 1 - cosine_distance — ANN quality
    "document":   "STGCN achieves 14 times...",  # display_text from ChromaDB
    "paper_id":   "stgcn_yu_2018",
    "chunk_type": "claim",
    "chunk_id":   "86ac1a06d297",
    "collection": "claims_and_findings",
    "metadata":   { full metadata dict },    # all ChromaDB metadata fields
    "intent":     "comparison",             # from query_handler
}
```

### `_retrieve_gaps(question: str) → list[dict]`

Reads `memory/gap_matrix.json`. Returns `[]` if file not found (with warning).

If `gaps` list is empty (only 1 paper indexed), returns a single result
explaining why no gaps were found.

For up to 10 gaps:
```python
doc = (
    f"Research gap: '{gap['method']}' has never been tested on "
    f"'{gap['dataset']}'. "
    f"Method appears in: {gap['method_used_in']}. "
    f"Dataset appears in: {gap['dataset_used_in']}."
)
```
`score = round(gap["gap_score"] / 10, 4)` — normalises gap_score to ~0-1 range.

### `print_results(results: list, question: str = "") → None`

Convenience display function. Prints each result with rank, score, chunk_type,
paper_id, and first 90 characters of document text.

---

## 13. `pipeline.py` — complete breakdown

**Purpose:** The single public entry point for the entire RAG system.
Agents, tests, and notebooks should only import from this file.
It ties all the steps together and provides a clean interface.

**Imports:**
```python
import logging
from pathlib import Path
```
All internal modules are imported lazily inside each function to keep
startup time fast and to avoid import errors if a dependency isn't installed.

### `index_paper(folder_path: str) → dict`

The most commonly called function. Runs Steps 1→2→3 in sequence.

**paper_id derivation:**
```python
folder   = Path(folder_path)
paper_id = paper_id_from_folder(str(folder))
```
Uses `paper_id_from_folder()` from utils/paper_id.py. The folder name IS
the paper ID — no arguments needed from the caller.

**Three steps printed with headers:**
```
Step 1/3 — Chunking...
  [chunker output: 379 chunks, counts by type]
Step 2/3 — Embedding...
  [progress bar]
Step 3/3 — Indexing into ChromaDB...
  [upsert counts per collection]
```

**Returns:**
```python
{
    "paper_id":    "stgcn_yu_2018",
    "total":       379,
    "collections": {"paper_sections": 14, "claims_and_findings": 163, "entities_global": 202}
}
```

**Does NOT call enrich()** — you call that separately after all papers.
This allows indexing 3 papers first then enriching once, rather than
re-enriching after every paper (which would be redundant).

### `index_all(memory_dir: str = "memory") → list[dict]`

Discovers paper folders automatically:
```python
folders = [
    f for f in memory.iterdir()
    if f.is_dir() and (f / "claims_output.json").exists()
]
```
Only includes folders that have `claims_output.json` — skips any other
folders in the memory directory.

Processes folders in `sorted()` order for consistent results.
Calls `index_paper()` for each and collects results.
Does NOT call enrich() — caller must do that.

### `enrich() → dict`

One-liner wrapper around `enricher.run_all_passes()`. Adds a header
print statement so the output is clear in terminal.

**When to call:**
- After indexing all initial papers
- After adding any new paper (re-runs all 3 passes on the full index)

### `query(question: str, top_k: int = 5, known_methods: list = None) → list[dict]`

One-liner wrapper around `retriever.retrieve()`. The `known_methods` parameter
lets callers pass a list of method names to enrich the query embedding.

Agents typically call it like:
```python
results = query("What are the limitations of STGCN?", top_k=5)
```

### `status() → None`

Inspection tool. Calls `collection_counts()` and checks for the two
output JSON files. Prints a formatted status table:
```
============================================================
  RAG SYSTEM STATUS
============================================================
  ChromaDB collections:
    paper_sections           : 14 docs
    claims_and_findings      : 163 docs
    entities_global          : 202 docs
    researcher_feedback      : 0 docs

  Memory files:
    contradiction_candidates.json     : 0.1 KB
    gap_matrix.json                   : 8.4 KB

  Total chunks indexed: 379
============================================================
```

---

## 14. The 4 ChromaDB collections — what lives where

### `paper_sections`

**Contains:** section chunks + figure chunks
**Document field:** Full raw section text / figure caption
**Key metadata filters:**
```python
{"chunk_type": {"$eq": "figure"}}          # only figures
{"chunk_type": {"$eq": "section"}}         # only sections
{"paper_id": {"$eq": "stgcn_yu_2018"}}    # specific paper
{"section_type": {"$eq": "Methods"}}       # specific section type
```
**Used by:** literature review agent, figure lookup, general semantic search

### `claims_and_findings`

**Contains:** claim chunks + limitation chunks + future_work chunks
**Document field:** Claim/limitation/future_work description text
**Key metadata filters:**
```python
{"chunk_type": {"$eq": "limitation"}}         # critique agent
{"claim_type": {"$eq": "comparative"}}        # comparison agent
{"claim_type": {"$eq": "performance"}}        # performance lookup
{"has_numeric_value": {"$eq": True}}          # contradiction detection
{"$and": [{"claim_type": {"$eq": "comparative"}},
          {"has_numeric_value": {"$eq": True}}]}  # enricher Pass 2
```
**Used by:** comparison agent, critique agent, literature review agent

### `entities_global`

**Contains:** entity chunks (method/dataset/metric/task)
**Document field:** Entity name text
**Key metadata filters:**
```python
{"entity_type": {"$eq": "method"}}            # gap matrix
{"entity_type": {"$eq": "dataset"}}           # gap matrix
{"also_in_papers": {"$ne": ""}}               # cross-paper entities
{"appears_in_n_papers": {"$gte": 2}}          # well-known methods
```
**Used by:** gap detection agent, entity lookup, cross-paper linking

### `researcher_feedback`

**Contains:** nothing at first — written at runtime by agents
**When used:** Every time a researcher rates a retrieved result as
relevant (+1) or irrelevant (-1).
**Used by:** RL training signal for Phase 4

---

## 15. The 2 output JSON files — structure and purpose

### `memory/contradiction_candidates.json`

Written by: `enricher._pass2_contradiction_candidates()`
Updated by: comparison agent (sets `confirmed: true`)
With 1 paper: always empty list `[]`

```json
[
    {
        "claim_a": {
            "chunk_id":      "86ac1a06d297",
            "paper_id":      "stgcn_yu_2018",
            "text":          "STGCN achieves 14 times acceleration...",
            "numeric_value": 14.0
        },
        "claim_b": {
            "chunk_id":      "a3b2c1d4e5f6",
            "paper_id":      "dcrnn_li_2018",
            "text":          "DCRNN training takes 2x longer than baseline...",
            "numeric_value": 2.0
        },
        "shared_entities": ["stgcn"],
        "confirmed": false
    }
]
```

### `memory/gap_matrix.json`

Written by: `enricher._pass3_gap_matrix()`
With 1 paper: 0 gaps, but matrix is built
With 2+ papers: actual gaps appear

```json
{
    "methods": {
        "stgcn":  ["stgcn_yu_2018"],
        "dcrnn":  ["dcrnn_li_2018"]
    },
    "datasets": {
        "pemsd7":   ["stgcn_yu_2018"],
        "metr-la":  ["dcrnn_li_2018"],
        "pems-bay": ["dcrnn_li_2018"]
    },
    "matrix": {
        "stgcn": {
            "pemsd7":   ["stgcn_yu_2018"],
            "metr-la":  [],
            "pems-bay": []
        },
        "dcrnn": {
            "pemsd7":   [],
            "metr-la":  ["dcrnn_li_2018"],
            "pems-bay": ["dcrnn_li_2018"]
        }
    },
    "gaps": [
        {
            "method":          "stgcn",
            "dataset":         "metr-la",
            "gap_score":       3,
            "method_used_in":  ["stgcn_yu_2018"],
            "dataset_used_in": ["dcrnn_li_2018"]
        }
    ]
}
```

---

## 16. File dependency graph

```
pipeline.py
    imports → chunker.py
                imports → utils/paper_id.py
                imports → utils/text_builder.py
    imports → embedder.py
    imports → indexer.py
                (used also by enricher.py and retriever.py)
    imports → enricher.py
                imports → indexer.py (get_collections, collection_counts)
    imports → retriever.py
                imports → embedder.py (embed_query)
                imports → indexer.py  (get_collections)
                imports → query_handler.py (parse_query)

query_handler.py    → no rag imports. Pure Python only.
utils/paper_id.py   → no rag imports. Only stdlib re, pathlib.
utils/text_builder.py → no imports at all. Pure Python string ops.
```

**Safe import order** (no circular imports):
```
utils/paper_id.py
utils/text_builder.py
chunker.py
embedder.py
indexer.py
enricher.py
query_handler.py
retriever.py
pipeline.py
```

---

## 17. How to run everything

### Install
```bash
pip install sentence-transformers chromadb
```

### Index one paper and query
```python
from rag.pipeline import index_paper, enrich, query, status

# Index
index_paper("memory/stgcn_yu_2018")
enrich()

# Check state
status()

# Query
results = query("What are the limitations of STGCN?", top_k=5)
for r in results:
    print(f"#{r['rank']} [{r['chunk_type']}] score={r['score']:.4f}")
    print(f"   {r['document'][:100]}")
    print(f"   paper: {r['paper_id']}")
```

### Index all papers at once
```python
from rag.pipeline import index_all, enrich

index_all("memory")   # finds all subfolders with claims_output.json
enrich()              # always call after indexing
```

### Add a second paper to existing index
```python
from rag.pipeline import index_paper, enrich

index_paper("memory/dcrnn_li_2018")
enrich()   # re-run — updates entity links, contradiction candidates, gap matrix
```

### Re-index a paper (after updating its Phase 1 output)
```python
from rag.indexer import delete_paper
from rag.pipeline import index_paper, enrich

delete_paper("stgcn_yu_2018")        # removes all chunks for this paper
index_paper("memory/stgcn_yu_2018")  # re-indexes from fresh JSONs
enrich()                              # re-run cross-paper analysis
```

### Run all tests in order
```bash
python test_chunker.py          # mock data test
python test_real_stgcn.py       # real data diagnostic
python test_embedder.py         # embedding quality
python test_indexer.py          # ChromaDB storage
python test_enricher.py         # cross-paper enrichment
python test_query_handler.py    # 17 intent detection tests (no DB needed)
python test_retriever.py        # end-to-end retrieval
python test_pipeline.py         # final system test
```
