# Agentic AI Research Assistant — RAG Pipeline (Phase 2)

A local vector database pipeline that turns structured paper JSONs from Phase 1 into a searchable, cross-paper research memory. Built for 5–6 research papers, designed to power six downstream AI agents.

---

## What this system does

Takes the structured output from Phase 1 (PDF parsing + NER + LLM claim extraction) and:

1. **Chunks** each paper into typed units — claims, limitations, future work, entities, sections, figures
2. **Embeds** every chunk using a local BGE model (768 dimensions, no API key needed)
3. **Indexes** chunks into ChromaDB across 4 collections with rich metadata
4. **Enriches** the index with cross-paper entity links, contradiction candidates, and a research gap matrix
5. **Retrieves** answers to natural language questions using semantic search + cross-encoder re-ranking

---

## Project structure

```
Research-Agent/
├── memory/                          ← one folder per paper (Phase 1 output)
│   └── stgcn_yu_2018/
│       ├── claims_output.json       ← claims, limitations, entities, metadata
│       ├── sections.json            ← full section text
│       └── figures.json             ← figure captions
│
├── rag/                             ← the full pipeline
│   ├── pipeline.py                  ← entry point (index + query)
│   ├── chunker.py                   ← step 1: JSON → chunk dicts
│   ├── embedder.py                  ← step 2: chunks → 768-dim vectors
│   ├── indexer.py                   ← step 3: chunks → ChromaDB
│   ├── enricher.py                  ← step 4: cross-paper linking
│   ├── query_handler.py             ← step 5: question → query plan
│   ├── retriever.py                 ← step 6: query plan → ranked results
│   ├── chroma_store/                ← ChromaDB data (auto-created, gitignore this)
│   └── utils/
│       ├── paper_id.py              ← shared ID convention
│       └── text_builder.py          ← embed text construction per chunk type
│
├── test_chunker.py                  ← mock data test
├── test_real_stgcn.py               ← real data diagnostic
├── test_embedder.py
├── test_indexer.py
├── test_enricher.py
├── test_query_handler.py
├── test_retriever.py
└── test_pipeline.py                 ← final end-to-end test
```

---

## Install

```bash
pip install sentence-transformers chromadb
```

That's all. No API keys. Everything runs locally.

**First run note:** The embedding model (`BAAI/bge-base-en-v1.5`, ~440 MB) and cross-encoder (`ms-marco-MiniLM-L-6-v2`, ~85 MB) download automatically on first use and cache to `~/.cache/huggingface/`.

---

## Quick start

### Index one paper

```python
from rag.pipeline import index_paper, enrich

index_paper("memory/stgcn_yu_2018")
enrich()
```

### Index all papers at once

```python
from rag.pipeline import index_all, enrich

index_all("memory")   # indexes every subfolder with claims_output.json
enrich()              # always run after indexing all papers
```

### Query

```python
from rag.pipeline import query

results = query("What are the limitations of STGCN?")

for r in results:
    print(f"[{r['chunk_type']}] score={r['score']:.4f}")
    print(f"  {r['document']}")
    print(f"  paper: {r['paper_id']}")
```

### Check system status

```python
from rag.pipeline import status
status()
```

---

## Adding a new paper

```python
from rag.pipeline import index_paper, enrich

# 1. Make sure Phase 1 has run and output is in memory/
index_paper("memory/dcrnn_li_2018")

# 2. Always re-run enrich after adding papers
#    This updates cross-paper links, contradiction candidates, gap matrix
enrich()
```

**Important:** Agree on the folder name with your collaborator before running Phase 1. The folder name becomes the `paper_id` used in both ChromaDB and the knowledge graph.

---

## ChromaDB collections

| Collection | Contents | Used for |
|---|---|---|
| `paper_sections` | Full section text + figure captions | Semantic search, literature review |
| `claims_and_findings` | Claims, limitations, future work | Comparison, critique, contradiction detection |
| `entities_global` | All method/dataset/metric/task entities | Gap detection, cross-paper linking |
| `researcher_feedback` | Relevance ratings from researcher | RL training signal (written at runtime) |

---

## Chunk types and collection routing

| Chunk type | Collection | Source field |
|---|---|---|
| `claim` | `claims_and_findings` | `claims_output.json → claims` |
| `limitation` | `claims_and_findings` | `claims_output.json → limitations` |
| `future_work` | `claims_and_findings` | `claims_output.json → future_work` |
| `entity` | `entities_global` | `claims_output.json → llm_entities` |
| `section` | `paper_sections` | `sections.json` |
| `figure` | `paper_sections` | `figures.json` |

---

## Query intent detection

The query handler automatically detects what kind of question is being asked and applies the right metadata filter before semantic search:

| Question type | Example | Filter applied |
|---|---|---|
| Limitation | "What are the weaknesses of STGCN?" | `chunk_type = limitation` |
| Future work | "What next steps do the authors suggest?" | `chunk_type = future_work` |
| Comparison | "How does STGCN compare to GCGRU?" | `claim_type = comparative` |
| Performance | "What RMSE does STGCN achieve?" | `claim_type = performance` |
| Gap | "What hasn't been tested on METR-LA?" | reads `gap_matrix.json` directly |
| Figure | "Show me the architecture diagram" | `chunk_type = figure` |
| Literature review | "Summarize what papers say about GCN" | semantic search only |
| General | anything else | semantic search only |

You can also filter by paper: `"What are the limitations of stgcn_yu_2018?"` automatically adds a `paper_id` filter.

---

## Retrieval pipeline (two stages)

```
Question
    │
    ▼
query_handler.py     detect intent → build metadata filter
    │
    ▼
embedder.py          embed question with BGE query prefix
    │
    ▼
ChromaDB ANN         fetch n candidates matching the filter
    │
    ▼
cross-encoder        re-score every (question, candidate) pair precisely
    │
    ▼
confidence weighting multiply score × chunk confidence
    │
    ▼
top-k results        sorted by final score, with full provenance
```

---

## Enrichment passes (run after all papers indexed)

**Pass 1 — Entity linking**
Finds entities that appear in multiple papers. Updates `also_in_papers` and `appears_in_n_papers` on entity chunks. Powers: "which papers use method X?"

**Pass 2 — Contradiction candidates**
Finds comparative claims with numeric values from different papers that mention the same methods. Writes `memory/contradiction_candidates.json`. Powers: the comparison agent.

**Pass 3 — Gap matrix**
Builds a method × dataset co-occurrence matrix. Empty cells = research gaps. Writes `memory/gap_matrix.json`. Powers: the gap detection agent.

---

## Running the tests

Run these in order. Each test validates one step before you build the next.

```bash
# step 1 — chunker (mock data)
python test_chunker.py

# step 1 — chunker (real STGCN data)
python test_real_stgcn.py

# step 2 — embedder
python test_embedder.py

# step 3 — indexer
python test_indexer.py

# step 4 — enricher
python test_enricher.py

# step 5 — query handler (no ChromaDB needed)
python test_query_handler.py

# step 6 — retriever
python test_retriever.py

# final — complete pipeline
python test_pipeline.py
```

**Expected results for the STGCN paper:**

| Step | Expected output |
|---|---|
| Chunker | 379 total chunks: 141 claims, 18 limitations, 4 future_work, 202 entities, 8 sections, 6 figures |
| Embedder | 379 × 768-dim unit vectors, ~60–120s on CPU |
| Indexer | paper_sections: 14, claims_and_findings: 163, entities_global: 202 |
| Enricher | 0 cross-paper results (expected with 1 paper) |
| Query handler | 17/17 intent tests pass |
| Retriever | 14x acceleration claim ranked #1 for training speed query |

---

## Coordination with knowledge graph (collaborator)

Both the RAG pipeline and the knowledge graph use the same `paper_id`. The folder name is the `paper_id`.

```python
# both sides call this on the same folder
from rag.utils.paper_id import paper_id_from_folder
paper_id = paper_id_from_folder("memory/stgcn_yu_2018")
# → "stgcn_yu_2018"
```

**Rule:** Agree on the folder name for each new paper before either side starts processing. One message, five seconds. That's the entire coordination protocol.

Do not derive the paper_id from the title, use sequential counters, or generate UUIDs — these all break the join between ChromaDB chunks and SQLite graph nodes.

---

## What changes when you add paper 2

After indexing a second paper and running `enrich()`:

- `also_in_papers` on entity chunks gets populated — e.g. "STGCN" appears in both papers
- `contradiction_candidates.json` gets real pairs — claims from different papers about the same method
- `gap_matrix.json` shows real gaps — method/dataset combinations no paper has tried

---

## What comes next (Phase 3)

The agents layer reads from this pipeline:

```python
from rag.pipeline import query

# comparison agent
results = query("How does STGCN compare to DCRNN?")

# critique agent
results = query("What are the limitations of STGCN?")

# gap detection agent
results = query("What research gaps exist?")

# literature review agent
results = query("Summarize the approaches to traffic forecasting")
```

Each agent wraps `query()` with an LLM call that synthesizes the retrieved chunks into a structured answer with citations.

---

## .gitignore additions

```
rag/chroma_store/
__pycache__/
*.pyc
.venv/
```
