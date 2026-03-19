## Part 3: SciBERT NER Entity Extraction

Takes the JSON outputs from `grobid_parser` and extracts **method, dataset, metric, and task** entities from every section (including detached Tables and Figures) using SciBERT fine-tuned on SciIE.

Every extracted entity carries full provenance: which section it came from, character offsets, and a confidence score. This is what Phase 2 (LLM claim extraction) and all downstream agents depend on.

---

### Setup

```bash
pip install -r requirements.txt
```

The model (`RJuro/SciNERTopic`) downloads automatically from HuggingFace on first run (~440MB). It is cached locally after that.

---

### Usage

#### Single paper 

```bash
# Input: folder produced by grobid_parser (must contain sections.json)
python -m ner_pipeline.cli path/to/grobid_output/

# On GPU (much faster for many papers)
python -m ner_pipeline.cli path/to/grobid_output/ --gpu 0

# Priority sections only (Abstract, Methods, Results, Tables, Figures, etc.) — faster, good for quick checks
python -m ner_pipeline.cli path/to/grobid_output/ --priority-only

# Save output to a different folder
python -m ner_pipeline.cli path/to/grobid_output/ --output path/to/ner_results/
```

#### Batch mode (many papers at once)

```bash
# Process all subfolders under all_papers/ that contain sections.json
# The model loads only once — efficient for large collections
python -m ner_pipeline.cli path/to/all_papers/ --batch
```

### Input

Your `grobid_output/` folder must contain `sections.json` (from `grobid_parser`/Hybrid module).
The pipeline will magically load `tables.json` and `figures.json` natively as pseudo-sections to extract entities from media components invisibly. 

Optionally, `metadata.json` is also read (title, authors, year, doi) to generate a stable `paper_id`.

---

### Output

A single `enriched_entities.json` is written into the same folder as `sections.json`.
