# Agentic AI Research Assistant

This repository contains the foundational modules for an AI-powered capable of modeling the entire research workflow: parsing structured information from PDFs, grounding claims to explicit entities, and eliminating LLM hallucinations.

The project is currently divided into four sequential parts: **Part 1: PDF Parsing**, **Part 2: GROBID Scientific Structure Parsing**, **Part 3: SciBERT NER Entity Extraction**, and **Part 4: LLM Claim Extraction**.

---

## Part 1: PDF Parser

The `module_parser` is a completely local, python-based parser that reads academic PDFs and outputs a strictly structured JSON file while preserving natural reading order. It intelligently extracts text, images, auto-detects synthetic figures (vector graphics), and native tables.

### Setup
1. Install the requirements (Requires PyMuPDF for advanced features, and pdfplumber as a table fallback):
   ```bash
   pip install pymupdf pdfplumber PyPDF2
   ```

### Usage
Run the parser using the modular CLI:

```bash
python -m modular_parser.cli "..\Data\your_paper.pdf" -o "output_folder" -a -i
```

**Flags:**
* `-o "output_folder"` : Where to save the output files (JSON, extracted images, text).
* `-a` : Auto-detect synthetic figures (diagrams) and save them as PNGs.
* `-i` : Extract standard rasterized images from the PDF.
* `--no-structured`: Disables the generation of the `structured_content.json` reading-order file.

**Output:**
The most important output is `output_folder/structured_content.json`, which contains the reading-order extracted text, table matrices, and image bounding boxes.

---

## Part 2: GROBID Scientific Structure Parser

The `grobid_parser` module extracts **semantic scientific structure** from research PDFs using **GROBID (GeneRation Of BIbliographic Data)**.

While the Phase 1 parser focuses on **layout and reading order**, the GROBID parser focuses on **scientific document structure and metadata**.

It converts PDFs into **TEI XML**, then extracts structured sections such as:

* Abstract
* Introduction
* Related Work
* Methods
* Experiments
* Results
* Discussion
* Conclusion

along with complete **paper metadata**.

---

## Requirements

GROBID must be running locally.

The easiest way is using Docker:

```bash
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

Verify that the service is running:

```bash
curl http://localhost:8070/api/isalive
```

Expected response:

```
true
```

---

## Usage

Run the GROBID parser using the CLI:

```bash
python -m grobid_parser.cli "..\Data\your_paper.pdf" -o "output_folder"
```

Or using the project entry point:

```bash
python run_grobid.py "..\Data\your_paper.pdf" -o "output_folder"
```

---

## CLI Options

* `-o "output_folder"` : Directory where parsed results will be saved
* `--grobid-url` : Custom GROBID server URL (default: `http://localhost:8070`)
* `--tei-cache` : Use a cached TEI XML file instead of calling GROBID
* `--no-consolidate` : Disable CrossRef metadata enrichment
* `--check` : Verify that the GROBID service is running
* `-v` : Print preview of extracted section text

Example:

```bash
python -m grobid_parser.cli paper.pdf -o parsed_output -v
```

---

## Output Files

The parser produces several structured outputs:

```
output_folder/
├── metadata.json
├── sections.json
├── tei_raw.xml
└── summary.txt
```

### metadata.json

Contains extracted paper metadata:

```
title
authors
year
venue
doi
abstract
```

---

### sections.json

Contains structured scientific sections extracted from the paper:

```json
[
  {
    "section_type": "Introduction",
    "heading": "1 Introduction",
    "text": "..."
  }
]
```

Sections are automatically classified into scientific categories such as:

* Abstract
* Introduction
* Related Work
* Methods
* Experiments
* Results
* Discussion
* Conclusion

---

### tei_raw.xml

Raw TEI XML returned by GROBID.

This file is saved to allow **re-parsing without reprocessing the PDF**, which significantly speeds up experiments.

---

### summary.txt

A human-readable summary of the extracted paper structure including metadata and section overview.

---
## Part 3: SciBERT NER Entity Extraction

Takes the `sections.json` output from `grobid_parser` and extracts **method, dataset, metric, and task** entities from every section using SciBERT fine-tuned on SciIE.

Every extracted entity carries full provenance: which section it came from, character offsets, and a confidence score. This is what Phase 2 (LLM claim extraction) and all downstream agents depend on.

---

### Setup

```bash
pip install -r requirements_ner.txt
```

The model (`RJuro/SciNERTopic`) downloads automatically from HuggingFace on first run (~440MB). It is cached locally after that.

---

### Usage

#### Single paper 

```bash
# Input: folder produced by grobid_parser (must contain sections.json)
python -m scibert_ner.cli path/to/grobid_output/

# On GPU (much faster for many papers)
python -m scibert_ner.cli path/to/grobid_output/ --gpu 0

# Priority sections only (Abstract, Methods, Results, etc.) — faster, good for quick checks
python -m scibert_ner.cli path/to/grobid_output/ --priority-only

# Save output to a different folder
python -m scibert_ner.cli path/to/grobid_output/ --output path/to/ner_results/
```

#### Batch mode (many papers at once)

```bash
# Process all subfolders under all_papers/ that contain sections.json
# The model loads only once — efficient for large collections
python -m scibert_ner.cli path/to/all_papers/ --batch
```

### Input

Your `grobid_output/` folder must contain `sections.json` (from `grobid_parser`):

Optionally, `metadata.json` is also read (title, authors, year, doi) to generate a stable `paper_id`.

---

### Output

A single `enriched_entities.json` is written into the same folder as `sections.json`:

```json
{
  "paper_id": "doi:10.18653_v1_...",
  "metadata": { "title": "...", "authors": [...], "year": "2023" },

  "ner_summary": {
    "total_entities": 47,
    "entity_type_counts": { "method": 18, "dataset": 9, "metric": 12, "task": 8 },
    "section_coverage": { "Methods": 21, "Results": 14, "Abstract": 7, "Introduction": 5 },
    "elapsed_seconds": 12.4,
    "model_used": "Jason9693/SciERC-scibert-cased"
  },

  "entities": [
    {
      "raw_text": "BERT",
      "text": "BERT",
      "entity_type": "method",
      "confidence": 0.9923,
      "start_char": 142,
      "end_char": 146,
      "section_type": "Methods",
      "section_heading": "3 Methodology",
      "also_in_sections": ["Introduction", "Results"]
    },
    ...
  ],

  "entity_index": {
    "method":  { "bert": [...], "lora": [...] },
    "dataset": { "squad": [...], "glue": [...] },
    "metric":  { "f1": [...], "bleu": [...] },
    "task":    { "question answering": [...] }
  },

  "entities_by_section": {
    "Methods":  [...],
    "Results":  [...],
    "Abstract": [...]
  }
}
```

---

## Part 4: LLM Claim Extractor

The `claim_extractor` module is **Phase 2** of the pipeline. It takes the outputs of Parts 2 and 3 — `sections.json` from GROBID and `enriched_entities.json` from SciBERT NER — and uses **Gemini 2.5 Flash** to extract four types of structured knowledge from every section of the paper:

* **Cleaned entities** — LLM-verified, normalized entities (fixes NER tokenization noise, merges fragments, removes false positives)
* **Claims** — Structured performance, comparative, and methodological claims grounded to specific entities
* **Limitations** — Explicit weakness or constraint statements the paper admits
* **Future work** — Statements proposing directions for future research

Results are merged back into `enriched_entities.json` **and** written as a standalone `claims_output.json`.

---

### Setup

```bash
pip install -r claim_extractor/requirements_claim.txt
```

Requirements: `google-genai>=0.8.0`

A Gemini API key is required. Get one free at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

### Usage

#### Single paper

```bash
# Minimum required: folder produced by grobid_parser (sections.json) and the ner results folder (enriched_entities.json)
python -m claim_extractor.cli --grobid-dir path/to/grobid_output/ --ner-dir path/to/ner_results/

#### Batch mode (many papers at once)

```bash
# Recursively walks --grobid-dir for folders matching grobid_output/sections.json
# and matches them to sibling ner_results/ folders
python -m claim_extractor.cli --grobid-dir path/to/all_papers/ --batch
```

---

### CLI Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--grobid-dir` | `-g` | *(required)* | Folder containing `sections.json` from `grobid_parser` |
| `--ner-dir` | `-n` | same as `--grobid-dir` | Folder containing `enriched_entities.json` from `scibert_ner` |
| `--api-key` | `-k` | `GEMINI_API_KEY` env var | Gemini API key |
| `--rpm` | | `12` | Requests per minute (free tier cap is 15) |
| `--batch` | | `False` | Batch mode: process all papers under `--grobid-dir` |

---

### Input

The pipeline reads two files:

| File | Source | Description |
|---|---|---|
| `sections.json` | `grobid_parser` output | Full text of each paper section |
| `enriched_entities.json` | `scibert_ner` output | NER entities grouped by section, used as hints for the LLM |

---

### Output

#### `claims_output.json` (standalone)

A clean, self-contained output file written next to `enriched_entities.json`

The existing NER output is **extended** with new LLM-extracted fields:

```
llm_entities           — deduplicated, normalized entity list
llm_entity_index       — entities grouped by type and lowercased name
claims                 — all extracted claims
limitations            — all limitation statements
future_work            — all future work statements
llm_results_by_section — per-section raw LLM output
llm_extraction_summary — run statistics
```

```json
{
  "paper_id": "doi:10.18653_v1_...",
  "metadata": { "title": "...", "authors": [...], "year": "2023" },

  "summary": {
    "total_llm_entities": 31,
    "llm_entity_type_counts": { "method": 12, "dataset": 7, "metric": 9, "task": 3 },
    "total_claims": 18,
    "claim_type_counts": { "performance": 10, "comparative": 5, "methodological": 3 },
    "total_limitations": 2,
    "total_future_work": 3,
    "elapsed_seconds": 48.2,
    "model_used": "gemini-2.5-flash",
    "sections_processed": 8
  },

  "entities": [
    {
      "text": "BERT",
      "entity_type": "method",
      "confidence": 0.97,
      "section_type": "Methods",
      "section_heading": "3 Methodology",
      "source": "llm",
      "also_in_sections": ["Introduction", "Results"]
    },
    ...
  ],

  "entity_index": {
    "method":  { "bert": [...], "lora": [...] },
    "dataset": { "squad": [...], "glue": [...] },
    "metric":  { "f1": [...], "bleu": [...] },
    "task":    { "question answering": [...] }
  },

  "claims": [
    {
      "claim_type": "performance",
      "description": "F-LSTM achieves a test perplexity of 35.1 on the One Billion Word Benchmark.",
      "entities_involved": ["F-LSTM", "One Billion Word Benchmark"],
      "value": 35.1,
      "confidence": 0.96,
      "section_type": "Results",
      "section_heading": "5 Results",
      "source": "llm"
    },
    ...
  ],

  "limitations": [
    {
      "text": "The model has not been evaluated on non-English corpora.",
      "entities_involved": [],
      "confidence": 0.88,
      "section_type": "Conclusion",
      "section_heading": "6 Conclusion",
      "source": "llm"
    }
  ],

  "future_work": [
    {
      "text": "We plan to explore multi-task training across all GLUE benchmarks.",
      "entities_involved": ["GLUE"],
      "confidence": 0.91,
      "section_type": "Conclusion",
      "section_heading": "6 Conclusion",
      "source": "llm"
    }
  ]
}
```
