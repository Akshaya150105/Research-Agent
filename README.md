# Agentic AI Research Assistant

This repository contains the foundational modules for an AI-powered assistant capable of modeling the entire research workflow: parsing structured information from PDFs, grounding claims to explicit entities, and eliminating LLM hallucinations.

The project is currently divided into four sequential modules. Each module contains its own dedicated documentation outlining setup and execution steps.

---

## Repository Modules

### [1. Modular PDF Parser](./modular_parser)
A completely local, python-based parser focused on layout mapping. It reads academic PDFs and outputs a strictly structured JSON file while preserving natural reading order. It intelligently extracts text, images, auto-detects synthetic figures (vector graphics), and native tables.

### [2. Hybrid GROBID Scientific Structure Parser](./grobid_parser)
Extracts semantic scientific structure from research PDFs using a hybrid of **GROBID** (for metadata and prose extraction) and **Docling** (for high-accuracy DataFrame table extraction). Produces strict, cleanly decoupled JSON payloads (`sections.json`, `tables.json`, `figures.json`).

### [3. SciBERT NER Entity Extraction](./ner_pipeline)
Takes the decoupled JSON outputs from the GROBID pipeline and extracts **method, dataset, metric, and task** entities across all paragraphs, natively including tables and figures, using a SciBERT model fine-tuned on SciIE.

### [4. LLM Claim Extractor](./claim_extractor)
Takes the outputs of the preceding steps and dynamically synthesizes the text and detached media files to use **Gemini 2.5 Flash** for extracting clean entities, grounded claims, limitations, and future work.

### [5. Knowledge Graph Population](./kg_population)
Consumes the structured JSON outputs from Phase 1 (claims extraction) and builds a fully typed, connected **Knowledge Graph** using SQLite and NetworkX. It uses an LLM for semantic entity clustering and deduplication to reliably map synonymous technical terms across distinct papers into a shared canonical form without losing intentionally nuanced distinctions.
