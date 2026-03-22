# Phase 2: Knowledge Graph Population

This directory contains scripts for **Phase 2** of the research agent pipeline. It consumes the JSON outputs generated in Phase 1 (claims extraction) and builds a fully typed Knowledge Graph using SQLite and NetworkX.

## Overview

The primary goal of this phase is to construct a structured representation of the knowledge extracted from research papers. 

Key features include:
- **Entity Deduplication Strategy**: Uses an LLM (Gemini) for semantic clustering to reliably merge true synonyms and surface variants of technical terms without conflating distinct concepts (e.g., merging "LSTM" and "Long Short-Term Memory", while keeping "Transformer" and "4-layer Transformer" separate).
- **Relational Mapping**: Maps typed entities (Methods, Datasets, Metrics, Tasks) to papers via explicit typed edges (`uses`, `evaluates_on`, `measures_with`, `addresses`).
- **Additional Nodes**: Extracts and stores explicit `LimitationStatement` and `FutureWork` nodes to track research gaps.

## Files

- `kg_population.py`: The main script that runs the knowledge graph population and entity clustering.
- `view_graph.py`: A utility script to visualize the constructed NetworkX knowledge graph.

## Setup & Requirements

Ensure you have the required dependencies installed:

```bash
pip install networkx requests matplotlib
```

An active Gemini API key is recommended for accurate LLM-driven entity clustering. If one is not provided, the pipeline falls back to exact text matching (Tier-1).

## Usage

### 1. Populating the Knowledge Graph

You can populate the knowledge graph by passing the Phase 1 extracted JSON outputs:

```bash
python kg_population.py \
    --inputs path/to/paper1.json path/to/paper2.json \
    --db shared_memory/research.db \
    --gexf shared_memory/knowledge_graph.gexf \
    --api-key YOUR_GEMINI_API_KEY
```

Alternatively, you can provide the API key via an environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
python kg_population.py --inputs ...
```

**Outputs:**
- SQLite database containing papers, entities, edges, claims, limitation statements, and future work.
- `.gexf` file containing the node-edge definitions exported from NetworkX.

### 2. Visualizing the Graph

Once the `.gexf` file is generated, you can visualize the graph using `view_graph.py`.

*Note: You may need to update the path within `view_graph.py` to match the exact location of your generated `.gexf` file (e.g., changing `memory/knowledge_graph.gexf` to `shared_memory/knowledge_graph.gexf`).*

```bash
python view_graph.py
```

This will save a `graph_preview.png` image displaying the knowledge graph, with different node types color-coded for clarity.

## Graph Schema

### Node Types
- `Paper`
- `Method`
- `Dataset`
- `Metric`
- `Task`
- `LimitationStatement`
- `FutureWork`

### Edge Types
- `uses` (Paper -> Method)
- `evaluates_on` (Paper -> Dataset)
- `measures_with` (Paper -> Metric)
- `addresses` (Paper -> Task)
- `has_limitation` (Paper -> LimitationStatement)
- `has_future_work` (Paper -> FutureWork)
