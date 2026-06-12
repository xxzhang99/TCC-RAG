# TCC-RAG

Temporal Causal Chain-guided Retrieval-Augmented Generation for Temporal Knowledge Graph Question Answering.

## Pipeline Overview

```
Step 1: Entity & Relation Recognition
  ├─ 1a: LLaMA NER + semantic/fuzzy KG entity grounding
  ├─ 1b: Entity conflict detection & LLM arbitration
  ├─ 1c: Relation structure extraction
  └─ 1d: Merge entities + relations

Step 2: Graph Construction
  └─ Build KG (MultiDiGraph) + indexed subj/obj relation structures

Step 3: Fact Retrieval
  ├─ Graph Retriever: route by qtype, retrieve temporal facts from indexed structures
  └─ Semantic Retriever: BGE-M3 dense embedding + FAISS (fallback for empty graph results)

Step 4: Causal Fact Filtering (CSEF)
  └─ Trigger detection → causal evidence scoring → fact minimization

Step 5: Answer Generation
  ├─ API mode: DeepSeek async with checkpoint (for large-scale runs)
  └─ Local mode: LLaMA 3.1 batched inference (GPU-only, no API dependency)

Step 6: Evaluation
  └─ Hit@k metrics by answer_type, qlabel, qtype
```

## Directory Structure

```
new_code/
├── config.py                    # All paths, API keys, model IDs, hyperparameters
├── main.py                      # Full pipeline entry point
├── utils.py                     # Shared utilities
├── dataset/MultiTQ/
│   ├── kg/
│   │   ├── full.txt             # Raw KG triples: S\tR\tO\tT
│   │   ├── entity2id.json
│   │   ├── relation2id.json
│   │   └── tkbc_processed_data/
│   ├── questions/
│   │   └── test.json            # Raw MultiTQ test questions
│   └── outputs/                 # All pipeline outputs (auto-created)
├── step1_entity_relation/
│   ├── entity_extract.py        # LLaMA NER + semantic/fuzzy entity grounding
│   ├── entity_compare.py        # Entity conflict detection & LLM arbitration
│   ├── relation_extract.py      # LLaMA relation structure extraction
│   └── merge_results.py         # Merge entities + relations into final JSON
├── step2_graph_construct/
│   └── build_graph.py           # Build indexed KG structures (production)
├── step3_retrieval/
│   ├── graph_retriever.py       # Graph-based temporal retrieval (6 query types)
│   └── semantic_retriever.py    # BGE-M3 semantic retrieval (fallback)
├── step4_causal_filter/
│   ├── causal_evidence_filter.py # CSEF: trigger + causal scoring + dedup
│   └── causal_filter.py          # LLM-based fact minimization + consistency check
├── step5_predict/
│   ├── predict_api.py           # DeepSeek API (async, checkpoint, retry)
│   └── predict_local.py         # LLaMA 3.1 local inference
└── step6_evaluate/
    └── evaluate.py              # Hit@k evaluation metrics
```

## Quick Start

### 1. Configuration

`config.py` is pre-configured. The dataset root is automatically resolved relative to this file (`new_code/dataset/MultiTQ`). You only need to set:

- `DEEPSEEK_API_KEY`: your API key (or set via env var `DEEPSEEK_API_KEY`)
- `CUDA_VISIBLE_DEVICES`: GPU devices to use (default: `"0,1"`)

### 2. Evaluate Pre-computed Results

Pre-computed results for DeepSeek-v3 and LLaMA 3.1 8B are included in `dataset/MultiTQ/outputs/`.

```bash
# Evaluate both models
python -m main --eval-only all

# Evaluate DeepSeek only
python -m main --eval-only deepseek

# Evaluate LLaMA only
python -m main --eval-only llama
```

### 3. Run the Full Pipeline

```bash
# Run full pipeline with DeepSeek API (Steps 1-6)
python -m main

# Run full pipeline with local LLaMA
python -m main --local

# Start from a specific step (skip earlier steps)
python -m main --start-step 3

# Skip causal filtering (Step 4)
python -m main --no-causal
```

## Query Types Supported

| QType | Description | Example |
|-------|-------------|---------|
| `equal` | Facts at a specific time | "Who cooperated with China on 2020-05-15?" |
| `equal_multi` | Facts in the same time period as reference | "Who praised Italy in the same month as China?" |
| `first_last` | Earliest or latest fact | "When did China first visit Japan?" |
| `before_after` | Facts before/after a time boundary | "After 2012, who accused Colombia?" |
| `before_last` | Last fact before a boundary | "Before X, who last did Y?" |
| `after_first` | First fact after a boundary | "After X, who first did Y?" |
