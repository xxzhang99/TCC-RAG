"""
Centralized configuration for the TCC-RAG pipeline.
All paths, API keys, model IDs, and GPU settings are managed here.
"""
import os

# ============================================================
# GPU Settings
# ============================================================
# Set CUDA_VISIBLE_DEVICES before importing torch
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# ============================================================
# API Configuration
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"

# ============================================================
# Model Configuration
# ============================================================
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SENTENCE_TRANSFORMER_MODEL = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BGE_M3_MODEL = "BAAI/bge-m3"

# ============================================================
# Dataset Paths (MultiTQ)
# ============================================================
# Directory layout (relative to this file):
#   new_code/
#     config.py          (this file)
#     dataset/MultiTQ/
#       kg/              (KG files)
#       questions/       (raw question files)
#       outputs/         (all pipeline inputs/outputs)
DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "MultiTQ")
)

# Knowledge Graph
KG_FULL_PATH           = os.path.join(DATASET_ROOT, "kg/full.txt")
KG_GRAPH_PATH          = os.path.join(DATASET_ROOT, "kg/graph.pkl")           # built by step2
KG_SUBJ_RELATIONS_PATH = os.path.join(DATASET_ROOT, "kg/subj_relations.pkl")  # built by step2
KG_OBJ_RELATIONS_PATH  = os.path.join(DATASET_ROOT, "kg/obj_relations.pkl")   # built by step2
KG_ENTITY2ID_PATH      = os.path.join(DATASET_ROOT, "kg/entity2id.json")
KG_RELATION2ID_PATH    = os.path.join(DATASET_ROOT, "kg/relation2id.json")
KG_ENT_ID_PATH         = os.path.join(DATASET_ROOT, "kg/tkbc_processed_data/ent_id")

# Raw questions
QUESTIONS_TEST_PATH = os.path.join(DATASET_ROOT, "questions/test.json")

# Pipeline outputs directory
OUTPUTS_DIR = os.path.join(DATASET_ROOT, "outputs")

# Step 1 output: merged entity + relation file (pipeline input for step 3+)
QUESTIONS_PROCESSED_PATH = os.path.join(OUTPUTS_DIR, "entities_relations_merged_v1.json")

# Step 3 outputs
GRAPH_RETRIEVE_OUTPUT    = os.path.join(OUTPUTS_DIR, "graph_retrieve_results.json")
SEMANTIC_RETRIEVE_OUTPUT = os.path.join(OUTPUTS_DIR, "semantic_retrieve_results.json")

# Step 4 output
CAUSAL_FILTER_OUTPUT = os.path.join(OUTPUTS_DIR, "causal_llm_filtered_results.json")

# Step 5 output
PREDICT_OUTPUT = os.path.join(OUTPUTS_DIR, "predictions.json")

# Pre-computed prediction results (for direct evaluation)
PREDICT_OUTPUT_DEEPSEEK = os.path.join(OUTPUTS_DIR, "result_Graph_Semantic_Causal_DS_async_all.json")
PREDICT_OUTPUT_LLAMA    = os.path.join(OUTPUTS_DIR, "result_Graph_Semantic_Causal_LLama3_ll.json")

# ============================================================
# NER / Entity Extraction  (Step 1 intermediates → outputs/)
# ============================================================
NER_OUTPUT_DIR = OUTPUTS_DIR

# Step 1a: LLaMA NER + KG grounding
NER_ENTITY_OUTPUT = os.path.join(NER_OUTPUT_DIR, "entities.json")

# Step 1b: Entity comparison / LLM arbitration
ENTITY_CONFLICT_OUTPUT      = os.path.join(NER_OUTPUT_DIR, "entity_conflicts.json")
ENTITY_COMPARE_OUTPUT       = os.path.join(NER_OUTPUT_DIR, "entities_compared.json")
ENTITY_COMPARE_ERROR_OUTPUT = os.path.join(NER_OUTPUT_DIR, "entity_compare_errors.json")

# Step 1c: Relation extraction
NER_RELATION_OUTPUT = os.path.join(NER_OUTPUT_DIR, "relations.json")

# Step 1d: Merged entity + relation output (== QUESTIONS_PROCESSED_PATH)
NER_MERGED_OUTPUT = QUESTIONS_PROCESSED_PATH

# Sentence-embedding cache for KG entities (generated on first run)
ENTITY_EMBED_CACHE = os.path.join(DATASET_ROOT, "kg/kg_entity_embeds.npy")

# ============================================================
# Generation Parameters
# ============================================================
# API generation
API_TEMPERATURE = 0.0
API_TOP_P = 0.9
API_MAX_CONCURRENCY = 32
API_SAVE_EVERY = 1000

# Local LLM generation
LOCAL_TEMPERATURE = 0.3
LOCAL_TOP_P = 0.9
LOCAL_BATCH_SIZE = 8
LOCAL_MAX_NEW_TOKENS = 256

# NER generation
NER_BATCH_SIZE = 64
NER_MAX_NEW_TOKENS = 128

# Semantic retrieval
SEMANTIC_TOPK = 20
SEMANTIC_EMBEDDING_SIZE = 1024

# Causal filter
CAUSAL_TRIGGER_L0 = 30
CAUSAL_TRIGGER_ALPHA = 0.5
CAUSAL_TRIGGER_BETA = 0.5
CAUSAL_TRIGGER_TAU = 0.75

# Entity matching
ENTITY_SEMANTIC_THRESHOLD = 0.85
