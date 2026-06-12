"""
TCC-RAG Pipeline - Main Entry Point (MultiTQ)

Runs the full pipeline end-to-end:
  Step 1: Entity & Relation Recognition
  Step 2: Graph Construction
  Step 3: Fact Retrieval (Graph + Semantic)
  Step 4: Causal Fact Filtering
  Step 5: Answer Generation (API or Local)

Usage:
    # Run full pipeline with API prediction
    python -m new_code.main

    # Run full pipeline with local LLM prediction
    python -m new_code.main --local

    # Run from a specific step (skip earlier steps)
    python -m new_code.main --start-step 3

    # Skip causal filtering
    python -m new_code.main --no-causal
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_ROOT, OUTPUTS_DIR, NER_OUTPUT_DIR,
    KG_FULL_PATH, KG_GRAPH_PATH, KG_ENT_ID_PATH, KG_ENTITY2ID_PATH,
    QUESTIONS_TEST_PATH, QUESTIONS_PROCESSED_PATH,
    NER_ENTITY_OUTPUT, ENTITY_COMPARE_OUTPUT,
    NER_RELATION_OUTPUT, NER_MERGED_OUTPUT,
    GRAPH_RETRIEVE_OUTPUT, SEMANTIC_RETRIEVE_OUTPUT, CAUSAL_FILTER_OUTPUT,
    PREDICT_OUTPUT, PREDICT_OUTPUT_DEEPSEEK, PREDICT_OUTPUT_LLAMA
)


def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}\n")


def check_file_exists(path, name):
    """Check if a required file exists."""
    if not os.path.exists(path):
        print(f"  [ERROR] {name} not found: {path}")
        return False
    print(f"  [OK] {name}: {path}")
    return True


def run_step1():
    """Step 1: Entity & Relation Recognition."""
    print_step(1, "Entity & Relation Recognition")

    # Check prerequisites
    if not check_file_exists(KG_ENT_ID_PATH, "KG entity list"):
        return False
    if not check_file_exists(KG_ENTITY2ID_PATH, "Entity2ID map"):
        return False
    if not check_file_exists(QUESTIONS_TEST_PATH, "Test questions"):
        return False

    # 1a: Entity Extraction (LLaMA NER + KG grounding)
    print("\n  [1a] Extracting entities with LLaMA NER...")
    from step1_entity_relation.entity_extract import main as entity_main
    entity_main()

    # 1b: Entity Comparison & LLM Arbitration
    print("\n  [1b] Comparing NER vs LLaMA entities and arbitrating conflicts...")
    from step1_entity_relation.entity_compare import (
        compare_entity_results, load_llama_model, process_entity_compare
    )
    from config import ENTITY_CONFLICT_OUTPUT, ENTITY_COMPARE_ERROR_OUTPUT
    compare_entity_results(
        ner_path=NER_ENTITY_OUTPUT,
        llama_path=NER_ENTITY_OUTPUT,   # same file; override if LLaMA NER is separate
        output_path=ENTITY_CONFLICT_OUTPUT,
    )
    model, tokenizer = load_llama_model()
    process_entity_compare(
        conflict_path=ENTITY_CONFLICT_OUTPUT,
        ner_path=NER_ENTITY_OUTPUT,
        output_path=ENTITY_COMPARE_OUTPUT,
        error_path=ENTITY_COMPARE_ERROR_OUTPUT,
        model=model,
        tokenizer=tokenizer,
    )

    # 1c: Relation Extraction (uses arbitrated entities)
    print("\n  [1c] Extracting relations with LLaMA...")
    from step1_entity_relation.relation_extract import (
        load_llama_model as load_rel_model, process_relations
    )
    rel_model, rel_tokenizer = load_rel_model()
    process_relations(ENTITY_COMPARE_OUTPUT, NER_RELATION_OUTPUT,
                      rel_model, rel_tokenizer)

    # 1d: Merge Results
    print("\n  [1d] Merging entity + relation results...")
    from step1_entity_relation.merge_results import merge_entities_and_relations
    merge_entities_and_relations(ENTITY_COMPARE_OUTPUT, NER_RELATION_OUTPUT,
                                  NER_MERGED_OUTPUT)

    return True


def run_step2():
    """Step 2: Graph Construction."""
    print_step(2, "Graph Construction")

    if not check_file_exists(KG_FULL_PATH, "KG triples (full.txt)"):
        return False

    print("\n  Building indexed graph structures...")
    from step2_graph_construct.build_graph import main as build_main
    build_main()

    return True


def run_step3():
    """Step 3: Fact Retrieval (Graph + Semantic)."""
    print_step(3, "Fact Retrieval")

    # Check prerequisites
    if not check_file_exists(KG_GRAPH_PATH, "KG graph pickle"):
        print("  -> Run Step 2 first to build the graph.")
        return False

    question_path = QUESTIONS_PROCESSED_PATH
    if not os.path.exists(question_path):
        # Fallback: check if NER merged output exists
        question_path = NER_MERGED_OUTPUT
    if not check_file_exists(question_path, "Processed questions"):
        print("  -> Run Step 1 first to process questions.")
        return False

    # 3a: Graph Retrieval
    print("\n  [3a] Running graph-based retrieval...")
    from step3_retrieval.graph_retriever import GraphRetriever
    import json

    retriever = GraphRetriever()
    with open(question_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    retriever.graph_retrieve(questions, save_path=GRAPH_RETRIEVE_OUTPUT)

    # 3b: Semantic Retrieval
    print("\n  [3b] Running semantic retrieval (BGE-M3)...")
    from step3_retrieval.semantic_retriever import retrieve_semantic_facts

    fact_list = []
    with open(KG_FULL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace("_", " ").split('\t')
            if len(parts) >= 4:
                fact_list.append(f"{parts[0]} {parts[1]} {parts[2]} in {parts[3]}")

    retrieve_semantic_facts(
        question_json=questions,
        fact_list=fact_list,
        output_path=SEMANTIC_RETRIEVE_OUTPUT
    )

    return True


def run_step4():
    """Step 4: Causal Fact Filtering."""
    print_step(4, "Causal Fact Filtering")

    if not check_file_exists(GRAPH_RETRIEVE_OUTPUT, "Graph retrieval results"):
        print("  -> Run Step 3 first.")
        return False

    print("\n  Running causal fact filtering...")
    from step4_causal_filter.causal_filter import process_causal_filter
    process_causal_filter()

    return True


def run_step5_api():
    """Step 5: Answer Generation (API mode)."""
    print_step(5, "Answer Generation (DeepSeek API)")

    if not check_file_exists(GRAPH_RETRIEVE_OUTPUT, "Graph retrieval results"):
        return False
    if not check_file_exists(SEMANTIC_RETRIEVE_OUTPUT, "Semantic retrieval results"):
        return False
    if not check_file_exists(CAUSAL_FILTER_OUTPUT, "Causal filter results"):
        print("  [WARN] Causal results not found, will proceed without causal facts.")

    print("\n  Running async API prediction...")
    from step5_predict.predict_api import process_predictions
    process_predictions()

    return True


def run_step6(target="all"):
    """Step 6: Evaluation.

    target:
      'deepseek' - evaluate DeepSeek results only
      'llama'    - evaluate LLaMA results only
      'all'      - evaluate both (default)
      'custom'   - evaluate PREDICT_OUTPUT (generated by step 5)
    """
    from step6_evaluate.evaluate import evaluate_file

    print_step(6, "Evaluation (Hit@1)")

    targets = {
        "deepseek": ("DeepSeek-v3",    PREDICT_OUTPUT_DEEPSEEK),
        "llama":    ("LLaMA 3.1 8B",   PREDICT_OUTPUT_LLAMA),
        "custom":   ("Step-5 output",  PREDICT_OUTPUT),
    }

    if target == "all":
        run_targets = [("DeepSeek-v3",  PREDICT_OUTPUT_DEEPSEEK),
                       ("LLaMA 3.1 8B", PREDICT_OUTPUT_LLAMA)]
    else:
        name, path = targets[target]
        run_targets = [(name, path)]

    for name, path in run_targets:
        if not check_file_exists(path, f"{name} predictions"):
            continue
        print(f"\n  >>> {name}")
        evaluate_file(path, k=1)

    return True
    """Step 5: Answer Generation (Local LLM mode)."""
    print_step(5, "Answer Generation (Local LLaMA)")

    if not check_file_exists(GRAPH_RETRIEVE_OUTPUT, "Graph retrieval results"):
        return False
    if not check_file_exists(SEMANTIC_RETRIEVE_OUTPUT, "Semantic retrieval results"):
        return False
    if not check_file_exists(CAUSAL_FILTER_OUTPUT, "Causal filter results"):
        print("  [WARN] Causal results not found, will proceed without causal facts.")

    print("\n  Loading LLaMA model and running prediction...")
    from step5_predict.predict_local import process_predictions, load_llama_model
    model, tokenizer = load_llama_model()
    process_predictions(model=model, tokenizer=tokenizer)

    return True


def main():
    parser = argparse.ArgumentParser(description="TCC-RAG Full Pipeline (MultiTQ)")
    parser.add_argument("--local", action="store_true",
                        help="Use local LLaMA for prediction instead of DeepSeek API")
    parser.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help="Start from this step (skip earlier steps)")
    parser.add_argument("--no-causal", action="store_true",
                        help="Skip causal fact filtering (Step 4)")
    parser.add_argument("--eval-only", choices=["deepseek", "llama", "all", "custom"],
                        default=None,
                        help="Only run evaluation on pre-computed results (skip Steps 1-5)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           TCC-RAG Pipeline (MultiTQ Dataset)            ║
╠══════════════════════════════════════════════════════════╣
║  Step 1: Entity & Relation Recognition                 ║
║    1a: LLaMA NER + KG entity grounding                 ║
║    1b: Entity comparison & LLM arbitration             ║
║    1c: Relation extraction                             ║
║    1d: Merge entity + relation results                 ║
║  Step 2: Graph Construction (Indexed KG)               ║
║  Step 3: Fact Retrieval (Graph + Semantic)              ║
║  Step 4: Causal Fact Filtering (CSEF)                  ║
║  Step 5: Answer Generation ({'Local LLaMA' if args.local else 'DeepSeek API'}){'       ' if args.local else '  '}║
║  Step 6: Evaluation (Hit@1)                            ║
╚══════════════════════════════════════════════════════════╝

  Dataset root: {DATASET_ROOT}
  Start step:   {args.start_step if not args.eval_only else 'N/A (--eval-only)'}
  Causal filter: {'Disabled' if args.no_causal else 'Enabled'}
  Prediction:   {'Local LLaMA' if args.local else 'DeepSeek API'}
""")

    # --eval-only: skip straight to step 6
    if args.eval_only:
        run_step6(target=args.eval_only)
        return

    # Ensure output directories exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(NER_OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    success = True

    # Run pipeline from start_step
    if args.start_step <= 1:
        success = run_step1()
        if not success:
            print("\n[FAILED] Step 1 failed. Aborting.")
            return

    if args.start_step <= 2:
        success = run_step2()
        if not success:
            print("\n[FAILED] Step 2 failed. Aborting.")
            return

    if args.start_step <= 3:
        success = run_step3()
        if not success:
            print("\n[FAILED] Step 3 failed. Aborting.")
            return

    if args.start_step <= 4 and not args.no_causal:
        success = run_step4()
        if not success:
            print("\n[FAILED] Step 4 failed. Aborting.")
            return
    elif args.no_causal:
        # Create empty causal results file so step 5 doesn't fail
        import json
        os.makedirs(os.path.dirname(CAUSAL_FILTER_OUTPUT), exist_ok=True)
        if not os.path.exists(CAUSAL_FILTER_OUTPUT):
            with open(CAUSAL_FILTER_OUTPUT, "w") as f:
                json.dump([], f)
            print("\n  [INFO] Causal filtering skipped. Empty placeholder created.")

    if args.start_step <= 5:
        if args.local:
            success = run_step5_local()
        else:
            success = run_step5_api()
        if not success:
            print("\n[FAILED] Step 5 failed. Aborting.")
            return

    # Step 6: Evaluation (always runs after step 5, or via --start-step 6)
    if args.start_step <= 6:
        run_step6(target="custom")

    elapsed = time.time() - start_time
    print(f"""
{'='*60}
  Pipeline Complete!
  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)
  Results: {PREDICT_OUTPUT}

  To evaluate pre-computed results:
    python -m new_code.main --eval-only all        # DeepSeek + LLaMA
    python -m new_code.main --eval-only deepseek
    python -m new_code.main --eval-only llama
{'='*60}
""")


if __name__ == "__main__":
    main()
