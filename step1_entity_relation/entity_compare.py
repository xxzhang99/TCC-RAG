"""
Step 1b: Entity Comparison and LLM Arbitration.

When NER-extracted entities and LLaMA-extracted entities disagree,
this step uses LLaMA 3.1 to arbitrate and select the correct entity set.

Pipeline:
  1. compare_entity_results: detect conflicts between NER and LLaMA entity sets.
  2. process_entity_compare: run LLaMA arbitration on conflicting items and
     write the resolved entities back.

The output (ENTITY_COMPARE_OUTPUT) replaces NER_ENTITY_OUTPUT as the input
to Step 1c (relation extraction).

Usage:
    python -m new_code.step1_entity_relation.entity_compare
"""
import json
import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLAMA_MODEL_ID,
    LOCAL_BATCH_SIZE, LOCAL_MAX_NEW_TOKENS,
    NER_ENTITY_OUTPUT, NER_OUTPUT_DIR,
    ENTITY_CONFLICT_OUTPUT, ENTITY_COMPARE_OUTPUT, ENTITY_COMPARE_ERROR_OUTPUT,
)


# ============================================================
# Helpers
# ============================================================

def normalize_entities(entities) -> set:
    """Normalise an entity list to a lowercase stripped set for comparison."""
    if not entities:
        return set()
    return {e.strip().lower() for e in entities if e}


# ============================================================
# Stage 1: Detect NER / LLaMA Conflicts
# ============================================================

def compare_entity_results(ner_path, llama_path, output_path):
    """
    Compare NER entity results with LLaMA entity results.
    Records items where the two sets differ into output_path.

    Args:
        ner_path:    Path to NER entity output JSON (list of items with "entities").
        llama_path:  Path to LLaMA NER entity output JSON.
        output_path: Path to write conflict records.
    """
    with open(ner_path, "r", encoding="utf-8") as f:
        ner_data = json.load(f)
    with open(llama_path, "r", encoding="utf-8") as f:
        llama_data = json.load(f)

    ner_dict   = {item["quid"]: item for item in ner_data}
    llama_dict = {item["quid"]: item for item in llama_data}

    conflict_records = []
    for quid, ner_item in ner_dict.items():
        if quid not in llama_dict:
            continue
        llama_item = llama_dict[quid]
        ner_set    = normalize_entities(ner_item.get("entities", []))
        llama_set  = normalize_entities(llama_item.get("entities", []))
        if ner_set != llama_set:
            conflict_records.append({
                "quid":           quid,
                "question":       ner_item.get("question"),
                "ner_entities":   list(ner_item.get("entities", [])),
                "llama_entities": list(llama_item.get("entities", [])),
                "answers":        ner_item.get("answers", []),
                "qtype":          ner_item.get("qtype"),
                "qlabel":         ner_item.get("qlabel"),
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conflict_records, f, indent=4, ensure_ascii=False)
    print(f"Detected {len(conflict_records)} entity conflicts. Saved to {output_path}")
    return conflict_records


# ============================================================
# Model Loading
# ============================================================

def load_llama_model(model_id=None):
    """Load LLaMA model for entity arbitration."""
    model_id = model_id or LLAMA_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ============================================================
# Batched Generation
# ============================================================

def batched_generate(model, tokenizer, prompts,
                     batch_size=None, max_new_tokens=None):
    """Generate responses in batches using LLaMA."""
    batch_size     = batch_size     or LOCAL_BATCH_SIZE
    max_new_tokens = max_new_tokens or LOCAL_MAX_NEW_TOKENS

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Entity Arbitration"):
        batch = prompts[i: i + batch_size]
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        inputs = tokenizer(
            chat_texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
            )

        for j, output in enumerate(outputs):
            gen_ids = output[len(inputs.input_ids[j]):]
            text    = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

        torch.cuda.empty_cache()
    return results


# ============================================================
# Arbitration Prompt
# ============================================================

_ARBITRATION_PROMPT = """You are an expert in entity disambiguation for temporal knowledge graphs.
Your goal is to determine which entity candidate correctly represents the intended real-world entities in a given question.

You will be provided with:
- A question from a temporal knowledge graph QA dataset.
- Two sets of entities:
  - "NER_Entities": obtained from a named-entity recognition and matching model.
  - "LLM_Entities": obtained from a large language model.

Carefully compare both sets of entities and the question meaning, then decide which set is more accurate or complete.
If neither fits, return `"chosen_entity": null` and `"type": "None"`.

Respond **only** with a valid JSON object in this format:
{{
    "chosen_entity": <list_of_entities_or_null>,
    "type": "<NER | LLM | Both | None>"
}}

### Rules
- `"type"` =
  - "NER" if the NER-based entities are more accurate,
  - "LLM" if the LLM-based entities are more accurate,
  - "Both" if both are equally correct or complementary (each captures part of the correct entities),
  - "None" if both are wrong or irrelevant.
- `"chosen_entity"` must be a JSON array of the final correct entities, or `null` if `"type"` is `"None"`.
- Output **only JSON**, no commentary or markdown.
- Ensure strict JSON syntax: use double quotes, commas between fields, and lowercase `null`.

---

### Examples

**Example 1**
Question: "After the Media Rights Group of Thailand, who was the first to denounce Thailand?"
NER_Entities: ["Media Rights Group (Thailand)", "Media Rights Group (Thailand)"]
LLM_Entities: ["Media Rights Group (Thailand)", "Thailand"]

Expected output:
{{
    "chosen_entity": ["Media Rights Group (Thailand)", "Thailand"],
    "type": "LLM"
}}

**Example 2**
Question: "Which country did China study before the religion of China?"
NER_Entities: ["Religion (China)", "Religion (China)"]
LLM_Entities: ["China", "China"]

Expected output:
{{
    "chosen_entity": ["China", "Religion (China)"],
    "type": "Both"
}}

**Example 3**
Question: "Who was the last person to visit China before the member of the Legislative Council of Iran?"
NER_Entities: ["China", "Islamic Council (Bahrain)"]
LLM_Entities: ["Pete Peterson", "China", "Member of Legislative (Govt) (Iran)"]

Expected output:
{{
    "chosen_entity": ["China", "Member of Legislative (Govt) (Iran)"],
    "type": "LLM"
}}

**Example 4**
Question: "Who attacked Iraq with small arms and light weapons after 9 August 2006?"
NER_Entities: ["Iraq"]
LLM_Entities: ["Iraq", "Citizen (Norway)"]

Expected output:
{{
    "chosen_entity": ["Iraq"],
    "type": "NER"
}}

**Example 5**
Question: "In which year did the United States' Council of Advisors to the Cabinet threaten Thailand?"
NER_Entities: ["Men (United States)", "Cabinet / Council of Ministers / Advisors (United States)", "Thailand"]
LLM_Entities: ["Thailand"]

Expected output:
{{
    "chosen_entity": ["Cabinet / Council of Ministers / Advisors (United States)", "Thailand"],
    "type": "Both"
}}

---

Now analyze the following case:

Question: "{question}"
NER_Entities: {ner_entities}
LLM_Entities: {llm_entities}
"""


# ============================================================
# Stage 2: LLM Arbitration on Conflicts
# ============================================================

def process_entity_compare(conflict_path, ner_path, output_path,
                            error_path, model, tokenizer):
    """
    Run LLM arbitration on conflict records and write resolved entities.

    For each conflict item, the LLM selects the correct entity set.
    The resolved entities are written back to a copy of the NER data,
    replacing the "entities" field for conflicting items.

    Args:
        conflict_path: Path to conflict records (from compare_entity_results).
        ner_path:      Path to original NER entity output (used as base).
        output_path:   Path to write updated entity results.
        error_path:    Path to write items where arbitration returned None.
        model:         Loaded LLaMA model.
        tokenizer:     Loaded LLaMA tokenizer.
    """
    with open(conflict_path, "r", encoding="utf-8") as f:
        conflicts = json.load(f)
    with open(ner_path, "r", encoding="utf-8") as f:
        ner_data = json.load(f)

    ner_index = {item["quid"]: item for item in ner_data}

    # Build prompts
    prompts = []
    for item in conflicts:
        ner_entities  = json.dumps(item.get("ner_entities",   []), ensure_ascii=False)
        llm_entities  = json.dumps(item.get("llama_entities", []), ensure_ascii=False)
        prompts.append(_ARBITRATION_PROMPT.format(
            question=item["question"],
            ner_entities=ner_entities,
            llm_entities=llm_entities,
        ))

    outputs = batched_generate(model, tokenizer, prompts)

    error_records = []
    for item, raw_out in zip(conflicts, outputs):
        try:
            clean = raw_out.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            chosen = parsed.get("chosen_entity", None)
            dtype  = parsed.get("type", "None")
        except json.JSONDecodeError:
            chosen = None
            dtype  = "None"

        # Write resolved entities back into the NER record
        quid = item["quid"]
        if quid in ner_index:
            ner_index[quid]["entities"]      = chosen or []
            ner_index[quid]["arbitration"]   = dtype

        if chosen is None or dtype == "None":
            error_records.append({**item, "arbitration": dtype})

    # Save main output (full NER data with resolved entities)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ner_data, f, indent=4, ensure_ascii=False)

    # Save error records
    Path(error_path).parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_records, f, indent=4, ensure_ascii=False)

    print(f"Entity arbitration done. {len(conflicts)} conflicts resolved.")
    print(f"  Errors (chosen=None): {len(error_records)}")
    print(f"  Output saved to {output_path}")
    print(f"  Error records saved to {error_path}")


# ============================================================
# Main
# ============================================================

def main():
    """
    Run entity conflict detection and LLM arbitration.

    Inputs:
        NER_ENTITY_OUTPUT         - Step 1a NER result (source of truth for base data)
        NER_ENTITY_OUTPUT         - also used as the NER side of comparison
        (LLaMA NER output is expected at ENTITY_COMPARE_OUTPUT before this runs,
         or pass a separate llama_path if available)

    For the standard pipeline the LLaMA NER path is the same file
    (entity_extract already stores llama-extracted entities in NER_ENTITY_OUTPUT).
    If a separate LLaMA NER file exists, pass it explicitly.
    """
    # Stage 1: detect conflicts
    # By default compare NER_ENTITY_OUTPUT against itself (same file) to allow
    # downstream override; in practice pass the correct llama_path if separate.
    llama_path = NER_ENTITY_OUTPUT  # override here if LLaMA NER is a separate file
    compare_entity_results(
        ner_path=NER_ENTITY_OUTPUT,
        llama_path=llama_path,
        output_path=ENTITY_CONFLICT_OUTPUT,
    )

    # Stage 2: LLM arbitration
    model, tokenizer = load_llama_model()
    process_entity_compare(
        conflict_path=ENTITY_CONFLICT_OUTPUT,
        ner_path=NER_ENTITY_OUTPUT,
        output_path=ENTITY_COMPARE_OUTPUT,
        error_path=ENTITY_COMPARE_ERROR_OUTPUT,
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
