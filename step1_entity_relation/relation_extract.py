"""
Step 1b: Relation Structure Extraction using LLaMA 3.1-8B-Instruct.

Given questions with recognized entities, extracts the semantic relation structure:
subject, relation, object, cue (before/after/first/last), reference_target.

Usage:
    python -m new_code.step1_entity_relation.relation_extract
"""
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLAMA_MODEL_ID, NER_ENTITY_OUTPUT, NER_RELATION_OUTPUT, NER_OUTPUT_DIR,
    LOCAL_BATCH_SIZE, LOCAL_MAX_NEW_TOKENS
)


# ============================================================
# Model Loading
# ============================================================

def load_llama_model(model_id=None):
    """Load LLaMA model for relation extraction."""
    model_id = model_id or LLAMA_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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
    for i in tqdm(range(0, len(prompts), batch_size), desc="Relation Extract"):
        batch = prompts[i: i + batch_size]
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
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
                top_p=0.9
            )

        for j, output in enumerate(outputs):
            gen_ids = output[len(inputs.input_ids[j]):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

        torch.cuda.empty_cache()
    return results


# ============================================================
# Relation Extraction Prompt
# ============================================================

RELATION_PROMPT_TEMPLATE = """You are an expert in temporal knowledge graph question understanding.
Given a natural language question and its recognized entities, identify the core semantic relation structure.
Your task is to output a single **valid JSON object** in the exact format below:

{{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "cue": "...",
    "reference_target": "..."
}}

### Field Definitions
- **subject**: The main entity performing or initiating the action. If not explicitly mentioned, set to "None".
- **relation**: The main predicate or action verb describing the interaction between entities. It may include auxiliary or descriptive words (e.g., "used small arms and light weapons", "use conventional military force", "Express_intent_to_cooperate"). Do not shorten it to a single verb or lemma.
- **object**: The entity, event, or object affected by the action. If missing, set to "None".
- **cue**: The temporal or comparative word that introduces a time or order constraint (e.g., "before", "after", "first", "last"). If not present, set to "None".
- **reference_target**: The entity or time expression that serves as the reference boundary for the cue (it can be a person, organization, country, or time). If missing, set to "None".

### Output Rules
- You must output **only one valid JSON object** - no explanations, comments, markdown formatting, or natural language text.
- All five keys must appear exactly as shown; if any information is unavailable, use "None" as the value.
- Do not wrap the output in code blocks (e.g., ```json).

---

### Examples

**Example 1**
Question: "Before the military of Taiwan, which country did China threaten last?"
Entities: ["Military (Taiwan)", "China"]

Expected output:
{{
    "subject": "China",
    "relation": "threaten",
    "object": "None",
    "cue": "before",
    "reference_target": "Military (Taiwan)"
}}

**Example 2**
Question: "When did China negotiate with the Iraqi Interim Government?"
Entities: ["China", "Iraqi Interim Government"]

Expected output:
{{
    "subject": "China",
    "relation": "negotiate",
    "object": "Iraqi Interim Government",
    "cue": "None",
    "reference_target": "None"
}}

**Example 3**
Question: "Who negotiated with Iraq last, before the Yemeni resistance movement?"
Entities: ["Iraq", "National Resistance Movement"]

Expected output:
{{
    "subject": "None",
    "relation": "negotiate",
    "object": "Iraq",
    "cue": "before",
    "reference_target": "National Resistance Movement"
}}

**Example 4**
Question: "Who was the first country that Ethiopia expressed optimism about?"
Entities: ["Ethiopia"]

Expected output:
{{
    "subject": "Ethiopia",
    "relation": "expressed optimism about",
    "object": "None",
    "cue": "first",
    "reference_target": "None"
}}

**Example 5**
Question: "After Thailand, who did the citizens of Australia support first?"
Entities: ["Thailand", "Citizen (Australia)"]

Expected output:
{{
    "subject": "Citizen (Australia)",
    "relation": "support",
    "object": "None",
    "cue": "after",
    "reference_target": "Thailand"
}}

**Example 6**
Question: "When did the Saudi Arabian Defence Forces first use small arms and light weapons against Ethiopia?"
Entities: ["Saudi Arabian Defence Forces", "Ethiopia"]

Expected output:
{{
    "subject": "Saudi Arabian Defence Forces",
    "relation": "use small arms and light weapons against",
    "object": "Ethiopia",
    "cue": "first",
    "reference_target": "None"
}}
---

Now analyze the following question:
Question: "{question}"
Entities: "{entities}"
"""


# ============================================================
# Processing
# ============================================================

def process_relations(input_path, output_path, model, tokenizer):
    """Extract relation structures from questions with entities."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [
        RELATION_PROMPT_TEMPLATE.format(
            question=item["question"],
            entities=item.get("entities", [])
        )
        for item in data
    ]

    outputs = batched_generate(model, tokenizer, prompts)

    final_results = []
    for item, out in zip(data, outputs):
        try:
            clean_out = out.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_out)
            record = {
                "quid": item.get("quid"),
                "question": item.get("question"),
                "entities": item.get("entities", []),
                "subject": parsed.get("subject", "None"),
                "relation": parsed.get("relation", "None"),
                "object": parsed.get("object", "None"),
                "cue": parsed.get("cue", "None"),
                "reference_target": parsed.get("reference_target", "None")
            }
        except json.JSONDecodeError:
            record = {
                "quid": item.get("quid"),
                "question": item.get("question"),
                "entities": item.get("entities", []),
                "subject": "None",
                "relation": "None",
                "object": "None",
                "cue": "None",
                "reference_target": "None",
                "raw_output": out.strip()
            }
        final_results.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"Saved relation results to {output_path}")


def main():
    """Run relation extraction on entity-extracted results."""
    model, tokenizer = load_llama_model()
    process_relations(NER_ENTITY_OUTPUT, NER_RELATION_OUTPUT, model, tokenizer)


if __name__ == "__main__":
    main()
