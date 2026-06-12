"""
Step 5b: Answer Generation using local LLaMA 3.1 model.

Reads graph retrieval results + semantic facts + causal filtered facts,
constructs prompts, and generates answers via local LLM inference.

Usage:
    python -m new_code.step5_predict.predict_local
"""
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLAMA_MODEL_ID, LOCAL_BATCH_SIZE, LOCAL_MAX_NEW_TOKENS,
    LOCAL_TEMPERATURE, LOCAL_TOP_P,
    GRAPH_RETRIEVE_OUTPUT, SEMANTIC_RETRIEVE_OUTPUT, CAUSAL_FILTER_OUTPUT,
    OUTPUTS_DIR
)


# ============================================================
# Model Loading
# ============================================================

def load_llama_model(model_id=None):
    """Load LLaMA model for answer generation."""
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

def batched_generate(model, tokenizer, prompts, batch_size=None, max_new_tokens=None):
    """Generate responses in batches using local LLM."""
    batch_size = batch_size or LOCAL_BATCH_SIZE
    max_new_tokens = max_new_tokens or LOCAL_MAX_NEW_TOKENS
    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Local LLM Generating"):
        batch = prompts[i: i + batch_size]
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True
            )
            for q in batch
        ]
        inputs = tokenizer(
            chat_texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=LOCAL_TEMPERATURE,
                top_p=LOCAL_TOP_P
            )

        for j, output in enumerate(outputs):
            gen_ids = output[len(inputs.input_ids[j]):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

        torch.cuda.empty_cache()

    return results


# ============================================================
# Prompt Template (same as API version)
# ============================================================

BASE_PROMPT = """You are a temporal knowledge question answering assistant..
Given a list of Historical facts and a Question, return ONLY a JSON object:{{"answers": [...]}}. Do NOT include any code block or explanation

Rules:
- Please must return all possible answers and do not repeat any Historical facts in the output.
- If multiple entities share the SAME earliest (or latest) timestamp that satisfies the question, return ALL of them.
- Every returned entity must appear EXACTLY as written in the Historical facts.
- Dates must follow ISO format (YYYY-MM-DD). If the question asks for:
   - year -> return YYYY
   - month -> return YYYY-MM
   - day -> return YYYY-MM-DD
- Relations are DIRECTIONAL; use only facts whose subject/object roles match the question and ignore facts with opposite direction or semantically different relations.

Here are some examples:
**Example 1**
Question: "After the Cabinet Council of Ministers of Peru, who was the first to express the intention to negotiate with Japan?"
Historical facts: [
    "Cabinet Council of Ministers Advisors (Peru) Express intent to meet or negotiate Japan 2010-02-20",
    "Brazil Express intent to meet or negotiate Japan in 2010-02-22",
    "Citizen (Vietnam) Express intent to meet or negotiate Japan in 2010-02-22",
    "Envoy (Canada) Express intent to meet or negotiate Japan in 2010-02-22"
]
Expected output:
{{
    "answers": ["Brazil", "Citizen (Vietnam)", "Envoy (Canada)"]
}}

**Example 2**
Question: "Before Malaysia, who last wanted to negotiate with the Governor of Malaysia?"
Historical facts: [
    "Malaysia Engage in negotiation Governor (Malaysia) in 2011-08-29",
    "Democratic Reform Movement Engage in negotiation Governor (Malaysia) in 2010-04-23",
    "Malaysia Express intent to meet or negotiate Governor (Malaysia) in 2013-10-17",
    "Citizen (Malaysia) Express intent to meet or negotiate Governor (Malaysia) in 2012-11-25"
]
Expected output:
{{
    "answers": ["Citizen (Malaysia)"]
}}

**Example 3**
Question: "When did Germany last visit Alexander Roth?"
Historical facts: [
    "Germany Make a visit Alexander Roth in 2006-09-23",
    "Germany Host a visit Alexander Roth in 2013-06-08",
    "Germany Consult Alexander Roth in 2015-05-16"
]
Expected output:
{{
    "answers": ["2006-09-23"]
}}

**Example 4**
Question: "Which country was accused by Colombia after 2012?"
Historical facts: [
    "Royal Administration (Spain) Accuse Colombia in 2013-02-28",
    "Portugal Accuse Colombia in 2013-08-20",
    "Portugal Accuse Colombia in 2015-07-07",
    "Portugal Accuse Colombia in 2015-09-07",
    "Portugal Accuse Colombia in 2015-09-15"
]
Expected output:
{{
    "answers": ["Royal Administration (Spain)", "Portugal"]
}}

Now analyze the following case:
Question: "{question}"
Historical facts: {fact_list}
"""


# ============================================================
# Main Processing
# ============================================================

def process_predictions(input_path=None, semantic_path=None, causal_path=None,
                        output_path=None, model=None, tokenizer=None):
    """Run prediction with local LLM."""
    input_path = input_path or GRAPH_RETRIEVE_OUTPUT
    semantic_path = semantic_path or SEMANTIC_RETRIEVE_OUTPUT
    causal_path = causal_path or CAUSAL_FILTER_OUTPUT
    output_path = output_path or os.path.join(OUTPUTS_DIR, "predictions_local.json")

    # Load data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(semantic_path, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)
    with open(causal_path, "r", encoding="utf-8") as f:
        causal_data = json.load(f)

    semantic_index = {item["quid"]: item.get("semantic_facts", []) for item in semantic_data}
    causal_index = {item["quid"]: item.get("final_facts", []) for item in causal_data}

    print(f"Loaded {len(data)} items.")

    # Build prompts with fact cascade
    prompts = []
    for item in data:
        retrieved = item.get("retrieved", {})
        qfacts = item.get("qfact", {})
        quid = item["quid"]
        semantic_facts = semantic_index.get(quid, [])
        causal_facts = causal_index.get(quid, [])

        used_graph_fact = False
        fact_list = []

        for rel, graph_facts in retrieved.items():
            if qfacts.get(rel):
                fact = qfacts[rel][0].split('\t')
                fact_list.append(f"{fact[0]} {fact[1]} {fact[2]} in {fact[3]}")
            if graph_facts:
                used_graph_fact = True
                for f in graph_facts:
                    p = f.split("\t")
                    fact_list.append(f"{p[0]} {p[1]} {p[2]} in {p[3]}")

        if not used_graph_fact:
            fact_list = semantic_facts

        if causal_facts:
            fact_list = causal_facts

        prompt = BASE_PROMPT.format(question=item["question"], fact_list=fact_list)
        prompts.append(prompt)

    # Generate
    if model is None or tokenizer is None:
        model, tokenizer = load_llama_model()

    outputs = batched_generate(model, tokenizer, prompts)

    # Parse results
    final_results = []
    for item, out_str in zip(data, outputs):
        try:
            clean_out = out_str.strip().replace("```json", "").replace("```", "").strip()
            model_answer = json.loads(clean_out)
        except json.JSONDecodeError:
            model_answer = {"answers": []}

        final_results.append({
            "quid": item.get("quid"),
            "question": item.get("question"),
            "model_answer": model_answer.get("answers", []),
            "qtype": item.get("qtype"),
            "qlabel": item.get("qlabel"),
            "answer_type": item.get("answer_type"),
            "ground_truth": item.get("answer", []),
        })

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"Done! Results saved to {output_path}")


def main():
    model, tokenizer = load_llama_model()
    process_predictions(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
