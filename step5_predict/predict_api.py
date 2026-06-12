"""
Step 5a: Answer Generation using DeepSeek API (async with checkpoint).

Reads graph retrieval results + semantic facts + causal filtered facts,
constructs prompts, and generates answers via async API calls.
Supports checkpoint resume and graceful interruption.

Usage:
    python -m new_code.step5_predict.predict_api
"""
import json
import os
import sys
import asyncio
import time
import signal
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    API_TEMPERATURE, API_TOP_P, API_MAX_CONCURRENCY, API_SAVE_EVERY,
    GRAPH_RETRIEVE_OUTPUT, SEMANTIC_RETRIEVE_OUTPUT, CAUSAL_FILTER_OUTPUT,
    PREDICT_OUTPUT
)


# ============================================================
# Global State
# ============================================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

STATS = {"total": 0, "success": 0, "fail": 0, "start_time": time.time()}
SHOULD_EXIT = False


def signal_handler(sig, frame):
    global SHOULD_EXIT
    print("\nCaught Ctrl+C, will exit after current chunk saves...")
    SHOULD_EXIT = True


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# API Call with Retry
# ============================================================

def ds_generate_single_with_retry(prompt, retries=2, backoff=1.5):
    """Call DeepSeek API with exponential backoff retry."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=API_TEMPERATURE,
                top_p=API_TOP_P
            )
            STATS["success"] += 1
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retries:
                STATS["fail"] += 1
                print(f"API Error (final attempt): {e}")
                return ""
            time.sleep(backoff ** attempt)


# ============================================================
# Async Wrappers
# ============================================================

async def ds_generate_async(prompt, sem):
    """Async wrapper with semaphore for concurrency control."""
    async with sem:
        STATS["total"] += 1
        return await asyncio.to_thread(ds_generate_single_with_retry, prompt)


async def ds_generate_batched_async(prompts, max_concurrency=None):
    """Run batch of prompts with async concurrency."""
    max_concurrency = max_concurrency or API_MAX_CONCURRENCY
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [asyncio.create_task(ds_generate_async(p, sem)) for p in prompts]
    results = await tqdm_asyncio.gather(*tasks)
    return results


# ============================================================
# Prompt Template
# ============================================================

BASE_PROMPT = """You are a temporal knowledge question answering assistant..
Given a list of Historical facts and a Question, return ONLY a JSON object:{{"answers": [...]}}.

Rules:
- Please must return all possible answers and do NOT provide explanations.
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
# Fact Cascade Logic
# ============================================================

def build_fact_list(item, semantic_index, causal_index):
    """
    Build fact list with priority cascade:
    1. Graph facts (primary)
    2. Semantic facts (fallback if no graph facts)
    3. Causal facts (override if available)
    """
    retrieved = item.get("retrieved", {})
    qfacts = item.get("qfact", {})
    quid = item["quid"]
    semantic_facts = semantic_index.get(quid, [])
    causal_facts = causal_index.get(quid, [])

    used_graph_fact = False
    fact_list = []

    for rel, graph_facts in retrieved.items():
        # Add query fact (the reference fact for the question)
        if qfacts.get(rel):
            fact = qfacts[rel][0].split('\t')
            fact_list.append(f"{fact[0]} {fact[1]} {fact[2]} in {fact[3]}")

        # Add retrieved graph facts
        if graph_facts:
            used_graph_fact = True
            for f in graph_facts:
                p = f.split("\t")
                fact_list.append(f"{p[0]} {p[1]} {p[2]} in {p[3]}")

    # Fallback to semantic facts if graph retrieval was empty
    if not used_graph_fact:
        fact_list = semantic_facts

    # Causal facts override everything (when available)
    if causal_facts:
        fact_list = causal_facts

    return fact_list


# ============================================================
# Main Processing
# ============================================================

def process_predictions(input_path=None, semantic_path=None, causal_path=None,
                        output_path=None, save_every=None, max_concurrency=None):
    """Run prediction with async API calls and checkpoint support."""
    input_path = input_path or GRAPH_RETRIEVE_OUTPUT
    semantic_path = semantic_path or SEMANTIC_RETRIEVE_OUTPUT
    causal_path = causal_path or CAUSAL_FILTER_OUTPUT
    output_path = output_path or PREDICT_OUTPUT
    save_every = save_every or API_SAVE_EVERY
    max_concurrency = max_concurrency or API_MAX_CONCURRENCY

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

    # Checkpoint resume
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            final_results = json.load(f)
        finished = len(final_results)
        print(f"Resuming from checkpoint: {finished}")
    else:
        final_results = []
        finished = 0

    # Process in chunks
    for start in range(finished, len(data), save_every):
        if SHOULD_EXIT:
            break

        end = min(start + save_every, len(data))
        chunk = data[start:end]

        # Build prompts
        prompts = []
        for item in chunk:
            fact_list = build_fact_list(item, semantic_index, causal_index)
            prompt = BASE_PROMPT.format(question=item["question"], fact_list=fact_list)
            prompts.append(prompt)

        # Async execution
        print(f"Processing {start} -> {end} | concurrency={max_concurrency}")
        outputs = asyncio.run(ds_generate_batched_async(prompts, max_concurrency))

        # Parse results
        for item, out_str in zip(chunk, outputs):
            try:
                clean_out = out_str.strip().replace("```json", "").replace("```", "").strip()
                model_answer = json.loads(clean_out)
            except:
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

        # Save checkpoint
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)

        elapsed = time.time() - STATS["start_time"]
        qps = STATS["total"] / elapsed if elapsed > 0 else 0
        print(f"Saved {len(final_results)} | QPS={qps:.2f} | Success={STATS['success']} | Fail={STATS['fail']}")

    print(f"Done! Results saved to {output_path}")


def main():
    process_predictions()


if __name__ == "__main__":
    main()
