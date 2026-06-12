"""
Step 4: Causal Fact Filtering.

Filters graph-retrieved facts to remove redundancy and keep only causally relevant ones.
Pipeline:
1. Check if causal filtering should be triggered (based on fact count & redundancy score)
2. Use LLM (CausalFactGenerator) to minimize the fact set
3. Verify consistency (CausalFactEvaluator) - filtered facts should yield same answer
4. Return filtered facts if consistent, otherwise fallback to original

Usage:
    python -m new_code.step4_causal_filter.causal_filter
"""
import json
import os
import sys
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    CAUSAL_TRIGGER_L0, CAUSAL_TRIGGER_ALPHA, CAUSAL_TRIGGER_BETA, CAUSAL_TRIGGER_TAU,
    GRAPH_RETRIEVE_OUTPUT, CAUSAL_FILTER_OUTPUT
)


# ============================================================
# Trigger Logic
# ============================================================

def normalize_answer(ans):
    """Normalize a single answer string for comparison."""
    if isinstance(ans, str):
        return ans.strip().lower()
    return str(ans).strip().lower()


def parse_signature(fact: str):
    """Extract (S, R, O) signature from a tab-separated fact string."""
    try:
        s, r, o, t = fact.split("\t")
        return (s.strip(), r.strip(), o.strip())
    except:
        return None


def causal_trigger_score(facts, L0=None, alpha=None, beta=None):
    """
    Compute causal filtering trigger score.
    Based on: length factor (how many facts) + redundancy factor (duplicate signatures).
    Returns: (score, length_factor, redundancy_factor)
    """
    L0 = L0 or CAUSAL_TRIGGER_L0
    alpha = alpha or CAUSAL_TRIGGER_ALPHA
    beta = beta or CAUSAL_TRIGGER_BETA

    N = len(facts)
    if N == 0:
        return 0.0, 0.0, 0.0

    # Length factor: ratio of fact count to threshold
    length_factor = N / L0

    # Redundancy factor: 1 - unique_signatures / total
    signatures = set()
    for f in facts:
        sig = parse_signature(f)
        if sig is not None:
            signatures.add(sig)

    redundancy_factor = 1 - len(signatures) / N

    score = alpha * length_factor + beta * redundancy_factor
    return score, length_factor, redundancy_factor


def should_trigger_causal_filter(facts, tau=None):
    """Check if causal filtering should be triggered for this fact set."""
    tau = tau or CAUSAL_TRIGGER_TAU
    score, _, _ = causal_trigger_score(facts)
    return score > tau


def answers_equal(ans1: dict, ans2: dict) -> bool:
    """
    Check if two answer dicts are semantically equivalent.
    Uses subset comparison (either is subset of the other).
    """
    if not isinstance(ans1, dict) or not isinstance(ans2, dict):
        return False
    if "answers" not in ans1 or "answers" not in ans2:
        return False

    a1 = ans1.get("answers", [])
    a2 = ans2.get("answers", [])

    if not isinstance(a1, list) or not isinstance(a2, list):
        return False

    set1 = set(normalize_answer(x) for x in a1)
    set2 = set(normalize_answer(x) for x in a2)

    return set1 <= set2 or set2 <= set1


# ============================================================
# Causal Fact Generator (Minimizer)
# ============================================================

class CausalFactGenerator:
    """Uses LLM to minimize a fact set while preserving answer-relevant facts."""

    def __init__(self, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL
        )
        self.model = DEEPSEEK_MODEL

    def generate(self, facts: list, question: str) -> list:
        """
        Input: redundant fact list + question
        Output: minimized fact list (or original on failure)
        """
        prompt = f"""You are a Causal Fact Minimizer.

Your task is to filter a list of historical facts based on a given question, keeping only the facts that are causally relevant to answering it.
Return ONLY a JSON object in this exact format:
{{"answers": ["fact1", "fact2", ..., "factN"]}}

Rules:
1. Use the question to decide which facts to keep. Delete any fact that does not directly help explain or answer the question.
2. If multiple facts convey the same information (e.g., same entities, same relation, overlapping or consecutive dates), keep only one representative fact.
3. Keep only the minimal necessary set of facts that is sufficient to answer the original question correctly.
4. The remaining facts must come exactly from the original list. Do NOT introduce any facts not in the input.
5. Remove any facts that are redundant, duplicate, or irrelevant for answering the original question.
6. Output only the JSON. Do not include any other text.

Question: "{question}"
Facts: {facts}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=0.9,
                stream=False,
            )
            result = resp.choices[0].message.content.strip()
            clean_out = result.replace("```json", "").replace("```", "").strip()
            causal_facts = json.loads(clean_out)
            return causal_facts
        except Exception as e:
            print(f"Generator JSON parse failed, falling back to original facts: {e}")
            return facts


# ============================================================
# Causal Fact Evaluator (Consistency Checker)
# ============================================================

class CausalFactEvaluator:
    """Answers a question using given facts to verify consistency."""

    def __init__(self, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL
        )
        self.model = DEEPSEEK_MODEL

    def answer_question(self, facts: list, question: str) -> dict:
        """
        Answer a question given facts.
        Returns: {"answers": [...]}
        """
        fact_list = []
        for f in facts:
            p = f.split("\t")
            if len(p) >= 4:
                fact_list.append(f"{p[0]} {p[1]} {p[2]} in {p[3]}")
            else:
                fact_list.append(f)

        prompt = f"""You are a temporal knowledge graph QA assistant.
Given a list of Historical facts and a Question, return ONLY a JSON object:{{"answers": [...]}}.

Rules:
- Please must return all possible answers and do NOT provide explanations.
- If multiple entities share the SAME earliest (or latest) timestamp that satisfies the question, return ALL of them.
- Every returned entity must appear EXACTLY as written in the Historical facts.
- Dates must follow ISO format (YYYY-MM-DD). If the question asks for:
   - year -> return YYYY
   - month -> return YYYY-MM
   - day -> return YYYY-MM-DD
- Relations are DIRECTIONAL; use only facts whose subject/object roles match the question.

Question: "{question}"
Historical facts: {fact_list}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                stream=False
            )
            raw = resp.choices[0].message.content.strip()
            clean_out = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_out)
        except:
            return {"answers": []}


# ============================================================
# CausalLLM Controller
# ============================================================

class CausalLLM:
    """
    Orchestrates causal fact filtering:
    1. Get answer from original facts (A1)
    2. Generate minimized facts
    3. Get answer from minimized facts (A2)
    4. If A1 == A2: return minimized facts; else: return original
    """

    def __init__(self, api_key=None):
        api_key = api_key or DEEPSEEK_API_KEY
        self.generator = CausalFactGenerator(api_key)
        self.evaluator = CausalFactEvaluator(api_key)

    def causal_filter(self, facts: list, question: str):
        """Run the full causal filtering pipeline."""
        # Step 1: Answer with original facts
        ans_original = self.evaluator.answer_question(facts, question)

        # Step 2: Generate minimized facts
        causal_result = self.generator.generate(facts, question)
        causal_facts = causal_result.get("answers", []) if isinstance(causal_result, dict) else causal_result

        # Step 3: Answer with minimized facts
        ans_causal = self.evaluator.answer_question(causal_facts, question)

        # Step 4: Consistency check
        if answers_equal(ans_original, ans_causal):
            return causal_facts
        else:
            return facts


# ============================================================
# Main Processing
# ============================================================

def process_causal_filter(input_path=None, output_path=None):
    """
    Run causal filtering on graph retrieval results.
    Only processes items where trigger score exceeds threshold.
    """
    input_path = input_path or GRAPH_RETRIEVE_OUTPUT
    output_path = output_path or CAUSAL_FILTER_OUTPUT

    with open(input_path, "r", encoding="utf-8") as f:
        question_json = json.load(f)

    causal_llm = CausalLLM()
    results = []

    for item in tqdm(question_json, desc="Causal Filtering"):
        question = item["question"]
        qfacts = item.get("qfact", {})
        retrieved_facts = item.get("retrieved", {})

        # Collect all facts for this question
        fact_list = []
        for rel, graph_facts in retrieved_facts.items():
            if qfacts.get(rel):
                fact_list.append(qfacts[rel][0])
            if graph_facts:
                fact_list.extend(graph_facts)

        # Check trigger condition
        if not should_trigger_causal_filter(fact_list):
            continue

        # Run causal filtering
        final_facts = causal_llm.causal_filter(fact_list, question)

        results.append({
            "quid": item.get("quid"),
            "question": question,
            "original_fact_num": len(fact_list),
            "final_fact_num": len(final_facts),
            "final_facts": final_facts,
            "qtype": item.get("qtype"),
            "qlabel": item.get("qlabel"),
            "answer_type": item.get("answer_type"),
            "answer": item.get("answer")
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Causal filtering done. {len(results)} items saved to {output_path}")


def main():
    process_causal_filter()


if __name__ == "__main__":
    main()
