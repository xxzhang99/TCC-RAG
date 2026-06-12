"""
Step 1a: Entity Extraction using LLaMA 3.1-8B-Instruct.

Extracts named entities from temporal knowledge graph questions,
then grounds them to KG entities via semantic + fuzzy matching.

Usage:
    python -m new_code.step1_entity_relation.entity_extract
"""
import json
import os
import sys
import torch
import numpy as np
import difflib
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLAMA_MODEL_ID, EMBEDDING_MODEL, ENTITY_SEMANTIC_THRESHOLD,
    NER_BATCH_SIZE, NER_MAX_NEW_TOKENS,
    KG_ENT_ID_PATH, KG_ENTITY2ID_PATH, QUESTIONS_TEST_PATH,
    NER_OUTPUT_DIR, NER_ENTITY_OUTPUT, ENTITY_EMBED_CACHE
)
from utils import normalize, extract_time, is_date_entity


# ============================================================
# Model Loading
# ============================================================

def load_llama_ner(model_id=None):
    """Load LLaMA model for entity recognition."""
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


def load_embed_model(all_entities, emb_file=None):
    """Load sentence embedding model and compute/cache entity embeddings."""
    emb_file = emb_file or ENTITY_EMBED_CACHE
    print("Loading embedding encoder...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda:0")

    if os.path.exists(emb_file):
        entity_embeds = np.load(emb_file)
        print(f"Loaded precomputed embeddings from {emb_file}")
    else:
        print("Encoding all KG entities (first run)...")
        entity_embeds = model.encode(
            all_entities, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )
        os.makedirs(os.path.dirname(emb_file), exist_ok=True)
        np.save(emb_file, entity_embeds)
        print(f"Entity embeddings saved to {emb_file}")
    return model, entity_embeds


def build_faiss_index(entity_embeds):
    """Build FAISS index for entity matching."""
    dim = entity_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(entity_embeds)
    print(f"FAISS index built with {len(entity_embeds)} entities.")
    return index


# ============================================================
# Entity Extraction (LLaMA NER)
# ============================================================

def build_ner_prompt(question):
    """Build the NER prompt for a single question."""
    return f"""You are an expert in entity recognition for temporal knowledge graphs.

Your task is to extract all **real-world entities or collective actor phrases** mentioned in the question.
Entities can include:
- **People** (e.g., "Catherine Ashton", "Donald Rumsfeld")
- **Organizations or government bodies** (e.g., "Cabinet Council of Ministers of Kazakhstan", "United Nations Security Council")
- **Countries or regions** (e.g., "China", "Ethiopia", "European Union")
- **Groups or collectives** (e.g., "other authorities and officials of Russia", "citizens of Belgium", "leaders of Iraq")
- **Role-based entities** (e.g., "the leader of Iraq", "the citizens of Belgium", "the military of Taiwan", "the president of the United States")
- **Organization/member composites** (e.g., "Hizbul Islam fighter", "Iraqi police officer", "Thai activist")
---

Must NOT include:
1. Temporal number expressions or dates, e.g.: "2011", "14 October 2015", "August 2013".
2. Abstract category words that do not refer to a specific actor, such as:"country", "organization", "people", "government", "leaders" (without a country or organization).
3. Relations, actions, or event phrases, e.g.: "conventional military forces", "humanitarian aid", "visit", "meeting", "agreement", "speech".

---

### Extraction principles
1. Always return **complete, continuous entity names** - do not split or truncate.
   - "Cabinet Council of Ministers of Kazakhstan"
   - "the citizens of Belgium"
   - "the leader of Iraq"
2. Combine descriptive entities if they refer to one actor or group.
   - "other authorities and officials of Russia"
   - "the military of Taiwan"
3. Each question contains **1-2 entities**, never fewer or more.
4. Output **only** a valid JSON list of strings (no markdown, no explanations).

---
### Examples

**Example 1**
Question: "After the Australian police, which country was the first to offer humanitarian aid to China?"
Expected Output: ["Australian police", "China"]

**Example 2**
Question: "Before the other authorities and officials of Russia, who made optimistic remarks about China?"
Expected Output: ["other authorities and officials of Russia", "China"]

**Example 3**
Question: "Before 14 October 2015, who made Burundi suffer from conventional military forces?"
Expected Output: ["Burundi"]

**Example 4**
Question: "Who replaced the leader of Iraq after the invasion by the United States?"
Expected Output: ["leader of Iraq", "United States"]

**Example 5**
Question: "In what year did China last appeal to Iraq?"
Expected Output: ["China", "Iraq"]

---

Now extract entities for the following question.
Question: "{question}"
"""


def llama_extract_entities(model, tokenizer, questions, batch_size=None, max_new_tokens=None):
    """
    Batch extract entities from questions using LLaMA 3.1.
    Returns: list of list of entity strings.
    """
    batch_size = batch_size or NER_BATCH_SIZE
    max_new_tokens = max_new_tokens or NER_MAX_NEW_TOKENS

    prompts = [build_ner_prompt(q) for q in questions]
    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="LLaMA NER"):
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
            clean = text.replace("```json", "").replace("```", "").strip()
            try:
                ents = json.loads(clean)
                ents = [normalize(e) for e in ents if isinstance(e, str) and e.strip()]
            except Exception:
                ents = []
            results.append(ents)

    return results


# ============================================================
# Entity Matching (Semantic + Fuzzy)
# ============================================================

def match_entities(question, entities, embed_model, all_entities, index,
                   mismatch_records=None, quid=None):
    """
    Match extracted entities to KG entities via semantic and fuzzy matching.
    Returns: list of matched entity dicts.
    """
    matched = []

    for ent in entities:
        text = normalize(ent)
        if is_date_entity(text):
            continue

        # Semantic matching
        cand_emb = embed_model.encode([text], normalize_embeddings=True)
        scores, idxs = index.search(cand_emb, 1)
        sem_match = all_entities[idxs[0][0]]
        sem_score = float(scores[0][0])

        # Fuzzy matching
        fuzzy_match = difflib.get_close_matches(text, all_entities, n=1)
        fuzzy_match = fuzzy_match[0] if fuzzy_match else None

        # Track mismatches for analysis
        if mismatch_records is not None and sem_match != fuzzy_match:
            mismatch_records.append({
                "qid": quid,
                "question": question,
                "ner_entity": text,
                "semantic_match": sem_match,
                "fuzzy_match": fuzzy_match
            })

        matched.append({
            "text": text,
            "semantic_match": sem_match,
            "fuzzy_match": fuzzy_match,
            "semantic_score": sem_score,
            "final_entity": sem_match if sem_score > ENTITY_SEMANTIC_THRESHOLD else fuzzy_match
        })

    return matched


# ============================================================
# Main Processing Pipeline
# ============================================================

def process_dataset(dataset, llama_model, llama_tokenizer,
                    embed_model, all_entities, index, mismatch_records):
    """Process a dataset: extract entities and match to KG."""
    questions = [item["question"] for item in dataset]
    ner_results = llama_extract_entities(llama_model, llama_tokenizer, questions)

    for item, entities in tqdm(zip(dataset, ner_results), total=len(dataset), desc="Matching"):
        matched = match_entities(
            item["question"], entities, embed_model, all_entities, index,
            mismatch_records=mismatch_records, quid=item.get("quid")
        )
        item["time"] = extract_time(item["question"])
        item["entities"] = [m["final_entity"] for m in matched if m["final_entity"]]
        item["ner_recognized_entities"] = entities
        item["semantic_top1_entities"] = [m["semantic_match"] for m in matched]
        item["fuzzy_top1_entities"] = [m["fuzzy_match"] for m in matched]


def main():
    """Run entity extraction on MultiTQ test set."""
    # Load KG entities
    with open(KG_ENT_ID_PATH, 'r', encoding='utf-8') as f:
        all_entities = [line.strip().split('\t')[0] for line in f]

    with open(KG_ENTITY2ID_PATH, 'r', encoding='utf-8') as f:
        entity2id_map = json.load(f)

    # Load models
    llama_model, llama_tokenizer = load_llama_ner()
    embed_model, entity_embeds = load_embed_model(all_entities)
    index = build_faiss_index(entity_embeds)

    # Load dataset
    with open(QUESTIONS_TEST_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Process
    mismatch_records = []
    process_dataset(dataset, llama_model, llama_tokenizer,
                    embed_model, all_entities, index, mismatch_records)

    # Save results
    os.makedirs(NER_OUTPUT_DIR, exist_ok=True)
    with open(NER_ENTITY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"Saved entity results to {NER_ENTITY_OUTPUT}")

    if mismatch_records:
        mismatch_path = os.path.join(NER_OUTPUT_DIR, "semantic_vs_fuzzy_mismatch.json")
        with open(mismatch_path, 'w', encoding='utf-8') as f:
            json.dump(mismatch_records, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(mismatch_records)} mismatches to {mismatch_path}")


if __name__ == "__main__":
    main()
