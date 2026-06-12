"""
Step 3b: Semantic Fact Retrieval using BGE-M3.

Provides semantic similarity-based retrieval as a fallback when graph retrieval
yields empty results. Uses dense embeddings + FAISS for efficient search.

Usage:
    python -m new_code.step3_retrieval.semantic_retriever
"""
import json
import os
import sys
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BGE_M3_MODEL, SEMANTIC_TOPK, SEMANTIC_EMBEDDING_SIZE,
    KG_FULL_PATH, QUESTIONS_PROCESSED_PATH, SEMANTIC_RETRIEVE_OUTPUT
)


class SemanticRetriever:
    """
    Semantic fact retrieval using BGE-M3 dense embeddings + FAISS.
    Given questions and a fact list, returns top-k most similar facts per question.
    """

    def __init__(self, model_name=None, question_json=None, fact_list=None,
                 embedding_size=None):
        model_name = model_name or BGE_M3_MODEL
        embedding_size = embedding_size or SEMANTIC_EMBEDDING_SIZE

        self.model = BGEM3FlagModel(model_name, use_fp16=True, devices=['cuda:0'])
        self.question_json = question_json or []
        self.questions = [item["question"] for item in self.question_json]
        self.qids = [item["quid"] for item in self.question_json]
        self.fact_list = fact_list or []
        self.embedding_size = embedding_size
        self.index = None

    def encode(self, texts):
        """Encode texts to dense vectors using BGE-M3."""
        output = self.model.encode_corpus(
            texts,
            convert_to_numpy=True,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return output["dense_vecs"]

    def build_index(self, n_clusters=1024, nprobe=128):
        """Build GPU-accelerated FAISS IVFFlat index."""
        quantizer = faiss.IndexFlatIP(self.embedding_size)
        index = faiss.IndexIVFFlat(
            quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT
        )
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.nprobe = nprobe
        self.index = index

    def search(self, topk=None):
        """Encode questions and facts, build index, and search."""
        topk = topk or SEMANTIC_TOPK
        question_embeddings = self.encode(self.questions)
        fact_embeddings = self.encode(self.fact_list)

        self.build_index()
        self.index.train(fact_embeddings)
        self.index.add(fact_embeddings)

        distances, idxs = self.index.search(question_embeddings, topk)
        return distances, idxs

    def build_results(self, distances, idxs):
        """Convert search indices to fact strings."""
        results = []
        for i in range(len(self.questions)):
            top_ids = idxs[i]
            top_facts = [self.fact_list[j] for j in top_ids]
            results.append({
                "quid": self.qids[i],
                "question": self.questions[i],
                "semantic_facts": top_facts
            })
        return results

    def run(self, topk=None):
        """Full pipeline: encode -> index -> search -> format results."""
        distances, idxs = self.search(topk=topk)
        return self.build_results(distances, idxs)

    def save(self, results, output_path):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


def retrieve_semantic_facts(model_name=None, question_json=None, fact_list=None,
                            topk=None, output_path=None):
    """
    Convenience function to run semantic retrieval end-to-end.
    Returns list of {quid, question, semantic_facts} dicts.
    """
    output_path = output_path or SEMANTIC_RETRIEVE_OUTPUT
    retriever = SemanticRetriever(model_name, question_json, fact_list)
    results = retriever.run(topk=topk)
    retriever.save(results, output_path)
    print(f"Semantic retrieval results saved to {output_path}")
    return results


# ============================================================
# Main
# ============================================================

def main():
    """Run semantic retrieval on MultiTQ questions."""
    # Load questions
    with open(QUESTIONS_PROCESSED_PATH, "r", encoding="utf-8") as f:
        question_data = json.load(f)

    # Load fact list from KG triples
    fact_list = []
    with open(KG_FULL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace("_", " ").split('\t')
            if len(parts) >= 4:
                fact_list.append(f"{parts[0]} {parts[1]} {parts[2]} in {parts[3]}")

    # Run retrieval
    retrieve_semantic_facts(
        question_json=question_data,
        fact_list=fact_list,
        topk=SEMANTIC_TOPK,
        output_path=SEMANTIC_RETRIEVE_OUTPUT
    )


if __name__ == "__main__":
    main()
