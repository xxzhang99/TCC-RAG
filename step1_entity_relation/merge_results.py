"""
Step 1c: Merge entity and relation extraction results into final question JSON.

Combines entity extraction output with relation structure output,
producing the final input for graph retrieval.

Usage:
    python -m new_code.step1_entity_relation.merge_results
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NER_ENTITY_OUTPUT, NER_RELATION_OUTPUT, NER_MERGED_OUTPUT, NER_OUTPUT_DIR


def merge_entities_and_relations(entity_path, relation_path, output_path):
    """
    Merge entity file and relation file into a single JSON.
    The relation file provides: subject, relation, object, cue, reference_target.
    The entity file provides: quid, question, entities, time, qtype, answers, etc.
    """
    with open(entity_path, "r", encoding="utf-8") as f:
        data_entity = json.load(f)
    with open(relation_path, "r", encoding="utf-8") as f:
        data_relation = json.load(f)

    # Index relations by quid
    relation_index = {item["quid"]: item for item in data_relation}

    # Fields to remove from final output (intermediate NER artifacts)
    remove_keys = ["ner_recognized_entities", "semantic_top1_entities", "fuzzy_top1_entities"]

    for item in data_entity:
        qid = item.get("quid")
        if qid in relation_index:
            rel_item = relation_index[qid]
            item["subject"] = rel_item.get("subject", "None")
            item["relation"] = rel_item.get("relation", "None")
            item["object"] = rel_item.get("object", "None")
            item["cue"] = rel_item.get("cue", "None")
            item["reference_target"] = rel_item.get("reference_target", "None")

        # Remove intermediate fields
        for k in remove_keys:
            item.pop(k, None)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_entity, f, ensure_ascii=False, indent=4)
    print(f"Merged results saved to {output_path} ({len(data_entity)} items)")


def main():
    """Merge entity and relation results."""
    merge_entities_and_relations(NER_ENTITY_OUTPUT, NER_RELATION_OUTPUT, NER_MERGED_OUTPUT)


if __name__ == "__main__":
    main()
