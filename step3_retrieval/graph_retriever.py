"""
Step 3a: Graph-based Fact Retrieval.

Routes questions by type (equal, equal_multi, first_last, before_after, before_last, after_first)
and retrieves relevant facts from the pre-built temporal knowledge graph structures.

Usage:
    python -m new_code.step3_retrieval.graph_retriever
"""
import os
import sys
import re
import json
import pickle
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from dateutil import parser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SENTENCE_TRANSFORMER_MODEL,
    KG_GRAPH_PATH, KG_SUBJ_RELATIONS_PATH, KG_OBJ_RELATIONS_PATH,
    KG_RELATION2ID_PATH, KG_ENTITY2ID_PATH,
    QUESTIONS_PROCESSED_PATH, GRAPH_RETRIEVE_OUTPUT
)


class GraphRetriever:
    """
    Graph-based temporal fact retrieval engine.
    Routes queries by type and retrieves facts using indexed structures.
    """

    def __init__(self, graph_path=None, subj_relations_path=None, obj_relations_path=None,
                 relation_path=None, entity_path=None):
        """Load all graph structures and initialize sentence transformer."""
        graph_path          = graph_path          or KG_GRAPH_PATH
        subj_relations_path = subj_relations_path or KG_SUBJ_RELATIONS_PATH
        obj_relations_path  = obj_relations_path  or KG_OBJ_RELATIONS_PATH
        relation_path       = relation_path       or KG_RELATION2ID_PATH
        entity_path         = entity_path         or KG_ENTITY2ID_PATH

        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        with open(subj_relations_path, 'rb') as f:
            self.subj_relations = pickle.load(f)
        with open(obj_relations_path, 'rb') as f:
            self.obj_relations = pickle.load(f)
        with open(relation_path, 'r') as f:
            self.relation_map = json.load(f)
            self.relation_map = {k.replace("_", " "): v for k, v in self.relation_map.items()}
        with open(entity_path, 'r') as f:
            self.entity_map = json.load(f)
            self.entity_map = {k.replace("_", " "): v for k, v in self.entity_map.items()}

        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device='cuda:0')

    # ============================================================
    # Main Retrieval Entry Point
    # ============================================================

    def graph_retrieve(self, question_json, save_path=None):
        """
        Retrieve facts for a list of questions.
        Routes each question to the appropriate handler based on qtype.
        """
        save_path = save_path or GRAPH_RETRIEVE_OUTPUT
        all_results = []
        empty_retrieved = []
        empty_count = 0

        for q in tqdm(question_json, desc="Graph Retrieval"):
            question = q["question"]
            quid = q["quid"]
            qtype = q.get("qtype", "")
            subject = q.get('subject', 'None')
            obj = q.get('object', 'None')
            relation = q.get("relation")

            # Validate entities exist in graph
            invalid_entity = False
            if subject != "None" and subject not in self.entity_map:
                invalid_entity = True
            if obj != "None" and obj not in self.entity_map:
                invalid_entity = True

            if invalid_entity:
                empty_retrieved.append({"quid": quid, "question": question})
                empty_count += 1
                all_results.append({
                    "quid": quid, "question": question,
                    "qfact": {}, "retrieved": {},
                    "qtype": qtype,
                    "qlabel": q.get('qlabel', ''),
                    "answer_type": q.get('answer_type', ''),
                    "answer": q.get('answers', [])
                })
                continue

            # Retrieve candidate relations via semantic similarity
            relation_candidates = self.relation_retrieve(relation)

            # Route to appropriate handler
            if qtype == 'equal':
                graph_facts = self._filter_equal(relation_candidates, q)
            elif qtype == 'equal_multi':
                graph_facts = self._filter_equal_multi(relation_candidates, q)
            elif qtype == 'first_last':
                graph_facts = self._filter_first_last(relation_candidates, q)
            elif qtype == 'before_after':
                graph_facts = self._filter_before_after(relation_candidates, q)
            elif qtype in ['before_last', 'after_first']:
                graph_facts = self._filter_bl_af(relation_candidates, q)
            else:
                tqdm.write(f"Unsupported qtype: {qtype}")
                continue

            # Check if retrieval is empty
            if isinstance(graph_facts, dict):
                retrieved_data = graph_facts.get("retrieved", {})
                is_all_empty = all(len(v) == 0 for v in retrieved_data.values())
                if is_all_empty:
                    empty_retrieved.append({"quid": quid, "question": question})
                    empty_count += 1
            elif not graph_facts:
                empty_retrieved.append({"quid": quid, "question": question})
                empty_count += 1

            result_item = {
                "quid": quid,
                "question": question,
                "qfact": graph_facts["qfact"] if isinstance(graph_facts, dict) else "",
                "retrieved": graph_facts["retrieved"] if isinstance(graph_facts, dict) else graph_facts,
                "qtype": qtype,
                "qlabel": q.get('qlabel', ''),
                "answer_type": q.get('answer_type', ''),
                "answer": q.get('answers', [])
            }
            all_results.append(result_item)

        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"Graph retrieve results saved to {save_path}")

        empty_save_path = save_path.replace(".json", "_empty.json")
        with open(empty_save_path, "w", encoding="utf-8") as f:
            json.dump(empty_retrieved, f, indent=4, ensure_ascii=False)
        print(f"Empty cases: {empty_count}, saved to {empty_save_path}")

        return all_results, empty_retrieved

    # ============================================================
    # Relation Retrieval (Semantic Similarity)
    # ============================================================

    def relation_retrieve(self, relation, top_k=3):
        """Find top-k matching relations from KG via sentence embedding similarity."""
        rels = [r.replace("_", " ") for r in self.relation_map.keys()]
        query_embedding = self.model.encode(relation.strip())
        passage_embedding = self.model.encode(rels)
        scores = util.dot_score(query_embedding, passage_embedding)
        indices = np.argsort(np.array(scores[0]))[::-1][:top_k]
        return [rels[i] for i in indices]

    # ============================================================
    # Time Utilities
    # ============================================================

    def match_time_by_level(self, dt_obj, timestamp_str, level):
        """Check if a timestamp matches at the given granularity (year/month/day)."""
        try:
            t2 = parser.parse(timestamp_str)
        except Exception:
            return False

        if level == 'year':
            return dt_obj.year == t2.year
        elif level == 'month':
            return dt_obj.year == t2.year and dt_obj.month == t2.month
        elif level == 'day':
            return dt_obj.date() == t2.date()
        return False

    def compare_by_level(self, t1, t2, level):
        """
        Compare t1 vs t2 at the given granularity.
        Returns: -1 (before), 0 (same), 1 (after)
        """
        if level == "year":
            if t1.year < t2.year: return -1
            elif t1.year > t2.year: return 1
            else: return 0

        if level == "month":
            if t1.year != t2.year:
                return -1 if t1.year < t2.year else 1
            if t1.month < t2.month: return -1
            elif t1.month > t2.month: return 1
            else: return 0

        if level == "day":
            if t1.date() < t2.date(): return -1
            elif t1.date() > t2.date(): return 1
            else: return 0

        return 0

    # ============================================================
    # Query Type Handlers
    # ============================================================

    def _filter_equal(self, relation_candidates, q):
        """Handle 'equal' type: facts matching a specific time point."""
        answer_type = q['answer_type']
        subject = q.get('subject', 'None')
        obj = q.get('object', 'None')
        result = {
            "qfact": {rel: [] for rel in relation_candidates},
            "retrieved": {rel: [] for rel in relation_candidates}
        }

        for rel in relation_candidates:
            if subject != 'None' and obj != 'None' and answer_type == 'time':
                edges = self.graph.get_edge_data(subject, obj)
                facts = [
                    f"{subject}\t{d.get('relation')}\t{obj}\t{d.get('timestamp')}"
                    for _, d in edges.items()
                    if d.get("relation") == rel
                ] if edges else []
                result["retrieved"][rel] = facts

            elif subject and subject != 'None' and obj == 'None' and answer_type == 'entity':
                facts = self.subj_relations.get((subject, rel), [])
                timestamp = q['time']
                time_level = q.get('time_level', '')
                if timestamp:
                    facts = [
                        f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                        for f in facts
                        if self.match_time_by_level(f[3], timestamp, time_level)
                    ]
                result["retrieved"][rel] = facts

            elif obj and obj != 'None' and subject == 'None' and answer_type == 'entity':
                facts = self.obj_relations.get((rel, obj), [])
                timestamp = q['time'][0]
                time_level = q.get('time_level', '')
                facts = [
                    f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                    for f in facts
                    if self.match_time_by_level(f[3], timestamp, time_level)
                ]
                result["retrieved"][rel] = facts
            else:
                tqdm.write(f"Unsupported case in _filter_equal, quid={q['quid']}")

        return result

    def _filter_equal_multi(self, relation_candidates, q):
        """Handle 'equal_multi' type: facts in the same time period as a reference event."""
        question = q['question']
        reference_target = q['reference_target']
        subject = q.get('subject', 'None')
        obj = q.get('object', 'None')
        result = {
            "qfact": {rel: [] for rel in relation_candidates},
            "retrieved": {rel: [] for rel in relation_candidates}
        }

        tag = 'sub_do'
        if 'the same' in question.lower():
            if subject == 'None':
                subject = reference_target
                tag = 'do_obj'
            elif obj == 'None':
                obj = reference_target

            edges_cur = self.graph.get_edge_data(subject, obj)

            for rel in relation_candidates:
                qfact_for_rel = []
                if edges_cur:
                    timestamps = [
                        d.get("timestamp")
                        for _, d in edges_cur.items()
                        if d.get("relation") == rel and d.get("timestamp")
                    ]
                else:
                    timestamps = []

                if timestamps:
                    qfact_for_rel = [f"{subject}\t{rel}\t{obj}\t{timestamps[-1]}"]
                else:
                    continue

                timestamp = timestamps[-1]
                time_level = q.get('time_level', '')

                if tag == 'do_obj':
                    pool = self.obj_relations.get((rel, obj), [])
                elif tag == 'sub_do':
                    pool = self.subj_relations.get((subject, rel), [])
                else:
                    continue

                facts = [
                    f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                    for f in pool
                    if self.match_time_by_level(f[3], timestamp, time_level)
                ]
                result["qfact"][rel] = qfact_for_rel
                result["retrieved"][rel] = facts

        elif 'first' in question.lower():
            timestamp = q['time'][0]
            time_level = q.get('time_level', '')
            for rel in relation_candidates:
                if subject and subject != 'None' and obj == 'None':
                    facts = self.subj_relations.get((subject, rel), [])
                    facts = [f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}" for f in facts
                             if self.match_time_by_level(f[3], timestamp, time_level)]
                    result["retrieved"][rel] = [facts[0]] if facts else []
                elif obj and obj != 'None' and subject == 'None':
                    facts = self.obj_relations.get((rel, obj), [])
                    facts = [f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}" for f in facts
                             if self.match_time_by_level(f[3], timestamp, time_level)]
                    result["retrieved"][rel] = [facts[0]] if facts else []

        elif 'last' in question.lower():
            timestamp = q['time'][0]
            time_level = q.get('time_level', '')
            for rel in relation_candidates:
                if subject and subject != 'None' and obj == 'None':
                    facts = self.subj_relations.get((subject, rel), [])
                    facts = [f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}" for f in facts
                             if self.match_time_by_level(f[3], timestamp, time_level)]
                    result["retrieved"][rel] = [facts[-1]] if facts else []
                elif obj and obj != 'None' and subject == 'None':
                    facts = self.obj_relations.get((rel, obj), [])
                    facts = [f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}" for f in facts
                             if self.match_time_by_level(f[3], timestamp, time_level)]
                    result["retrieved"][rel] = [facts[-1]] if facts else []
        else:
            tqdm.write(f"Unsupported case in _filter_equal_multi, quid={q['quid']}")

        return result

    def _filter_first_last(self, relation_candidates, q):
        """Handle 'first_last' type: earliest or latest fact."""
        question = q['question']
        answer_type = q['answer_type']
        subject = q.get('subject', 'None')
        obj = q.get('object', 'None')
        result = {
            "qfact": {rel: [] for rel in relation_candidates},
            "retrieved": {rel: [] for rel in relation_candidates}
        }

        for rel in relation_candidates:
            if subject != 'None' and obj != 'None' and answer_type == 'time':
                edges = self.graph.get_edge_data(subject, obj)
                facts = [
                    f"{subject}\t{d.get('relation')}\t{obj}\t{d.get('timestamp')}"
                    for _, d in edges.items()
                    if d.get("relation") == rel
                ] if edges else []

                if 'first' in question.lower():
                    result["retrieved"][rel] = [f for f in facts
                        if f.split('\t')[-1] == facts[0].split('\t')[-1]] if facts else []
                elif 'last' in question.lower():
                    result["retrieved"][rel] = [f for f in facts
                        if f.split('\t')[-1] == facts[-1].split('\t')[-1]] if facts else []

            elif subject != 'None' and obj == 'None' and answer_type == 'entity':
                facts = self.subj_relations.get((subject, rel), [])
                facts = [f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}" for f in facts]
                if 'first' in question.lower():
                    result["retrieved"][rel] = [f for f in facts
                        if f.split('\t')[-1] == facts[0].split('\t')[-1]] if facts else []
                elif 'last' in question.lower():
                    result["retrieved"][rel] = [f for f in facts
                        if f.split('\t')[-1] == facts[-1].split('\t')[-1]] if facts else []

            elif obj != 'None' and subject == 'None' and answer_type == 'entity':
                facts = self.obj_relations.get((rel, obj), [])
                facts = [f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}" for f in facts]
                if 'first' in question.lower():
                    result["retrieved"][rel] = [f for f in facts
                        if f.split('\t')[-1] == facts[0].split('\t')[-1]] if facts else []
                elif 'last' in question.lower():
                    result["retrieved"][rel] = [f for f in facts
                        if f.split('\t')[-1] == facts[-1].split('\t')[-1]] if facts else []
            else:
                tqdm.write(f"Unsupported case in _filter_first_last, quid={q['quid']}")

        return result

    def _filter_before_after(self, relation_candidates, q):
        """Handle 'before_after' type: facts before/after a time boundary."""
        question = q["question"].lower()
        subject = q.get("subject", "None")
        obj = q.get("object", "None")
        reference_target = q.get("reference_target", "")
        time_list = q.get("time", [])
        time_level = q.get("time_level", "")

        result = {
            "qfact": {rel: [] for rel in relation_candidates},
            "retrieved": {rel: [] for rel in relation_candidates}
        }

        global_time = parser.parse(time_list[0]) if time_list else None

        for rel in relation_candidates:
            boundary_time = global_time
            qfact_for_rel = []

            if boundary_time is None:
                if reference_target in self.entity_map:
                    if subject != "None" and obj == "None":
                        edges = self.graph.get_edge_data(subject, reference_target)
                    elif obj != "None" and subject == "None":
                        edges = self.graph.get_edge_data(reference_target, obj)
                    else:
                        edges = None

                    if edges:
                        timestamps = [
                            d["timestamp"] for _, d in edges.items()
                            if d.get("relation") == rel and d.get("timestamp")
                        ]
                        if timestamps:
                            t_last = parser.parse(timestamps[-1])
                            boundary_time = t_last
                            if subject != "None":
                                qfact_for_rel = [f"{subject}\t{rel}\t{reference_target}\t{t_last.date()}"]
                            else:
                                qfact_for_rel = [f"{reference_target}\t{rel}\t{obj}\t{t_last.date()}"]
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            if global_time is None:
                result["qfact"][rel] = qfact_for_rel

            # Get fact pool
            if subject == "None":
                pool = self.obj_relations.get((rel, obj), [])
            elif obj == "None":
                pool = self.subj_relations.get((subject, rel), [])
            else:
                continue

            # Filter by temporal direction
            if "before" in question:
                facts = [
                    f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                    for f in pool
                    if self.compare_by_level(f[3], boundary_time, time_level) == -1
                ] if pool and boundary_time else []
            elif "after" in question:
                facts = [
                    f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                    for f in pool
                    if self.compare_by_level(f[3], boundary_time, time_level) == 1
                ] if pool and boundary_time else []
            else:
                facts = []

            result["retrieved"][rel] = facts[:10] if facts else []

        return result

    def _filter_bl_af(self, relation_candidates, q):
        """Handle 'before_last' and 'after_first' types."""
        question = q["question"].lower()
        subject = q.get("subject", "None")
        obj = q.get("object", "None")
        reference_target = q.get("reference_target", "")
        time_list = q.get("time", [])
        time_level = q.get("time_level", "")

        result = {
            "qfact": {rel: [] for rel in relation_candidates},
            "retrieved": {rel: [] for rel in relation_candidates}
        }

        global_time = parser.parse(time_list[0]) if time_list else None

        for rel in relation_candidates:
            boundary_time = global_time
            qfact_for_rel = []

            if boundary_time is None:
                if reference_target in self.entity_map:
                    if subject != "None" and obj == "None":
                        edges = self.graph.get_edge_data(subject, reference_target)
                    elif obj != "None" and subject == "None":
                        edges = self.graph.get_edge_data(reference_target, obj)
                    else:
                        edges = None

                    if edges:
                        timestamps = [
                            d["timestamp"] for _, d in edges.items()
                            if d.get("relation") == rel and d.get("timestamp")
                        ]
                        if timestamps:
                            t_last = parser.parse(timestamps[-1])
                            boundary_time = t_last
                            if subject != "None":
                                qfact_for_rel = [f"{subject}\t{rel}\t{reference_target}\t{t_last.date()}"]
                            else:
                                qfact_for_rel = [f"{reference_target}\t{rel}\t{obj}\t{t_last.date()}"]
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            if global_time is None:
                result["qfact"][rel] = qfact_for_rel

            # Get fact pool
            if subject == "None":
                pool = self.obj_relations.get((rel, obj), [])
            elif obj == "None":
                pool = self.subj_relations.get((subject, rel), [])
            else:
                continue

            # Filter by combined temporal direction
            if "before" in question and "last" in question:
                facts = [
                    f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                    for f in pool
                    if self.compare_by_level(f[3], boundary_time, time_level) == -1
                ] if pool and boundary_time else []
                # Take facts with the latest timestamp (the "last" before boundary)
                result["retrieved"][rel] = [f for f in facts
                    if f.split('\t')[-1] == facts[-1].split('\t')[-1]] if facts else []

            elif "after" in question and "first" in question:
                facts = [
                    f"{f[0]}\t{f[2]}\t{f[1]}\t{f[3].date()}"
                    for f in pool
                    if self.compare_by_level(f[3], boundary_time, time_level) == 1
                ] if pool and boundary_time else []
                # Take facts with the earliest timestamp (the "first" after boundary)
                result["retrieved"][rel] = [f for f in facts
                    if f.split('\t')[-1] == facts[0].split('\t')[-1]] if facts else []
            else:
                result["retrieved"][rel] = []

        return result


# ============================================================
# Main
# ============================================================

def main():
    """Run graph retrieval on processed questions."""
    retriever = GraphRetriever()

    with open(QUESTIONS_PROCESSED_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    retriever.graph_retrieve(questions, save_path=GRAPH_RETRIEVE_OUTPUT)


if __name__ == '__main__':
    main()
