"""
Step 2b: Build full bidirectional causal cue graphs (non-indexed version).

This version builds complete MultiDiGraph causal cue graphs (not indexed subgraphs).
Each causal edge stores full metadata (timestamps, facts, direction).
Use this when you need the full causal graph structure for analysis.

For the lightweight indexed version used in production retrieval, see build_graph.py.

Usage:
    python -m new_code.step2_graph_construct.build_graph_full
"""
import os
import sys
import pickle
import networkx as nx
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KG_FULL_PATH, DATASET_ROOT
from step2_graph_construct.build_graph import load_temporal_kg


# ============================================================
# Build Full Causal Cue Graphs
# ============================================================

def build_bidirectional_causal_graphs(kg):
    """
    Build two full causal cue graphs:
    - cue_graph_obj: Object-centered (who interacted with same entity before/after)
    - cue_graph_subj: Subject-centered (same entity interacted with whom before/after)

    Each edge stores full metadata including timestamps, facts, and cue direction.
    """
    if kg is None:
        raise ValueError("KG not loaded.")

    cue_graph_obj = nx.MultiDiGraph()
    cue_graph_subj = nx.MultiDiGraph()

    obj_relations = defaultdict(list)
    subj_relations = defaultdict(list)

    # Step 1: Collect facts
    for u, v, _, data in kg.edges(keys=True, data=True):
        rel = data.get("relation")
        ts = data.get("timestamp")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue

        obj_relations[(v, rel)].append((u, v, rel, t))
        subj_relations[(u, rel)].append((u, v, rel, t))

    # Step 2: Add bidirectional causal edges
    def add_causal_edges(cue_graph, facts_dict, mode):
        for key, facts in facts_dict.items():
            facts.sort(key=lambda x: x[3])
            for i in range(len(facts) - 1):
                a1, b1, r1, t1 = facts[i]
                a2, b2, r2, t2 = facts[i + 1]

                fact1 = f"{a1}\t{r1}\t{b1}\t{t1.date()}"
                fact2 = f"{a2}\t{r2}\t{b2}\t{t2.date()}"

                key_after = f"{r1}_{t1.isoformat()}_{t2.isoformat()}"
                key_before = f"{r1}_{t2.isoformat()}_{t1.isoformat()}"

                # "after" edge: earlier -> later
                cue_graph.add_edge(
                    (a1 if mode == 'obj' else b1),
                    (a2 if mode == 'obj' else b2),
                    key=key_after,
                    cue="after",
                    relation=r1,
                    time_1=t1.isoformat(),
                    time_2=t2.isoformat(),
                    fact_1=fact1,
                    fact_2=fact2,
                    center=key[0],
                    mode=mode
                )

                # "before" edge: later -> earlier
                cue_graph.add_edge(
                    (a2 if mode == 'obj' else b2),
                    (a1 if mode == 'obj' else b1),
                    key=key_before,
                    cue="before",
                    relation=r1,
                    time_1=t2.isoformat(),
                    time_2=t1.isoformat(),
                    fact_1=fact2,
                    fact_2=fact1,
                    center=key[0],
                    mode=mode
                )

    add_causal_edges(cue_graph_obj, obj_relations, mode='obj')
    add_causal_edges(cue_graph_subj, subj_relations, mode='subj')

    print(f"Object-centered Causal Graph: {cue_graph_obj.number_of_edges():,} edges")
    print(f"Subject-centered Causal Graph: {cue_graph_subj.number_of_edges():,} edges")
    return cue_graph_obj, cue_graph_subj


# ============================================================
# Main
# ============================================================

def main():
    """Build full causal cue graphs and save as pickle."""
    output_obj_path = os.path.join(DATASET_ROOT, "kg/causal_cue_graph_obj.pkl")
    output_sub_path = os.path.join(DATASET_ROOT, "kg/causal_cue_graph_sub.pkl")

    kg = load_temporal_kg(KG_FULL_PATH)
    cue_graph_obj, cue_graph_subj = build_bidirectional_causal_graphs(kg)

    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)

    with open(output_obj_path, 'wb') as f:
        pickle.dump(cue_graph_obj, f)
    print(f"Full causal graph (obj) saved to: {output_obj_path}")

    with open(output_sub_path, 'wb') as f:
        pickle.dump(cue_graph_subj, f)
    print(f"Full causal graph (subj) saved to: {output_sub_path}")


if __name__ == "__main__":
    main()
