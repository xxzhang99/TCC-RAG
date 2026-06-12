"""
Step 2a: Build Temporal Knowledge Graph with indexed structures.

Loads full.txt KG triples and constructs:
1. NetworkX MultiDiGraph (the full KG)
2. subj_relations index: (subject, relation) -> list of (s, o, r, datetime) facts
3. obj_relations index: (relation, object) -> list of (s, o, r, datetime) facts

This is the "light" indexed version used for efficient graph retrieval.

Usage:
    python -m new_code.step2_graph_construct.build_graph
"""
import os
import sys
import pickle
import networkx as nx
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KG_FULL_PATH, KG_GRAPH_PATH,
    KG_SUBJ_RELATIONS_PATH, KG_OBJ_RELATIONS_PATH,
)


# ============================================================
# Load Temporal Knowledge Graph
# ============================================================

def load_temporal_kg(kg_file_path) -> nx.MultiDiGraph:
    """
    Load full.txt into a NetworkX MultiDiGraph.
    Format: entity1 <tab> relation <tab> entity2 <tab> timestamp
    """
    graph = nx.MultiDiGraph()

    with open(kg_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            subj, rel, obj = parts[:3]
            timestamp = parts[3] if len(parts) > 3 else None

            if not subj or not obj:
                continue

            # Validate timestamp
            try:
                if timestamp:
                    datetime.fromisoformat(timestamp)
            except Exception:
                continue

            edge_key = f"{rel}_{timestamp or 'static'}_{line_num}"
            graph.add_edge(subj, obj, key=edge_key, relation=rel, timestamp=timestamp)

    print(f"KG loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    return graph


# ============================================================
# Build Indexed Structures + Causal Cue Graphs
# ============================================================

def build_indexed_structures(kg):
    """
    Build indexed data structures from the KG:
    - subj_relations: (subject, relation) -> [(s, o, r, datetime), ...]
    - obj_relations:  (relation, object)  -> [(s, o, r, datetime), ...]
    """
    if kg is None:
        raise ValueError("KG not loaded.")

    obj_relations  = defaultdict(list)
    subj_relations = defaultdict(list)

    for u, v, _, data in kg.edges(keys=True, data=True):
        rel = data.get("relation")
        ts  = data.get("timestamp")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue

        # Key format matches graph_retriever.py's lookup:
        # subj_relations[(subject, relation)] and obj_relations[(relation, object)]
        obj_relations[(rel, v)].append((u, v, rel, t))
        subj_relations[(u, rel)].append((u, v, rel, t))

    print(f"Subject relations: {len(subj_relations):,} groups")
    print(f"Object relations:  {len(obj_relations):,} groups")

    return kg, subj_relations, obj_relations


# ============================================================
# Save Structures
# ============================================================

def save_structures(kg, subj_relations, obj_relations,
                    graph_path=None, subj_path=None, obj_path=None):
    """Save all constructed structures to pickle files."""
    graph_path = graph_path or KG_GRAPH_PATH
    subj_path  = subj_path  or KG_SUBJ_RELATIONS_PATH
    obj_path   = obj_path   or KG_OBJ_RELATIONS_PATH

    for path in [graph_path, subj_path, obj_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(graph_path, 'wb') as f:
        pickle.dump(kg, f)
    print(f"KG graph saved to:        {graph_path}")

    with open(subj_path, 'wb') as f:
        pickle.dump(subj_relations, f)
    print(f"Subject relations saved to: {subj_path}")

    with open(obj_path, 'wb') as f:
        pickle.dump(obj_relations, f)
    print(f"Object relations saved to:  {obj_path}")


# ============================================================
# Main
# ============================================================

def main():
    """Build graph and all indexed structures from full.txt."""
    kg = load_temporal_kg(KG_FULL_PATH)
    kg, subj_relations, obj_relations = build_indexed_structures(kg)
    save_structures(kg, subj_relations, obj_relations)
    print("All graph structures built and saved successfully.")


if __name__ == "__main__":
    main()
