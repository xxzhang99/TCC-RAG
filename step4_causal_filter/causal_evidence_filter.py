"""
Step 4: Causal Evidence Filter (CEF).

Supports two input modes:

  semantic  (primary): input is semantic_retrieve_results.json whose facts are
            in the format "S R O in YYYY-MM-DD".  Facts are first converted to
            the tab-separated "S\\tR\\tO\\tDATE" format required by CSEF, then
            filtered.  This is the main use case.

  graph     (secondary): input is graph_retrieve_results.json whose facts are
            already tab-separated.  Filtering is applied directly.

Filtering pipeline (same for both modes):
  1. Trigger gate: causal_trigger_score > tau  (from causal_filter.py).
     Fact sets below the threshold are kept unchanged.
  2. Cue / anchor inference: from entities_relations_merged file (primary)
     with regex fallback on the question text.
  3. Pre-deduplication by (s, r, o): one boundary-representative fact per group.
  4. CSEF pipeline (csef.py): Stage-1 dedup -> Stage-2 selection
     -> Stage-3 optional LLM verification.

Output schema:
    quid / question / original_fact_num / final_fact_num / final_facts /
    qtype / qlabel / answer_type / answer

Usage:
    # semantic mode (default)
    python -m new_code.step4_causal_filter.causal_evidence_filter

    # graph mode
    python -m new_code.step4_causal_filter.causal_evidence_filter --mode graph
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEEPSEEK_API_KEY,
    CAUSAL_TRIGGER_L0, CAUSAL_TRIGGER_ALPHA, CAUSAL_TRIGGER_BETA, CAUSAL_TRIGGER_TAU,
    GRAPH_RETRIEVE_OUTPUT, SEMANTIC_RETRIEVE_OUTPUT, CAUSAL_FILTER_OUTPUT,
    QUESTIONS_PROCESSED_PATH,
)

from step4_causal_filter.csef import CSEF, parse_fact


# ============================================================
# Semantic fact format conversion
# ============================================================

# Global entity set built from the ER lookup; populated once via
# _build_global_entities() at startup.
_GLOBAL_ENTITIES: set = set()


def _build_global_entities(er_lookup: dict):
    """Populate _GLOBAL_ENTITIES from the loaded ER lookup (call once)."""
    global _GLOBAL_ENTITIES
    _GLOBAL_ENTITIES = set()
    for v in er_lookup.values():
        for e in (v.get("entities") or []):
            if e and e != "None":
                _GLOBAL_ENTITIES.add(e.lower())


def semantic_to_tab(fact_str: str, entities=None) -> str:
    """
    Convert a semantic fact string to tab-separated format.

    Input:  "Subject Relation Object in YYYY-MM-DD"
    Output: "Subject\\tRelation\\tObject\\tYYYY-MM-DD"

    Subject and Object boundaries are located by matching known entity names
    (question entities first, then the global entity set).  Falls back to
    first-token / last-token heuristic if no entity match is found.

    Returns None if the input cannot be parsed.
    """
    m = re.match(r"^(.+)\s+in\s+(\d{4}-\d{2}-\d{2})\s*$", fact_str.strip())
    if not m:
        return None
    sro, date = m.group(1).strip(), m.group(2)

    cands = sorted(
        [e for e in (entities or []) if e and e != "None"],
        key=len, reverse=True,
    )

    subj = obj = rel = None

    for ent in cands:
        if not sro.lower().startswith(ent.lower()):
            continue
        subj = sro[:len(ent)]
        rest = sro[len(ent):].strip()

        # Try question entities for O boundary
        for ent2 in cands:
            if ent2.lower() != ent.lower() and rest.lower().endswith(ent2.lower()):
                obj = rest[-len(ent2):]
                rel = rest[:-len(ent2)].strip()
                break

        # Try global entity set for O boundary
        if obj is None:
            for ge in sorted(_GLOBAL_ENTITIES, key=len, reverse=True):
                if rest.lower().endswith(ge):
                    obj = rest[-len(ge):]
                    rel = rest[:-len(ge)].strip()
                    break

        # Last-resort: trailing parenthesised phrase or last token as O
        if obj is None:
            pm = re.search(r"(\S+\s*\([^)]+\)|\S+)\s*$", rest)
            if pm:
                obj = pm.group(0).strip()
                rel = rest[:pm.start()].strip()

        if subj and rel and obj:
            break

    # Final fallback: first / middle / last token split
    if not (subj and rel and obj):
        tokens = sro.split()
        if len(tokens) < 3:
            return None
        subj = tokens[0]
        rel  = " ".join(tokens[1:-1])
        obj  = tokens[-1]

    return f"{subj}\t{rel}\t{obj}\t{date}"


def convert_semantic_facts(raw_facts: list, entities=None) -> list:
    """
    Convert a list of semantic fact strings to tab-separated format.
    Silently drops facts that cannot be parsed.
    """
    result = []
    for f in raw_facts:
        tab = semantic_to_tab(f, entities)
        if tab and parse_fact(tab) is not None:
            result.append(tab)
    return result


# ============================================================
# ER lookup loader
# ============================================================

def load_er_lookup(er_path: str) -> dict:
    """Load entities_relations_merged JSON and index by quid."""
    if not er_path or not os.path.exists(er_path):
        return {}
    with open(er_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["quid"]: item for item in data}


# ============================================================
# Cue normalisation
# ============================================================

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_RAW_CUE_MAP = {}
for _p in {"before", "last before"}:                          _RAW_CUE_MAP[_p] = "before"
for _p in {"after", "for the first time after"}:              _RAW_CUE_MAP[_p] = "after"
for _p in {"first", "for the first time"}:                    _RAW_CUE_MAP[_p] = "first"
for _p in {"last", "for the last time"}:                      _RAW_CUE_MAP[_p] = "last"
for _p in {
    "same", "on", "in", "at what time",
    "in the same month as", "in the same month of", "in the same month",
    "on the same day as", "on the same day of", "on the same day",
    "same day as", "same day of", "same day",
    "in the same year as", "in the same year of", "in the same year",
    "on the same year as", "on the same year of",
    "same year as", "same year of", "same year",
    "on the same month as", "on the same month of",
    "same month as", "same month of", "same month",
    "same as", "the same",
    "exact month", "exact month when", "exact month in which",
    "in which month", "in the year of",
}:                                                            _RAW_CUE_MAP[_p] = "same"

_QTYPE_TO_CUE = {
    "after_first":  "after",
    "before_last":  "before",
    "equal":        "same",
    "equal_multi":  "same",
}


def _normalise_raw_cue(raw: str):
    if not raw or raw == "None":
        return None
    low = raw.strip().lower()
    if low in _RAW_CUE_MAP:
        return _RAW_CUE_MAP[low]
    date_like = re.search(
        r"\b(\d{1,2}\s+)?(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+\d{4}\b"
        r"|\bin\s+\d{4}\b|\bon\s+\d{4}\b", low)
    if date_like:
        return "same"
    return None


# ============================================================
# Anchor parsing helpers
# ============================================================

def _parse_iso_date(s: str):
    parts = s.strip().split("-")
    try:
        if len(parts) == 3: return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2: return datetime(int(parts[0]), int(parts[1]), 1)
        if len(parts) == 1 and len(parts[0]) == 4: return datetime(int(parts[0]), 1, 1)
    except ValueError:
        pass
    return None


def _parse_date_from_text(text: str):
    m = re.search(
        r"(?:(\d{1,2})\s+)?(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})", text, re.IGNORECASE)
    if m:
        day   = int(m.group(1)) if m.group(1) else 1
        month = _MONTHS[m.group(2).lower()]
        year  = int(m.group(3))
        try:    return datetime(year, month, day)
        except: return datetime(year, month, 1)
    m = re.search(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
        try:    return datetime(y, mo, d)
        except: return datetime(y, mo, 1)
    m = re.search(r"\b(?:in|on)\s+(\d{4})\b", text, re.IGNORECASE)
    if m:
        return datetime(int(m.group(1)), 1, 1)
    return None


def _anchor_from_entity(ref_entity: str, fact_list, cue: str):
    if not ref_entity or ref_entity == "None":
        return None
    ref_lower = ref_entity.lower()
    ts_hits = []
    for f in fact_list:
        p = parse_fact(f)
        if p is None:
            continue
        s, r, o, ts = p
        if ref_lower in s.lower() or ref_lower in o.lower():
            ts_hits.append(ts)
    if not ts_hits:
        return None
    return min(ts_hits) if cue in ("after", "first", "same") else max(ts_hits)


def _anchor_from_question_regex(question: str, fact_list, cue: str):
    kw = None
    for k in ("before", "after", "first", "last", "same"):
        if re.search(r"\b" + k + r"\b", question, re.IGNORECASE):
            kw = k
            break
    if kw is None:
        return None
    m = re.search(kw + r"\s+([A-Z][\w().'-]*(?:\s+[A-Z][\w().'-]*)*)",
                  question, re.IGNORECASE)
    if not m:
        return None
    ref = m.group(1).strip().strip(".,?")
    return _anchor_from_entity(ref, fact_list, cue)


# ============================================================
# Unified cue / anchor inference
# ============================================================

def infer_cue_and_anchor(item, fact_list, er_lookup=None):
    """
    Infer (cue, anchor_ts) with priority:
      1. ER file: cue field -> time list -> reference_target -> raw cue date
      2. Regex fallback on question text
      3. qtype keyword fallback for cue
    """
    quid     = item.get("quid")
    qtype    = (item.get("qtype") or "").lower()
    question = item.get("question", "")

    er = (er_lookup or {}).get(quid, {})

    # Cue from ER file
    raw_cue = str(er.get("cue", "None")) if er else "None"
    cue = _normalise_raw_cue(raw_cue)

    if cue is None:
        cue = _QTYPE_TO_CUE.get(qtype)
        if cue is None:
            if qtype == "before_after":
                cue = "after" if "after" in question.lower() else "before"
            elif qtype == "first_last":
                cue = "first" if "first" in question.lower() else "last"
            else:
                for k in ("before", "after", "first", "last", "same"):
                    if re.search(r"\b" + k + r"\b", question, re.IGNORECASE):
                        cue = k
                        break

    # Anchor from ER file
    anchor_ts = None
    if er:
        for t in (er.get("time") or []):
            anchor_ts = _parse_iso_date(str(t))
            if anchor_ts:
                break
        if anchor_ts is None:
            ref = er.get("reference_target", "None")
            if ref and ref != "None":
                anchor_ts = _anchor_from_entity(ref, fact_list, cue or "after")
        if anchor_ts is None and raw_cue not in ("None", "before", "after", "first", "last", "same"):
            anchor_ts = _parse_date_from_text(raw_cue)

    if anchor_ts is None:
        anchor_ts = _parse_date_from_text(question)
    if anchor_ts is None and cue in ("before", "after", "same", "first", "last"):
        anchor_ts = _anchor_from_question_regex(question, fact_list, cue)

    return cue, anchor_ts


# ============================================================
# Trigger logic
# ============================================================

def parse_signature(fact: str):
    """Extract (S, R, O) signature from a tab-separated fact string."""
    try:
        s, r, o, t = fact.split("\t")
        return (s.strip(), r.strip(), o.strip())
    except Exception:
        return None


def causal_trigger_score(facts, L0=None, alpha=None, beta=None):
    """
    Compute causal filtering trigger score.
    score = alpha * (N / L0) + beta * (1 - unique_signatures / N)
    Returns: (score, length_factor, redundancy_factor)
    """
    L0    = L0    or CAUSAL_TRIGGER_L0
    alpha = alpha or CAUSAL_TRIGGER_ALPHA
    beta  = beta  or CAUSAL_TRIGGER_BETA
    N = len(facts)
    if N == 0:
        return 0.0, 0.0, 0.0
    length_factor     = N / L0
    signatures        = {parse_signature(f) for f in facts if parse_signature(f)}
    redundancy_factor = 1 - len(signatures) / N
    return alpha * length_factor + beta * redundancy_factor, length_factor, redundancy_factor


def should_trigger_causal_filter(facts, tau=None):
    """Check if causal filtering should be triggered for this fact set."""
    tau = tau or CAUSAL_TRIGGER_TAU
    score, _, _ = causal_trigger_score(facts)
    return score > tau


# ============================================================
# Pre-deduplication by (s, r, o)
# ============================================================

def prededup_by_sro(facts, cue, anchor_ts=None):
    """
    Group by (s, r, o), apply cue directional constraint, keep one
    boundary-representative fact per group.
    """
    groups = defaultdict(list)
    for f in facts:
        p = parse_fact(f)
        if p is None:
            continue
        s, r, o, ts = p
        if cue == "after"  and anchor_ts is not None and ts <= anchor_ts: continue
        if cue == "before" and anchor_ts is not None and ts >= anchor_ts: continue
        if cue == "same"   and anchor_ts is not None and ts.date() != anchor_ts.date(): continue
        groups[(s, r, o)].append((ts, f))

    result = []
    for entries in groups.values():
        entries.sort(key=lambda x: x[0])
        result.append(entries[0][1] if cue in ("after", "first", None) else entries[-1][1])
    return result


# ============================================================
# Main processing
# ============================================================

def process_causal_evidence_filter(input_path=None, output_path=None,
                                   er_path=None, mode="semantic"):
    """
    Run causal evidence filtering.

    Args:
        input_path:  Path to input JSON.
                     semantic mode: semantic_retrieve_results.json
                                    (facts under key "semantic_facts")
                     graph mode:    graph_retrieve_results.json
                                    (facts under keys "qfact" / "retrieved")
        output_path: Output file path.
        er_path:     Path to entities_relations_merged file.
                     Defaults to QUESTIONS_PROCESSED_PATH from config.
        mode:        "semantic" (default) or "graph".
    """
    if mode == "semantic":
        input_path  = input_path  or SEMANTIC_RETRIEVE_OUTPUT
    else:
        input_path  = input_path  or GRAPH_RETRIEVE_OUTPUT
    output_path = output_path or CAUSAL_FILTER_OUTPUT
    er_path     = er_path     or QUESTIONS_PROCESSED_PATH

    with open(input_path, "r", encoding="utf-8") as f:
        question_json = json.load(f)

    er_lookup = load_er_lookup(er_path)
    if er_lookup:
        _build_global_entities(er_lookup)
        print(f"Loaded ER lookup: {len(er_lookup)} entries.")
    else:
        print("No ER lookup found; using regex-only cue/anchor inference.")

    cef = CSEF(api_key=DEEPSEEK_API_KEY)
    results = []

    for item in tqdm(question_json, desc=f"CEF [{mode}]"):
        question = item["question"]
        quid     = item.get("quid")

        # Collect facts depending on mode
        if mode == "semantic":
            entities  = (er_lookup.get(quid) or {}).get("entities") or []
            fact_list = convert_semantic_facts(item.get("semantic_facts", []), entities)
        else:
            qfacts    = item.get("qfact", {})
            retrieved = item.get("retrieved", {})
            fact_list = []
            for rel, graph_facts in retrieved.items():
                if qfacts.get(rel):
                    fact_list.append(qfacts[rel][0])
                if graph_facts:
                    fact_list.extend(graph_facts)

        original_count = len(fact_list)

        # Trigger gate
        if should_trigger_causal_filter(fact_list):
            er_item        = er_lookup.get(quid) or {}
            meta           = {
                "quid":     quid,
                "question": question,
                "qtype":    er_item.get("qtype") or item.get("qtype"),
            }
            cue, anchor_ts = infer_cue_and_anchor(meta, fact_list, er_lookup)
            target_rel     = item.get("relation_canonical")

            deduped = prededup_by_sro(fact_list, cue, anchor_ts)
            if not deduped:
                deduped = fact_list

            final_facts, _stats = cef.filter(
                deduped, question, cue=cue,
                anchor_ts=anchor_ts, target_relation=target_rel,
            )
        else:
            final_facts = fact_list

        results.append({
            "quid":              quid,
            "question":          question,
            "original_fact_num": original_count,
            "final_fact_num":    len(final_facts),
            "final_facts":       final_facts,
            "qtype":             item.get("qtype"),
            "qlabel":            item.get("qlabel"),
            "answer_type":       item.get("answer_type"),
            "answer":            item.get("answer"),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    triggered = sum(1 for r in results if r["original_fact_num"] != r["final_fact_num"])
    print(f"CEF [{mode}] done. {len(results)} items saved to {output_path}")
    print(f"  Filtered: {triggered} / {len(results)} items "
          f"({100 * triggered / max(len(results), 1):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Causal Evidence Filter")
    parser.add_argument("--mode", choices=["semantic", "graph"],
                        default="semantic",
                        help="Input format: semantic (default) or graph")
    parser.add_argument("--input",  default=None, help="Override input JSON path")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    parser.add_argument("--er",     default=None, help="Override ER lookup JSON path")
    args = parser.parse_args()

    process_causal_evidence_filter(
        input_path=args.input,
        output_path=args.output,
        er_path=args.er,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
