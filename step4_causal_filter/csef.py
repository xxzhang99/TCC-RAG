"""
Step 4 (v2): Causal Sufficient Evidence Filter (CSEF).

Implements the post-revision design described in Section 3.3 of the paper:

  * Stage 1 (always-on, deterministic, O(|F|)): Causal-Equivalent Deduplication
      via the query-conditioned signature  sigma(e | Q) = (h, r, o, kappa(tau, c_hat, tau_b)).
  * Stage 2 (always-on, deterministic, O(|F^d|)): Greedy Causal Sufficiency
      Selection enforcing (i) cue coverage, (ii) boundary preservation,
      (iii) chain connectivity. Falls back to F^d if (iii) is violated.
  * Stage 3 (optional, LLM-gated by gamma): Mutual Entailment Verification.
      Triggered ONLY when |F^d| / |F^c| > gamma. Compares the structured
      causal traces of F^d and F^c rather than answer-level consistency.

Hyperparameters (interpretable, training-free):
  * DELTA_DAYS = 1   : kappa temporal-bucket width for range cues (before/after).
  * GAMMA      = 3.0 : verifier trigger threshold on |F^d| / |F^c|.

Replaces the legacy `causal_filter.py` whose trigger formula
  score = alpha * (N / L0) + beta * (1 - unique/N) > tau
and answer-level consistency check correspond to the OLD design and are no
longer reflected in the paper.

Usage:
    python -m new_code.step4_causal_filter.csef
"""
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    GRAPH_RETRIEVE_OUTPUT, CAUSAL_FILTER_OUTPUT,
)


# ============================================================
# Hyperparameters (Section 3.3 of the paper)
# ============================================================
DELTA_DAYS = 1     # kappa temporal-bucket width for before/after cues
GAMMA      = 3.0   # |F^d|/|F^c| ratio threshold for the optional verifier


# ============================================================
# Fact (de)serialization helpers
# ============================================================

def parse_fact(fact_str: str):
    """Parse a tab-separated fact 's\\tr\\to\\tt' into (s, r, o, datetime)."""
    parts = fact_str.split("\t")
    if len(parts) < 4:
        return None
    s, r, o, t = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
    try:
        ts = datetime.fromisoformat(t)
    except Exception:
        return None
    return (s, r, o, ts)


def fact_str(s, r, o, ts) -> str:
    return f"{s}\t{r}\t{o}\t{ts.date()}"


# ============================================================
# kappa(tau, c_hat, tau_b): cue-specific temporal normalization
# ============================================================

def kappa(ts: datetime, cue: str, anchor_ts=None,
          ordinal=None, delta_days: int = DELTA_DAYS):
    """
    Cue-specific temporal normalization function used inside the causal
    signature. Two facts that share kappa() are causally equivalent under Q.

      * 'before' / 'after' : bucketize ts into delta_days-wide buckets,
                             relative to the anchor timestamp tau_b if given.
      * 'same'             : preserve the exact day (only same-day facts collide).
      * 'first' / 'last'   : encode ordinal rank within the (s, r, o) group
                             (caller pre-computes `ordinal`).
      * default            : exact day.
    """
    if cue in ("before", "after"):
        if anchor_ts is None:
            return ts.toordinal() // delta_days
        offset_days = (ts - anchor_ts).days
        return offset_days // delta_days
    if cue == "same":
        return ts.toordinal()
    if cue in ("first", "last"):
        return ordinal
    return ts.toordinal()


# ============================================================
# Stage 1: Causal-Equivalent Deduplication
# ============================================================

def stage1_deduplicate(facts, cue: str, anchor_ts=None):
    """
    Partition F into equivalence classes under the signature
        sigma(e | Q) = (h, r, o, kappa(tau, c_hat, tau_b))
    and return F^d with one canonical representative per class.

    Returns:
        F_d:   list of representative fact strings (sorted by ts ascending).
        groups: dict signature -> list of (s, r, o, ts) members.
    """
    parsed = [parse_fact(f) for f in facts]
    parsed = [p for p in parsed if p is not None]

    # Pre-compute ordinal ranks for first/last cues so we can pass into kappa.
    ordinal_map = {}
    if cue in ("first", "last"):
        bucket = defaultdict(list)
        for s, r, o, ts in parsed:
            bucket[(s, r, o)].append(ts)
        for k in bucket:
            bucket[k].sort()
        for (s, r, o), tss in bucket.items():
            n = len(tss)
            for rank, ts in enumerate(tss):
                ordinal_map[(s, r, o, ts)] = rank if cue == "first" else (n - 1 - rank)

    groups = defaultdict(list)
    for s, r, o, ts in parsed:
        ord_rank = ordinal_map.get((s, r, o, ts))
        sig = (s, r, o, kappa(ts, cue, anchor_ts=anchor_ts, ordinal=ord_rank))
        groups[sig].append((s, r, o, ts))

    F_d = []
    for sig, members in groups.items():
        rep = min(members, key=lambda x: x[3])  # earliest ts as canonical
        F_d.append(fact_str(*rep))
    F_d.sort(key=lambda f: parse_fact(f)[3])
    return F_d, groups


# ============================================================
# Stage 2: Greedy Causal Sufficiency Selection
# ============================================================

def _cue_consistent(parsed_fact, cue, anchor_ts):
    """Whether `parsed_fact`'s ts satisfies cue c_hat relative to anchor tau_b."""
    _, _, _, ts = parsed_fact
    if cue == "before":
        return anchor_ts is None or ts < anchor_ts
    if cue == "after":
        return anchor_ts is None or ts > anchor_ts
    if cue == "same":
        return anchor_ts is None or ts.date() == anchor_ts.date()
    return True  # first / last handled in boundary preservation


def _chain_connected(F_c_parsed, cue, target_relation):
    """
    Lightweight connectivity check for condition (iii):
    consecutive facts share the queried relation (or relation is unspecified)
    and respect the temporal direction implied by the cue.
    """
    if len(F_c_parsed) < 2:
        return True
    for i in range(len(F_c_parsed) - 1):
        _, r1, _, t1 = F_c_parsed[i]
        _, r2, _, t2 = F_c_parsed[i + 1]
        if target_relation and (r1 != target_relation or r2 != target_relation):
            return False
        if cue in ("before", "first", "after", "last") and t2 < t1:
            return False
    return True


def stage2_select(F_d, groups, cue: str, anchor_ts=None, target_relation=None):
    """
    Greedy selection of a minimal F^c subset of F^d satisfying:
      (i)   cue coverage         : every cue-consistent class is represented;
      (ii)  boundary preservation: position-sensitive cues keep boundary events;
      (iii) chain connectivity   : consecutive facts form a coherent step.

    Falls back to F^d if (iii) is violated.
    Returns the F^c list of fact strings (sorted by ts ascending).
    """
    parsed = [parse_fact(f) for f in F_d if parse_fact(f) is not None]
    if not parsed:
        return F_d

    F_c = []

    # (ii) Boundary preservation -----------------------------------------------
    if cue == "first":
        F_c.append(min(parsed, key=lambda x: x[3]))
    elif cue == "last":
        F_c.append(max(parsed, key=lambda x: x[3]))
    elif cue == "same" and anchor_ts is not None:
        for p in parsed:
            if p[3].date() == anchor_ts.date():
                F_c.append(p)

    # (i) Cue coverage ---------------------------------------------------------
    for sig, members in groups.items():
        rep = min(members, key=lambda x: x[3])
        if not _cue_consistent(rep, cue, anchor_ts):
            continue
        if rep not in F_c:
            F_c.append(rep)

    # Dedup by (s, r, o, ts) and sort
    seen = {}
    for p in F_c:
        seen[(p[0], p[1], p[2], p[3])] = p
    F_c = sorted(seen.values(), key=lambda x: x[3])

    # (iii) Chain connectivity --------------------------------------------------
    if not _chain_connected(F_c, cue, target_relation):
        return F_d  # safe fallback

    return [fact_str(*p) for p in F_c]


# ============================================================
# Stage 3: Mutual Entailment Verification (optional)
# ============================================================

class MutualEntailmentVerifier:
    """
    LLM-based verifier that compares the structured causal traces of F^d and F^c
    rather than their answers, addressing the criticism in R5 Q4 that
    answer-level consistency implicitly assumes the unfiltered evidence is
    already correct.
    """

    TRACE_PROMPT = """You are a Temporal Causal Trace Generator.

Given a Question and a list of Historical facts, output ONLY a JSON object that
captures (a) the key events relevant to the question and (b) their temporal
ordering.

Format:
{{"trace": [{{"event": "<fact text>", "ts": "YYYY-MM-DD"}}, ...]}}

Rules:
- Each event must come EXACTLY from the input facts (verbatim).
- Events must be sorted by ts in ascending order.
- Drop any fact irrelevant to the question.
- Return valid JSON only. No prose, no code fences.

Question: "{question}"
Historical facts: {facts}
"""

    def __init__(self, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL

    def trace(self, question: str, facts):
        """Return P_F = [(event, ts), ...] for fact set F."""
        prompt = self.TRACE_PROMPT.format(question=question, facts=facts)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=0.9,
                stream=False,
            )
            raw = resp.choices[0].message.content.strip()
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return [(it["event"], it.get("ts", "")) for it in data.get("trace", [])]
        except Exception as e:
            print(f"[verifier] trace generation failed: {e}")
            return []

    @staticmethod
    def _entails(a, b):
        """a entails b iff every (event, ts) in b appears in a."""
        return set(b).issubset(set(a))

    def verify(self, F_d, F_c, question: str) -> bool:
        """
        Generate P_{F^d} and P_{F^c}, accept F^c iff the two traces mutually
        entail each other (i.e., share the same set of (event, ts) tuples).
        """
        P_d = self.trace(question, F_d)
        P_c = self.trace(question, F_c)
        return self._entails(P_d, P_c) and self._entails(P_c, P_d)


# ============================================================
# CSEF Orchestrator
# ============================================================

class CSEF:
    """End-to-end Causal Sufficient Evidence Filter."""

    def __init__(self, api_key=None):
        self.verifier = MutualEntailmentVerifier(api_key=api_key)

    def filter(self, facts, question: str, cue: str,
               anchor_ts=None, target_relation=None):
        """
        Args:
            facts:           list of '\\t'-separated fact strings.
            question:        natural-language query.
            cue:             one of {'before','after','same','first','last', None}.
            anchor_ts:       boundary timestamp tau_b (datetime or None).
            target_relation: grounded relation r* used in connectivity check.

        Returns:
            (F_final, stats)
        """
        # ----- Stage 1 -----
        F_d, groups = stage1_deduplicate(facts, cue=cue, anchor_ts=anchor_ts)
        # ----- Stage 2 -----
        F_c = stage2_select(F_d, groups, cue=cue,
                            anchor_ts=anchor_ts, target_relation=target_relation)

        stats = {
            "n_input":            len(facts),
            "n_dedup":            len(F_d),
            "n_select":           len(F_c),
            "verifier_triggered": False,
            "verifier_accepted":  None,
            "ratio":              None,
        }

        # ----- Stage 3 (optional) -----
        if F_c and len(F_c) > 0 and len(F_d) > 0:
            ratio = len(F_d) / len(F_c)
            stats["ratio"] = ratio
            if ratio > GAMMA:
                stats["verifier_triggered"] = True
                accepted = self.verifier.verify(F_d, F_c, question)
                stats["verifier_accepted"] = bool(accepted)
                return (F_c if accepted else F_d), stats

        return F_c, stats


# ============================================================
# Batch entry point (mirrors causal_filter.py I/O contract)
# ============================================================

def process_csef(input_path=None, output_path=None):
    input_path = input_path or GRAPH_RETRIEVE_OUTPUT
    output_path = output_path or CAUSAL_FILTER_OUTPUT
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    csef = CSEF()
    out = []
    for item in tqdm(items, desc="CSEF"):
        question = item.get("question", "")
        cue = item.get("cue")
        anchor_ts = None
        anchor_str = item.get("anchor_ts")
        if anchor_str:
            try:
                anchor_ts = datetime.fromisoformat(anchor_str)
            except Exception:
                anchor_ts = None
        target_rel = item.get("relation_canonical")

        # Collect candidate facts (mirror existing pipeline).
        qfacts = item.get("qfact", {})
        retrieved = item.get("retrieved", {})
        fact_list = []
        for rel, gfacts in retrieved.items():
            if qfacts.get(rel):
                fact_list.append(qfacts[rel][0])
            if gfacts:
                fact_list.extend(gfacts)

        F_final, stats = csef.filter(
            fact_list, question, cue=cue,
            anchor_ts=anchor_ts, target_relation=target_rel,
        )

        out.append({
            "quid":               item.get("quid"),
            "question":           question,
            "cue":                cue,
            "anchor_ts":          anchor_str,
            "n_input":            stats["n_input"],
            "n_dedup":            stats["n_dedup"],
            "n_select":           stats["n_select"],
            "verifier_triggered": stats["verifier_triggered"],
            "verifier_accepted":  stats["verifier_accepted"],
            "ratio":              stats["ratio"],
            "final_facts":        F_final,
            "qtype":              item.get("qtype"),
            "qlabel":             item.get("qlabel"),
            "answer_type":        item.get("answer_type"),
            "answer":             item.get("answer"),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    n_trig = sum(1 for r in out if r["verifier_triggered"])
    n_acc  = sum(1 for r in out if r["verifier_accepted"] is True)
    pct    = 100.0 * n_trig / max(len(out), 1)
    print(f"CSEF done. {len(out)} items.")
    print(f"  Verifier triggered on {n_trig} ({pct:.2f}%).")
    print(f"  Verifier accepted F^c on {n_acc} of those.")


if __name__ == "__main__":
    process_csef()
