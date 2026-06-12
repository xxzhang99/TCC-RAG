"""
Step 6: Evaluation - Compute Hit@k metrics.

Evaluates prediction results against ground truth answers.
Reports metrics grouped by answer_type, qlabel, and qtype.

Usage:
    python -m new_code.step6_evaluate.evaluate [path_to_predictions.json]
"""
import json
import os
import sys
import re
import string
import ast
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREDICT_OUTPUT


# ============================================================
# Text Normalization
# ============================================================

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s).lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(s1: str, s2: str) -> bool:
    """Return true if normalized s2 is contained in normalized s1."""
    return normalize(s2) in normalize(s1)


# ============================================================
# Prediction Handling
# ============================================================

def normalize_prediction(prediction):
    """Ensure prediction becomes a list of strings."""
    if isinstance(prediction, list):
        return prediction
    if isinstance(prediction, (int, float)):
        return [str(prediction)]
    if isinstance(prediction, str):
        try:
            return json.loads(prediction) if prediction.strip().startswith("[") else [prediction]
        except:
            return [prediction]
    return []


def top_k(pred_list, k):
    """Take top-k predictions."""
    if isinstance(pred_list, list):
        return pred_list[:k]
    return [pred_list]


# ============================================================
# Hit Metric
# ============================================================

def eval_hit(pred_list, gold_list):
    """
    Hit@k: 1 if any prediction matches any gold answer, else 0.
    """
    if isinstance(gold_list, str):
        try:
            gold_list = ast.literal_eval(gold_list)
        except:
            gold_list = [gold_list]

    for pred in pred_list:
        for gold in gold_list:
            if match(pred, gold):
                return 1
    return 0


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_items(items, k=1):
    """Evaluate a list of QA prediction items."""
    hit_list = []
    wrong_cases = []
    wrong_full = []

    stats_by_answer_type = defaultdict(lambda: {"hit": 0, "total": 0})
    stats_by_qlabel = defaultdict(lambda: {"hit": 0, "total": 0})

    qtypes = ["equal", "equal_multi", "before_after", "first_last", "before_last", "after_first"]
    stats_by_qtype = {q: {"hit": 0, "total": 0} for q in qtypes}

    for item in items:
        pred = normalize_prediction(item["model_answer"])
        gold = item["ground_truth"]
        qtype = item.get("qtype", "")
        qlabel = item.get("qlabel", "")
        level = item.get("time_level", "None")
        atype = item.get("answer_type", "")

        pred_k = top_k(pred, k)
        hit = eval_hit(pred_k, gold)
        hit_list.append(hit)

        if hit == 0:
            wrong_cases.append({
                "question": item["question"],
                "prediction": pred_k,
                "gold_answer": gold,
                "qtype": qtype,
                "qlabel": qlabel,
                "answer_type": atype,
                "time_level": level
            })
            wrong_full.append(item)

        stats_by_answer_type[atype]["hit"] += hit
        stats_by_answer_type[atype]["total"] += 1
        stats_by_qlabel[qlabel]["hit"] += hit
        stats_by_qlabel[qlabel]["total"] += 1

        if qtype in stats_by_qtype:
            stats_by_qtype[qtype]["hit"] += hit
            stats_by_qtype[qtype]["total"] += 1

    return {
        "hit_rate": sum(hit_list) / len(hit_list) if hit_list else 0,
        "total": len(hit_list),
        "stats_by_answer_type": stats_by_answer_type,
        "stats_by_qlabel": stats_by_qlabel,
        "stats_by_qtype": stats_by_qtype,
        "wrong_cases": wrong_cases,
        "wrong_full": wrong_full
    }


def evaluate_file(json_path, k=1, save_wrong=True):
    """Evaluate a prediction JSON file and print results."""
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    result = evaluate_items(items, k=k)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results (Hit@{k})")
    print(f"{'='*50}")
    hit = result["hit_rate"] * 100
    print(f"\nOverall Hit@{k}: {hit:.2f}% ({int(result['hit_rate'] * result['total'])}/{result['total']})")

    print("\nBy Answer Type:")
    for atype, s in result["stats_by_answer_type"].items():
        acc = s["hit"] * 100 / s["total"] if s["total"] > 0 else 0
        print(f"  {atype}: {acc:.2f}% ({s['hit']}/{s['total']})")

    print("\nBy QLabel:")
    for label, s in result["stats_by_qlabel"].items():
        acc = s["hit"] * 100 / s["total"] if s["total"] > 0 else 0
        print(f"  {label}: {acc:.2f}% ({s['hit']}/{s['total']})")

    print("\nBy QType:")
    for qtype, s in result["stats_by_qtype"].items():
        if s["total"] > 0:
            acc = s["hit"] * 100 / s["total"]
            print(f"  {qtype}: {acc:.2f}% ({s['hit']}/{s['total']})")

    # Save wrong cases
    if save_wrong:
        base_dir = os.path.dirname(json_path)
        file_name = os.path.splitext(os.path.basename(json_path))[0]
        save_dir = os.path.join(base_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)

        wrong_cases_path = os.path.join(save_dir, f"wrong_cases_k{k}.json")
        wrong_full_path = os.path.join(save_dir, f"wrong_full_k{k}.json")

        with open(wrong_cases_path, "w", encoding="utf-8") as f:
            json.dump(result["wrong_cases"], f, indent=2, ensure_ascii=False)
        with open(wrong_full_path, "w", encoding="utf-8") as f:
            json.dump(result["wrong_full"], f, indent=2, ensure_ascii=False)

        print(f"\nWrong cases saved to:\n  {wrong_cases_path}\n  {wrong_full_path}")

    return result


def main():
    """Evaluate prediction results."""
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = PREDICT_OUTPUT

    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    evaluate_file(json_path, k=k)


if __name__ == "__main__":
    main()
