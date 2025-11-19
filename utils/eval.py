import re
import string
from collections import Counter
from typing import Callable, Dict, Any
from datasets import Dataset


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ["yes", "no"] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ["yes", "no"] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def evaluate_qa_system(ds_val: Dataset, predict_fn: Callable[[str, int], str], n: int = 100, k: int = 5) -> Dict[str, Any]:
    """
    Evaluate a QA system on the first n validation examples with EM and F1.
    predict_fn(question, k) -> predicted answer (string)
    """
    idxs = list(range(min(n, len(ds_val))))
    ems, f1s = [], []

    for i in idxs:
        ex = ds_val[i]
        q = ex["question"]
        ground_truth = ex["answer"]

        pred = predict_fn(q, k)

        print(f"Q: {q}")
        print(f"Pred: {pred}")
        print(f"Ground Truth: {ground_truth}")

        ems.append(exact_match_score(pred, ground_truth))
        f1s.append(f1_score(pred, ground_truth))

    m = len(idxs) if idxs else 1
    return {
        "n": len(idxs),
        "k": k,
        "EM": sum(ems) / m,
        "F1": sum(f1s) / m,
    }