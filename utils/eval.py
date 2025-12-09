import json
import re
import string
from collections import Counter
from typing import Callable, Dict, Any
from datasets import Dataset
from langchain_openai import ChatOpenAI
from utils.prompts import LLM_EVAL_SYSTEM_PROMPT, build_llm_eval_prompt
from dotenv import load_dotenv
from rouge_score import rouge_scorer

def normalize_answer(s):
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


def f1_score(prediction, ground_truth):
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


def exact_match_score(prediction, ground_truth):
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
def llm_eval_score(question, gold_answer, model_answer):
    user_prompt = build_llm_eval_prompt(question, gold_answer, model_answer)
    resp = llm.invoke(
        [
            {"role": "system", "content": LLM_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    raw = resp.content.strip()
    print(raw)
    ans = json.loads(raw)
    ans["score"] = float(ans["score"])

    return ans

def evaluate_qa_system(ds_val, predict_fn, n = 100, k = 5):

    idxs = list(range(min(n, len(ds_val))))
    ems, f1s, llm_evals = [], [], []

    for i in idxs:
        ex = ds_val[i]
        q  = ex["question"]
        gt = ex["answer"]

        pred = predict_fn(q, k)

        print(f"Q: {q}")
        print(f"Pred: {pred}")
        print(f"Ground Truth: {gt}\n")

        ems.append(exact_match_score(pred, gt))
        f1s.append(f1_score(pred, gt))
        llm_evals.append(llm_eval_score(q, gt, pred)["score"])

    m = len(idxs) or 1

    return {
        "n": m,
        "k": k,
        "EM": sum(ems)/m,
        "F1": sum(f1s)/m,
        "LLM Eval": sum(llm_evals)/m,
    }