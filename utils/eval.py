import json
import re
import string
from collections import Counter
from typing import Callable, Dict, Any, List, Optional
from datasets import Dataset
from langchain_openai import ChatOpenAI
from utils.prompts import LLM_EVAL_SYSTEM_PROMPT, build_llm_eval_prompt
from utils.phoenix_config import (
    get_tracer, is_phoenix_enabled,
    EVAL_QUESTION, EVAL_GROUND_TRUTH, EVAL_PREDICTION,
    EVAL_EM_SCORE, EVAL_F1_SCORE, EVAL_LLM_SCORE, EVAL_ERROR_CATEGORY,
    EVAL_DATASET, EVAL_RETRIEVAL_MODE, EVAL_ARCHITECTURE,
    ERROR_CATEGORIES
)
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

class EvalSchema(BaseModel):
    score: float
    explanation: str

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
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0).with_structured_output(EvalSchema)
def llm_eval_score(question, gold_answer, model_answer):
    user_prompt = build_llm_eval_prompt(question, gold_answer, model_answer)
    resp = llm.invoke(
        [
            {"role": "system", "content": LLM_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {"score": resp.score, "explanation": resp.explanation}


def categorize_error(
    question: str,
    prediction: str,
    ground_truth: str,
    em_score: float,
    f1_score: float,
    llm_score: float,
    retrieved_docs: Optional[List[str]] = None
) -> str:
    """
    Categorize the type of error based on metrics and context.
    
    Args:
        question: The question being answered
        prediction: The model's prediction
        ground_truth: The correct answer
        em_score: Exact match score (0.0 or 1.0)
        f1_score: F1 score (0.0 to 1.0)
        llm_score: LLM evaluation score (0.0 to 1.0)
        retrieved_docs: Optional list of retrieved document contents
        
    Returns:
        Error category string from ERROR_CATEGORIES
    """
    # Perfect match - no error
    if em_score == 1.0:
        return "none"
    
    # Check for answer type mismatch (yes/no vs entity)
    gt_lower = ground_truth.lower().strip()
    pred_lower = prediction.lower().strip()
    
    if gt_lower in ["yes", "no"] and pred_lower not in ["yes", "no"]:
        return "answer_wrong_type"
    if pred_lower in ["yes", "no"] and gt_lower not in ["yes", "no"]:
        return "answer_wrong_type"
    
    # Check for hallucination (answer not supported by context)
    if retrieved_docs and llm_score < 0.5:
        # Check if ground truth or prediction appears in retrieved docs
        all_docs_text = " ".join(retrieved_docs).lower()
        gt_in_context = gt_lower in all_docs_text or any(
            word in all_docs_text for word in gt_lower.split() if len(word) > 3
        )
        pred_in_context = pred_lower in all_docs_text or any(
            word in all_docs_text for word in pred_lower.split() if len(word) > 3
        )
        
        if not pred_in_context and not gt_in_context:
            return "retrieval_irrelevant"
        elif not pred_in_context:
            return "answer_hallucination"
    
    # Check for incomplete answer
    pred_words = len(prediction.split())
    gt_words = len(ground_truth.split())
    
    if pred_words < gt_words * 0.5 and f1_score < 0.5:
        return "answer_incomplete"
    
    # Check for insufficient retrieval (low F1 but high LLM score suggests retrieval issue)
    if f1_score < 0.3 and llm_score > 0.6:
        return "retrieval_insufficient"
    
    # Partial match categories
    if f1_score >= 0.7:
        return "answer_partial_match"
    elif f1_score >= 0.3:
        return "answer_incorrect"
    else:
        return "answer_completely_wrong"

def evaluate_qa_system(
    ds_val,
    predict_fn,
    n=100,
    k=5,
    dataset_name: str = "unknown",
    retrieval_mode: str = "unknown",
    architecture: str = "single_hop",
    export_data: bool = False
) -> Dict[str, Any]:
    """
    Evaluate QA system with Phoenix tracing and error categorization.
    
    Args:
        ds_val: Validation dataset
        predict_fn: Function that takes (question, k) and returns prediction
        n: Number of samples to evaluate
        k: Number of documents to retrieve
        dataset_name: Name of the dataset
        retrieval_mode: Retrieval mode (faiss, bm25, hybrid)
        architecture: Architecture type (single_hop, multi_hop)
        export_data: Whether to return data for export
        
    Returns:
        Dictionary with metrics and optionally evaluation data
    """
    idxs = list(range(min(n, len(ds_val))))
    ems, f1s, llm_evals = [], [], []
    
    # Collect data for export if requested
    eval_data = {
        "questions": [],
        "predictions": [],
        "ground_truths": [],
        "em_scores": [],
        "f1_scores": [],
        "llm_scores": [],
        "error_categories": [],
        "llm_explanations": []
    }
    
    # Get tracer if Phoenix is enabled
    tracer = get_tracer() if is_phoenix_enabled() else None

    for i in idxs:
        ex = ds_val[i]
        q = ex["question"]
        gt = ex["answer"]

        # Create evaluation span if Phoenix is enabled
        span = None
        if tracer:
            span = tracer.start_span("evaluation")
            span.set_attribute(EVAL_QUESTION, q)
            span.set_attribute(EVAL_GROUND_TRUTH, gt)
            span.set_attribute(EVAL_DATASET, dataset_name)
            span.set_attribute(EVAL_RETRIEVAL_MODE, retrieval_mode)
            span.set_attribute(EVAL_ARCHITECTURE, architecture)
            span.set_attribute("eval.sample_index", i)
            span.set_attribute("eval.k", k)

        try:
            # Get prediction (this will be auto-traced by Phoenix if enabled)
            pred = predict_fn(q, k)
            
            # Calculate metrics
            em = exact_match_score(pred, gt)
            f1 = f1_score(pred, gt)
            llm_eval_result = llm_eval_score(q, gt, pred)
            llm_score = llm_eval_result["score"]
            llm_explanation = llm_eval_result.get("explanation", "")
            
            # Categorize error
            # Note: We don't have retrieved_docs here, but we can still categorize
            error_category = categorize_error(q, pred, gt, em, f1, llm_score, retrieved_docs=None)
            
            # Add attributes to span
            if span:
                span.set_attribute(EVAL_PREDICTION, pred)
                span.set_attribute(EVAL_EM_SCORE, float(em))
                span.set_attribute(EVAL_F1_SCORE, float(f1))
                span.set_attribute(EVAL_LLM_SCORE, float(llm_score))
                span.set_attribute(EVAL_ERROR_CATEGORY, error_category)
                span.set_attribute("eval.is_correct", em == 1.0)
                
                # Mark span as error if not correct
                if em == 0:
                    span.set_status(Status(StatusCode.ERROR, error_category))
                else:
                    span.set_status(Status(StatusCode.OK))
            
            ems.append(em)
            f1s.append(f1)
            llm_evals.append(llm_score)
            
            # Collect data for export
            if export_data:
                eval_data["questions"].append(q)
                eval_data["predictions"].append(pred)
                eval_data["ground_truths"].append(gt)
                eval_data["em_scores"].append(em)
                eval_data["f1_scores"].append(f1)
                eval_data["llm_scores"].append(llm_score)
                eval_data["error_categories"].append(error_category)
                eval_data["llm_explanations"].append(llm_explanation)
            
            print(f"Q: {q}")
            print(f"Pred: {pred}")
            print(f"Ground Truth: {gt}")
            if is_phoenix_enabled():
                print(f"Error Category: {error_category}")
            print()
            
        except Exception as e:
            # Handle errors in evaluation
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute(EVAL_ERROR_CATEGORY, "evaluation_error")
            print(f"Error evaluating sample {i}: {e}")
            # Add zero scores for failed evaluations
            ems.append(0.0)
            f1s.append(0.0)
            llm_evals.append(0.0)
        finally:
            if span:
                span.end()

    m = len(idxs) or 1
    
    result = {
        "n": m,
        "k": k,
        "EM": sum(ems)/m,
        "F1": sum(f1s)/m,
        "LLM Eval": sum(llm_evals)/m,
    }
    
    # Add evaluation data if requested
    if export_data:
        result["eval_data"] = eval_data
    
    return result