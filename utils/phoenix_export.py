"""
Phoenix Export Module

This module handles exporting evaluation data to Phoenix for analysis.
"""

import pandas as pd
from typing import List, Dict, Optional
from utils.phoenix_config import is_phoenix_enabled, get_phoenix_manager


def export_evaluation_to_phoenix(
    questions: List[str],
    predictions: List[str],
    ground_truths: List[str],
    em_scores: List[float],
    f1_scores: List[float],
    llm_scores: List[float],
    error_categories: List[str],
    dataset_name: str,
    retrieval_mode: str,
    architecture: str,
    llm_explanations: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Export evaluation results as a pandas DataFrame for Phoenix analysis.
    
    Args:
        questions: List of questions
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        em_scores: List of exact match scores
        f1_scores: List of F1 scores
        llm_scores: List of LLM evaluation scores
        error_categories: List of error category strings
        dataset_name: Name of the dataset
        retrieval_mode: Retrieval mode used
        architecture: Architecture type (single_hop, multi_hop)
        llm_explanations: Optional list of LLM evaluation explanations
        
    Returns:
        pandas DataFrame with evaluation data, or None if Phoenix not enabled
    """
    if not is_phoenix_enabled():
        return None
    
    # Create DataFrame
    data = {
        "question": questions,
        "prediction": predictions,
        "ground_truth": ground_truths,
        "em_score": em_scores,
        "f1_score": f1_scores,
        "llm_score": llm_scores,
        "error_category": error_categories,
        "is_correct": [em == 1.0 for em in em_scores],
        "dataset": [dataset_name] * len(questions),
        "retrieval_mode": [retrieval_mode] * len(questions),
        "architecture": [architecture] * len(questions),
    }
    
    if llm_explanations:
        data["llm_explanation"] = llm_explanations
    
    df = pd.DataFrame(data)
    
    # Note: Phoenix doesn't have a direct log_dataset method in the current API
    # The data is already in traces, but we can return the DataFrame for
    # programmatic analysis or export to CSV/JSON
    return df


def export_evaluation_summary(
    eval_data: Dict[str, List],
    dataset_name: str,
    retrieval_mode: str,
    architecture: str,
) -> Dict[str, any]:
    """
    Create a summary of evaluation results for analysis.
    
    Args:
        eval_data: Dictionary with evaluation data from evaluate_qa_system
        dataset_name: Name of the dataset
        retrieval_mode: Retrieval mode used
        architecture: Architecture type
        
    Returns:
        Dictionary with summary statistics
    """
    if not eval_data:
        return {}
    
    error_categories = eval_data.get("error_categories", [])
    em_scores = eval_data.get("em_scores", [])
    f1_scores = eval_data.get("f1_scores", [])
    
    # Count error categories
    error_counts = {}
    for category in error_categories:
        error_counts[category] = error_counts.get(category, 0) + 1
    
    # Calculate statistics
    total = len(em_scores)
    correct = sum(em_scores)
    error_rate = (total - correct) / total if total > 0 else 0.0
    
    summary = {
        "dataset": dataset_name,
        "retrieval_mode": retrieval_mode,
        "architecture": architecture,
        "total_samples": total,
        "correct": correct,
        "error_rate": error_rate,
        "avg_f1": sum(f1_scores) / total if total > 0 else 0.0,
        "error_distribution": error_counts,
        "top_error_category": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else "none",
    }
    
    return summary

