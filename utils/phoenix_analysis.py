"""
Phoenix Error Analysis Module

This module provides programmatic access to Phoenix trace data for error analysis,
pattern identification, and comparative evaluation of RAG configurations.

Uses Phoenix Client API to query spans and generate analysis reports.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime
import webbrowser

try:
    import phoenix as px
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("Warning: phoenix package not available. Install with: pip install arize-phoenix")

from utils.phoenix_config import get_phoenix_url, is_phoenix_enabled, ERROR_CATEGORIES


def get_phoenix_client(endpoint: str = "http://127.0.0.1:6006"):
    """
    Get Phoenix client for API access.
    
    Args:
        endpoint: Phoenix server endpoint URL (default: http://127.0.0.1:6006)
    
    Returns:
        Phoenix Client instance
    
    Raises:
        RuntimeError: If Phoenix is not available or cannot connect
    """
    if not PHOENIX_AVAILABLE:
        raise RuntimeError("Phoenix package not installed. Install with: pip install arize-phoenix")
    
    # Try to get URL from config if Phoenix was initialized in this process
    if is_phoenix_enabled():
        config_url = get_phoenix_url()
        if config_url:
            endpoint = config_url
    
    try:
        # Connect to Phoenix instance
        client = px.Client(endpoint=endpoint)
        return client
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Phoenix at {endpoint}: {e}\n"
            f"Make sure Phoenix server is running: phoenix serve"
        )


def fetch_eval_dataframe(
    project_name: Optional[str] = None,
    endpoint: str = "http://127.0.0.1:6006"
) -> pd.DataFrame:
    """
    Fetch evaluation spans as a pandas DataFrame.
    
    Args:
        project_name: Optional project name to filter by
        endpoint: Phoenix server endpoint URL
    
    Returns:
        DataFrame with evaluation span data
    """
    client = get_phoenix_client(endpoint=endpoint)
    
    try:
        # Get spans DataFrame from Phoenix
        # Try multiple methods as Phoenix API may vary by version
        spans_df = None
        
        # Try new API first, then fall back to deprecated method
        try:
            # Try new API (Phoenix 4.0+)
            if hasattr(client, 'spans') and hasattr(client.spans, 'get_spans_dataframe'):
                spans_df = client.spans.get_spans_dataframe()
            # Fall back to deprecated method
            elif hasattr(client, 'get_spans_dataframe'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    spans_df = client.get_spans_dataframe()
            elif hasattr(client, 'query_spans'):
                spans_df = client.query_spans()
            else:
                raise AttributeError("Could not find spans query method on Phoenix client")
        except Exception as method_error:
            # Try using the px module directly as fallback
            try:
                import phoenix as px
                session = px.active_session()
                if session:
                    spans_df = session.get_spans_dataframe()
                else:
                    raise RuntimeError("No active Phoenix session found")
            except Exception as e2:
                raise RuntimeError(f"Failed to fetch spans: {method_error}")
        
        if spans_df is None or spans_df.empty:
            print("No spans found in Phoenix. Run evaluations with --phoenix flag to populate data.")
            return pd.DataFrame()
        
        # Filter to evaluation spans
        if 'name' not in spans_df.columns:
            return pd.DataFrame()
        
        eval_spans = spans_df[spans_df['name'] == 'evaluation'].copy()
        
        if eval_spans.empty:
            return pd.DataFrame()
        
        # Extract attributes (Phoenix 12.x stores them in attributes.eval as nested dict)
        if 'attributes.eval' in eval_spans.columns:
            eval_attrs = eval_spans['attributes.eval'].apply(
                lambda x: x if isinstance(x, dict) else {}
            )
            
            # Extract each field from the nested dictionary
            eval_spans['question'] = eval_attrs.apply(lambda x: x.get('question'))
            eval_spans['ground_truth'] = eval_attrs.apply(lambda x: x.get('ground_truth'))
            eval_spans['prediction'] = eval_attrs.apply(lambda x: x.get('prediction'))
            eval_spans['em_score'] = eval_attrs.apply(lambda x: x.get('em_score'))
            eval_spans['f1_score'] = eval_attrs.apply(lambda x: x.get('f1_score'))
            eval_spans['llm_score'] = eval_attrs.apply(lambda x: x.get('llm_score'))
            eval_spans['error_category'] = eval_attrs.apply(lambda x: x.get('error_category'))
            eval_spans['dataset'] = eval_attrs.apply(lambda x: x.get('dataset'))
            eval_spans['retrieval_mode'] = eval_attrs.apply(lambda x: x.get('retrieval_mode'))
            eval_spans['architecture'] = eval_attrs.apply(lambda x: x.get('architecture'))
            eval_spans['is_correct'] = eval_attrs.apply(lambda x: x.get('is_correct'))
            eval_spans['sample_index'] = eval_attrs.apply(lambda x: x.get('sample_index'))
            eval_spans['k'] = eval_attrs.apply(lambda x: x.get('k'))
            
            # Convert numeric columns
            for col in ["em_score", "f1_score", "llm_score", "sample_index", "k"]:
                if col in eval_spans.columns:
                    eval_spans[col] = pd.to_numeric(eval_spans[col], errors="coerce")
        elif 'attributes' in eval_spans.columns:
            # Fallback: try old flattened format
            attrs_df = pd.json_normalize(eval_spans['attributes'])
            eval_spans = pd.concat([eval_spans.drop('attributes', axis=1), attrs_df], axis=1)
        
        return eval_spans
        
    except Exception as e:
        print(f"Error fetching spans: {e}")
        return pd.DataFrame()


def analyze_error_patterns(
    project_name: Optional[str] = None,
    min_occurrences: int = 2,
    endpoint: str = "http://127.0.0.1:6006"
) -> Dict[str, Any]:
    """
    Analyze error patterns across evaluation runs.
    
    Args:
        project_name: Project to analyze (currently unused, kept for API consistency)
        min_occurrences: Minimum occurrences to report a pattern
        endpoint: Phoenix server endpoint URL
    
    Returns:
        Dictionary with error analysis results
    """
    eval_df = fetch_eval_dataframe(project_name, endpoint=endpoint)
    
    if eval_df.empty:
        return {"error": "No evaluation data found. Run evaluations with --phoenix flag."}
    
    # Check if we have the error_category column after extraction
    if 'error_category' not in eval_df.columns:
        print(f"Available columns: {list(eval_df.columns)}")
        return {"error": "Could not find error_category column in span data. Extraction may have failed."}
    
    error_col = 'error_category'
    is_correct_col = 'is_correct'
    em_col = 'em_score'
    f1_col = 'f1_score'
    llm_col = 'llm_score'
    
    total_samples = len(eval_df)
    
    # Identify errors (where status is ERROR or is_correct is False)
    if is_correct_col:
        errors = eval_df[eval_df[is_correct_col] == False]
        correct = eval_df[eval_df[is_correct_col] == True]
    else:
        # Fallback to status code
        errors = eval_df[eval_df.get('status_code', '') == 'ERROR']
        correct = eval_df[eval_df.get('status_code', '') == 'OK']
    
    # Error category distribution
    error_counts = Counter(errors[error_col].dropna())
    error_distribution = {
        cat: {
            "count": count,
            "percentage": (count / total_samples) * 100 if total_samples > 0 else 0,
            "description": ERROR_CATEGORIES.get(cat, "Unknown error")
        }
        for cat, count in error_counts.items()
        if count >= min_occurrences
    }
    
    # Performance metrics
    metrics = {
        "total_samples": total_samples,
        "correct": len(correct),
        "incorrect": len(errors),
        "accuracy": (len(correct) / total_samples * 100) if total_samples > 0 else 0,
    }
    
    # Add score metrics if available
    if em_col:
        metrics["avg_em_score"] = float(pd.to_numeric(eval_df[em_col], errors='coerce').mean())
    if f1_col:
        metrics["avg_f1_score"] = float(pd.to_numeric(eval_df[f1_col], errors='coerce').mean())
    if llm_col:
        metrics["avg_llm_score"] = float(pd.to_numeric(eval_df[llm_col], errors='coerce').mean())
    
    # Top error categories
    top_errors = sorted(
        error_distribution.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:5]
    
    return {
        "metrics": metrics,
        "error_distribution": error_distribution,
        "top_errors": [{"category": cat, **data} for cat, data in top_errors],
        "total_error_types": len(error_distribution),
    }


def compare_retrieval_modes(
    project_name: Optional[str] = None,
    modes: Optional[List[str]] = None,
    endpoint: str = "http://127.0.0.1:6006"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare performance across different retrieval modes (FAISS, BM25, hybrid).
    
    Args:
        project_name: Project to analyze
        modes: List of retrieval modes to compare (None for all)
        endpoint: Phoenix server endpoint URL
    
    Returns:
        Dictionary with comparison results by retrieval mode
    """
    eval_df = fetch_eval_dataframe(project_name, endpoint=endpoint)
    
    if eval_df.empty:
        return {"error": "No evaluation data found."}
    
    # Find retrieval_mode column
    mode_col = next((c for c in ['eval.retrieval_mode', 'retrieval_mode', 'attributes.eval.retrieval_mode'] if c in eval_df.columns), None)
    if not mode_col:
        return {"error": "Could not find retrieval_mode column in span data."}
    
    # Find metric columns
    is_correct_col = next((c for c in ['eval.is_correct', 'is_correct'] if c in eval_df.columns), None)
    em_col = next((c for c in ['eval.em_score', 'em_score'] if c in eval_df.columns), None)
    f1_col = next((c for c in ['eval.f1_score', 'f1_score'] if c in eval_df.columns), None)
    llm_col = next((c for c in ['eval.llm_score', 'llm_score'] if c in eval_df.columns), None)
    error_col = next((c for c in ['eval.error_category', 'error_category'] if c in eval_df.columns), None)
    
    # Filter by modes if specified
    if modes:
        eval_df = eval_df[eval_df[mode_col].isin(modes)]
    
    # Group by retrieval mode
    results = {}
    for mode in eval_df[mode_col].unique():
        if pd.isna(mode):
            continue
        
        mode_df = eval_df[eval_df[mode_col] == mode]
        mode_results = {
            "total_samples": len(mode_df),
        }
        
        if is_correct_col:
            correct_count = mode_df[is_correct_col].sum()
            mode_results["accuracy"] = (correct_count / len(mode_df) * 100) if len(mode_df) > 0 else 0
        
        if em_col:
            mode_results["avg_em_score"] = float(pd.to_numeric(mode_df[em_col], errors='coerce').mean())
        if f1_col:
            mode_results["avg_f1_score"] = float(pd.to_numeric(mode_df[f1_col], errors='coerce').mean())
        if llm_col:
            mode_results["avg_llm_score"] = float(pd.to_numeric(mode_df[llm_col], errors='coerce').mean())
        
        if error_col and is_correct_col:
            errors = mode_df[mode_df[is_correct_col] == False]
            mode_results["error_distribution"] = dict(Counter(errors[error_col].dropna()))
        
        results[mode] = mode_results
    
    return results


def compare_architectures(
    project_name: Optional[str] = None,
    endpoint: str = "http://127.0.0.1:6006"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare single-hop vs multi-hop RAG architectures.
    
    Args:
        project_name: Project to analyze
        endpoint: Phoenix server endpoint URL
    
    Returns:
        Dictionary with comparison results by architecture
    """
    eval_df = fetch_eval_dataframe(project_name, endpoint=endpoint)
    
    if eval_df.empty:
        return {"error": "No evaluation data found."}
    
    # Find architecture column
    arch_col = next((c for c in ['eval.architecture', 'architecture', 'attributes.eval.architecture'] if c in eval_df.columns), None)
    if not arch_col:
        return {"error": "Could not find architecture column in span data."}
    
    # Find metric columns
    is_correct_col = next((c for c in ['eval.is_correct', 'is_correct'] if c in eval_df.columns), None)
    em_col = next((c for c in ['eval.em_score', 'em_score'] if c in eval_df.columns), None)
    f1_col = next((c for c in ['eval.f1_score', 'f1_score'] if c in eval_df.columns), None)
    llm_col = next((c for c in ['eval.llm_score', 'llm_score'] if c in eval_df.columns), None)
    error_col = next((c for c in ['eval.error_category', 'error_category'] if c in eval_df.columns), None)
    mode_col = next((c for c in ['eval.retrieval_mode', 'retrieval_mode'] if c in eval_df.columns), None)
    
    results = {}
    for arch in eval_df[arch_col].unique():
        if pd.isna(arch):
            continue
        
        arch_df = eval_df[eval_df[arch_col] == arch]
        arch_results = {
            "total_samples": len(arch_df),
        }
        
        if is_correct_col:
            correct_count = arch_df[is_correct_col].sum()
            arch_results["accuracy"] = (correct_count / len(arch_df) * 100) if len(arch_df) > 0 else 0
        
        if em_col:
            arch_results["avg_em_score"] = float(pd.to_numeric(arch_df[em_col], errors='coerce').mean())
        if f1_col:
            arch_results["avg_f1_score"] = float(pd.to_numeric(arch_df[f1_col], errors='coerce').mean())
        if llm_col:
            arch_results["avg_llm_score"] = float(pd.to_numeric(arch_df[llm_col], errors='coerce').mean())
        
        if error_col and is_correct_col:
            errors = arch_df[arch_df[is_correct_col] == False]
            arch_results["error_distribution"] = dict(Counter(errors[error_col].dropna()))
        
        if mode_col:
            arch_results["retrieval_modes"] = list(arch_df[mode_col].unique())
        
        results[arch] = arch_results
    
    return results


def identify_failure_clusters(
    project_name: Optional[str] = None,
    error_category: Optional[str] = None,
    top_n: int = 10,
    endpoint: str = "http://127.0.0.1:6006"
) -> List[Dict[str, Any]]:
    """
    Identify clusters of similar failures for targeted debugging.
    
    Args:
        project_name: Project to analyze
        error_category: Specific error category to focus on (None for all)
        top_n: Number of top failure examples to return
        endpoint: Phoenix server endpoint URL
    
    Returns:
        List of failure examples with context
    """
    eval_df = fetch_eval_dataframe(project_name, endpoint=endpoint)
    
    if eval_df.empty:
        return []
    
    # Find necessary columns
    is_correct_col = next((c for c in ['eval.is_correct', 'is_correct'] if c in eval_df.columns), None)
    error_col = next((c for c in ['eval.error_category', 'error_category'] if c in eval_df.columns), None)
    question_col = next((c for c in ['eval.question', 'question'] if c in eval_df.columns), None)
    pred_col = next((c for c in ['eval.prediction', 'prediction'] if c in eval_df.columns), None)
    gt_col = next((c for c in ['eval.ground_truth', 'ground_truth'] if c in eval_df.columns), None)
    llm_col = next((c for c in ['eval.llm_score', 'llm_score'] if c in eval_df.columns), None)
    em_col = next((c for c in ['eval.em_score', 'em_score'] if c in eval_df.columns), None)
    f1_col = next((c for c in ['eval.f1_score', 'f1_score'] if c in eval_df.columns), None)
    mode_col = next((c for c in ['eval.retrieval_mode', 'retrieval_mode'] if c in eval_df.columns), None)
    arch_col = next((c for c in ['eval.architecture', 'architecture'] if c in eval_df.columns), None)
    
    # Filter to errors
    if is_correct_col:
        errors = eval_df[eval_df[is_correct_col] == False].copy()
    else:
        errors = eval_df[eval_df.get('status_code', '') == 'ERROR'].copy()
    
    if error_category and error_col:
        errors = errors[errors[error_col] == error_category]
    
    if errors.empty:
        return []
    
    # Sort by worst performance (lowest LLM score if available)
    if llm_col:
        errors = errors.sort_values(llm_col, ascending=True)
    
    # Extract top N failures
    failures = []
    for _, row in errors.head(top_n).iterrows():
        failure = {}
        
        if question_col:
            failure["question"] = str(row.get(question_col, ""))
        if pred_col:
            failure["prediction"] = str(row.get(pred_col, ""))
        if gt_col:
            failure["ground_truth"] = str(row.get(gt_col, ""))
        if error_col:
            failure["error_category"] = str(row.get(error_col, ""))
        if em_col:
            failure["em_score"] = float(pd.to_numeric(row.get(em_col, 0), errors='coerce'))
        if f1_col:
            failure["f1_score"] = float(pd.to_numeric(row.get(f1_col, 0), errors='coerce'))
        if llm_col:
            failure["llm_score"] = float(pd.to_numeric(row.get(llm_col, 0), errors='coerce'))
        if mode_col:
            failure["retrieval_mode"] = str(row.get(mode_col, ""))
        if arch_col:
            failure["architecture"] = str(row.get(arch_col, ""))
        
        failure["span_id"] = str(row.get("context.span_id", row.get("span_id", "")))
        
        failures.append(failure)
    
    return failures


def generate_error_report(
    project_name: Optional[str] = None,
    output_format: str = "text",
    endpoint: str = "http://127.0.0.1:6006"
) -> str:
    """
    Generate a comprehensive error analysis report.
    
    Args:
        project_name: Project to analyze
        output_format: Output format ("text", "markdown", "html")
        endpoint: Phoenix server endpoint URL
    
    Returns:
        Formatted report string
    """
    # Gather all analysis data
    patterns = analyze_error_patterns(project_name, endpoint=endpoint)
    retrieval_comparison = compare_retrieval_modes(project_name, endpoint=endpoint)
    architecture_comparison = compare_architectures(project_name, endpoint=endpoint)
    top_failures = identify_failure_clusters(project_name, top_n=5, endpoint=endpoint)
    
    # Generate report based on format
    if output_format == "text":
        return _generate_text_report(patterns, retrieval_comparison, architecture_comparison, top_failures, project_name)
    elif output_format == "markdown":
        return _generate_markdown_report(patterns, retrieval_comparison, architecture_comparison, top_failures, project_name)
    elif output_format == "html":
        return _generate_html_report(patterns, retrieval_comparison, architecture_comparison, top_failures, project_name)
    else:
        raise ValueError(f"Unsupported format: {output_format}")


def open_phoenix_ui(
    filter_query: Optional[str] = None,
    endpoint: str = "http://127.0.0.1:6006"
):
    """
    Open Phoenix UI in web browser, optionally with a filter applied.
    
    Args:
        filter_query: Optional filter query to pre-apply in Phoenix UI
        endpoint: Phoenix server endpoint URL
    """
    # Try to get URL from config if Phoenix was initialized in this process
    if is_phoenix_enabled():
        config_url = get_phoenix_url()
        if config_url:
            endpoint = config_url
    
    url = endpoint
    if filter_query:
        # Phoenix UI supports URL parameters for filters
        url = f"{endpoint}?filter={filter_query}"
    
    print(f"Opening Phoenix UI: {url}")
    webbrowser.open(url)


def _generate_text_report(patterns, retrieval_comp, arch_comp, failures, project_name):
    """Generate plain text report."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Phoenix Error Analysis Report")
    lines.append(f"Project: {project_name or 'All Projects'}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Check for errors in data fetch
    if "error" in patterns:
        lines.append(f"ERROR: {patterns['error']}")
        lines.append("")
        return "\n".join(lines)
    
    # Overall metrics
    if "metrics" in patterns:
        lines.append("Overall Performance:")
        lines.append("-" * 80)
        m = patterns["metrics"]
        lines.append(f"  Total Samples: {m['total_samples']}")
        lines.append(f"  Accuracy: {m['accuracy']:.2f}%")
        if 'avg_em_score' in m:
            lines.append(f"  Average EM Score: {m['avg_em_score']:.3f}")
        if 'avg_f1_score' in m:
            lines.append(f"  Average F1 Score: {m['avg_f1_score']:.3f}")
        if 'avg_llm_score' in m:
            lines.append(f"  Average LLM Score: {m['avg_llm_score']:.3f}")
        lines.append("")
    
    # Top errors
    if "top_errors" in patterns and patterns["top_errors"]:
        lines.append("Top Error Categories:")
        lines.append("-" * 80)
        for err in patterns["top_errors"]:
            lines.append(f"  {err['category']}: {err['count']} ({err['percentage']:.1f}%)")
            lines.append(f"    → {err['description']}")
        lines.append("")
    
    # Retrieval mode comparison
    if retrieval_comp and "error" not in retrieval_comp:
        lines.append("Retrieval Mode Comparison:")
        lines.append("-" * 80)
        for mode, stats in retrieval_comp.items():
            lines.append(f"  {mode}:")
            lines.append(f"    Samples: {stats['total_samples']}")
            if 'accuracy' in stats:
                lines.append(f"    Accuracy: {stats['accuracy']:.2f}%")
            if 'avg_f1_score' in stats:
                lines.append(f"    Avg F1: {stats['avg_f1_score']:.3f}")
        lines.append("")
    
    # Architecture comparison
    if arch_comp and "error" not in arch_comp:
        lines.append("Architecture Comparison:")
        lines.append("-" * 80)
        for arch, stats in arch_comp.items():
            lines.append(f"  {arch}:")
            lines.append(f"    Samples: {stats['total_samples']}")
            if 'accuracy' in stats:
                lines.append(f"    Accuracy: {stats['accuracy']:.2f}%")
            if 'avg_f1_score' in stats:
                lines.append(f"    Avg F1: {stats['avg_f1_score']:.3f}")
        lines.append("")
    
    # Top failures
    if failures:
        lines.append("Top Failure Examples:")
        lines.append("-" * 80)
        for i, fail in enumerate(failures, 1):
            lines.append(f"  [{i}] {fail.get('error_category', 'Unknown')}")
            if 'question' in fail:
                lines.append(f"      Q: {fail['question'][:100]}...")
            if 'prediction' in fail:
                lines.append(f"      Pred: {fail['prediction'][:80]}...")
            if 'ground_truth' in fail:
                lines.append(f"      GT: {fail['ground_truth'][:80]}...")
            scores = []
            if 'em_score' in fail:
                scores.append(f"EM: {fail['em_score']:.2f}")
            if 'f1_score' in fail:
                scores.append(f"F1: {fail['f1_score']:.2f}")
            if 'llm_score' in fail:
                scores.append(f"LLM: {fail['llm_score']:.2f}")
            if scores:
                lines.append(f"      Scores - {', '.join(scores)}")
            lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def _generate_markdown_report(patterns, retrieval_comp, arch_comp, failures, project_name):
    """Generate Markdown report."""
    lines = []
    lines.append(f"# Phoenix Error Analysis Report")
    lines.append(f"**Project:** {project_name or 'All Projects'}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if "error" in patterns:
        lines.append(f"**ERROR:** {patterns['error']}")
        return "\n".join(lines)
    
    # Overall metrics
    if "metrics" in patterns:
        lines.append("## Overall Performance")
        m = patterns["metrics"]
        lines.append(f"- **Total Samples:** {m['total_samples']}")
        lines.append(f"- **Accuracy:** {m['accuracy']:.2f}%")
        if 'avg_em_score' in m:
            lines.append(f"- **Average EM Score:** {m['avg_em_score']:.3f}")
        if 'avg_f1_score' in m:
            lines.append(f"- **Average F1 Score:** {m['avg_f1_score']:.3f}")
        if 'avg_llm_score' in m:
            lines.append(f"- **Average LLM Score:** {m['avg_llm_score']:.3f}")
        lines.append("")
    
    # Top errors
    if "top_errors" in patterns and patterns["top_errors"]:
        lines.append("## Top Error Categories")
        lines.append("| Category | Count | Percentage | Description |")
        lines.append("|----------|-------|------------|-------------|")
        for err in patterns["top_errors"]:
            lines.append(f"| {err['category']} | {err['count']} | {err['percentage']:.1f}% | {err['description']} |")
        lines.append("")
    
    # Retrieval comparison
    if retrieval_comp and "error" not in retrieval_comp:
        lines.append("## Retrieval Mode Comparison")
        lines.append("| Mode | Samples | Accuracy | Avg F1 |")
        lines.append("|------|---------|----------|--------|")
        for mode, stats in retrieval_comp.items():
            acc = f"{stats['accuracy']:.2f}%" if 'accuracy' in stats else "N/A"
            f1 = f"{stats['avg_f1_score']:.3f}" if 'avg_f1_score' in stats else "N/A"
            lines.append(f"| {mode} | {stats['total_samples']} | {acc} | {f1} |")
        lines.append("")
    
    # Architecture comparison
    if arch_comp and "error" not in arch_comp:
        lines.append("## Architecture Comparison")
        lines.append("| Architecture | Samples | Accuracy | Avg F1 |")
        lines.append("|--------------|---------|----------|--------|")
        for arch, stats in arch_comp.items():
            acc = f"{stats['accuracy']:.2f}%" if 'accuracy' in stats else "N/A"
            f1 = f"{stats['avg_f1_score']:.3f}" if 'avg_f1_score' in stats else "N/A"
            lines.append(f"| {arch} | {stats['total_samples']} | {acc} | {f1} |")
        lines.append("")
    
    # Top failures
    if failures:
        lines.append("## Top Failure Examples")
        for i, fail in enumerate(failures, 1):
            lines.append(f"### {i}. {fail.get('error_category', 'Unknown')}")
            if 'question' in fail:
                lines.append(f"- **Question:** {fail['question'][:150]}...")
            if 'prediction' in fail:
                lines.append(f"- **Prediction:** {fail['prediction'][:100]}...")
            if 'ground_truth' in fail:
                lines.append(f"- **Ground Truth:** {fail['ground_truth'][:100]}...")
            scores = []
            if 'em_score' in fail:
                scores.append(f"EM={fail['em_score']:.2f}")
            if 'f1_score' in fail:
                scores.append(f"F1={fail['f1_score']:.2f}")
            if 'llm_score' in fail:
                scores.append(f"LLM={fail['llm_score']:.2f}")
            if scores:
                lines.append(f"- **Scores:** {', '.join(scores)}")
            lines.append("")
    
    return "\n".join(lines)


def _generate_html_report(patterns, retrieval_comp, arch_comp, failures, project_name):
    """Generate HTML report with basic styling."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phoenix Error Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }}
        .metric p {{ margin: 5px 0; }}
        .failure {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
        .success {{ color: #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Phoenix Error Analysis Report</h1>
        <p><strong>Project:</strong> {project_name or 'All Projects'}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    if "error" in patterns:
        html += f"""
        <p class="error">ERROR: {patterns['error']}</p>
    </div>
</body>
</html>
"""
        return html
    
    # Add metrics
    if "metrics" in patterns:
        m = patterns["metrics"]
        html += """
        <h2>Overall Performance</h2>
        <div class="metric">
"""
        html += f"            <p><strong>Total Samples:</strong> {m['total_samples']}</p>\n"
        html += f"            <p><strong>Accuracy:</strong> <span class='success'>{m['accuracy']:.2f}%</span></p>\n"
        if 'avg_em_score' in m:
            html += f"            <p><strong>Average EM Score:</strong> {m['avg_em_score']:.3f}</p>\n"
        if 'avg_f1_score' in m:
            html += f"            <p><strong>Average F1 Score:</strong> {m['avg_f1_score']:.3f}</p>\n"
        if 'avg_llm_score' in m:
            html += f"            <p><strong>Average LLM Score:</strong> {m['avg_llm_score']:.3f}</p>\n"
        html += """
        </div>
"""
    
    # Add top errors table
    if "top_errors" in patterns and patterns["top_errors"]:
        html += """
        <h2>Top Error Categories</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Count</th>
                <th>Percentage</th>
                <th>Description</th>
            </tr>
"""
        for err in patterns["top_errors"]:
            html += f"""
            <tr>
                <td>{err['category']}</td>
                <td>{err['count']}</td>
                <td>{err['percentage']:.1f}%</td>
                <td>{err['description']}</td>
            </tr>
"""
        html += "        </table>\n"
    
    html += """
    </div>
</body>
</html>
"""
    
    return html
