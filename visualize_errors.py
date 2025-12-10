"""
Phoenix Error Visualization and Analysis CLI

This script provides command-line access to Phoenix trace data for error analysis
and visualization. It connects to Phoenix server and generates reports or opens
the Phoenix UI with pre-configured filters.

Usage:
    # Open Phoenix UI with all traces
    python visualize_errors.py --interactive
    
    # Open Phoenix UI filtered to specific error category
    python visualize_errors.py --interactive --filter "eval.error_category == 'answer_hallucination'"
    
    # Generate text report
    python visualize_errors.py --report --format text
    
    # Generate HTML report and save to file
    python visualize_errors.py --report --format html --output error_report.html
    
    # Compare retrieval modes
    python visualize_errors.py --compare_retrieval_modes
    
    # Compare architectures
    python visualize_errors.py --compare_architectures
    
    # Show top failure examples
    python visualize_errors.py --show_failures --error_category answer_hallucination --top 10
"""

import argparse
import sys
from typing import Optional

from utils.phoenix_analysis import (
    analyze_error_patterns,
    compare_retrieval_modes,
    compare_architectures,
    identify_failure_clusters,
    generate_error_report,
    open_phoenix_ui,
    PHOENIX_AVAILABLE,
    get_phoenix_client
)


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_dict(data: dict, indent: int = 0):
    """Pretty print a dictionary."""
    spaces = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{spaces}{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print(f"{spaces}{key}: {len(value)} items")
        else:
            print(f"{spaces}{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Phoenix Error Analysis and Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Open Phoenix UI
  python visualize_errors.py --interactive
  
  # Generate comprehensive report
  python visualize_errors.py --report --format markdown --output report.md
  
  # Compare configurations
  python visualize_errors.py --compare_retrieval_modes
  python visualize_errors.py --compare_architectures
  
  # Show specific failures
  python visualize_errors.py --show_failures --error_category answer_hallucination --top 5
"""
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('Modes')
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Open Phoenix UI in browser"
    )
    mode_group.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive error analysis report"
    )
    mode_group.add_argument(
        "--compare_retrieval_modes",
        action="store_true",
        help="Compare FAISS vs BM25 vs hybrid retrieval"
    )
    mode_group.add_argument(
        "--compare_architectures",
        action="store_true",
        help="Compare single-hop vs multi-hop RAG"
    )
    mode_group.add_argument(
        "--show_failures",
        action="store_true",
        help="Show top failure examples"
    )
    mode_group.add_argument(
        "--analyze_patterns",
        action="store_true",
        help="Analyze error patterns and distribution"
    )
    
    # Options
    options_group = parser.add_argument_group('Options')
    options_group.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name to analyze (default: all projects)"
    )
    options_group.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter query for Phoenix UI (e.g., 'eval.error_category == \"answer_hallucination\"')"
    )
    options_group.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown", "html"],
        default="text",
        help="Report output format (default: text)"
    )
    options_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for report (default: print to console)"
    )
    options_group.add_argument(
        "--error_category",
        type=str,
        default=None,
        help="Specific error category to analyze"
    )
    options_group.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top failures to show (default: 10)"
    )
    options_group.add_argument(
        "--phoenix_url",
        type=str,
        default="http://127.0.0.1:6006",
        help="Phoenix server URL (default: http://127.0.0.1:6006)"
    )
    options_group.add_argument(
        "--phoenix_port",
        type=int,
        default=None,
        help="Phoenix server port (overrides --phoenix_url port)"
    )
    
    args = parser.parse_args()
    
    # Check if Phoenix is available
    if not PHOENIX_AVAILABLE:
        print("ERROR: Phoenix package not installed.")
        print("Install with: pip install arize-phoenix")
        sys.exit(1)
    
    # Determine Phoenix URL
    phoenix_url = args.phoenix_url
    if args.phoenix_port:
        # Override port if specified
        phoenix_url = f"http://127.0.0.1:{args.phoenix_port}"
    
    # Try to connect to Phoenix server
    try:
        import phoenix as px
        print(f"Connecting to Phoenix server at {phoenix_url}...")
        
        # Test connection by creating client
        client = px.Client(endpoint=phoenix_url)
        print(f"✓ Successfully connected to Phoenix server")
        
    except Exception as e:
        print(f"ERROR: Cannot connect to Phoenix server at {phoenix_url}")
        print(f"Details: {e}")
        print("\nMake sure Phoenix server is running:")
        print("  phoenix serve")
        print("\nThen run evaluations with --phoenix flag:")
        print("  python run_singlehop.py --phoenix --dataset_name hotpot --n_eval 10")
        sys.exit(1)
    
    # If no mode specified, show help
    if not any([
        args.interactive,
        args.report,
        args.compare_retrieval_modes,
        args.compare_architectures,
        args.show_failures,
        args.analyze_patterns
    ]):
        parser.print_help()
        sys.exit(0)
    
    # Handle interactive mode
    if args.interactive:
        print_section("Opening Phoenix UI")
        import webbrowser
        url = phoenix_url
        if args.filter:
            # Phoenix UI supports URL parameters for filters
            url = f"{phoenix_url}?filter={args.filter}"
        
        print(f"Opening: {url}")
        webbrowser.open(url)
        print("Phoenix UI opened in browser.")
        if args.filter:
            print(f"Filter applied: {args.filter}")
        return
    
    # Handle report generation
    if args.report:
        print_section("Generating Error Analysis Report")
        report = generate_error_report(
            project_name=args.project,
            output_format=args.format,
            endpoint=phoenix_url
        )
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)
        return
    
    # Handle retrieval mode comparison
    if args.compare_retrieval_modes:
        print_section("Retrieval Mode Comparison")
        comparison = compare_retrieval_modes(
            project_name=args.project,
            endpoint=phoenix_url
        )
        
        if "error" in comparison:
            print(f"ERROR: {comparison['error']}")
        else:
            for mode, stats in comparison.items():
                print(f"\n{mode}:")
                print_dict(stats, indent=1)
        return
    
    # Handle architecture comparison
    if args.compare_architectures:
        print_section("Architecture Comparison")
        comparison = compare_architectures(
            project_name=args.project,
            endpoint=phoenix_url
        )
        
        if "error" in comparison:
            print(f"ERROR: {comparison['error']}")
        else:
            for arch, stats in comparison.items():
                print(f"\n{arch}:")
                print_dict(stats, indent=1)
        return
    
    # Handle failure clustering
    if args.show_failures:
        print_section(f"Top {args.top} Failure Examples")
        failures = identify_failure_clusters(
            project_name=args.project,
            error_category=args.error_category,
            top_n=args.top,
            endpoint=phoenix_url
        )
        
        if not failures:
            print("No failures found.")
        else:
            for i, fail in enumerate(failures, 1):
                print(f"\n[{i}] {fail.get('error_category', 'Unknown')}")
                print(f"  Question: {fail.get('question', '')[:120]}...")
                print(f"  Prediction: {fail.get('prediction', '')[:100]}...")
                print(f"  Ground Truth: {fail.get('ground_truth', '')[:100]}...")
                
                scores = []
                if 'em_score' in fail:
                    scores.append(f"EM: {fail['em_score']:.2f}")
                if 'f1_score' in fail:
                    scores.append(f"F1: {fail['f1_score']:.2f}")
                if 'llm_score' in fail:
                    scores.append(f"LLM: {fail['llm_score']:.2f}")
                if scores:
                    print(f"  Scores: {', '.join(scores)}")
                
                if 'retrieval_mode' in fail:
                    print(f"  Retrieval: {fail['retrieval_mode']}")
                if 'architecture' in fail:
                    print(f"  Architecture: {fail['architecture']}")
        return
    
    # Handle error pattern analysis
    if args.analyze_patterns:
        print_section("Error Pattern Analysis")
        patterns = analyze_error_patterns(
            project_name=args.project,
            endpoint=phoenix_url
        )
        
        if "error" in patterns:
            print(f"ERROR: {patterns['error']}")
        else:
            if "metrics" in patterns:
                print("\nOverall Metrics:")
                print_dict(patterns["metrics"], indent=1)
            
            if "top_errors" in patterns and patterns["top_errors"]:
                print("\nTop Error Categories:")
                for err in patterns["top_errors"]:
                    print(f"  {err['category']}: {err['count']} ({err['percentage']:.1f}%)")
                    print(f"    → {err['description']}")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

