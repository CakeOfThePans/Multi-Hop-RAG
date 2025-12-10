"""
Phoenix Tracing Utilities

This module provides decorators and context managers for tracing RAG operations
with Phoenix observability.
"""

import time
from typing import List, Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from langchain_core.documents import Document

from utils.phoenix_config import (
    get_tracer,
    is_phoenix_enabled,
    RETRIEVAL_QUERY,
    RETRIEVAL_METHOD,
    RETRIEVAL_K,
    RETRIEVAL_NUM_DOCS,
    RETRIEVAL_LATENCY_MS,
    LLM_OPERATION,
    MULTIHOP_SUBQUESTION,
    MULTIHOP_COMPOSED_QUERY,
    MULTIHOP_INTERMEDIATE_ANSWER,
    MULTIHOP_HOP_NUMBER,
    MULTIHOP_TOTAL_HOPS,
)


def trace_retrieval(method: str = "unknown"):
    """
    Decorator factory to trace retrieval operations.
    
    Args:
        method: Retrieval method name ("faiss", "bm25", "hybrid", "rerank")
    
    Usage:
        @trace_retrieval(method="faiss")
        def similarity_search(self, query, k=5):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, query: str, k: int = 5, *args, **kwargs):
            if not is_phoenix_enabled():
                return func(self, query, k, *args, **kwargs)
            
            tracer = get_tracer()
            span = tracer.start_span("retrieval")
            
            start_time = time.time()
            
            try:
                with trace.use_span(span):
                    # Set retrieval attributes
                    span.set_attribute(RETRIEVAL_QUERY, query)
                    span.set_attribute(RETRIEVAL_METHOD, method)
                    span.set_attribute(RETRIEVAL_K, k)
                    
                    # Execute retrieval
                    results = func(self, query, k, *args, **kwargs)
                    
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Set result attributes
                    span.set_attribute(RETRIEVAL_NUM_DOCS, len(results) if results else 0)
                    span.set_attribute(RETRIEVAL_LATENCY_MS, latency_ms)
                    
                    # Add document snippets (first 200 chars of top doc)
                    if results and len(results) > 0:
                        top_doc_snippet = results[0].page_content[:200].replace("\n", " ")
                        span.set_attribute("retrieval.top_doc_snippet", top_doc_snippet)
                    
                    span.set_status(Status(StatusCode.OK))
                    return results
                    
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute(RETRIEVAL_LATENCY_MS, latency_ms)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                span.end()
        
        return wrapper
    return decorator


def trace_rerank(func: Callable) -> Callable:
    """
    Decorator to trace reranking operations.
    
    Usage:
        @trace_rerank
        def rerank(self, query, docs, top_k=5):
            ...
    """
    @wraps(func)
    def wrapper(self, query: str, docs: List[Document], top_k: int = 5, *args, **kwargs):
        if not is_phoenix_enabled():
            return func(self, query, docs, top_k, *args, **kwargs)
        
        tracer = get_tracer()
        span = tracer.start_span("rerank")
        
        start_time = time.time()
        
        try:
            with trace.use_span(span):
                # Set reranking attributes
                span.set_attribute(RETRIEVAL_QUERY, query)
                span.set_attribute("rerank.input_docs", len(docs))
                span.set_attribute("rerank.top_k", top_k)
                
                # Execute reranking
                results = func(self, query, docs, top_k, *args, **kwargs)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Set result attributes
                span.set_attribute("rerank.output_docs", len(results) if results else 0)
                span.set_attribute(RETRIEVAL_LATENCY_MS, latency_ms)
                
                span.set_status(Status(StatusCode.OK))
                return results
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute(RETRIEVAL_LATENCY_MS, latency_ms)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()
    
    return wrapper


@contextmanager
def trace_rag_pipeline(question: str, architecture: str = "single_hop"):
    """
    Context manager to trace an entire RAG pipeline.
    
    Args:
        question: The question being answered
        architecture: Architecture type ("single_hop" or "multi_hop")
    
    Usage:
        with trace_rag_pipeline(question, "single_hop"):
            # RAG operations here
            ...
    """
    if not is_phoenix_enabled():
        yield
        return
    
    tracer = get_tracer()
    span = tracer.start_span("rag_pipeline")
    
    try:
        with trace.use_span(span):
            span.set_attribute("rag.question", question)
            span.set_attribute("rag.architecture", architecture)
            yield span
            span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


@contextmanager
def trace_multihop_decomposition(question: str, max_hops: int):
    """
    Context manager to trace question decomposition in multi-hop RAG.
    
    Args:
        question: The original question
        max_hops: Maximum number of hops
    
    Usage:
        with trace_multihop_decomposition(question, max_hops):
            subqs = decomposer(question)
    """
    if not is_phoenix_enabled():
        yield
        return
    
    tracer = get_tracer()
    span = tracer.start_span("multihop.decomposition")
    
    try:
        with trace.use_span(span):
            span.set_attribute("rag.question", question)
            span.set_attribute(MULTIHOP_TOTAL_HOPS, max_hops)
            yield span
            span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


@contextmanager
def trace_multihop_hop(hop_number: int, subquestion: str, composed_query: str):
    """
    Context manager to trace a single hop in multi-hop RAG.
    
    Args:
        hop_number: The hop number (1-indexed)
        subquestion: The sub-question for this hop
        composed_query: The composed query used for retrieval
    
    Usage:
        with trace_multihop_hop(1, subq, composed):
            # Hop operations here
    """
    if not is_phoenix_enabled():
        yield
        return
    
    tracer = get_tracer()
    span = tracer.start_span("multihop.hop")
    
    try:
        with trace.use_span(span):
            span.set_attribute(MULTIHOP_HOP_NUMBER, hop_number)
            span.set_attribute(MULTIHOP_SUBQUESTION, subquestion)
            span.set_attribute(MULTIHOP_COMPOSED_QUERY, composed_query)
            yield span
            span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


@contextmanager
def trace_multihop_synthesis(question: str, total_hops: int):
    """
    Context manager to trace final answer synthesis in multi-hop RAG.
    
    Args:
        question: The original question
        total_hops: Total number of hops completed
    
    Usage:
        with trace_multihop_synthesis(question, len(hops)):
            final = llm.invoke(...)
    """
    if not is_phoenix_enabled():
        yield
        return
    
    tracer = get_tracer()
    span = tracer.start_span("multihop.synthesis")
    
    try:
        with trace.use_span(span):
            span.set_attribute("rag.question", question)
            span.set_attribute(MULTIHOP_TOTAL_HOPS, total_hops)
            span.set_attribute(LLM_OPERATION, "synthesis")
            yield span
            span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()

