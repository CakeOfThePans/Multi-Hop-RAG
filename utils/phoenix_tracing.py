"""
Phoenix Tracing Utilities

This module provides decorators and context managers for tracing RAG operations
with Phoenix observability.
"""

import time
from typing import List, Optional, Callable, Any
from functools import wraps
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

