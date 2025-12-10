"""
Phoenix Configuration Module

This module handles Phoenix observability platform initialization, configuration,
and lifecycle management for the Multi-Hop RAG system.

Key responsibilities:
- Launch Phoenix server
- Configure OpenTelemetry instrumentation
- Set up tracing exporters
- Define custom span attributes for RAG operations
- Manage server lifecycle
"""

import os
import atexit
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


class PhoenixManager:
    """
    Manages Phoenix session lifecycle and OpenTelemetry configuration.
    """
    
    def __init__(self):
        self.tracer_provider: Optional[TracerProvider] = None
        self.span_processor: Optional[BatchSpanProcessor] = None
        self.port: int = 6006
        self.host: str = "127.0.0.1"
        self.project_name: str = "multi_hop_rag"
        self._is_initialized: bool = False
        self._phoenix_url: Optional[str] = None
    
    def initialize(
        self,
        project_name: str = "multi_hop_rag",
        port: int = 6006,
        host: str = "127.0.0.1",
        auto_instrument: bool = True,
        verbose: bool = False,
    ) -> Optional[TracerProvider]:
        """
        Initialize Phoenix session and configure OpenTelemetry tracing.
        
        Args:
            project_name: Name of the project for Phoenix session
            port: Port number for Phoenix server (default: 6006)
            host: Host address for Phoenix server (default: 127.0.0.1)
            auto_instrument: Enable automatic instrumentation for LangChain
            verbose: Enable verbose logging
            
        Returns:
            TracerProvider object, or None if initialization fails
        """
        if self._is_initialized:
            print(f"Phoenix already initialized. Server at {self._phoenix_url}")
            return self.tracer_provider
        
        self.port = port
        self.host = host
        self.project_name = project_name
        
        # Use direct OTLP configuration (phoenix.otel.register has issues with batch processing)
        try:
            trace_endpoint = f"http://{host}:{port}/v1/traces"
            print(f"Initializing Phoenix tracing to: {trace_endpoint}")
            
            # Create resource with project name
            resource = Resource.create(attributes={"service.name": project_name})
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Create OTLP exporter
            exporter = OTLPSpanExporter(
                endpoint=trace_endpoint,
                timeout=10
            )
            
            # Create and add batch span processor
            self.span_processor = BatchSpanProcessor(exporter)
            self.tracer_provider.add_span_processor(self.span_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            self._phoenix_url = f"http://{host}:{port}"
            self._is_initialized = True
            
            print(f"Phoenix initialized successfully")
            print(f"   Project: {project_name}")
            print(f"   Server URL: {self._phoenix_url}")
            print(f"   Trace endpoint: {trace_endpoint}")
            print(f"   Processor: BatchSpanProcessor")
            
            # Auto-instrument if requested
            if auto_instrument:
                try:
                    from openinference.instrumentation.langchain import LangChainInstrumentor
                    LangChainInstrumentor().instrument(tracer_provider=self.tracer_provider)
                    print(f"   LangChain auto-instrumentation: enabled")
                except ImportError:
                    print(f"   LangChain auto-instrumentation: skipped (not installed)")
                except Exception as e:
                    print(f"   LangChain auto-instrumentation: failed ({e})")
            
            # Test connection
            import requests
            try:
                response = requests.get(f"http://{host}:{port}", timeout=2)
                print(f"   Phoenix server responding: HTTP {response.status_code}")
            except Exception as conn_err:
                print(f"   Warning: Could not verify Phoenix server connection: {conn_err}")
            
            # Register cleanup on exit
            atexit.register(self.shutdown)
            
            return self.tracer_provider
            
        except Exception as e:
            print(f"Failed to initialize Phoenix: {e}")
            print("   Continuing without Phoenix observability...")
            print("   Make sure Phoenix server is running: phoenix serve")
            self._is_initialized = False
            return None
    
    def shutdown(self):
        """
        Gracefully shutdown Phoenix session and cleanup resources.
        """
        if not self._is_initialized:
            return
        
        try:
            if self.tracer_provider:
                print("Flushing remaining spans to Phoenix...")
                # Flush any remaining spans (important for batched spans)
                self.tracer_provider.force_flush(timeout_millis=5000)
                print("Flush complete")
            
            self._is_initialized = False
            print("Phoenix tracing shutdown complete")
            
        except Exception as e:
            print(f"Error during Phoenix shutdown: {e}")
    
    def get_tracer(self, name: str = "multi_hop_rag"):
        """
        Get an OpenTelemetry tracer for creating custom spans.
        
        Args:
            name: Name of the tracer
            
        Returns:
            Tracer object
        """
        if not self._is_initialized or not self.tracer_provider:
            # Return a no-op tracer if Phoenix is not initialized
            return trace.NoOpTracer()
        
        return trace.get_tracer(name, tracer_provider=self.tracer_provider)
    
    def is_initialized(self) -> bool:
        """Check if Phoenix is initialized."""
        return self._is_initialized
    
    def get_url(self) -> Optional[str]:
        """Get the Phoenix server URL."""
        return self._phoenix_url


# Global Phoenix manager instance
_phoenix_manager: Optional[PhoenixManager] = None


def initialize_phoenix(
    project_name: str = "multi_hop_rag",
    port: int = 6006,
    host: str = "127.0.0.1",
    auto_instrument: bool = True,
    verbose: bool = False,
) -> Optional[TracerProvider]:
    """
    Initialize Phoenix observability platform.
    
    This is the main entry point for Phoenix initialization. It creates a
    global PhoenixManager instance and configures OpenTelemetry tracing.
    
    Note: The Phoenix server must be started separately before calling this function.
    Start it with: `phoenix serve` or `python -m phoenix.server.main`
    
    Args:
        project_name: Name of the project for Phoenix session
        port: Port number for Phoenix server (default: 6006)
        host: Host address for Phoenix server (default: 127.0.0.1)
        auto_instrument: Enable automatic instrumentation for LangChain
        verbose: Enable verbose logging
        
    Returns:
        TracerProvider object, or None if initialization fails
        
    Example:
        >>> # First, start Phoenix server in a separate terminal:
        >>> # phoenix serve
        >>> 
        >>> # Then in your code:
        >>> tracer_provider = initialize_phoenix(project_name="hotpot_eval", port=6006)
        >>> # Run your RAG evaluation
        >>> # Traces will automatically be sent to Phoenix
    """
    global _phoenix_manager
    
    if _phoenix_manager is None:
        _phoenix_manager = PhoenixManager()
    
    return _phoenix_manager.initialize(
        project_name=project_name,
        port=port,
        host=host,
        auto_instrument=auto_instrument,
        verbose=verbose,
    )


def get_phoenix_manager() -> Optional[PhoenixManager]:
    """
    Get the global Phoenix manager instance.
    
    Returns:
        PhoenixManager instance, or None if not initialized
    """
    return _phoenix_manager


def shutdown_phoenix():
    """
    Shutdown Phoenix session and cleanup resources.
    
    This should be called when the application is terminating to ensure
    all traces are flushed and resources are cleaned up.
    """
    global _phoenix_manager
    
    if _phoenix_manager:
        _phoenix_manager.shutdown()
        _phoenix_manager = None


def get_tracer(name: str = "multi_hop_rag"):
    """
    Get an OpenTelemetry tracer for creating custom spans.
    
    Args:
        name: Name of the tracer
        
    Returns:
        Tracer object (or NoOpTracer if Phoenix not initialized)
    """
    if _phoenix_manager:
        return _phoenix_manager.get_tracer(name)
    return trace.NoOpTracer()


def is_phoenix_enabled() -> bool:
    """
    Check if Phoenix is initialized and enabled.
    
    Returns:
        True if Phoenix is initialized, False otherwise
    """
    return _phoenix_manager is not None and _phoenix_manager.is_initialized()


def get_phoenix_url() -> Optional[str]:
    """
    Get the Phoenix server URL.
    
    Returns:
        URL string if Phoenix is initialized, None otherwise
    """
    if _phoenix_manager:
        return _phoenix_manager.get_url()
    return None


# RAG-specific span attribute definitions
# These constants define the attribute keys used for RAG operations

# Retrieval attributes
RETRIEVAL_QUERY = "retrieval.query"
RETRIEVAL_METHOD = "retrieval.method"  # "faiss" | "bm25" | "hybrid"
RETRIEVAL_K = "retrieval.k"
RETRIEVAL_DOCUMENTS = "retrieval.documents"
RETRIEVAL_SCORES = "retrieval.scores"
RETRIEVAL_LATENCY_MS = "retrieval.latency_ms"
RETRIEVAL_NUM_DOCS = "retrieval.num_documents"

# LLM attributes
LLM_MODEL = "llm.model"
LLM_TEMPERATURE = "llm.temperature"
LLM_PROMPT = "llm.prompt"
LLM_RESPONSE = "llm.response"
LLM_TOKENS_INPUT = "llm.tokens.input"
LLM_TOKENS_OUTPUT = "llm.tokens.output"
LLM_TOKENS_TOTAL = "llm.tokens.total"
LLM_LATENCY_MS = "llm.latency_ms"
LLM_OPERATION = "llm.operation"  # "answer_generation" | "decomposition" | "synthesis" | "query_composition"

# Evaluation attributes
EVAL_QUESTION = "eval.question"
EVAL_GROUND_TRUTH = "eval.ground_truth"
EVAL_PREDICTION = "eval.prediction"
EVAL_EM_SCORE = "eval.em_score"
EVAL_F1_SCORE = "eval.f1_score"
EVAL_LLM_SCORE = "eval.llm_score"
EVAL_ERROR_CATEGORY = "eval.error_category"
EVAL_DATASET = "eval.dataset"
EVAL_RETRIEVAL_MODE = "eval.retrieval_mode"
EVAL_ARCHITECTURE = "eval.architecture"  # "single_hop" | "multi_hop"

# Multi-hop specific attributes
MULTIHOP_SUBQUESTION = "multihop.subquestion"
MULTIHOP_COMPOSED_QUERY = "multihop.composed_query"
MULTIHOP_INTERMEDIATE_ANSWER = "multihop.intermediate_answer"
MULTIHOP_HOP_NUMBER = "multihop.hop_number"
MULTIHOP_TOTAL_HOPS = "multihop.total_hops"

# Error categories
ERROR_CATEGORIES = {
    "retrieval_no_results": "No documents retrieved",
    "retrieval_irrelevant": "Retrieved documents not relevant",
    "retrieval_insufficient": "Insufficient context retrieved",
    "answer_hallucination": "Answer not supported by context",
    "answer_incomplete": "Answer missing key information",
    "answer_wrong_type": "Answer format mismatch",
    "decomposition_poor": "Question decomposition failed",
    "reasoning_error": "Logical reasoning error",
    "synthesis_error": "Final answer synthesis failed",
    "none": "No error detected",
}

