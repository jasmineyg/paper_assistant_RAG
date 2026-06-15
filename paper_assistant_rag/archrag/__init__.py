"""ArchRAG-style offline indexing and online retrieval components."""

from paper_assistant_rag.archrag.adaptive_filter import AdaptiveFilteringGenerator
from paper_assistant_rag.archrag.hierarchical_index import HierarchicalIndex
from paper_assistant_rag.archrag.hierarchical_retriever import HierarchicalRetriever
from paper_assistant_rag.archrag.pipeline import ArchRAGPipeline, build_archrag_pipeline
from paper_assistant_rag.archrag.query_processing import (
    build_retrieval_query,
    get_beam_width,
    get_rerank_weights,
    rewrite_query,
)

__all__ = [
    "AdaptiveFilteringGenerator",
    "ArchRAGPipeline",
    "HierarchicalIndex",
    "HierarchicalRetriever",
    "build_retrieval_query",
    "build_archrag_pipeline",
    "get_beam_width",
    "get_rerank_weights",
    "rewrite_query",
]
