"""Online multi-level retrieval over entities and communities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from paper_assistant_rag.archrag.query_processing import build_retrieval_query, rewrite_query
from paper_assistant_rag.archrag_generation import (
    archrag_level_results_to_documents,
    rerank_archrag_chunks,
)
from paper_assistant_rag.archrag.hierarchical_index import HierarchicalIndex


class HierarchicalRetriever:
    """Retrieve high-level communities, mid-level communities, and L0 entities."""

    def __init__(self, index: HierarchicalIndex, embeddings, vectorstore=None, llm=None) -> None:
        self.index = index
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.llm = llm

    def retrieve(
        self,
        query: str,
        top_k_per_level: int = 5,
        max_levels: int | None = None,
        final_chunk_limit: int = 10,
        chat_history: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Return level results and source-chunk evidence candidates."""
        rewritten_query = rewrite_query(
            query,
            llm=self.llm,
            chat_history=chat_history,
        )
        retrieval_query = build_retrieval_query(rewritten_query)
        query_type = str(rewritten_query["query_type"])
        search_result = self.index.search_text(
            query=retrieval_query,
            embeddings=self.embeddings,
            top_k_per_level=top_k_per_level,
            max_levels=max_levels,
            query_type=query_type,
        )
        evidence_documents = (
            rerank_archrag_chunks(
                query=retrieval_query,
                level_results=search_result["level_results"],
                vectorstore=self.vectorstore,
                embeddings=self.embeddings,
                limit=final_chunk_limit,
                query_type=query_type,
            )
            if self.vectorstore is not None
            else archrag_level_results_to_documents(
                search_result["level_results"],
                limit=final_chunk_limit,
            )
        )
        return {
            "search": search_result,
            "evidence_documents": evidence_documents,
            "rewritten_query": rewritten_query,
            "entry_nodes": search_result.get("entry_nodes", []),
            "retrieval_paths": search_result.get("retrieval_paths", []),
            "query_type": query_type,
        }


__all__ = ["HierarchicalRetriever"]
