"""Online multi-level retrieval over entities and communities."""

from __future__ import annotations

from typing import Any

from paper_assistant_rag.archrag_generation import (
    archrag_level_results_to_documents,
    rerank_archrag_chunks,
)
from paper_assistant_rag.archrag.hierarchical_index import HierarchicalIndex


class HierarchicalRetriever:
    """Retrieve high-level communities, mid-level communities, and L0 entities."""

    def __init__(self, index: HierarchicalIndex, embeddings, vectorstore=None) -> None:
        self.index = index
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k_per_level: int = 5,
        max_levels: int | None = None,
        final_chunk_limit: int = 10,
    ) -> dict[str, Any]:
        """Return level results and source-chunk evidence candidates."""
        search_result = self.index.search_text(
            query=query,
            embeddings=self.embeddings,
            top_k_per_level=top_k_per_level,
            max_levels=max_levels,
        )
        evidence_documents = (
            rerank_archrag_chunks(
                query=query,
                level_results=search_result["level_results"],
                vectorstore=self.vectorstore,
                embeddings=self.embeddings,
                limit=final_chunk_limit,
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
        }


__all__ = ["HierarchicalRetriever"]
