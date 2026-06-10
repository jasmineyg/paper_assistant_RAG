"""Online multi-level retrieval over entities and communities."""

from __future__ import annotations

from typing import Any

from paper_assistant_rag.archrag_generation import archrag_level_results_to_documents
from paper_assistant_rag.archrag.hierarchical_index import HierarchicalIndex


class HierarchicalRetriever:
    """Retrieve high-level communities, mid-level communities, and L0 entities."""

    def __init__(self, index: HierarchicalIndex, embeddings) -> None:
        self.index = index
        self.embeddings = embeddings

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
        return {
            "search": search_result,
            "evidence_documents": archrag_level_results_to_documents(
                search_result["level_results"],
                limit=final_chunk_limit,
            ),
        }


__all__ = ["HierarchicalRetriever"]
