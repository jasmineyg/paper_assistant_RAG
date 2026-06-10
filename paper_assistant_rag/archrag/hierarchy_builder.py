"""Iterative hierarchical attributed community construction."""

from __future__ import annotations

from typing import Any

from paper_assistant_rag.archrag_hierarchy import build_hierarchical_communities
from paper_assistant_rag.archrag_types import ArchIndex


class HierarchicalCommunityBuilder:
    """Implement ArchRAG Algorithm 1 over KG entities and relations."""

    def __init__(
        self,
        max_levels: int = 3,
        min_nodes_per_level: int = 5,
        similarity_top_k: int = 5,
        similarity_threshold: float = 0.65,
        community_algorithm: str = "louvain",
        summary_concurrency: int = 12,
    ) -> None:
        self.max_levels = max_levels
        self.min_nodes_per_level = min_nodes_per_level
        self.similarity_top_k = similarity_top_k
        self.similarity_threshold = similarity_threshold
        self.community_algorithm = community_algorithm
        self.summary_concurrency = summary_concurrency

    def build(self, entities: list[dict[str, Any]], relations: list[dict[str, Any]], llm, embeddings) -> ArchIndex:
        """Build L0 entities and iterative higher-level attributed communities."""
        return build_hierarchical_communities(
            entities=entities,
            relations=relations,
            llm=llm,
            embeddings=embeddings,
            max_levels=self.max_levels,
            min_nodes_per_level=self.min_nodes_per_level,
            similarity_top_k=self.similarity_top_k,
            similarity_threshold=self.similarity_threshold,
            community_algorithm=self.community_algorithm,
            summary_concurrency=self.summary_concurrency,
        )


__all__ = ["HierarchicalCommunityBuilder"]
