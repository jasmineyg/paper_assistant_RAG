"""Attribute-aware graph augmentation for ArchRAG."""

from __future__ import annotations

from typing import Any

import networkx as nx

from paper_assistant_rag.archrag_hierarchy import build_attributed_entity_graph


class AttributeAwareGraphAugmentor:
    """Build an entity graph using KG structure and textual-attribute similarity."""

    def __init__(
        self,
        similarity_top_k: int = 5,
        similarity_threshold: float = 0.65,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self.similarity_top_k = similarity_top_k
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.beta = beta

    def build(self, entities: list[dict[str, Any]], relations: list[dict[str, Any]], embeddings) -> nx.Graph:
        """Return an augmented weighted attributed graph."""
        return build_attributed_entity_graph(
            entities=entities,
            relations=relations,
            embeddings=embeddings,
            similarity_top_k=self.similarity_top_k,
            similarity_threshold=self.similarity_threshold,
            relation_weight=self.alpha,
            attribute_weight=self.beta,
        )


__all__ = ["AttributeAwareGraphAugmentor"]
