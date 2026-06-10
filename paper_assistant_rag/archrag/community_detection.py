"""Weighted attributed community detection for ArchRAG."""

from __future__ import annotations

import networkx as nx

from paper_assistant_rag.archrag_hierarchy import detect_attributed_communities


class WeightedCommunityDetector:
    """Detect weighted communities while preserving a Leiden-compatible interface."""

    def __init__(self, algorithm: str = "louvain") -> None:
        self.algorithm = algorithm

    def detect(self, graph: nx.Graph) -> list[set[str]]:
        """Return weighted communities for an attributed graph."""
        return detect_attributed_communities(graph, algorithm=self.algorithm)


__all__ = ["WeightedCommunityDetector"]
