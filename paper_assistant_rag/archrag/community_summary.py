"""LLM community summary generation for ArchRAG."""

from __future__ import annotations

from typing import Any

from paper_assistant_rag.archrag_hierarchy import summarize_community
from paper_assistant_rag.archrag_types import ArchNode


class CommunitySummaryGenerator:
    """Generate retrieval-ready summaries for entity/community groups."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def summarize(self, community_nodes: list[ArchNode], internal_edges: list[dict[str, Any]]) -> str:
        """Summarize one attributed community using the configured LLM."""
        return summarize_community(community_nodes, internal_edges, self.llm)


__all__ = ["CommunitySummaryGenerator"]
