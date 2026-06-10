"""Adaptive filtering-based generation for ArchRAG."""

from __future__ import annotations

from typing import Any

from paper_assistant_rag.archrag_generation import adaptive_filter_level_results, merge_filtered_reports


class AdaptiveFilteringGenerator:
    """Generate per-level reports, rank them, and merge the final answer."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def filter_levels(self, query: str, level_results: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Return adaptive filtering reports for retrieved hierarchy levels."""
        return adaptive_filter_level_results(query=query, level_results=level_results, llm=self.llm)

    def generate(self, query: str, reports: list[dict[str, Any]], response_format: str) -> dict[str, Any]:
        """Merge ranked filtering reports into a final answer."""
        return merge_filtered_reports(query=query, reports=reports, llm=self.llm, response_format=response_format)


__all__ = ["AdaptiveFilteringGenerator"]
