"""C-HNSW-compatible hierarchical index facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_assistant_rag.archrag_index import build_archrag_index, hierarchical_search, load_archrag_index
from paper_assistant_rag.archrag_types import ArchIndex


class HierarchicalIndex:
    """Unified multi-level index over L0 entities and higher-level communities."""

    def __init__(self, index: ArchIndex) -> None:
        self.index = index

    @classmethod
    def build(cls, hierarchy: ArchIndex, m_neighbors: int, archrag_dir: Path | None = None) -> "HierarchicalIndex":
        """Build C-HNSW-like intra/inter links for a hierarchy."""
        return cls(build_archrag_index(hierarchy=hierarchy, m_neighbors=m_neighbors, archrag_dir=archrag_dir))

    @classmethod
    def load(cls, archrag_dir: Path) -> "HierarchicalIndex":
        """Load a persisted hierarchy index from disk."""
        return cls(load_archrag_index(archrag_dir))

    def search(self, query_embedding: list[float], top_k_per_level: int) -> dict[str, list[dict[str, Any]]]:
        """Search each level using an already-computed query embedding."""
        from paper_assistant_rag.archrag_index import search_layer

        results: dict[str, list[dict[str, Any]]] = {}
        current_start = self.index.entry_node_id
        for level in sorted(self.index.layers, reverse=True):
            level_results = search_layer(
                layer=self.index.layers[level],
                query_embedding=query_embedding,
                start_node_id=current_start,
                top_k=top_k_per_level,
            )
            results[f"level_{level}"] = level_results
            if level_results:
                current_start = self.index.inter_links.get(str(level_results[0]["node_id"]))
        return results

    def search_text(self, query: str, embeddings, top_k_per_level: int, max_levels: int | None = None) -> dict[str, Any]:
        """Embed a query and run top-down hierarchical search."""
        return hierarchical_search(
            arch_index=self.index,
            query=query,
            embeddings=embeddings,
            top_k_per_level=top_k_per_level,
            max_levels=max_levels,
        )


__all__ = ["HierarchicalIndex"]
