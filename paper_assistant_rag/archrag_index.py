"""C-HNSW-like hierarchical indexing and search for ArchRAG nodes."""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Any

from paper_assistant_rag.archrag_types import ArchIndex, ArchLayer, ArchNode, load_arch_index, save_arch_index


def build_archrag_index(
    hierarchy: ArchIndex,
    embeddings=None,
    m_neighbors: int = 8,
    archrag_dir: Path | None = None,
) -> ArchIndex:
    """Build intra-layer nearest-neighbor links and inter-layer descent links."""
    indexed = ArchIndex(
        layers=hierarchy.layers,
        inter_links=dict(hierarchy.inter_links),
        entry_node_id=hierarchy.entry_node_id,
        metadata=dict(hierarchy.metadata),
    )
    for layer in indexed.layers.values():
        layer.intra_links = _nearest_neighbor_links(layer, m_neighbors=m_neighbors)
    indexed.inter_links = _inter_layer_links(indexed)
    if indexed.entry_node_id is None and indexed.layers:
        top_layer = indexed.layers[max(indexed.layers)]
        indexed.entry_node_id = _central_node(top_layer.nodes)
    indexed.metadata["m_neighbors"] = m_neighbors
    indexed.metadata["index_kind"] = "python_c_hnsw_like"
    if archrag_dir is not None:
        build_config = dict(indexed.metadata)
        save_arch_index(indexed, archrag_dir, build_config=build_config)
    return indexed


def search_layer(
    layer: ArchLayer,
    query_embedding: list[float],
    start_node_id: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """Search one layer from a start node by greedily expanding intra-layer links."""
    if not layer.nodes or top_k <= 0:
        return []
    start = start_node_id if start_node_id in layer.nodes else _central_node(layer.nodes)
    if start is None:
        return []

    visited: set[str] = set()
    candidates: list[tuple[float, str]] = [(-_score_node(layer.nodes[start], query_embedding), start)]
    best_by_id: dict[str, float] = {}
    expansions = max(len(layer.nodes), top_k * 8)

    while candidates and len(visited) < expansions:
        negative_score, node_id = heapq.heappop(candidates)
        if node_id in visited:
            continue
        visited.add(node_id)
        score = -negative_score
        best_by_id[node_id] = max(score, best_by_id.get(node_id, -1.0))
        for neighbor_id in layer.intra_links.get(node_id, []):
            if neighbor_id in visited or neighbor_id not in layer.nodes:
                continue
            neighbor_score = _score_node(layer.nodes[neighbor_id], query_embedding)
            heapq.heappush(candidates, (-neighbor_score, neighbor_id))

    if len(best_by_id) < min(top_k, len(layer.nodes)):
        for node_id, node in layer.nodes.items():
            best_by_id.setdefault(node_id, _score_node(node, query_embedding))

    ranked = sorted(best_by_id.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [_node_result(layer.nodes[node_id], score) for node_id, score in ranked]


def hierarchical_search(
    arch_index: ArchIndex,
    query: str,
    embeddings,
    top_k_per_level: int = 5,
    max_levels: int | None = None,
) -> dict[str, Any]:
    """Run top-down hierarchical search from the highest layer to level 0."""
    if not arch_index.layers:
        raise ValueError("ArchRAG index is empty. Run `uv run python main.py archrag-build` first.")
    query_embedding = [float(value) for value in embeddings.embed_query(query)]
    available_levels = sorted(arch_index.layers, reverse=True)
    if max_levels is not None and max_levels > 0:
        min_allowed_level = max(0, max(available_levels) - max_levels + 1)
        available_levels = [level for level in available_levels if level >= min_allowed_level]
    current_start = arch_index.entry_node_id
    level_results: dict[int, list[dict[str, Any]]] = {}
    path: list[dict[str, Any]] = []

    for level in available_levels:
        layer = arch_index.layers[level]
        results = search_layer(
            layer=layer,
            query_embedding=query_embedding,
            start_node_id=current_start,
            top_k=top_k_per_level,
        )
        level_results[level] = results
        if not results:
            current_start = None
            continue
        best_node_id = str(results[0]["node_id"])
        path.append({"level": level, "node_id": best_node_id, "score": results[0]["score"]})
        current_start = _next_lower_start(arch_index, best_node_id, next_level=level - 1)

    return {
        "query": query,
        "level_results": level_results,
        "path": path,
        "query_embedding_dim": len(query_embedding),
    }


def load_archrag_index(archrag_dir: Path) -> ArchIndex:
    """Load a persisted ArchRAG index from disk."""
    return load_arch_index(archrag_dir)


def _nearest_neighbor_links(layer: ArchLayer, m_neighbors: int) -> dict[str, list[str]]:
    """Return top-m cosine neighbors for every node in one layer."""
    links: dict[str, list[str]] = {}
    node_ids = sorted(layer.nodes)
    for node_id in node_ids:
        node = layer.nodes[node_id]
        scored = [
            (_cosine(node.embedding, layer.nodes[other_id].embedding), other_id)
            for other_id in node_ids
            if other_id != node_id
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        links[node_id] = [other_id for _score, other_id in scored[: max(0, m_neighbors)]]
    return links


def _inter_layer_links(index: ArchIndex) -> dict[str, str]:
    """Return one descent link from each upper node to the best lower node."""
    links: dict[str, str] = {}
    for level in sorted(index.layers):
        if level == 0 or level - 1 not in index.layers:
            continue
        lower_nodes = index.layers[level - 1].nodes
        for node in index.layers[level].nodes.values():
            child_candidates = [child_id for child_id in node.children if child_id in lower_nodes]
            if child_candidates:
                links[node.node_id] = _nearest_node_id(node.embedding, {child_id: lower_nodes[child_id] for child_id in child_candidates})
                continue
            nearest = _nearest_node_id(node.embedding, lower_nodes)
            if nearest:
                links[node.node_id] = nearest
    return links


def _next_lower_start(index: ArchIndex, node_id: str, next_level: int) -> str | None:
    """Choose the next layer's start node for top-down search."""
    if next_level < 0 or next_level not in index.layers:
        return None
    linked = index.inter_links.get(node_id)
    if linked in index.layers[next_level].nodes:
        return linked
    node = index.find_node(node_id)
    if node is not None:
        for child_id in node.children:
            if child_id in index.layers[next_level].nodes:
                return child_id
    return _central_node(index.layers[next_level].nodes)


def _nearest_node_id(vector: list[float], nodes: dict[str, ArchNode]) -> str:
    """Find the node whose embedding is closest to the given vector."""
    if not nodes:
        return ""
    return sorted(nodes.values(), key=lambda node: (-_cosine(vector, node.embedding), node.node_id))[0].node_id


def _central_node(nodes: dict[str, ArchNode]) -> str | None:
    """Pick a deterministic fallback entry node for a layer."""
    if not nodes:
        return None
    return sorted(nodes.values(), key=lambda node: (-len(node.children), -len(node.source_chunks), node.node_id))[0].node_id


def _score_node(node: ArchNode, query_embedding: list[float]) -> float:
    """Score one node against the query embedding."""
    return _cosine(query_embedding, node.embedding)


def _node_result(node: ArchNode, score: float) -> dict[str, Any]:
    """Convert a searched node into a structured result row."""
    return {
        "node_id": node.node_id,
        "level": node.level,
        "node_type": node.node_type,
        "score": float(score),
        "name": node.name,
        "text": node.text,
        "summary": node.summary,
        "source_chunks": list(node.source_chunks),
        "children": list(node.children),
        "metadata": dict(node.metadata),
    }


def _cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two dense vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


__all__ = [
    "build_archrag_index",
    "search_layer",
    "hierarchical_search",
    "load_archrag_index",
]
