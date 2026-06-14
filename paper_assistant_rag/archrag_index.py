"""C-HNSW-like hierarchical indexing and search for ArchRAG nodes."""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any

import numpy as np

from paper_assistant_rag.archrag_types import ArchIndex, ArchLayer, ArchNode, load_arch_index, save_arch_index
from paper_assistant_rag.ui import create_progress


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
        for node_id, score in _score_all_nodes(layer, query_embedding).items():
            best_by_id.setdefault(node_id, score)

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
    """Return top-m cosine neighbors for every node in one layer using blockwise NumPy."""
    links: dict[str, list[str]] = {node_id: [] for node_id in layer.nodes}
    if m_neighbors <= 0 or len(layer.nodes) <= 1:
        return links

    node_ids, vectors = _normalized_embedding_matrix(layer)
    if len(node_ids) <= 1:
        return links

    top_k = min(m_neighbors, len(node_ids) - 1)
    block_size = _similarity_block_size(len(node_ids))
    with create_progress() as progress:
        task = progress.add_task(
            f"Building level {layer.level} intra-layer links",
            total=len(node_ids),
        )
        for start in range(0, len(node_ids), block_size):
            end = min(start + block_size, len(node_ids))
            similarity_block = vectors[start:end] @ vectors.T
            _fill_neighbor_links_from_block(
                links=links,
                node_ids=node_ids,
                similarity_block=similarity_block,
                block_start=start,
                top_k=top_k,
            )
            progress.advance(task, advance=end - start)
    return links


def _normalized_embedding_matrix(layer: ArchLayer) -> tuple[list[str], np.ndarray]:
    """Return node ids and an L2-normalized float32 embedding matrix for one layer."""
    return _normalized_nodes_matrix(layer.nodes)


def _normalized_nodes_matrix(nodes: dict[str, ArchNode]) -> tuple[list[str], np.ndarray]:
    """Return node ids and an L2-normalized float32 embedding matrix for node dicts."""
    node_ids: list[str] = []
    rows: list[np.ndarray] = []
    expected_dim: int | None = None
    for node_id in sorted(nodes):
        vector = np.asarray(nodes[node_id].embedding, dtype=np.float32)
        if vector.size == 0:
            continue
        if expected_dim is None:
            expected_dim = int(vector.size)
        if vector.ndim != 1 or vector.size != expected_dim:
            continue
        node_ids.append(node_id)
        rows.append(vector)
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)

    matrix = np.stack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    valid_mask = (norms[:, 0] > 0.0) & np.isfinite(norms[:, 0])
    if not np.all(valid_mask):
        matrix = matrix[valid_mask]
        node_ids = [node_id for node_id, is_valid in zip(node_ids, valid_mask.tolist(), strict=False) if is_valid]
        norms = norms[valid_mask]
    matrix = matrix / np.maximum(norms, 1e-12)
    return node_ids, matrix


def _similarity_block_size(node_count: int) -> int:
    """Choose a bounded block size for dense layer-link similarity."""
    if node_count >= 10000:
        return 256
    if node_count >= 3000:
        return 512
    return 1024


def _fill_neighbor_links_from_block(
    links: dict[str, list[str]],
    node_ids: list[str],
    similarity_block: np.ndarray,
    block_start: int,
    top_k: int,
) -> None:
    """Fill top-k neighbor ids from one dense similarity block."""
    for row_offset in range(similarity_block.shape[0]):
        node_index = block_start + row_offset
        row = similarity_block[row_offset]
        row[node_index] = -np.inf
        candidate_indices = np.argpartition(row, -top_k)[-top_k:]
        ranked_indices = candidate_indices[np.argsort(row[candidate_indices])[::-1]]
        links[node_ids[node_index]] = [node_ids[int(index)] for index in ranked_indices if np.isfinite(row[index])]


def _inter_layer_links(index: ArchIndex) -> dict[str, str]:
    """Return one descent link from each upper node to the best lower node."""
    links: dict[str, str] = {}
    for level in sorted(index.layers):
        if level == 0 or level - 1 not in index.layers:
            continue
        lower_layer = index.layers[level - 1]
        lower_nodes = lower_layer.nodes
        lower_node_ids, lower_vectors = _normalized_embedding_matrix(lower_layer)
        if not lower_node_ids:
            continue
        lower_positions = {node_id: position for position, node_id in enumerate(lower_node_ids)}
        upper_nodes = list(index.layers[level].nodes.values())
        with create_progress() as progress:
            task = progress.add_task(
                f"Building level {level}->{level - 1} inter-layer links",
                total=len(upper_nodes),
            )
            for node in upper_nodes:
                child_positions = [
                    lower_positions[child_id]
                    for child_id in node.children
                    if child_id in lower_positions
                ]
                if child_positions:
                    links[node.node_id] = _nearest_from_matrix(
                        query_vector=node.embedding,
                        candidate_ids=lower_node_ids,
                        candidate_vectors=lower_vectors,
                        candidate_positions=child_positions,
                    )
                    progress.advance(task)
                    continue
                nearest = _nearest_from_matrix(
                    query_vector=node.embedding,
                    candidate_ids=lower_node_ids,
                    candidate_vectors=lower_vectors,
                    candidate_positions=None,
                )
                if nearest:
                    links[node.node_id] = nearest
                progress.advance(task)
    return links


def _nearest_from_matrix(
    query_vector: list[float],
    candidate_ids: list[str],
    candidate_vectors: np.ndarray,
    candidate_positions: list[int] | None,
) -> str:
    """Find the nearest candidate using normalized matrix multiplication."""
    query = _normalized_query_vector(query_vector, candidate_vectors.shape[1] if candidate_vectors.ndim == 2 else 0)
    if query is None or not candidate_ids:
        return ""
    if candidate_positions:
        positions = np.asarray(candidate_positions, dtype=np.int64)
        scores = candidate_vectors[positions] @ query
        best_position = int(positions[int(np.argmax(scores))])
        return candidate_ids[best_position]
    scores = candidate_vectors @ query
    return candidate_ids[int(np.argmax(scores))]


def _score_all_nodes(layer: ArchLayer, query_embedding: list[float]) -> dict[str, float]:
    """Score all layer nodes against a query embedding with one vectorized pass."""
    node_ids, vectors = _normalized_embedding_matrix(layer)
    query = _normalized_query_vector(query_embedding, vectors.shape[1] if vectors.ndim == 2 else 0)
    if query is None:
        return {}
    scores = vectors @ query
    return {node_id: float(score) for node_id, score in zip(node_ids, scores.tolist(), strict=False)}


def _normalized_query_vector(vector: list[float], expected_dim: int) -> np.ndarray | None:
    """Return an L2-normalized query vector or None when dimensions are invalid."""
    if expected_dim <= 0 or len(vector) != expected_dim:
        return None
    query = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return query / max(norm, 1e-12)


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


def _cosine(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    """Compute cosine similarity for two dense vectors."""
    if len(left) == 0 or len(right) == 0 or len(left) != len(right):
        return 0.0
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    dot = float(left_vector @ right_vector)
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


__all__ = [
    "build_archrag_index",
    "search_layer",
    "hierarchical_search",
    "load_archrag_index",
]
