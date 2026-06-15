"""C-HNSW-like hierarchical indexing and search for ArchRAG nodes."""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any

import numpy as np

from paper_assistant_rag.archrag.query_processing import get_beam_width
from paper_assistant_rag.archrag_types import ArchIndex, ArchLayer, ArchNode, load_arch_index, save_arch_index
from paper_assistant_rag.ui import create_progress

DEFAULT_BEAM_WIDTH = 3
LOCAL_SCORE_WEIGHT = 0.85
PARENT_ROUTE_WEIGHT = 0.15


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
    start_node_ids = [start_node_id] if start_node_id else []
    return _search_layer_from_entries(
        layer=layer,
        query_embedding=query_embedding,
        start_node_ids=start_node_ids,
        top_k=top_k,
    )


def select_entry_nodes(
    query_vec: list[float] | np.ndarray,
    top_layer_nodes: dict[str, ArchNode] | list[ArchNode],
    K: int = 3,
) -> list[dict[str, Any]]:
    """Select the top-K highest-layer nodes by query embedding similarity."""
    nodes = (
        top_layer_nodes
        if isinstance(top_layer_nodes, dict)
        else {node.node_id: node for node in top_layer_nodes}
    )
    if not nodes or K <= 0:
        return []
    layer = ArchLayer(level=max(node.level for node in nodes.values()), nodes=nodes)
    scores = _score_all_nodes(layer, list(query_vec))
    if scores:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: min(K, len(scores))]
    else:
        fallback_id = _central_node(nodes)
        ranked = [(fallback_id, 0.0)] if fallback_id else []
    return [
        {
            "node_id": node_id,
            "score": float(score),
            "level": nodes[node_id].level,
            "node_type": nodes[node_id].node_type,
            "name": nodes[node_id].name,
        }
        for node_id, score in ranked
    ]


def _search_layer_from_entries(
    layer: ArchLayer,
    query_embedding: list[float],
    start_node_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Search one layer from multiple entry nodes and merge their traversal."""
    if not layer.nodes or top_k <= 0:
        return []
    starts = [node_id for node_id in dict.fromkeys(start_node_ids) if node_id in layer.nodes]
    if not starts:
        fallback = _central_node(layer.nodes)
        starts = [fallback] if fallback else []
    if not starts:
        return []

    visited: set[str] = set()
    candidates: list[tuple[float, str]] = [
        (-_score_node(layer.nodes[node_id], query_embedding), node_id)
        for node_id in starts
    ]
    heapq.heapify(candidates)
    best_by_id: dict[str, float] = {}
    expansions = min(len(layer.nodes), max(32, top_k * 8, len(starts) * 8))

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
    beam_width: int | None = None,
    query_type: str = "fact",
    entry_k: int = 3,
) -> dict[str, Any]:
    """Run parent-constrained top-down beam search from the highest layer."""
    query_embedding = [float(value) for value in embeddings.embed_query(query)]
    return hierarchical_search_by_embedding(
        arch_index=arch_index,
        query_embedding=query_embedding,
        query=query,
        top_k_per_level=top_k_per_level,
        max_levels=max_levels,
        beam_width=beam_width,
        query_type=query_type,
        entry_k=entry_k,
    )


def hierarchical_search_by_embedding(
    arch_index: ArchIndex,
    query_embedding: list[float],
    top_k_per_level: int = 5,
    max_levels: int | None = None,
    beam_width: int | None = None,
    query: str = "",
    query_type: str = "fact",
    entry_k: int = 3,
) -> dict[str, Any]:
    """Run parent-constrained beam search with an already-computed query vector."""
    if not arch_index.layers:
        raise ValueError("ArchRAG index is empty. Run `uv run python main.py archrag-build` first.")
    available_levels = sorted(arch_index.layers, reverse=True)
    if max_levels is not None and max_levels > 0:
        min_allowed_level = max(0, max(available_levels) - max_levels + 1)
        available_levels = [level for level in available_levels if level >= min_allowed_level]
    level_results: dict[int, list[dict[str, Any]]] = {}
    path: list[dict[str, Any]] = []
    beam_trace: dict[int, list[str]] = {}
    candidate_counts: dict[int, int] = {}
    fallback_levels: list[int] = []
    parent_beam: list[dict[str, Any]] = []
    beam_widths: dict[int, int] = {}
    entry_nodes: list[dict[str, Any]] = []

    for level_index, level in enumerate(available_levels):
        layer = arch_index.layers[level]
        current_beam_width = (
            max(1, int(beam_width))
            if beam_width is not None
            else get_beam_width(query_type, depth=level_index)
        )
        beam_widths[level] = current_beam_width
        level_top_k = max(1, int(top_k_per_level), current_beam_width)
        if level_index == 0:
            entry_nodes = select_entry_nodes(
                query_vec=query_embedding,
                top_layer_nodes=layer.nodes,
                K=entry_k,
            )
            results = _search_layer_from_entries(
                layer=layer,
                query_embedding=query_embedding,
                start_node_ids=[str(row["node_id"]) for row in entry_nodes],
                top_k=level_top_k,
            )
            for result in results:
                result["local_score"] = float(result["score"])
                result["route_score"] = float(result["score"])
                result["parent_score"] = None
                result["parent_node_ids"] = []
            candidate_counts[level] = len(layer.nodes)
        else:
            parent_candidates = _child_candidates(
                arch_index=arch_index,
                parent_beam=parent_beam,
                layer=layer,
            )
            candidate_counts[level] = len(parent_candidates)
            if parent_candidates:
                results = _rank_child_candidates(
                    layer=layer,
                    query_embedding=query_embedding,
                    parent_candidates=parent_candidates,
                    top_k=level_top_k,
                )
            else:
                fallback_levels.append(level)
                fallback_start = _fallback_lower_start(arch_index, parent_beam, layer)
                results = search_layer(
                    layer=layer,
                    query_embedding=query_embedding,
                    start_node_id=fallback_start,
                    top_k=level_top_k,
                )
                for result in results:
                    result["local_score"] = float(result["score"])
                    result["route_score"] = float(result["score"])
                    result["parent_score"] = None
                    result["parent_node_ids"] = []
        level_results[level] = results
        if not results:
            parent_beam = []
            beam_trace[level] = []
            continue
        parent_beam = results[:current_beam_width]
        beam_trace[level] = [str(result["node_id"]) for result in parent_beam]
        path.append(
            {
                "level": level,
                "node_id": str(results[0]["node_id"]),
                "score": float(results[0]["score"]),
            }
        )

    return {
        "query": query,
        "query_type": query_type,
        "level_results": level_results,
        "path": path,
        "retrieval_paths": _retrieval_paths(level_results),
        "entry_nodes": entry_nodes,
        "beam_width": max(beam_widths.values(), default=DEFAULT_BEAM_WIDTH),
        "beam_widths": beam_widths,
        "beam_trace": beam_trace,
        "candidate_counts": candidate_counts,
        "fallback_levels": fallback_levels,
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


def _child_candidates(
    arch_index: ArchIndex,
    parent_beam: list[dict[str, Any]],
    layer: ArchLayer,
) -> dict[str, list[dict[str, Any]]]:
    """Map eligible lower-layer children to the beam parents that reach them."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for parent_result in parent_beam:
        parent_id = str(parent_result.get("node_id", ""))
        parent = arch_index.find_node(parent_id)
        child_ids = list(parent.children) if parent is not None else []
        linked = arch_index.inter_links.get(parent_id)
        if linked:
            child_ids.append(linked)
        for child_id in dict.fromkeys(child_ids):
            if child_id in layer.nodes:
                candidates.setdefault(child_id, []).append(parent_result)
    return candidates


def _rank_child_candidates(
    layer: ArchLayer,
    query_embedding: list[float],
    parent_candidates: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank only children reachable from the current beam."""
    ranked: list[dict[str, Any]] = []
    for child_id, parents in parent_candidates.items():
        node = layer.nodes[child_id]
        local_score = _score_node(node, query_embedding)
        parent_score = max(float(parent.get("route_score", parent.get("score", 0.0))) for parent in parents)
        route_score = LOCAL_SCORE_WEIGHT * local_score + PARENT_ROUTE_WEIGHT * parent_score
        result = _node_result(node, route_score)
        result.update(
            {
                "local_score": float(local_score),
                "route_score": float(route_score),
                "parent_score": float(parent_score),
                "parent_node_ids": [str(parent.get("node_id", "")) for parent in parents],
            }
        )
        ranked.append(result)
    ranked.sort(key=lambda result: (-float(result["score"]), str(result["node_id"])))
    return ranked[:top_k]


def _fallback_lower_start(
    arch_index: ArchIndex,
    parent_beam: list[dict[str, Any]],
    layer: ArchLayer,
) -> str | None:
    """Choose a deterministic lower-layer start when hierarchy links are missing."""
    for parent_result in parent_beam:
        linked = arch_index.inter_links.get(str(parent_result.get("node_id", "")))
        if linked in layer.nodes:
            return linked
    return _central_node(layer.nodes)


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


def _retrieval_paths(
    level_results: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten route-aware node results for diagnostics and downstream clients."""
    paths: list[dict[str, Any]] = []
    for level in sorted(level_results, reverse=True):
        for result in level_results[level]:
            paths.append(
                {
                    "level": level,
                    "node_id": str(result.get("node_id", "")),
                    "parent_node_ids": list(result.get("parent_node_ids", [])),
                    "local_score": float(result.get("local_score", result.get("score", 0.0))),
                    "route_score": float(result.get("route_score", result.get("score", 0.0))),
                }
            )
    return paths


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
    "hierarchical_search_by_embedding",
    "load_archrag_index",
    "select_entry_nodes",
]
