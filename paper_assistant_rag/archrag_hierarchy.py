"""Hierarchical attributed community construction for ArchRAG-style retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from paper_assistant_rag.archrag_types import ArchIndex, ArchLayer, ArchNode, save_arch_index
from paper_assistant_rag.communities import PREDICATE_WEIGHTS
from paper_assistant_rag.kg import ENTITIES_FILE, RELATIONS_FILE
from paper_assistant_rag.models import build_embeddings, build_llm
from paper_assistant_rag.paths import DEFAULT_ARCHRAG_DIR, DEFAULT_EMBED_BATCH_SIZE
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, create_progress


def build_archrag_hierarchy_cache(
    graph_dir: Path,
    archrag_dir: Path,
    max_levels: int,
    min_nodes_per_level: int,
    similarity_top_k: int,
    similarity_threshold: float,
    community_algorithm: str,
) -> ArchIndex:
    """Build and save hierarchical attributed communities from the KG cache."""
    settings = Settings.from_env()
    entities = _read_jsonl(graph_dir / ENTITIES_FILE)
    relations = _read_jsonl(graph_dir / RELATIONS_FILE)
    if not entities:
        raise ValueError(f"No KG entities found at {graph_dir / ENTITIES_FILE}. Run `uv run python main.py kg-build` first.")

    embeddings = build_embeddings(settings)
    llm = build_llm(settings)
    hierarchy = build_hierarchical_communities(
        entities=entities,
        relations=relations,
        llm=llm,
        embeddings=embeddings,
        max_levels=max_levels,
        min_nodes_per_level=min_nodes_per_level,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
        community_algorithm=community_algorithm,
    )
    hierarchy.metadata.update(
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "graph_dir": str(graph_dir),
            "archrag_dir": str(archrag_dir),
            "entities": len(entities),
            "relations": len(relations),
            "max_levels": max_levels,
            "min_nodes_per_level": min_nodes_per_level,
            "similarity_top_k": similarity_top_k,
            "similarity_threshold": similarity_threshold,
            "community_algorithm": community_algorithm,
        }
    )
    save_arch_index(hierarchy, archrag_dir, build_config=hierarchy.metadata)
    return hierarchy


def build_attributed_entity_graph(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    embeddings,
    similarity_top_k: int = 5,
    similarity_threshold: float = 0.65,
    relation_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> nx.Graph:
    """Build an attributed entity graph with KG relation edges and attribute-similarity edges."""
    graph = nx.Graph()
    texts = [_entity_text(entity) for entity in entities]
    vectors = _embed_texts(embeddings, texts, "Embedding entity textual attributes")
    for entity, text, vector in zip(entities, texts, vectors, strict=False):
        entity_id = str(entity.get("entity_id", "")).strip()
        if not entity_id:
            continue
        graph.add_node(
            entity_id,
            node_type="entity",
            name=str(entity.get("name", "")),
            text=text,
            summary=str(entity.get("description", "")) or text,
            embedding=vector,
            source_chunks=_source_chunk_keys(entity.get("source_chunks", [])),
            raw_source_chunks=_source_chunks(entity.get("source_chunks", [])),
            metadata={
                "entity_type": str(entity.get("type", "Other")),
                "aliases": entity.get("aliases", []),
                "attributes": entity.get("attributes", {}),
            },
        )

    for relation in relations:
        subject_id = str(relation.get("subject_id", "")).strip()
        object_id = str(relation.get("object_id", "")).strip()
        if subject_id not in graph or object_id not in graph or subject_id == object_id:
            continue
        predicate = str(relation.get("predicate", "other"))
        weight = relation_weight * PREDICATE_WEIGHTS.get(predicate, 0.5)
        _add_or_update_edge(
            graph,
            subject_id,
            object_id,
            weight=weight,
            edge_type="relation",
            relation=relation,
        )

    _add_attribute_similarity_edges(
        graph=graph,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
        attribute_weight=attribute_weight,
    )
    return graph


def detect_attributed_communities(graph: nx.Graph, algorithm: str = "louvain") -> list[set[str]]:
    """Detect weighted communities from an attributed graph."""
    if graph.number_of_nodes() == 0:
        return []
    normalized = algorithm.strip().lower()
    if normalized in {"louvain", "auto"}:
        try:
            return [
                set(community)
                for community in nx.algorithms.community.louvain_communities(
                    graph,
                    weight="weight",
                    seed=42,
                )
            ]
        except AttributeError:
            normalized = "greedy"
    if normalized in {"greedy", "greedy_modularity"}:
        return [
            set(community)
            for community in nx.algorithms.community.greedy_modularity_communities(
                graph,
                weight="weight",
            )
        ]
    if normalized in {"label", "label_propagation"}:
        return [
            set(community)
            for community in nx.algorithms.community.asyn_lpa_communities(
                graph,
                weight="weight",
                seed=42,
            )
        ]
    raise ValueError("community_algorithm must be one of: louvain, greedy, label")


def summarize_community(
    community_nodes: list[ArchNode],
    internal_edges: list[dict[str, Any]],
    llm,
) -> str:
    """Generate an LLM community summary, falling back to a deterministic summary if needed."""
    fallback = _deterministic_summary(community_nodes, internal_edges)
    if llm is None:
        return fallback
    facts = _summary_facts(community_nodes, internal_edges)
    prompt = f"""
You summarize one attributed community in an academic-paper knowledge graph.
Use only the supplied facts. Do not invent methods, datasets, metrics, or findings.

Return a concise English summary with these sections:
- Community theme
- Key entities
- Key relations
- Related methods, datasets, metrics, findings, limitations, or future work
- Source chunk ids

Facts:
{facts[:7000]}
""".strip()
    try:
        response = llm.invoke(prompt)
        text = _response_text(response).strip()
        return text or fallback
    except Exception as exc:
        return fallback + f"\nSummary generation warning: {exc}"


def build_hierarchical_communities(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    llm,
    embeddings,
    max_levels: int = 3,
    min_nodes_per_level: int = 5,
    similarity_top_k: int = 5,
    similarity_threshold: float = 0.65,
    community_algorithm: str = "louvain",
) -> ArchIndex:
    """Build level-0 entity nodes and repeated higher-level attributed communities."""
    max_levels = max(1, max_levels)
    entity_graph = build_attributed_entity_graph(
        entities=entities,
        relations=relations,
        embeddings=embeddings,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
    )
    level0_nodes = {
        node_id: ArchNode(
            node_id=node_id,
            level=0,
            node_type="entity",
            name=str(data.get("name", node_id)),
            text=str(data.get("text", "")),
            summary=str(data.get("summary", "")),
            embedding=[float(value) for value in data.get("embedding", [])],
            source_chunks=list(data.get("source_chunks", [])),
            children=[],
            parents=[],
            metadata={
                **dict(data.get("metadata", {})),
                "raw_source_chunks": list(data.get("raw_source_chunks", [])),
            },
        )
        for node_id, data in entity_graph.nodes(data=True)
    }
    layers: dict[int, ArchLayer] = {0: ArchLayer(level=0, nodes=level0_nodes)}
    current_graph = entity_graph
    current_nodes = level0_nodes

    for next_level in range(1, max_levels):
        if len(current_nodes) < max(2, min_nodes_per_level):
            break
        communities = _ordered_communities(detect_attributed_communities(current_graph, algorithm=community_algorithm))
        if not communities:
            break
        if len(communities) == len(current_nodes) and len(current_nodes) > min_nodes_per_level:
            communities = [set(current_nodes)]

        new_nodes = _community_nodes_for_level(
            level=next_level,
            communities=communities,
            child_nodes=current_nodes,
            lower_graph=current_graph,
            llm=llm,
            embeddings=embeddings,
        )
        if not new_nodes:
            break
        layers[next_level] = ArchLayer(level=next_level, nodes=new_nodes)
        for parent in new_nodes.values():
            for child_id in parent.children:
                child = current_nodes.get(child_id)
                if child is not None and parent.node_id not in child.parents:
                    child.parents.append(parent.node_id)

        if len(new_nodes) == 1:
            current_nodes = new_nodes
            break
        current_graph = _build_upper_graph(
            parent_nodes=new_nodes,
            lower_graph=current_graph,
            similarity_top_k=similarity_top_k,
            similarity_threshold=similarity_threshold,
        )
        current_nodes = new_nodes

    entry_node_id = _choose_entry_node(layers[max(layers)].nodes) if layers else None
    index = ArchIndex(layers=layers, entry_node_id=entry_node_id)
    index.metadata = {
        "levels": len(layers),
        "layer_node_counts": {str(level): len(layer.nodes) for level, layer in layers.items()},
        "hierarchy_kind": "hierarchical_attributed_communities",
    }
    return index


def _community_nodes_for_level(
    level: int,
    communities: list[set[str]],
    child_nodes: dict[str, ArchNode],
    lower_graph: nx.Graph,
    llm,
    embeddings,
) -> dict[str, ArchNode]:
    """Create community nodes for one upper hierarchy level."""
    nodes: dict[str, ArchNode] = {}
    for index, members in enumerate(communities, start=1):
        children = sorted(member for member in members if member in child_nodes)
        if not children:
            continue
        community_id = f"l{level}_c{index:04d}_{_hash_id('|'.join(children))[:8]}"
        member_nodes = [child_nodes[child_id] for child_id in children]
        internal_edges = _internal_edge_rows(lower_graph, children, child_nodes)
        summary = summarize_community(member_nodes, internal_edges, llm)
        source_chunks = _unique_value(item for node in member_nodes for item in node.source_chunks)
        raw_source_chunks = _unique_raw_source_chunks(member_nodes)
        text = _community_text(community_id, member_nodes, internal_edges, summary)
        vector = _embed_texts(embeddings, [summary or text], f"Embedding level {level} community {index}")[0]
        node = ArchNode(
            node_id=community_id,
            level=level,
            node_type="community",
            name=f"Level {level} community {index}",
            text=text,
            summary=summary,
            embedding=vector,
            source_chunks=source_chunks,
            children=children,
            parents=[],
            metadata={
                "member_count": len(children),
                "child_level": level - 1,
                "internal_edge_count": len(internal_edges),
                "key_children": [node.name for node in member_nodes[:20]],
                "raw_source_chunks": raw_source_chunks,
            },
        )
        nodes[node.node_id] = node
    return nodes


def _build_upper_graph(
    parent_nodes: dict[str, ArchNode],
    lower_graph: nx.Graph,
    similarity_top_k: int,
    similarity_threshold: float,
) -> nx.Graph:
    """Build the attributed graph used to cluster the next hierarchy level."""
    graph = nx.Graph()
    child_to_parent = {
        child_id: parent.node_id
        for parent in parent_nodes.values()
        for child_id in parent.children
    }
    for node in parent_nodes.values():
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            name=node.name,
            text=node.text,
            summary=node.summary,
            embedding=node.embedding,
            source_chunks=node.source_chunks,
            metadata=node.metadata,
        )
    for left, right, data in lower_graph.edges(data=True):
        parent_left = child_to_parent.get(left)
        parent_right = child_to_parent.get(right)
        if not parent_left or not parent_right or parent_left == parent_right:
            continue
        _add_or_update_edge(
            graph,
            parent_left,
            parent_right,
            weight=float(data.get("weight", 1.0)),
            edge_type="structural_rollup",
            relation={"description": f"Rolled up from lower edge {left} -- {right}"},
        )
    _add_attribute_similarity_edges(
        graph=graph,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
        attribute_weight=1.0,
    )
    return graph


def _add_attribute_similarity_edges(
    graph: nx.Graph,
    similarity_top_k: int,
    similarity_threshold: float,
    attribute_weight: float,
) -> None:
    """Add KNN-style attribute-similarity edges to an attributed graph."""
    node_ids = sorted(graph.nodes)
    for node_id in node_ids:
        vector = graph.nodes[node_id].get("embedding", [])
        scored: list[tuple[float, str]] = []
        for other_id in node_ids:
            if other_id == node_id:
                continue
            score = _cosine(vector, graph.nodes[other_id].get("embedding", []))
            if score >= similarity_threshold:
                scored.append((score, other_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for score, other_id in scored[: max(0, similarity_top_k)]:
            _add_or_update_edge(
                graph,
                node_id,
                other_id,
                weight=attribute_weight * score,
                edge_type="attribute_similarity",
                relation={"description": f"attribute similarity {score:.4f}", "similarity": score},
            )


def _add_or_update_edge(
    graph: nx.Graph,
    left: str,
    right: str,
    weight: float,
    edge_type: str,
    relation: dict[str, Any],
) -> None:
    """Accumulate edge weight and provenance between two graph nodes."""
    if graph.has_edge(left, right):
        edge = graph[left][right]
        edge["weight"] = float(edge.get("weight", 0.0)) + float(weight)
        edge.setdefault("edge_types", Counter())[edge_type] += 1
        edge.setdefault("relations", []).append(relation)
        return
    graph.add_edge(
        left,
        right,
        weight=float(weight),
        edge_types=Counter({edge_type: 1}),
        relations=[relation],
    )


def _internal_edge_rows(graph: nx.Graph, children: list[str], child_nodes: dict[str, ArchNode]) -> list[dict[str, Any]]:
    """Collect the strongest internal edges for one detected community."""
    child_set = set(children)
    rows: list[dict[str, Any]] = []
    for left, right, data in graph.edges(data=True):
        if left not in child_set or right not in child_set:
            continue
        rows.append(
            {
                "subject_id": left,
                "subject": child_nodes.get(left, ArchNode(left, 0, "entity", left, "", "")).name,
                "object_id": right,
                "object": child_nodes.get(right, ArchNode(right, 0, "entity", right, "", "")).name,
                "weight": float(data.get("weight", 0.0)),
                "edge_types": dict(data.get("edge_types", {})),
                "relations": data.get("relations", [])[:8],
            }
        )
    rows.sort(key=lambda row: (-float(row["weight"]), row["subject"], row["object"]))
    return rows[:60]


def _deterministic_summary(community_nodes: list[ArchNode], internal_edges: list[dict[str, Any]]) -> str:
    """Build a non-LLM community summary fallback."""
    names = [node.name for node in community_nodes[:12]]
    chunks = _unique_value(item for node in community_nodes for item in node.source_chunks)[:20]
    lines = [
        f"Community theme: {', '.join(names[:5]) if names else 'mixed paper concepts'}",
        f"Key entities: {', '.join(names) if names else 'none'}",
        "Key relations:",
    ]
    for edge in internal_edges[:12]:
        lines.append(f"- {edge['subject']} -- {edge['object']} (weight={edge['weight']:.3f})")
    lines.append(f"Source chunk ids: {', '.join(chunks) if chunks else 'none'}")
    return "\n".join(lines)


def _summary_facts(community_nodes: list[ArchNode], internal_edges: list[dict[str, Any]]) -> str:
    """Format community facts for the LLM summary prompt."""
    lines = ["Nodes:"]
    for node in community_nodes[:50]:
        lines.append(f"- {node.name} ({node.node_type}): {node.summary or node.text}")
        if node.source_chunks:
            lines.append(f"  source_chunks: {', '.join(node.source_chunks[:8])}")
    lines.append("Internal edges:")
    for edge in internal_edges[:50]:
        descriptions = []
        for relation in edge.get("relations", [])[:4]:
            if isinstance(relation, dict):
                descriptions.append(str(relation.get("description") or relation.get("predicate") or "relation"))
        lines.append(
            f"- {edge['subject']} -> {edge['object']} weight={edge['weight']:.3f}; "
            f"{'; '.join(descriptions)}"
        )
    return "\n".join(lines)


def _community_text(
    community_id: str,
    member_nodes: list[ArchNode],
    internal_edges: list[dict[str, Any]],
    summary: str,
) -> str:
    """Build the full textual attribute for a community node."""
    child_lines = [f"- {node.name}: {node.summary or node.text}" for node in member_nodes[:30]]
    edge_lines = [f"- {edge['subject']} -- {edge['object']} ({edge['weight']:.3f})" for edge in internal_edges[:30]]
    return "\n".join(
        [
            f"Community: {community_id}",
            "Summary:",
            summary,
            "Children:",
            *child_lines,
            "Internal edges:",
            *edge_lines,
        ]
    )


def _ordered_communities(communities: list[set[str]]) -> list[set[str]]:
    """Sort communities deterministically by size and first node id."""
    return sorted(
        [set(community) for community in communities if community],
        key=lambda members: (-len(members), sorted(members)[0]),
    )


def _choose_entry_node(nodes: dict[str, ArchNode]) -> str | None:
    """Pick a deterministic top-layer entry node."""
    if not nodes:
        return None
    return sorted(nodes.values(), key=lambda node: (-len(node.source_chunks), node.node_id))[0].node_id


def _entity_text(entity: dict[str, Any]) -> str:
    """Build the textual attribute used to embed one KG entity."""
    aliases = entity.get("aliases", [])
    alias_text = ", ".join(str(alias) for alias in aliases[:8]) if isinstance(aliases, list) else ""
    attributes = entity.get("attributes", {})
    attribute_text = json.dumps(attributes, ensure_ascii=False, sort_keys=True) if isinstance(attributes, dict) else ""
    return "\n".join(
        [
            f"Name: {entity.get('name', '')}",
            f"Type: {entity.get('type', '')}",
            f"Aliases: {alias_text}",
            f"Description: {entity.get('description', '')}",
            f"Attributes: {attribute_text}",
        ]
    )


def _embed_texts(embeddings, texts: list[str], description: str) -> list[list[float]]:
    """Embed texts in the project's configured batch size."""
    if not texts:
        return []
    vectors: list[list[float]] = []
    with create_progress() as progress:
        task = progress.add_task(description, total=len(texts))
        for start in range(0, len(texts), DEFAULT_EMBED_BATCH_SIZE):
            batch = texts[start : start + DEFAULT_EMBED_BATCH_SIZE]
            vectors.extend([[float(value) for value in vector] for vector in embeddings.embed_documents(batch)])
            progress.advance(task, advance=len(batch))
    return vectors


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


def _source_chunks(raw_refs: Any) -> list[dict[str, str]]:
    """Normalize raw KG source chunk references."""
    refs: list[dict[str, str]] = []
    if not isinstance(raw_refs, list):
        return refs
    for ref in raw_refs:
        if not isinstance(ref, dict):
            continue
        refs.append({str(key): str(value) for key, value in ref.items()})
    return refs


def _source_chunk_keys(raw_refs: Any) -> list[str]:
    """Extract stable chunk ids from KG source references."""
    keys: list[str] = []
    for ref in _source_chunks(raw_refs):
        key = str(ref.get("stable_chunk_id") or ref.get("chunk_key") or "").strip()
        if not key:
            source = str(ref.get("source", ""))
            page = str(ref.get("page", ""))
            chunk_id = str(ref.get("chunk_id", ""))
            key = "|".join(part for part in [source, page, chunk_id] if part)
        if key and key not in keys:
            keys.append(key)
    return keys


def _unique_value(values) -> list[str]:
    """Return unique non-empty strings in first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _unique_raw_source_chunks(nodes: list[ArchNode]) -> list[dict[str, str]]:
    """Merge raw source references from child nodes without duplicates."""
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for node in nodes:
        raw_refs = node.metadata.get("raw_source_chunks", [])
        if not isinstance(raw_refs, list):
            continue
        for raw in raw_refs:
            if not isinstance(raw, dict):
                continue
            ref = {str(key): str(value) for key, value in raw.items()}
            key = (
                str(ref.get("stable_chunk_id") or ref.get("chunk_key") or ""),
                str(ref.get("source", "")),
                str(ref.get("page", "")),
                str(ref.get("chunk_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
            if len(refs) >= 80:
                return refs
    return refs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL dictionaries, returning an empty list when missing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, strict=False)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _response_text(response) -> str:
    """Extract text from a LangChain response object."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _hash_id(value: str) -> str:
    """Return a stable SHA1 hex digest for ids."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


__all__ = [
    "build_archrag_hierarchy_cache",
    "build_attributed_entity_graph",
    "detect_attributed_communities",
    "summarize_community",
    "build_hierarchical_communities",
    "DEFAULT_ARCHRAG_DIR",
]
