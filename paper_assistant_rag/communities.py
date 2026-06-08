"""Single-level KG community detection and community-summary indexing."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_community.vectorstores import FAISS

from paper_assistant_rag.kg import ENTITIES_FILE, RELATIONS_FILE
from paper_assistant_rag.models import build_embeddings, build_llm
from paper_assistant_rag.paths import DEFAULT_EMBED_BATCH_SIZE
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, create_progress

COMMUNITIES_FILE = "communities.jsonl"
COMMUNITY_MANIFEST_FILE = "community_manifest.json"

PREDICATE_WEIGHTS = {
    "proposes": 2.0,
    "has_component": 2.0,
    "solves": 1.8,
    "has_finding": 1.8,
    "reports_result": 1.6,
    "evaluated_on": 1.5,
    "evaluated_by": 1.4,
    "improves_on": 1.4,
    "compares_with": 1.2,
    "has_limitation": 1.2,
    "has_future_work": 1.2,
    "uses": 1.0,
    "defines": 1.0,
    "has_property": 1.0,
    "formulates_as": 1.0,
    "outputs": 1.0,
    "supports": 1.0,
    "similar_to": 0.8,
    "extends": 0.8,
    "mentions": 0.4,
    "other": 0.2,
}

ENTITY_TYPE_ORDER = [
    "Contribution",
    "Method",
    "Model",
    "Module",
    "Problem",
    "Scenario",
    "Dataset",
    "Metric",
    "Finding",
    "Result",
    "Limitation",
    "FutureWork",
    "Task",
    "Application",
    "Concept",
    "Paper",
]


def build_community_index(
    graph_dir: Path,
    community_index_dir: Path,
    algorithm: str,
    resolution: float,
    max_summary_entities: int,
    max_summary_relations: int,
    llm_summaries: bool,
    summary_concurrency: int,
) -> None:
    settings = Settings.from_env()
    entities = _read_jsonl(graph_dir / ENTITIES_FILE)
    relations = _read_jsonl(graph_dir / RELATIONS_FILE)
    if not entities:
        raise ValueError(f"No entities found at {graph_dir / ENTITIES_FILE}. Run kg-build first.")

    console.print("[bold]Building single-level KG communities[/bold]")
    embeddings = build_embeddings(settings)
    entity_texts = [_entity_embedding_text(entity) for entity in entities]
    entity_vectors = _embed_texts(embeddings, entity_texts, description="Embedding entity descriptions")
    graph = _build_graph(entities, relations, entity_vectors)
    raw_communities = _detect_communities(graph, algorithm=algorithm, resolution=resolution)
    communities = _community_records(
        graph=graph,
        raw_communities=raw_communities,
        relations=relations,
        max_summary_entities=max_summary_entities,
        max_summary_relations=max_summary_relations,
    )

    if llm_summaries:
        llm = build_llm(settings)
        communities = _refine_summaries_with_llm(
            llm=llm,
            communities=communities,
            concurrency=summary_concurrency,
        )

    _write_jsonl(graph_dir / COMMUNITIES_FILE, communities)
    _write_community_index(community_index_dir, communities, embeddings)
    _write_manifest(
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        algorithm=algorithm,
        resolution=resolution,
        entities=len(entities),
        relations=len(relations),
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        communities=len(communities),
        llm_summaries=llm_summaries,
        summary_concurrency=summary_concurrency,
    )
    console.print(f"Saved communities to [cyan]{graph_dir / COMMUNITIES_FILE}[/cyan]")
    console.print(f"Saved community FAISS index to [cyan]{community_index_dir}[/cyan]")
    console.print(
        f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges | "
        f"communities: {len(communities)}"
    )


def _build_graph(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    entity_vectors: list[list[float]],
) -> nx.Graph:
    graph = nx.Graph()
    for entity, vector in zip(entities, entity_vectors, strict=False):
        entity_id = str(entity.get("entity_id", "")).strip()
        if not entity_id:
            continue
        graph.add_node(
            entity_id,
            name=str(entity.get("name", "")),
            type=str(entity.get("type", "Other")),
            description=str(entity.get("description", "")),
            aliases=list(entity.get("aliases", [])) if isinstance(entity.get("aliases"), list) else [],
            source_chunks=list(entity.get("source_chunks", []))
            if isinstance(entity.get("source_chunks"), list)
            else [],
            embedding=vector,
        )

    for relation in relations:
        subject_id = str(relation.get("subject_id", "")).strip()
        object_id = str(relation.get("object_id", "")).strip()
        if not subject_id or not object_id or subject_id == object_id:
            continue
        if subject_id not in graph or object_id not in graph:
            continue
        predicate = str(relation.get("predicate", "other"))
        relation_weight = PREDICATE_WEIGHTS.get(predicate, 0.5)
        semantic_weight = _semantic_edge_weight(
            graph.nodes[subject_id].get("embedding", []),
            graph.nodes[object_id].get("embedding", []),
        )
        weight = relation_weight + semantic_weight
        if graph.has_edge(subject_id, object_id):
            edge = graph[subject_id][object_id]
            edge["weight"] = float(edge.get("weight", 0.0)) + weight
            edge["relation_count"] = int(edge.get("relation_count", 0)) + 1
            edge.setdefault("predicates", Counter())[predicate] += 1
            edge.setdefault("relation_ids", []).append(str(relation.get("relation_id", "")))
        else:
            graph.add_edge(
                subject_id,
                object_id,
                weight=weight,
                relation_count=1,
                predicates=Counter({predicate: 1}),
                relation_ids=[str(relation.get("relation_id", ""))],
            )
    return graph


def _detect_communities(graph: nx.Graph, algorithm: str, resolution: float) -> list[set[str]]:
    normalized = algorithm.strip().lower()
    if normalized in {"louvain", "auto"}:
        try:
            return [
                set(community)
                for community in nx.algorithms.community.louvain_communities(
                    graph,
                    weight="weight",
                    resolution=resolution,
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

    raise ValueError("algorithm must be one of: louvain, greedy, label")


def _community_records(
    graph: nx.Graph,
    raw_communities: list[set[str]],
    relations: list[dict[str, Any]],
    max_summary_entities: int,
    max_summary_relations: int,
) -> list[dict[str, Any]]:
    entity_to_community: dict[str, int] = {}
    ordered_communities = sorted(raw_communities, key=lambda members: (-len(members), sorted(members)[0]))
    for index, members in enumerate(ordered_communities, start=1):
        for entity_id in members:
            entity_to_community[entity_id] = index

    relation_rows_by_community: dict[int, list[dict[str, Any]]] = {}
    for relation in relations:
        subject_id = str(relation.get("subject_id", ""))
        object_id = str(relation.get("object_id", ""))
        community_index = entity_to_community.get(subject_id)
        if community_index is None or entity_to_community.get(object_id) != community_index:
            continue
        relation_rows_by_community.setdefault(community_index, []).append(relation)

    records: list[dict[str, Any]] = []
    for community_index, members in enumerate(ordered_communities, start=1):
        community_id = f"c{community_index:04d}"
        member_rows = [_entity_row(graph, entity_id) for entity_id in members]
        member_rows.sort(key=_entity_sort_key)
        key_entities = member_rows[:max_summary_entities]
        key_relations = _key_relations(
            graph=graph,
            relations=relation_rows_by_community.get(community_index, []),
            max_relations=max_summary_relations,
        )
        source_chunks = _top_source_chunks(member_rows, key_relations)
        type_counts = Counter(row["type"] for row in member_rows)
        predicate_counts = Counter(relation["predicate"] for relation in relation_rows_by_community.get(community_index, []))
        summary = _heuristic_summary(
            community_id=community_id,
            key_entities=key_entities,
            key_relations=key_relations,
            type_counts=type_counts,
            predicate_counts=predicate_counts,
            source_chunks=source_chunks,
        )
        records.append(
            {
                "community_id": community_id,
                "algorithm_level": "single",
                "member_count": len(member_rows),
                "community_members": [row["entity_id"] for row in member_rows],
                "key_entities": key_entities,
                "key_relations": key_relations,
                "entity_type_counts": dict(type_counts),
                "predicate_counts": dict(predicate_counts),
                "source_chunks": source_chunks,
                "community_summary": summary,
            }
        )
    return records


def _entity_row(graph: nx.Graph, entity_id: str) -> dict[str, Any]:
    data = graph.nodes[entity_id]
    return {
        "entity_id": entity_id,
        "name": str(data.get("name", "")),
        "type": str(data.get("type", "Other")),
        "description": str(data.get("description", "")),
        "source_chunks": list(data.get("source_chunks", [])),
        "degree": int(graph.degree(entity_id)),
        "weighted_degree": float(graph.degree(entity_id, weight="weight")),
    }


def _key_relations(graph: nx.Graph, relations: list[dict[str, Any]], max_relations: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for relation in relations:
        subject_id = str(relation.get("subject_id", ""))
        object_id = str(relation.get("object_id", ""))
        if subject_id not in graph or object_id not in graph:
            continue
        predicate = str(relation.get("predicate", "other"))
        score = (
            PREDICATE_WEIGHTS.get(predicate, 0.5)
            + 0.02 * float(graph.degree(subject_id, weight="weight"))
            + 0.02 * float(graph.degree(object_id, weight="weight"))
        )
        ranked.append(
            {
                "relation_id": str(relation.get("relation_id", "")),
                "subject_id": subject_id,
                "subject": str(graph.nodes[subject_id].get("name", "")),
                "predicate": predicate,
                "object_id": object_id,
                "object": str(graph.nodes[object_id].get("name", "")),
                "description": str(relation.get("description", "")),
                "evidence": relation.get("evidence", []),
                "source_chunks": relation.get("source_chunks", []),
                "score": score,
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), row["predicate"], row["subject"], row["object"]))
    return ranked[:max_relations]


def _heuristic_summary(
    community_id: str,
    key_entities: list[dict[str, Any]],
    key_relations: list[dict[str, Any]],
    type_counts: Counter,
    predicate_counts: Counter,
    source_chunks: list[dict[str, str]],
) -> str:
    by_type: dict[str, list[str]] = {}
    for entity in key_entities:
        by_type.setdefault(entity["type"], []).append(entity["name"])

    lines = [
        f"Community {community_id}",
        f"Theme: {_theme_from_entities(key_entities)}",
        f"Entity type distribution: {_counter_text(type_counts)}",
        f"Relation distribution: {_counter_text(predicate_counts)}",
        f"Core methods/models/modules: {_names_for_types(by_type, ['Method', 'Model', 'Module', 'Contribution'])}",
        f"Core problems/scenarios/tasks: {_names_for_types(by_type, ['Problem', 'Scenario', 'Task', 'Application'])}",
        f"Datasets and metrics: {_names_for_types(by_type, ['Dataset', 'Metric'])}",
        f"Findings/results/limitations/future work: {_names_for_types(by_type, ['Finding', 'Result', 'Limitation', 'FutureWork'])}",
        "Key entities:",
    ]
    for entity in key_entities[:12]:
        lines.append(f"- {entity['name']} ({entity['type']}): {entity['description']}")

    lines.append("Key relations:")
    for relation in key_relations[:12]:
        lines.append(
            "- "
            f"{relation['subject']} --{relation['predicate']}--> {relation['object']}: "
            f"{relation['description']}"
        )

    lines.append("Source chunks:")
    for ref in source_chunks[:10]:
        lines.append(f"- {ref.get('source', '')} | page {ref.get('page', '')} | chunk {ref.get('chunk_id', '')}")
    return "\n".join(lines)


def _refine_summaries_with_llm(
    llm,
    communities: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    max_workers = max(1, concurrency)
    with create_progress() as progress:
        task = progress.add_task("Refining community summaries", total=len(communities))
        if max_workers == 1:
            for index, community in enumerate(communities):
                communities[index] = _refine_single_summary(llm, community)
                progress.advance(task)
            return communities

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_refine_single_summary, llm, community): index
                for index, community in enumerate(communities)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                communities[index] = future.result()
                progress.advance(task)
    return communities


def _refine_single_summary(llm, community: dict[str, Any]) -> dict[str, Any]:
    refined = dict(community)
    prompt = f"""
You summarize one academic-paper knowledge-graph community for retrieval.
Use only the structured facts below.
Write a concise English summary with these parts:
1. Community topic.
2. Core methods/problems/datasets/metrics.
3. Key findings or limitations.
4. Why this community is useful for retrieval.

Facts:
{refined["community_summary"][:6000]}
""".strip()
    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", response)
        refined["community_summary"] = str(text).strip() or refined["community_summary"]
    except Exception as exc:
        refined["summary_error"] = str(exc)
    return refined


def _write_community_index(community_index_dir: Path, communities: list[dict[str, Any]], embeddings) -> None:
    community_index_dir.mkdir(parents=True, exist_ok=True)
    texts = [community["community_summary"] for community in communities]
    vectors = _embed_texts(embeddings, texts, description="Embedding community summaries")
    metadatas = [_community_metadata(community) for community in communities]
    text_embeddings = list(zip(texts, vectors))
    vectorstore = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)
    vectorstore.save_local(str(community_index_dir))


def _community_metadata(community: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_type": "community",
        "community_id": community["community_id"],
        "source": f"Community {community['community_id']}",
        "page": "-",
        "chunk_id": "-",
        "member_count": str(community["member_count"]),
        "key_entities": ", ".join(row["name"] for row in community.get("key_entities", [])[:12]),
        "source_chunks_json": json.dumps(community.get("source_chunks", [])[:30], ensure_ascii=False),
    }


def _embed_texts(embeddings, texts: list[str], description: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    with create_progress() as progress:
        task = progress.add_task(description, total=len(texts))
        for start in range(0, len(texts), DEFAULT_EMBED_BATCH_SIZE):
            batch = texts[start : start + DEFAULT_EMBED_BATCH_SIZE]
            vectors.extend(embeddings.embed_documents(batch))
            progress.advance(task, advance=len(batch))
    return vectors


def _entity_embedding_text(entity: dict[str, Any]) -> str:
    aliases = entity.get("aliases", [])
    alias_text = ", ".join(str(alias) for alias in aliases[:8]) if isinstance(aliases, list) else ""
    return "\n".join(
        [
            f"Name: {entity.get('name', '')}",
            f"Type: {entity.get('type', '')}",
            f"Aliases: {alias_text}",
            f"Description: {entity.get('description', '')}",
        ]
    )


def _entity_sort_key(entity: dict[str, Any]) -> tuple[int, float, str]:
    try:
        type_rank = ENTITY_TYPE_ORDER.index(entity["type"])
    except ValueError:
        type_rank = len(ENTITY_TYPE_ORDER)
    return (type_rank, -float(entity.get("weighted_degree", 0.0)), entity["name"].lower())


def _theme_from_entities(entities: list[dict[str, Any]]) -> str:
    names = [entity["name"] for entity in entities[:5] if entity.get("name")]
    return "; ".join(names) if names else "mixed academic-paper concepts"


def _counter_text(counter: Counter) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(8))


def _names_for_types(by_type: dict[str, list[str]], types: list[str]) -> str:
    names: list[str] = []
    for entity_type in types:
        names.extend(by_type.get(entity_type, [])[:5])
    return "; ".join(names[:12]) if names else "none"


def _top_source_chunks(
    entity_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    refs_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for entity in entity_rows:
        for raw_ref in entity.get("source_chunks", []):
            _count_ref(raw_ref, counts, refs_by_key)
    for relation in relation_rows:
        for raw_ref in relation.get("source_chunks", []):
            _count_ref(raw_ref, counts, refs_by_key)

    refs = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [refs_by_key[key] for key, _count in refs[:30]]


def _count_ref(
    raw_ref: Any,
    counts: Counter[tuple[str, str, str, str]],
    refs_by_key: dict[tuple[str, str, str, str], dict[str, str]],
) -> None:
    if not isinstance(raw_ref, dict):
        return
    source = str(raw_ref.get("source", ""))
    page = str(raw_ref.get("page", ""))
    chunk_id = str(raw_ref.get("chunk_id", ""))
    stable_chunk_id = str(raw_ref.get("stable_chunk_id", "") or raw_ref.get("chunk_key", ""))
    key = (stable_chunk_id, source, page, chunk_id)
    if not any(key):
        return
    counts[key] += 1
    refs_by_key[key] = {
        "stable_chunk_id": stable_chunk_id,
        "source": source,
        "page": page,
        "chunk_id": chunk_id,
    }


def _semantic_edge_weight(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm)) * 0.25


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, strict=False)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(
    graph_dir: Path,
    community_index_dir: Path,
    algorithm: str,
    resolution: float,
    entities: int,
    relations: int,
    graph_nodes: int,
    graph_edges: int,
    communities: int,
    llm_summaries: bool,
    summary_concurrency: int,
) -> None:
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "resolution": resolution,
        "graph_dir": str(graph_dir),
        "community_index_dir": str(community_index_dir),
        "files": {
            "communities": COMMUNITIES_FILE,
            "community_index": str(community_index_dir),
        },
        "counts": {
            "entities": entities,
            "relations": relations,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "communities": communities,
        },
        "llm_summaries": llm_summaries,
        "summary_concurrency": summary_concurrency,
        "status": "ok",
    }
    (graph_dir / COMMUNITY_MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
