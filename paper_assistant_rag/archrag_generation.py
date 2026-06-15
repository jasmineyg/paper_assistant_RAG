"""Adaptive filtering and answer generation for ArchRAG hierarchical results."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from langchain_core.documents import Document

from paper_assistant_rag.archrag.query_processing import (
    build_retrieval_query,
    get_rerank_weights,
    rewrite_query,
)
from paper_assistant_rag.archrag_index import hierarchical_search_by_embedding
from paper_assistant_rag.archrag_types import ArchIndex
from paper_assistant_rag.retrieval import clean_model_output, normalize_text

CHUNK_SEMANTIC_WEIGHT = 0.65
HIERARCHY_NODE_WEIGHT = 0.25
KEYWORD_WEIGHT = 0.10
DEFAULT_MAX_CHUNKS_PER_NODE = 3
DEFAULT_MAX_CHUNKS_PER_PAPER = 6


def adaptive_filter_level_results(
    query: str,
    level_results: dict[int, list[dict[str, Any]]],
    llm,
) -> list[dict[str, Any]]:
    """Ask the LLM to extract query-relevant points from each hierarchy level."""
    reports: list[dict[str, Any]] = []
    for level in sorted(level_results, reverse=True):
        nodes = level_results[level]
        if not nodes:
            continue
        evidence_lookup = {f"E{index}": node for index, node in enumerate(nodes, start=1)}
        prompt = f"""
You are filtering hierarchical ArchRAG retrieval results for an academic-paper QA task.
Extract only information that is useful for answering the query.
Do not answer the query yet. Do not invent facts.

Query:
{query}

Level {level} results:
{_level_evidence_text(evidence_lookup)}

Return valid JSON only:
{{
  "points": [
    {{
      "description": "query-relevant fact or reason this result matters",
      "score": 0-100,
      "source": "E1",
      "level": {level}
    }}
  ]
}}

Scoring:
- 90-100: directly answers the query.
- 70-89: strong supporting context.
- 40-69: related but incomplete.
- 0-39: weak or not useful.
""".strip()
        try:
            response = llm.invoke(prompt)
            parsed = _parse_json_object(_response_text(response))
            points = _points_from_response(parsed, level, evidence_lookup)
        except Exception as exc:
            points = _heuristic_points(level, nodes, warning=str(exc))
        reports.append({"level": level, "points": points})
    return reports


def merge_filtered_reports(
    query: str,
    reports: list[dict[str, Any]],
    llm,
    response_format: str = "Chinese answer with [S#] citations",
    source_rows_override: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge filtered hierarchy reports into a final cited answer."""
    points = _rank_points(reports)
    source_rows = source_rows_override if source_rows_override is not None else _source_rows_from_points(points)
    prompt = f"""
You are an academic-paper RAG assistant.
Answer only from the filtered ArchRAG evidence below. Do not invent facts.
Use Chinese. Cite source chunks with [S1], [S2], etc. after key claims.
If evidence is insufficient, say so clearly.

Query:
{query}

Response format:
{response_format}

Filtered points:
{_points_text(points)}

Source chunks:
{_sources_text(source_rows)}
""".strip()
    try:
        response = llm.invoke(prompt)
        answer = clean_model_output(_response_text(response))
    except Exception as exc:
        answer = _fallback_answer(points, source_rows, exc)
    return {
        "answer": answer,
        "sources": source_rows,
        "points": points,
    }


def generate_archrag_answer(
    query: str,
    arch_index: ArchIndex,
    llm,
    embeddings,
    vectorstore=None,
    top_k_per_level: int = 5,
    max_levels: int | None = None,
    final_chunk_limit: int = 10,
    entry_k: int = 3,
    chat_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run hierarchical search, adaptive filtering, and final answer merging."""
    rewritten_query = rewrite_query(query, llm=llm, chat_history=chat_history)
    query_type = str(rewritten_query["query_type"])
    retrieval_query = build_retrieval_query(rewritten_query)
    query_embedding = [float(value) for value in embeddings.embed_query(retrieval_query)]
    search_result = hierarchical_search_by_embedding(
        arch_index=arch_index,
        query_embedding=query_embedding,
        query=retrieval_query,
        top_k_per_level=top_k_per_level,
        max_levels=max_levels,
        query_type=query_type,
        entry_k=entry_k,
    )
    reranked_documents = (
        rerank_archrag_chunks(
            query=retrieval_query,
            level_results=search_result["level_results"],
            vectorstore=vectorstore,
            embeddings=embeddings,
            limit=final_chunk_limit,
            query_embedding=query_embedding,
            query_type=query_type,
        )
        if vectorstore is not None
        else archrag_level_results_to_documents(
            search_result["level_results"],
            limit=final_chunk_limit,
        )
    )
    source_rows = _source_rows_from_ranked_documents(reranked_documents)
    reports = adaptive_filter_level_results(
        query=query,
        level_results=search_result["level_results"],
        llm=llm,
    )
    merged = merge_filtered_reports(
        query=query,
        reports=reports,
        llm=llm,
        response_format="Structured Chinese answer with explicit [S#] citations",
        source_rows_override=source_rows,
    )
    chunk_scores = _chunk_score_rows(reranked_documents)
    return {
        "answer": merged["answer"],
        "sources": merged["sources"],
        "rewritten_query": rewritten_query,
        "entry_nodes": search_result.get("entry_nodes", []),
        "retrieval_paths": search_result.get("retrieval_paths", []),
        "final_chunks": source_rows,
        "chunk_scores": chunk_scores,
        "query_type": query_type,
        "debug_info": {
            "search": search_result,
            "reports": reports,
            "points": merged["points"],
            "used_source_chunks": [row["stable_chunk_id"] for row in merged["sources"]],
            "rewritten_query": rewritten_query,
            "entry_nodes": search_result.get("entry_nodes", []),
            "retrieval_paths": search_result.get("retrieval_paths", []),
            "chunk_scores": chunk_scores,
            "query_type": query_type,
        },
    }


def archrag_level_results_to_documents(
    level_results: dict[int, list[dict[str, Any]]],
    limit: int,
) -> list[tuple[Document, float]]:
    """Convert hierarchy search results into ranked pseudo-documents for evaluation metrics."""
    rows: list[tuple[Document, float]] = []
    for level in sorted(level_results, reverse=True):
        for node in level_results[level]:
            score = float(node.get("score", 0.0))
            for ref in _node_source_refs(node):
                metadata = {
                    "document_type": "archrag_node",
                    "source": ref.get("source", "unknown"),
                    "page": ref.get("page", "?"),
                    "chunk_id": ref.get("chunk_id", "?"),
                    "stable_chunk_id": ref.get("stable_chunk_id", ""),
                    "archrag_level": str(level),
                    "archrag_node_id": str(node.get("node_id", "")),
                    "archrag_node_name": str(node.get("name", "")),
                    "archrag_node_type": str(node.get("node_type", "")),
                    "archrag_node_score": f"{score:.6f}",
                    "archrag_source_chunks_json": json.dumps(node.get("source_chunks", [])[:30], ensure_ascii=False),
                    "community_id": str(node.get("node_id", "")),
                }
                text = normalize_text(str(node.get("summary") or node.get("text") or ""))
                rows.append((Document(page_content=text, metadata=metadata), score))
    deduped: list[tuple[Document, float]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for doc, score in sorted(rows, key=lambda item: (-item[1], str(item[0].metadata.get("archrag_node_id", "")))):
        key = (
            str(doc.metadata.get("stable_chunk_id", "")),
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("page", "")),
            str(doc.metadata.get("chunk_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, score))
        if len(deduped) >= limit:
            break
    return deduped


def rerank_archrag_chunks(
    query: str,
    level_results: dict[int, list[dict[str, Any]]],
    vectorstore,
    embeddings,
    limit: int,
    semantic_weight: float | None = None,
    hierarchy_weight: float | None = None,
    keyword_weight: float | None = None,
    max_chunks_per_node: int = DEFAULT_MAX_CHUNKS_PER_NODE,
    max_chunks_per_paper: int = DEFAULT_MAX_CHUNKS_PER_PAPER,
    query_embedding: list[float] | np.ndarray | None = None,
    query_type: str | None = None,
) -> list[tuple[Document, float]]:
    """Rerank source chunks using query semantics, hierarchy score, and keywords."""
    if limit <= 0:
        return []
    chunk_lookup = _vectorstore_chunk_lookup(vectorstore)
    if not chunk_lookup:
        return archrag_level_results_to_documents(level_results, limit=limit)

    query_embedding = np.asarray(
        embeddings.embed_query(query) if query_embedding is None else query_embedding,
        dtype=np.float32,
    )
    weights = _resolve_rerank_weights(
        query_type=query_type,
        semantic_weight=semantic_weight,
        hierarchy_weight=hierarchy_weight,
        keyword_weight=keyword_weight,
    )
    query_terms = _query_terms(query)
    candidates: dict[str, dict[str, Any]] = {}
    for level in sorted(level_results, reverse=True):
        for node in level_results[level]:
            node_score = float(node.get("route_score", node.get("score", 0.0)))
            node_score_01 = _cosine_to_unit_interval(node_score)
            for ref in _node_source_refs(node):
                stable_id = str(ref.get("stable_chunk_id", "")).strip()
                stored = chunk_lookup.get(stable_id)
                if stored is None:
                    continue
                previous = candidates.get(stable_id)
                if previous is not None and float(previous["node_score_01"]) >= node_score_01:
                    continue
                candidates[stable_id] = {
                    "stable_id": stable_id,
                    "document": stored["document"],
                    "position": stored["position"],
                    "node": node,
                    "level": level,
                    "node_score": node_score,
                    "node_score_01": node_score_01,
                }

    ranked: list[tuple[Document, float]] = []
    for candidate in candidates.values():
        document = candidate["document"]
        chunk_vector = _reconstruct_vector(vectorstore, int(candidate["position"]))
        semantic_score = _cosine_dense(query_embedding, chunk_vector)
        semantic_score_01 = _cosine_to_unit_interval(semantic_score)
        keyword_score = _keyword_relevance(query_terms, document.page_content)
        final_score = (
            weights["semantic"] * semantic_score_01
            + weights["hierarchy"] * float(candidate["node_score_01"])
            + weights["keyword"] * keyword_score
        )
        node = candidate["node"]
        metadata = dict(document.metadata)
        metadata.update(
            {
                "archrag_level": str(candidate["level"]),
                "archrag_node_id": str(node.get("node_id", "")),
                "archrag_node_name": str(node.get("name", "")),
                "archrag_node_type": str(node.get("node_type", "")),
                "archrag_node_score": f"{float(candidate['node_score']):.6f}",
                "chunk_semantic_score": f"{semantic_score:.6f}",
                "chunk_semantic_score_01": f"{semantic_score_01:.6f}",
                "chunk_hierarchy_score": f"{float(candidate['node_score_01']):.6f}",
                "chunk_keyword_score": f"{keyword_score:.6f}",
                "chunk_relevance_score": f"{final_score:.6f}",
                "chunk_rerank_weights_json": json.dumps(weights, sort_keys=True),
                "query_type": query_type or "legacy",
                "community_id": str(node.get("node_id", "")),
            }
        )
        ranked.append((Document(page_content=document.page_content, metadata=metadata), final_score))

    ranked.sort(
        key=lambda item: (
            -item[1],
            str(item[0].metadata.get("source", "")),
            str(item[0].metadata.get("stable_chunk_id", "")),
        )
    )
    return _apply_chunk_quotas(
        ranked,
        limit=limit,
        max_chunks_per_node=max_chunks_per_node,
        max_chunks_per_paper=max_chunks_per_paper,
    )


def _resolve_rerank_weights(
    query_type: str | None,
    semantic_weight: float | None,
    hierarchy_weight: float | None,
    keyword_weight: float | None,
) -> dict[str, float]:
    """Resolve dynamic defaults while preserving explicit legacy overrides."""
    defaults = (
        get_rerank_weights(query_type)
        if query_type is not None
        else {
            "semantic": CHUNK_SEMANTIC_WEIGHT,
            "hierarchy": HIERARCHY_NODE_WEIGHT,
            "keyword": KEYWORD_WEIGHT,
        }
    )
    return {
        "semantic": float(defaults["semantic"] if semantic_weight is None else semantic_weight),
        "hierarchy": float(defaults["hierarchy"] if hierarchy_weight is None else hierarchy_weight),
        "keyword": float(defaults["keyword"] if keyword_weight is None else keyword_weight),
    }


def _chunk_score_rows(
    ranked_documents: list[tuple[Document, float]],
) -> list[dict[str, Any]]:
    """Expose final and component scores in a stable JSON-friendly structure."""
    rows: list[dict[str, Any]] = []
    for document, score in ranked_documents:
        metadata = document.metadata
        try:
            weights = json.loads(str(metadata.get("chunk_rerank_weights_json", "{}")))
        except json.JSONDecodeError:
            weights = {}
        rows.append(
            {
                "stable_chunk_id": str(metadata.get("stable_chunk_id", "")),
                "score": float(score),
                "semantic": float(
                    metadata.get(
                        "chunk_semantic_score_01",
                        metadata.get("chunk_semantic_score", 0.0),
                    )
                ),
                "hierarchy": float(metadata.get("chunk_hierarchy_score", 0.0)),
                "keyword": float(metadata.get("chunk_keyword_score", 0.0)),
                "weights": weights,
            }
        )
    return rows


def _level_evidence_text(evidence_lookup: dict[str, dict[str, Any]]) -> str:
    """Format one level's node results for the adaptive-filter prompt."""
    blocks: list[str] = []
    for source_id, node in evidence_lookup.items():
        blocks.append(
            "\n".join(
                [
                    f"{source_id}:",
                    f"node_id: {node.get('node_id', '')}",
                    f"level: {node.get('level', '')}",
                    f"type: {node.get('node_type', '')}",
                    f"name: {node.get('name', '')}",
                    f"score: {float(node.get('score', 0.0)):.4f}",
                    f"source_chunks: {', '.join(str(item) for item in node.get('source_chunks', [])[:12])}",
                    f"text: {normalize_text(str(node.get('summary') or node.get('text') or ''))[:1800]}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _points_from_response(
    parsed: dict[str, Any],
    level: int,
    evidence_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize LLM point JSON and attach source-node metadata."""
    raw_points = parsed.get("points", [])
    if not isinstance(raw_points, list):
        return []
    points: list[dict[str, Any]] = []
    for raw in raw_points:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", "")).strip()
        node = evidence_lookup.get(source) or _first_node(evidence_lookup)
        points.append(_point_from_node(raw, level, node, source))
    return points


def _heuristic_points(level: int, nodes: list[dict[str, Any]], warning: str) -> list[dict[str, Any]]:
    """Create deterministic filtering points when the LLM filter fails."""
    points: list[dict[str, Any]] = []
    for index, node in enumerate(nodes, start=1):
        raw = {
            "description": normalize_text(str(node.get("summary") or node.get("text") or ""))[:600],
            "score": max(0.0, min(100.0, float(node.get("score", 0.0)) * 100.0)),
            "source": f"E{index}",
            "level": level,
        }
        point = _point_from_node(raw, level, node, f"E{index}")
        point["filter_warning"] = warning
        points.append(point)
    return points


def _point_from_node(raw: dict[str, Any], level: int, node: dict[str, Any], source: str) -> dict[str, Any]:
    """Attach a raw filter point to its retrieved hierarchy node."""
    try:
        score = max(0.0, min(100.0, float(raw.get("score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "description": str(raw.get("description", "")).strip(),
        "score": score,
        "source": source,
        "level": int(raw.get("level", level) or level),
        "node_id": str(node.get("node_id", "")),
        "node_name": str(node.get("name", "")),
        "source_chunks": [str(item) for item in node.get("source_chunks", [])],
        "source_refs": _node_source_refs(node),
    }


def _rank_points(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten and rank filtered points by score and hierarchy level."""
    points = [point for report in reports for point in report.get("points", []) if float(point.get("score", 0.0)) >= 35.0]
    points.sort(key=lambda point: (-float(point.get("score", 0.0)), -int(point.get("level", 0)), str(point.get("node_id", ""))))
    return points[:24]


def _source_rows_from_points(points: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build printable source rows from filtered point chunk references."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for point in points:
        score = float(point.get("score", 0.0))
        for ref in point.get("source_refs", []):
            stable_id = str(ref.get("stable_chunk_id", "")).strip()
            fallback_key = "|".join(
                [
                    stable_id,
                    str(ref.get("source", "")),
                    str(ref.get("page", "")),
                    str(ref.get("chunk_id", "")),
                ]
            )
            if fallback_key in seen:
                continue
            seen.add(fallback_key)
            rows.append(
                {
                    "id": f"S{len(rows) + 1}",
                    "stable_chunk_id": stable_id,
                    "source": str(ref.get("source", "unknown")),
                    "page": str(ref.get("page", "?")),
                    "chunk": str(ref.get("chunk_id", "?")),
                    "score": f"{score:.1f}",
                    "snippet": str(ref.get("snippet", "") or stable_id),
                }
            )
            if len(rows) >= 12:
                return rows
    return rows


def _source_rows_from_ranked_documents(
    ranked_documents: list[tuple[Document, float]],
) -> list[dict[str, str]]:
    """Build final answer citations from query-reranked real chunk documents."""
    rows: list[dict[str, str]] = []
    for index, (document, score) in enumerate(ranked_documents, start=1):
        metadata = document.metadata
        rows.append(
            {
                "id": f"S{index}",
                "stable_chunk_id": str(metadata.get("stable_chunk_id", "")),
                "source": str(metadata.get("source", "unknown")),
                "page": str(metadata.get("page", "?")),
                "chunk": str(metadata.get("chunk_id", "?")),
                "score": f"{float(score):.4f}",
                "snippet": normalize_text(document.page_content)[:1600],
            }
        )
    return rows


def _node_source_refs(node: dict[str, Any]) -> list[dict[str, str]]:
    """Extract source chunk references from a hierarchy node result."""
    metadata = node.get("metadata", {})
    raw_refs = metadata.get("raw_source_chunks", []) if isinstance(metadata, dict) else []
    refs: list[dict[str, str]] = []
    if isinstance(raw_refs, list):
        for raw in raw_refs:
            if not isinstance(raw, dict):
                continue
            refs.append(
                {
                    "stable_chunk_id": str(raw.get("stable_chunk_id") or raw.get("chunk_key") or ""),
                    "source": str(raw.get("source", "unknown")),
                    "page": str(raw.get("page", "?")),
                    "chunk_id": str(raw.get("chunk_id", "?")),
                    "snippet": str(raw.get("text", "")),
                }
            )
    if refs:
        return refs
    return [
        {
            "stable_chunk_id": str(chunk_id),
            "source": "unknown",
            "page": "?",
            "chunk_id": str(chunk_id),
            "snippet": str(chunk_id),
        }
        for chunk_id in node.get("source_chunks", [])
    ]


def _vectorstore_chunk_lookup(vectorstore) -> dict[str, dict[str, Any]]:
    """Map stable chunk ids to their stored documents and FAISS positions."""
    docstore = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
    position_to_doc_id = getattr(vectorstore, "index_to_docstore_id", {})
    if not isinstance(docstore, dict) or not isinstance(position_to_doc_id, dict):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for position, doc_id in position_to_doc_id.items():
        document = docstore.get(doc_id)
        if not isinstance(document, Document):
            continue
        stable_id = str(document.metadata.get("stable_chunk_id", "")).strip()
        if stable_id:
            lookup[stable_id] = {"document": document, "position": int(position)}
    return lookup


def _reconstruct_vector(vectorstore, position: int) -> np.ndarray:
    """Read one stored FAISS vector for exact query-to-chunk cosine scoring."""
    try:
        return np.asarray(vectorstore.index.reconstruct(position), dtype=np.float32)
    except Exception:
        return np.empty(0, dtype=np.float32)


def _cosine_dense(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity for dense vectors, or zero for invalid rows."""
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or left.size != right.size:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(left @ right) / denominator


def _cosine_to_unit_interval(score: float) -> float:
    """Map cosine similarity from [-1, 1] into [0, 1]."""
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def _query_terms(query: str) -> list[str]:
    """Extract Latin tokens and Chinese bigrams for lightweight lexical matching."""
    normalized = normalize_text(query)
    latin_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_+\-]{1,}", normalized)
    if latin_terms:
        return list(dict.fromkeys(latin_terms))

    terms: list[str] = []
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    for sequence in chinese_sequences:
        terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return list(dict.fromkeys(terms))


def _keyword_relevance(query_terms: list[str], text: str) -> float:
    """Return the fraction of query terms present in the chunk text."""
    if not query_terms:
        return 0.0
    searchable = normalize_text(text)
    searchable_lower = searchable.lower()
    score = 0.0
    for term in query_terms:
        if term in searchable:
            score += 1.0
        elif term.lower() in searchable_lower:
            score += 0.5
    return score / len(query_terms)


def _apply_chunk_quotas(
    ranked: list[tuple[Document, float]],
    limit: int,
    max_chunks_per_node: int,
    max_chunks_per_paper: int,
) -> list[tuple[Document, float]]:
    """Apply diversity quotas, then relax them only when too few rows survive."""
    selected: list[tuple[Document, float]] = []
    selected_ids: set[str] = set()
    node_counts: dict[str, int] = {}
    paper_counts: dict[str, int] = {}
    for document, score in ranked:
        node_id = str(document.metadata.get("archrag_node_id", ""))
        paper = str(document.metadata.get("source", "unknown"))
        if node_counts.get(node_id, 0) >= max(1, max_chunks_per_node):
            continue
        if paper_counts.get(paper, 0) >= max(1, max_chunks_per_paper):
            continue
        stable_id = str(document.metadata.get("stable_chunk_id", ""))
        selected.append((document, score))
        selected_ids.add(stable_id)
        node_counts[node_id] = node_counts.get(node_id, 0) + 1
        paper_counts[paper] = paper_counts.get(paper, 0) + 1
        if len(selected) >= limit:
            return selected

    for document, score in ranked:
        stable_id = str(document.metadata.get("stable_chunk_id", ""))
        if stable_id in selected_ids:
            continue
        selected.append((document, score))
        selected_ids.add(stable_id)
        if len(selected) >= limit:
            break
    return selected


def _points_text(points: list[dict[str, Any]]) -> str:
    """Format filtered points for the final merge prompt."""
    lines: list[str] = []
    for index, point in enumerate(points, start=1):
        source_ids = ", ".join(str(ref.get("stable_chunk_id", "")) for ref in point.get("source_refs", [])[:6])
        lines.append(
            f"{index}. level={point.get('level')} node={point.get('node_id')} "
            f"score={float(point.get('score', 0.0)):.1f} chunks={source_ids}\n"
            f"   {point.get('description', '')}"
        )
    return "\n".join(lines) if lines else "No useful points survived filtering."


def _sources_text(source_rows: list[dict[str, str]]) -> str:
    """Format source rows for the final merge prompt."""
    if not source_rows:
        return "No concrete source chunks were available."
    return "\n".join(
        f"[{row['id']}] {row['source']} | page {row['page']} | chunk {row['chunk']} | "
        f"{row['stable_chunk_id']} | score {row['score']}\n{row.get('snippet', '')}"
        for row in source_rows
    )


def _fallback_answer(points: list[dict[str, Any]], source_rows: list[dict[str, str]], exc: Exception) -> str:
    """Return a conservative answer when final answer generation fails."""
    citations = " ".join(f"[{row['id']}]" for row in source_rows[:3])
    if not points:
        return f"证据不足，无法基于当前 ArchRAG 检索结果回答。生成阶段错误：{exc}"
    lines = ["基于当前 ArchRAG 层次检索结果，可以提炼出以下要点："]
    for point in points[:6]:
        lines.append(f"- {point.get('description', '')} {citations}".strip())
    lines.append(f"不确定性：最终合并阶段调用模型失败，以上为过滤点的保守汇总。错误：{exc}")
    return "\n".join(lines)


def _first_node(evidence_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the first available evidence node for malformed LLM source ids."""
    return next(iter(evidence_lookup.values())) if evidence_lookup else {}


def _response_text(response) -> str:
    """Extract text from a LangChain response object."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain or fenced LLM output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM did not return a JSON object")
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object")
    return parsed


__all__ = [
    "adaptive_filter_level_results",
    "merge_filtered_reports",
    "generate_archrag_answer",
    "archrag_level_results_to_documents",
    "rerank_archrag_chunks",
]
