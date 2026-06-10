"""Adaptive filtering and answer generation for ArchRAG hierarchical results."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.documents import Document

from paper_assistant_rag.archrag_index import hierarchical_search
from paper_assistant_rag.archrag_types import ArchIndex
from paper_assistant_rag.retrieval import clean_model_output, normalize_text


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
) -> dict[str, Any]:
    """Merge filtered hierarchy reports into a final cited answer."""
    points = _rank_points(reports)
    source_rows = _source_rows_from_points(points)
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
    top_k_per_level: int = 5,
    max_levels: int | None = None,
) -> dict[str, Any]:
    """Run hierarchical search, adaptive filtering, and final answer merging."""
    search_result = hierarchical_search(
        arch_index=arch_index,
        query=query,
        embeddings=embeddings,
        top_k_per_level=top_k_per_level,
        max_levels=max_levels,
    )
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
    )
    return {
        "answer": merged["answer"],
        "sources": merged["sources"],
        "debug_info": {
            "search": search_result,
            "reports": reports,
            "points": merged["points"],
            "used_source_chunks": [row["stable_chunk_id"] for row in merged["sources"]],
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
        for chunk_id in node.get("source_chunks", [])[:12]
    ]


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
        f"[{row['id']}] {row['source']} | page {row['page']} | chunk {row['chunk']} | {row['stable_chunk_id']}"
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
]
