"""Graph-assisted online retrieval over cached KG extraction files."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from paper_assistant_rag.documents import stable_chunk_id
from paper_assistant_rag.kg import ENTITIES_FILE, ENTITY_CHUNK_LINKS_FILE, RELATIONS_FILE
from paper_assistant_rag.retrieval import (
    hybrid_search_with_score,
    is_reference_like,
    normalize_text,
    retrieve_chunks_with_score,
)

GRAPH_RRF_K = 60
BASE_RRF_WEIGHT = 1.0
GRAPH_RRF_WEIGHT = 2.7

GRAPH_QUERY_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "does",
    "approach",
    "approaches",
    "data",
    "dataset",
    "datasets",
    "for",
    "from",
    "framework",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "paper",
    "papers",
    "please",
    "show",
    "summarize",
    "summary",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}

PREDICATE_INTENT_TERMS = {
    "proposes": ["propose", "proposed", "contribution", "提出", "贡献", "创新"],
    "uses": ["use", "uses", "component", "architecture", "framework", "method", "使用", "组件", "结构", "框架", "方法", "流程"],
    "improves": ["improve", "improves", "better", "提升", "改进", "优于"],
    "compares_with": ["compare", "comparison", "baseline", "versus", "对比", "比较", "基线"],
    "evaluated_on": ["evaluate", "evaluated", "experiment", "dataset", "实验", "评测", "数据集"],
    "reports_metric": ["metric", "result", "accuracy", "auc", "指标", "结果", "性能"],
    "addresses_problem": ["problem", "challenge", "motivation", "解决", "问题", "挑战", "动机"],
    "has_limitation": ["limitation", "weakness", "drawback", "局限", "缺点", "不足"],
    "extends": ["extend", "extends", "扩展"],
    "mentions": ["mention", "mentions", "提到"],
}


PREDICATE_INTENT_TERMS.update(
    {
        "proposes": [
            "propose",
            "proposed",
            "introduce",
            "introduced",
            "contribution",
            "novel",
            "\u63d0\u51fa",
            "\u4ecb\u7ecd",
            "\u8d21\u732e",
            "\u521b\u65b0",
        ],
        "uses": [
            "use",
            "uses",
            "employ",
            "employs",
            "framework",
            "method",
            "\u4f7f\u7528",
            "\u91c7\u7528",
            "\u65b9\u6cd5",
            "\u6846\u67b6",
        ],
        "has_component": [
            "component",
            "module",
            "architecture",
            "layer",
            "block",
            "\u7ec4\u4ef6",
            "\u6a21\u5757",
            "\u7ed3\u6784",
            "\u67b6\u6784",
        ],
        "evaluated_on": [
            "evaluate",
            "evaluated",
            "experiment",
            "dataset",
            "benchmark",
            "\u5b9e\u9a8c",
            "\u8bc4\u6d4b",
            "\u6570\u636e\u96c6",
        ],
        "evaluated_by": [
            "metric",
            "accuracy",
            "auc",
            "f1",
            "measure",
            "\u6307\u6807",
            "\u8bc4\u4ef7",
            "\u8861\u91cf",
        ],
        "reports_result": [
            "result",
            "performance",
            "achieve",
            "outperform",
            "\u7ed3\u679c",
            "\u6027\u80fd",
            "\u8fbe\u5230",
        ],
        "has_finding": [
            "finding",
            "conclusion",
            "ablation",
            "observation",
            "\u7ed3\u8bba",
            "\u53d1\u73b0",
            "\u5b9e\u9a8c\u7ed3\u8bba",
            "\u6d88\u878d",
        ],
        "solves": [
            "solve",
            "solves",
            "address",
            "challenge",
            "problem",
            "\u89e3\u51b3",
            "\u95ee\u9898",
            "\u6311\u6218",
        ],
        "addresses_scenario": [
            "scenario",
            "setting",
            "application",
            "domain",
            "\u573a\u666f",
            "\u5e94\u7528",
            "\u9886\u57df",
        ],
        "improves_on": [
            "improve",
            "improves",
            "better",
            "outperform",
            "\u6539\u8fdb",
            "\u63d0\u5347",
            "\u4f18\u4e8e",
        ],
        "improves": ["improve", "improves", "better", "\u6539\u8fdb", "\u63d0\u5347"],
        "compares_with": [
            "compare",
            "comparison",
            "baseline",
            "versus",
            "\u5bf9\u6bd4",
            "\u6bd4\u8f83",
            "\u57fa\u7ebf",
        ],
        "has_limitation": [
            "limitation",
            "weakness",
            "drawback",
            "failure",
            "\u5c40\u9650",
            "\u7f3a\u70b9",
            "\u4e0d\u8db3",
        ],
        "has_future_work": [
            "future",
            "future work",
            "extension",
            "\u672a\u6765",
            "\u5c55\u671b",
            "\u540e\u7eed",
        ],
        "similar_to": ["similar", "related", "\u76f8\u4f3c", "\u76f8\u5173"],
        "defines": ["define", "definition", "\u5b9a\u4e49"],
        "has_property": ["property", "invariant", "characteristic", "\u6027\u8d28", "\u7279\u6027"],
        "formulates_as": ["formulate", "formulation", "objective", "\u5efa\u6a21", "\u8868\u8ff0", "\u76ee\u6807\u51fd\u6570"],
        "outputs": ["output", "predict", "score", "\u8f93\u51fa", "\u9884\u6d4b", "\u5f97\u5206"],
        "supports": ["support", "evidence", "\u652f\u6301", "\u8bc1\u660e"],
        "reports_metric": ["metric", "result", "accuracy", "auc", "\u6307\u6807", "\u7ed3\u679c", "\u6027\u80fd"],
        "addresses_problem": ["problem", "challenge", "motivation", "\u89e3\u51b3", "\u95ee\u9898", "\u6311\u6218", "\u52a8\u673a"],
    }
)


@dataclass(frozen=True)
class ChunkRef:
    chunk_key: str
    source: str
    page: str
    chunk_id: str
    stable_chunk_id: str


@dataclass
class GraphCache:
    entities: dict[str, dict[str, Any]]
    relations: list[dict[str, Any]]
    entity_chunks: dict[str, list[ChunkRef]]
    chunk_entities: dict[str, set[str]]
    entity_relations: dict[str, list[dict[str, Any]]]


def retrieve_graph_chunks_with_score(
    vectorstore,
    query: str,
    k: int,
    include_references: bool,
    graph_dir: Path,
) -> list[tuple[Document, float]]:
    """Retrieve chunks by fusing chunk hybrid search with cached KG signals."""
    cache = load_graph_cache(graph_dir)
    if not cache.entities and not cache.relations:
        return retrieve_chunks_with_score(
            vectorstore,
            query=query,
            k=k,
            include_references=include_references,
        )

    raw_k = max(k * 8, k + 30)
    base_results = hybrid_search_with_score(vectorstore, query=query, k=raw_k)
    doc_index = _build_doc_index(vectorstore)
    seed_chunk_ids = _seed_chunk_ids(base_results[: max(k, 8)])
    seed_sources = _seed_sources(base_results[: max(k, 8)])
    graph_scores = _score_graph_chunks(
        cache=cache,
        query=query,
        seed_chunk_ids=seed_chunk_ids,
        seed_sources=seed_sources,
    )

    merged: dict[str, dict[str, Any]] = {}
    for rank, (doc, _score) in enumerate(base_results, start=1):
        chunk_key = _doc_chunk_key(doc)
        entry = merged.setdefault(chunk_key, {"doc": doc, "graph_score": 0.0})
        entry["base_rank"] = rank
        entry["base_score"] = 1.0 / (GRAPH_RRF_K + rank)

    ranked_graph = sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)
    for rank, (chunk_key, graph_score) in enumerate(ranked_graph, start=1):
        doc = doc_index.get(chunk_key)
        if doc is None:
            continue
        entry = merged.setdefault(chunk_key, {"doc": doc, "graph_score": 0.0})
        entry["graph_rank"] = rank
        entry["graph_score"] = graph_score
        entry["graph_rank_score"] = 1.0 / (GRAPH_RRF_K + rank)

    ranked = sorted(
        merged.values(),
        key=lambda entry: (
            -_combined_score(entry),
            entry.get("graph_rank", 10**9),
            entry.get("base_rank", 10**9),
        ),
    )

    selected: list[tuple[Document, float]] = []
    reference_like: list[tuple[Document, float]] = []
    for entry in ranked:
        doc = _with_graph_metadata(entry["doc"], entry)
        score = _combined_score(entry)
        if include_references or not is_reference_like(doc.page_content):
            selected.append((doc, score))
        else:
            reference_like.append((doc, score))
        if len(selected) >= k:
            return selected[:k]

    return (selected + reference_like)[:k]


def graph_cache_available(graph_dir: Path) -> bool:
    return (
        (graph_dir / ENTITIES_FILE).exists()
        and (graph_dir / RELATIONS_FILE).exists()
        and (graph_dir / ENTITY_CHUNK_LINKS_FILE).exists()
    )


def load_graph_cache(graph_dir: Path) -> GraphCache:
    graph_dir = graph_dir.resolve()
    signature = _graph_signature(graph_dir)
    return _load_graph_cache_cached(str(graph_dir), signature)


@lru_cache(maxsize=4)
def _load_graph_cache_cached(graph_dir_text: str, signature: tuple[tuple[str, int, int], ...]) -> GraphCache:
    graph_dir = Path(graph_dir_text)
    entities = {
        str(entity.get("entity_id", "")): entity
        for entity in _read_jsonl(graph_dir / ENTITIES_FILE)
        if entity.get("entity_id")
    }
    relations = _read_jsonl(graph_dir / RELATIONS_FILE)
    links = _read_jsonl(graph_dir / ENTITY_CHUNK_LINKS_FILE)

    entity_chunks: dict[str, list[ChunkRef]] = {}
    chunk_entities: dict[str, set[str]] = {}
    for link in links:
        entity_id = str(link.get("entity_id", ""))
        ref = _chunk_ref(link)
        if not entity_id or not ref.chunk_key:
            continue
        entity_chunks.setdefault(entity_id, []).append(ref)
        chunk_entities.setdefault(ref.chunk_key, set()).add(entity_id)

    for entity_id, entity in entities.items():
        for raw_ref in entity.get("source_chunks", []):
            if not isinstance(raw_ref, dict):
                continue
            ref = _chunk_ref(raw_ref)
            if not ref.chunk_key:
                continue
            _append_ref_once(entity_chunks.setdefault(entity_id, []), ref)
            chunk_entities.setdefault(ref.chunk_key, set()).add(entity_id)

    entity_relations: dict[str, list[dict[str, Any]]] = {}
    for relation in relations:
        subject_id = str(relation.get("subject_id", ""))
        object_id = str(relation.get("object_id", ""))
        if subject_id:
            entity_relations.setdefault(subject_id, []).append(relation)
        if object_id and object_id != subject_id:
            entity_relations.setdefault(object_id, []).append(relation)

    return GraphCache(
        entities=entities,
        relations=relations,
        entity_chunks=entity_chunks,
        chunk_entities=chunk_entities,
        entity_relations=entity_relations,
    )


def _score_graph_chunks(
    cache: GraphCache,
    query: str,
    seed_chunk_ids: dict[str, int],
    seed_sources: set[str],
) -> dict[str, float]:
    normalized_query = normalize_text(query).lower()
    query_terms = _query_terms(normalized_query)
    chunk_scores: dict[str, float] = {}
    matched_entity_scores: dict[str, float] = {}

    for entity_id, entity in cache.entities.items():
        direct_score = _entity_direct_match_score(entity, normalized_query, query_terms)
        type_score = _entity_type_intent_score(str(entity.get("type", "")), normalized_query)
        if type_score and not (direct_score or _refs_touch_sources(cache.entity_chunks.get(entity_id, []), seed_sources)):
            type_score = 0.0
        score = direct_score + type_score
        if score <= 0:
            continue
        matched_entity_scores[entity_id] = score
        for ref in cache.entity_chunks.get(entity_id, []):
            _add_chunk_score(chunk_scores, ref.chunk_key, score)

    for relation in cache.relations:
        score = _relation_match_score(
            relation=relation,
            cache=cache,
            normalized_query=normalized_query,
            query_terms=query_terms,
            matched_entity_scores=matched_entity_scores,
            seed_sources=seed_sources,
        )
        if score <= 0:
            continue
        for ref in _relation_chunk_refs(relation):
            _add_chunk_score(chunk_scores, ref.chunk_key, score)

    _expand_from_seed_chunks(
        cache=cache,
        seed_chunk_ids=seed_chunk_ids,
        chunk_scores=chunk_scores,
    )
    return chunk_scores


def _entity_direct_match_score(entity: dict[str, Any], normalized_query: str, query_terms: set[str]) -> float:
    names = _entity_names(entity)
    exact_score = 0.0
    for name in names:
        normalized_name = normalize_text(name).lower()
        if not normalized_name or len(normalized_name) < 3:
            continue
        if normalized_name in normalized_query:
            exact_score = max(exact_score, 10.0 + min(len(normalized_name), 80) / 20.0)

    search_text = _entity_search_text(entity)
    overlap_score = _weighted_overlap_score(query_terms, search_text)
    return exact_score + overlap_score


def _relation_match_score(
    relation: dict[str, Any],
    cache: GraphCache,
    normalized_query: str,
    query_terms: set[str],
    matched_entity_scores: dict[str, float],
    seed_sources: set[str],
) -> float:
    subject_id = str(relation.get("subject_id", ""))
    object_id = str(relation.get("object_id", ""))
    endpoint_score = 0.25 * (
        matched_entity_scores.get(subject_id, 0.0)
        + matched_entity_scores.get(object_id, 0.0)
    )

    search_text = _relation_search_text(relation, cache)
    overlap_score = _weighted_overlap_score(query_terms, search_text)
    predicate = str(relation.get("predicate", ""))
    intent_score = _predicate_intent_score(predicate, normalized_query)

    if intent_score and not (endpoint_score or overlap_score or _relation_touches_sources(relation, seed_sources)):
        intent_score = 0.0

    return endpoint_score + overlap_score + intent_score


def _expand_from_seed_chunks(
    cache: GraphCache,
    seed_chunk_ids: dict[str, int],
    chunk_scores: dict[str, float],
) -> None:
    for seed_chunk_id, seed_rank in seed_chunk_ids.items():
        seed_weight = 2.5 / math.sqrt(seed_rank)
        for entity_id in cache.chunk_entities.get(seed_chunk_id, set()):
            for ref in cache.entity_chunks.get(entity_id, []):
                _add_chunk_score(chunk_scores, ref.chunk_key, seed_weight)
            for relation in cache.entity_relations.get(entity_id, []):
                relation_weight = seed_weight * 0.6
                for ref in _relation_chunk_refs(relation):
                    _add_chunk_score(chunk_scores, ref.chunk_key, relation_weight)


def _build_doc_index(vectorstore) -> dict[str, Document]:
    doc_index: dict[str, Document] = {}
    for doc in _iter_vectorstore_documents(vectorstore):
        stable_id = _doc_chunk_key(doc)
        if stable_id:
            doc_index[stable_id] = doc
        legacy_key = _legacy_doc_key(doc)
        if legacy_key:
            doc_index.setdefault(legacy_key, doc)
    return doc_index


def _iter_vectorstore_documents(vectorstore) -> list[Document]:
    docstore_dict = getattr(vectorstore.docstore, "_dict", {})
    if not isinstance(docstore_dict, dict):
        return []
    return [doc for doc in docstore_dict.values() if isinstance(doc, Document)]


def _seed_chunk_ids(results: list[tuple[Document, float]]) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for rank, (doc, _score) in enumerate(results, start=1):
        chunk_key = _doc_chunk_key(doc)
        if chunk_key:
            seeds[chunk_key] = rank
    return seeds


def _seed_sources(results: list[tuple[Document, float]]) -> set[str]:
    sources: set[str] = set()
    for doc, _score in results[:5]:
        source = str(doc.metadata.get("source", ""))
        if source:
            sources.add(source)
    return sources


def _doc_chunk_key(doc: Document) -> str:
    stable_id = str(doc.metadata.get("stable_chunk_id", "")).strip()
    if stable_id:
        return stable_id
    stable_id = stable_chunk_id(doc)
    doc.metadata["stable_chunk_id"] = stable_id
    return stable_id


def _legacy_doc_key(doc: Document) -> str:
    metadata = doc.metadata
    source = str(metadata.get("source", ""))
    page = str(metadata.get("page", ""))
    chunk_id = str(metadata.get("chunk_id", ""))
    if not source or not page or not chunk_id:
        return ""
    return f"{source}|{page}|{chunk_id}"


def _chunk_ref(raw: dict[str, Any]) -> ChunkRef:
    stable_id = str(raw.get("stable_chunk_id", "") or raw.get("chunk_key", "")).strip()
    source = str(raw.get("source", "")).strip()
    page = str(raw.get("page", "")).strip()
    chunk_id = str(raw.get("chunk_id", "")).strip()
    chunk_key = stable_id or (f"{source}|{page}|{chunk_id}" if source and page and chunk_id else "")
    return ChunkRef(
        chunk_key=chunk_key,
        source=source,
        page=page,
        chunk_id=chunk_id,
        stable_chunk_id=stable_id,
    )


def _relation_chunk_refs(relation: dict[str, Any]) -> list[ChunkRef]:
    refs: list[ChunkRef] = []
    for raw_ref in relation.get("source_chunks", []):
        if not isinstance(raw_ref, dict):
            continue
        ref = _chunk_ref(raw_ref)
        if ref.chunk_key:
            _append_ref_once(refs, ref)
    return refs


def _append_ref_once(refs: list[ChunkRef], ref: ChunkRef) -> None:
    if all(existing.chunk_key != ref.chunk_key for existing in refs):
        refs.append(ref)


def _with_graph_metadata(doc: Document, entry: dict[str, Any]) -> Document:
    metadata = dict(doc.metadata)
    if "graph_rank" in entry:
        metadata["graph_rank"] = str(entry["graph_rank"])
        metadata["graph_score"] = f"{float(entry.get('graph_score', 0.0)):.4f}"
    if "base_rank" in entry:
        metadata["base_rank"] = str(entry["base_rank"])
    return Document(page_content=doc.page_content, metadata=metadata)


def _combined_score(entry: dict[str, Any]) -> float:
    return (
        BASE_RRF_WEIGHT * float(entry.get("base_score", 0.0))
        + GRAPH_RRF_WEIGHT * float(entry.get("graph_rank_score", 0.0))
    )


def _entity_names(entity: dict[str, Any]) -> list[str]:
    names = [str(entity.get("name", "")).strip(), str(entity.get("canonical_name", "")).strip()]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        names.extend(str(alias).strip() for alias in aliases)
    return [name for name in names if name]


def _entity_search_text(entity: dict[str, Any]) -> str:
    parts = [
        str(entity.get("name", "")),
        str(entity.get("canonical_name", "")),
        str(entity.get("type", "")),
        str(entity.get("description", "")),
    ]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        parts.extend(str(alias) for alias in aliases)
    observed_types = entity.get("observed_types", [])
    if isinstance(observed_types, list):
        parts.extend(str(value) for value in observed_types)
    return normalize_text(" ".join(parts)).lower()


def _relation_search_text(relation: dict[str, Any], cache: GraphCache) -> str:
    subject = cache.entities.get(str(relation.get("subject_id", "")), {})
    obj = cache.entities.get(str(relation.get("object_id", "")), {})
    parts = [
        str(relation.get("predicate", "")),
        str(relation.get("description", "")),
        str(subject.get("name", "")),
        str(obj.get("name", "")),
    ]
    evidence = relation.get("evidence", [])
    if isinstance(evidence, list):
        parts.extend(str(item) for item in evidence)
    return normalize_text(" ".join(parts)).lower()


def _weighted_overlap_score(query_terms: set[str], search_text: str) -> float:
    if not query_terms:
        return 0.0
    score = 0.0
    for term in query_terms:
        if term not in search_text:
            continue
        score += 1.0 if len(term) < 6 else 1.5
    return score


def _entity_type_intent_score(entity_type: str, normalized_query: str) -> float:
    entity_type = entity_type.lower()
    if entity_type == "contribution" and any(term in normalized_query for term in ["contribution", "novelty", "\u8d21\u732e", "\u521b\u65b0", "\u63d0\u51fa"]):
        return 4.5
    if entity_type == "finding" and any(term in normalized_query for term in ["finding", "conclusion", "ablation", "\u53d1\u73b0", "\u7ed3\u8bba", "\u5b9e\u9a8c\u7ed3\u8bba"]):
        return 4.5
    if entity_type == "futurework" and any(term in normalized_query for term in ["future", "future work", "\u672a\u6765", "\u5c55\u671b", "\u540e\u7eed"]):
        return 4.0
    if entity_type in {"problem", "scenario"} and any(term in normalized_query for term in ["problem", "challenge", "scenario", "setting", "\u95ee\u9898", "\u6311\u6218", "\u573a\u666f"]):
        return 4.0
    if entity_type == "module" and any(term in normalized_query for term in ["component", "module", "architecture", "\u7ec4\u4ef6", "\u6a21\u5757", "\u7ed3\u6784", "\u67b6\u6784"]):
        return 3.0
    if entity_type == "limitation" and any(term in normalized_query for term in ["limitation", "weakness", "局限", "缺点", "不足"]):
        return 4.0
    if entity_type == "dataset" and any(term in normalized_query for term in ["dataset", "data", "数据集", "数据"]):
        return 3.0
    if entity_type in {"method", "model"} and any(term in normalized_query for term in ["method", "model", "framework", "方法", "模型", "框架"]):
        return 2.0
    if entity_type == "metric" and any(term in normalized_query for term in ["metric", "result", "指标", "结果"]):
        return 3.0
    return 0.0


def _predicate_intent_score(predicate: str, normalized_query: str) -> float:
    terms = PREDICATE_INTENT_TERMS.get(predicate, [])
    if not any(term in normalized_query for term in terms):
        return 0.0
    if predicate in {
        "proposes",
        "evaluated_on",
        "evaluated_by",
        "reports_result",
        "has_finding",
        "solves",
        "has_limitation",
        "has_future_work",
        "reports_metric",
    }:
        return 4.0
    return 2.0


def _relation_touches_sources(relation: dict[str, Any], seed_sources: set[str]) -> bool:
    if not seed_sources:
        return False
    return _refs_touch_sources(_relation_chunk_refs(relation), seed_sources)


def _refs_touch_sources(refs: list[ChunkRef], seed_sources: set[str]) -> bool:
    if not seed_sources:
        return False
    return any(ref.source in seed_sources for ref in refs)


def _query_terms(normalized_query: str) -> set[str]:
    terms = {
        term.strip("-_")
        for term in re.findall(r"[a-z][a-z0-9_-]{1,}", normalized_query)
        if len(term.strip("-_")) >= 3 and term.strip("-_") not in GRAPH_QUERY_STOPWORDS
    }
    return terms


def _add_chunk_score(chunk_scores: dict[str, float], chunk_key: str, score: float) -> None:
    if not chunk_key:
        return
    chunk_scores[chunk_key] = chunk_scores.get(chunk_key, 0.0) + score


def _graph_signature(graph_dir: Path) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for name in [ENTITIES_FILE, RELATIONS_FILE, ENTITY_CHUNK_LINKS_FILE]:
        path = graph_dir / name
        if not path.exists():
            signature.append((name, 0, 0))
            continue
        stat = path.stat()
        signature.append((name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
    "method",
    "methods",
    "model",
    "models",
