"""ArchRAG-lite retrieval that uses graph/community signals as a paper gate."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

from paper_assistant_rag.community_retrieval import _community_search
from paper_assistant_rag.graph_retrieval import graph_cache_available, retrieve_graph_chunks_with_score
from paper_assistant_rag.indexing import index_exists
from paper_assistant_rag.retrieval import (
    hybrid_search_with_score,
    is_reference_like,
    keyword_search_with_score,
    normalize_text,
    retrieve_chunks_with_score,
)

QueryType = Literal["specific", "paper_level", "abstract"]

HYBRID_PAPER_WEIGHT = 1.0
COMMUNITY_PAPER_WEIGHT = 1.2
GRAPH_PAPER_WEIGHT = 1.0
PAPER_RRF_K = 60


@dataclass
class PaperCandidate:
    """A paper-level candidate with fused scores and source diagnostics."""

    source: str
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


def detect_query_type(query: str) -> QueryType:
    """Classify the query so community evidence is used at the right granularity."""
    normalized = normalize_text(query).lower()
    if _contains_any(
        normalized,
        [
            "哪篇论文",
            "哪些论文",
            "相关论文",
            "提出了某方法",
            "which paper",
            "which papers",
            "related papers",
        ],
    ):
        return "paper_level"
    if _contains_any(
        normalized,
        [
            "总结",
            "对比",
            "共同问题",
            "研究脉络",
            "有哪些方向",
            "这些论文",
            "summarize",
            "compare",
            "common problem",
            "research trend",
        ],
    ):
        return "abstract"
    return "specific"


def retrieve_candidate_papers(
    query: str,
    vectorstore,
    graph_dir: Path | None = None,
    community_index_dir: Path | None = None,
    top_papers: int = 5,
    community_k: int = 3,
) -> list[str]:
    """Return the most relevant paper/source names without returning final chunks."""
    return [
        candidate.source
        for candidate in retrieve_candidate_papers_with_scores(
            query=query,
            vectorstore=vectorstore,
            graph_dir=graph_dir,
            community_index_dir=community_index_dir,
            top_papers=top_papers,
            community_k=community_k,
        )
    ]


def retrieve_candidate_papers_with_scores(
    query: str,
    vectorstore,
    graph_dir: Path | None = None,
    community_index_dir: Path | None = None,
    top_papers: int = 5,
    community_k: int = 3,
) -> list[PaperCandidate]:
    """Fuse hybrid, graph, and community paper signals into ranked paper candidates."""
    top_papers = max(1, top_papers)
    candidates: dict[str, PaperCandidate] = {}
    raw_k = max(top_papers * 8, top_papers + 20)

    hybrid_results = hybrid_search_with_score(vectorstore, query=query, k=raw_k)
    _add_doc_paper_votes(
        candidates,
        hybrid_results,
        signal="hybrid",
        weight=HYBRID_PAPER_WEIGHT,
    )

    if graph_dir is not None and graph_cache_available(graph_dir):
        graph_results = retrieve_graph_chunks_with_score(
            vectorstore=vectorstore,
            query=query,
            k=raw_k,
            include_references=True,
            graph_dir=graph_dir,
        )
        _add_doc_paper_votes(
            candidates,
            graph_results,
            signal="graph",
            weight=GRAPH_PAPER_WEIGHT,
        )

    if community_index_dir is not None and index_exists(community_index_dir):
        community_results = _community_search(
            query=query,
            community_index_dir=community_index_dir,
            community_k=max(community_k, top_papers),
        )
        _add_community_paper_votes(
            candidates,
            community_results,
            weight=COMMUNITY_PAPER_WEIGHT,
        )

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, candidate.source.lower()),
    )
    return ranked[:top_papers]


def retrieve_chunks_within_candidate_papers(
    query: str,
    candidate_papers: list[str],
    vectorstore,
    top_k: int = 8,
    per_paper_k: int = 5,
    fetch_k: int = 50,
    include_references: bool = False,
    paper_scores: dict[str, float] | None = None,
) -> list[tuple[Document, float]]:
    """Run chunk-level retrieval only inside the selected candidate papers."""
    if not candidate_papers:
        return retrieve_chunks_with_score(
            vectorstore,
            query=query,
            k=top_k,
            include_references=include_references,
        )

    paper_scores = paper_scores or {}
    results: list[tuple[Document, float]] = []
    for paper in candidate_papers:
        paper_filter = _paper_metadata_filter(paper)
        try:
            paper_results = hybrid_search_with_score(
                vectorstore,
                query=query,
                k=max(per_paper_k * 3, per_paper_k),
                metadata_filter=paper_filter,
                fetch_k=fetch_k,
            )
        except Exception:
            paper_results = _fallback_keyword_results_for_paper(
                vectorstore=vectorstore,
                query=query,
                paper=paper,
                k=max(per_paper_k * 3, per_paper_k),
            )
        for doc, score in paper_results[:per_paper_k]:
            metadata = dict(doc.metadata)
            metadata["candidate_paper"] = paper
            metadata["paper_score"] = f"{paper_scores.get(paper, 0.0):.6f}"
            metadata["document_type"] = str(metadata.get("document_type", "chunk"))
            results.append((Document(page_content=doc.page_content, metadata=metadata), float(score)))

    if not results:
        return retrieve_chunks_with_score(
            vectorstore,
            query=query,
            k=top_k,
            include_references=include_references,
        )
    return results


def rerank_candidate_chunks(
    query: str,
    chunks: list[tuple[Document, float]],
    top_k: int,
    include_references: bool = False,
) -> list[tuple[Document, float]]:
    """Lightweight local reranker for gated chunk candidates."""
    if not chunks:
        return []

    chunk_scores = [float(score) for _doc, score in chunks]
    max_chunk_score = max(chunk_scores) if chunk_scores else 0.0
    max_paper_score = max((_paper_score(doc) for doc, _score in chunks), default=0.0)
    query_terms = _query_terms(query)

    ranked: list[tuple[float, int, Document]] = []
    for rank, (doc, score) in enumerate(chunks, start=1):
        chunk_relevance = _normalize_score(float(score), max_chunk_score)
        keyword_score = _keyword_overlap_score(query_terms, doc)
        paper_score = _normalize_score(_paper_score(doc), max_paper_score)
        reference_penalty = 1.0 if is_reference_like(doc.page_content) else 0.0
        section_bonus = _section_match_score(query_terms, doc)
        final_score = (
            0.50 * chunk_relevance
            + 0.25 * keyword_score
            + 0.15 * paper_score
            + 0.05 * section_bonus
            - 0.10 * reference_penalty
        )
        metadata = dict(doc.metadata)
        metadata["chunk_relevance_score"] = f"{chunk_relevance:.4f}"
        metadata["keyword_score"] = f"{keyword_score:.4f}"
        metadata["section_match_score"] = f"{section_bonus:.4f}"
        metadata["reference_penalty"] = f"{reference_penalty:.1f}"
        metadata["gated_final_score"] = f"{final_score:.4f}"
        ranked.append((final_score, -rank, Document(page_content=doc.page_content, metadata=metadata)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[Document, float]] = []
    reference_like: list[tuple[Document, float]] = []
    for final_score, _negative_rank, doc in ranked:
        if include_references or not is_reference_like(doc.page_content):
            selected.append((doc, final_score))
        else:
            reference_like.append((doc, final_score))
        if len(selected) >= top_k:
            return selected[:top_k]
    return (selected + reference_like)[:top_k]


def retrieve_archrag_gated_chunks_with_score(
    vectorstore,
    query: str,
    k: int,
    include_references: bool,
    graph_dir: Path | None,
    community_index_dir: Path | None,
    candidate_papers: int = 5,
    per_paper_k: int = 5,
    community_k: int = 3,
    include_community_docs: bool = False,
) -> list[tuple[Document, float]]:
    """Two-stage ArchRAG-lite retrieval: paper gate first, precise chunks second."""
    query_type = detect_query_type(query)
    paper_candidates = retrieve_candidate_papers_with_scores(
        query=query,
        vectorstore=vectorstore,
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        top_papers=candidate_papers,
        community_k=community_k,
    )
    if not paper_candidates:
        return retrieve_chunks_with_score(
            vectorstore,
            query=query,
            k=k,
            include_references=include_references,
        )

    paper_names = [candidate.source for candidate in paper_candidates]
    paper_scores = {candidate.source: candidate.score for candidate in paper_candidates}
    per_paper = max(per_paper_k, 2 if query_type == "paper_level" else 1)
    chunk_candidates = retrieve_chunks_within_candidate_papers(
        query=query,
        candidate_papers=paper_names,
        vectorstore=vectorstore,
        top_k=max(k * 3, k + 10),
        per_paper_k=per_paper,
        fetch_k=max(candidate_papers * per_paper * 5, 50),
        include_references=include_references,
        paper_scores=paper_scores,
    )
    selected = rerank_candidate_chunks(
        query=query,
        chunks=chunk_candidates,
        top_k=k,
        include_references=include_references,
    )
    selected = _with_gated_debug_metadata(
        selected,
        query_type=query_type,
        paper_candidates=paper_candidates,
    )

    if include_community_docs and query_type == "abstract" and community_index_dir is not None:
        selected = _prepend_limited_community_docs(
            selected=selected,
            query=query,
            community_index_dir=community_index_dir,
            community_k=community_k,
            k=k,
        )
    return selected


def _add_doc_paper_votes(
    candidates: dict[str, PaperCandidate],
    results: list[tuple[Document, float]],
    signal: str,
    weight: float,
) -> None:
    seen_sources: set[str] = set()
    for rank, (doc, _score) in enumerate(results, start=1):
        source = _paper_source(doc)
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        _add_vote(candidates, source, signal, rank, weight)


def _add_community_paper_votes(
    candidates: dict[str, PaperCandidate],
    results: list[tuple[Document, float]],
    weight: float,
) -> None:
    seen_sources: set[str] = set()
    for community_rank, (community_doc, _score) in enumerate(results, start=1):
        for ref_rank, source in enumerate(_community_sources(community_doc), start=1):
            if source in seen_sources:
                continue
            seen_sources.add(source)
            adjusted_rank = community_rank + max(ref_rank - 1, 0) * 0.1
            _add_vote(candidates, source, "community", adjusted_rank, weight)


def _add_vote(
    candidates: dict[str, PaperCandidate],
    source: str,
    signal: str,
    rank: float,
    weight: float,
) -> None:
    candidate = candidates.setdefault(source, PaperCandidate(source=source))
    score = weight / (PAPER_RRF_K + rank)
    candidate.score += score
    candidate.signals[signal] = candidate.signals.get(signal, 0.0) + score
    previous_rank = candidate.ranks.get(signal)
    if previous_rank is None or rank < previous_rank:
        candidate.ranks[signal] = int(math.ceil(rank))


def _paper_source(doc: Document) -> str:
    metadata = doc.metadata
    return str(metadata.get("source") or metadata.get("paper_id") or "").strip()


def _community_sources(community_doc: Document) -> list[str]:
    raw_refs = str(community_doc.metadata.get("source_chunks_json", "[]"))
    try:
        refs = json.loads(raw_refs)
    except json.JSONDecodeError:
        return []

    sources: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        source = str(ref.get("source", "")).strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def _paper_metadata_filter(paper: str):
    normalized_paper = _normalize_paper_name(paper)

    def matches(metadata: dict[str, Any]) -> bool:
        values = [
            str(metadata.get("source", "")),
            str(metadata.get("source_path", "")),
            str(metadata.get("paper_id", "")),
        ]
        return any(_normalize_paper_name(value) == normalized_paper for value in values)

    return matches


def _fallback_keyword_results_for_paper(
    vectorstore,
    query: str,
    paper: str,
    k: int,
) -> list[tuple[Document, float]]:
    return keyword_search_with_score(
        vectorstore,
        query=query,
        k=k,
        metadata_filter=_paper_metadata_filter(paper),
    )


def _query_terms(query: str) -> set[str]:
    normalized = normalize_text(query).lower()
    english_terms = {
        term.strip("-_")
        for term in re.findall(r"[a-z][a-z0-9_-]{1,}", normalized)
        if len(term.strip("-_")) >= 3
    }
    chinese_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
    return {term for term in english_terms.union(chinese_terms) if term}


def _keyword_overlap_score(query_terms: set[str], doc: Document) -> float:
    if not query_terms:
        return 0.0
    searchable = normalize_text(
        " ".join(
            [
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("section_title", "")),
                str(doc.metadata.get("section_type", "")),
                doc.page_content,
            ]
        )
    ).lower()
    hits = 0.0
    for term in query_terms:
        if term in searchable:
            hits += 1.5 if len(term) >= 6 else 1.0
    return min(1.0, hits / max(len(query_terms), 1))


def _section_match_score(query_terms: set[str], doc: Document) -> float:
    section_text = normalize_text(
        f"{doc.metadata.get('section_title', '')} {doc.metadata.get('section_type', '')}"
    ).lower()
    if not section_text or not query_terms:
        return 0.0
    return 1.0 if any(term in section_text for term in query_terms) else 0.0


def _paper_score(doc: Document) -> float:
    try:
        return float(doc.metadata.get("paper_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize_score(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def _normalize_paper_name(value: str) -> str:
    path_name = Path(value).name if value else ""
    stem = Path(path_name).stem if path_name else value
    return normalize_text(stem or value).lower()


def _with_gated_debug_metadata(
    selected: list[tuple[Document, float]],
    query_type: QueryType,
    paper_candidates: list[PaperCandidate],
) -> list[tuple[Document, float]]:
    candidates_json = json.dumps(
        [
            {
                "source": candidate.source,
                "score": round(candidate.score, 6),
                "signals": {key: round(value, 6) for key, value in candidate.signals.items()},
                "ranks": candidate.ranks,
            }
            for candidate in paper_candidates
        ],
        ensure_ascii=False,
    )
    decorated: list[tuple[Document, float]] = []
    for doc, score in selected:
        metadata = dict(doc.metadata)
        metadata["query_type"] = query_type
        metadata["candidate_papers_json"] = candidates_json
        decorated.append((Document(page_content=doc.page_content, metadata=metadata), score))
    return decorated


def _prepend_limited_community_docs(
    selected: list[tuple[Document, float]],
    query: str,
    community_index_dir: Path,
    community_k: int,
    k: int,
) -> list[tuple[Document, float]]:
    community_limit = min(2, max(1, k // 4))
    community_docs = [
        (_with_community_metadata(doc, rank), 1.0 / (PAPER_RRF_K + rank))
        for rank, (doc, _score) in enumerate(
            _community_search(query, community_index_dir, community_limit or community_k),
            start=1,
        )
    ][:community_limit]
    if not community_docs:
        return selected[:k]
    chunk_limit = max(k - len(community_docs), 0)
    return (selected[:chunk_limit] + community_docs)[:k]


def _with_community_metadata(doc: Document, rank: int) -> Document:
    metadata = dict(doc.metadata)
    metadata["document_type"] = "community"
    metadata["community_rank"] = str(rank)
    metadata["chunk_id"] = str(metadata.get("chunk_id", "-"))
    metadata["page"] = str(metadata.get("page", "-"))
    return Document(page_content=doc.page_content, metadata=metadata)


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)
