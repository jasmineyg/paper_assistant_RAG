"""Retrieval post-processing, reference filtering, context formatting, and answer cleanup."""

from __future__ import annotations

import re
import unicodedata
from math import log1p

from langchain_core.documents import Document

RRF_K = 60
VECTOR_RRF_WEIGHT = 1.0
KEYWORD_RRF_WEIGHT = 2.0

QUERY_STOPWORDS = {
    "about",
    "algorithm",
    "an",
    "and",
    "approach",
    "describe",
    "explain",
    "flow",
    "for",
    "from",
    "how",
    "in",
    "method",
    "methods",
    "of",
    "on",
    "paper",
    "please",
    "process",
    "summarize",
    "summary",
    "the",
    "to",
    "what",
    "with",
    "workflow",
}

SUMMARY_INTENT_TERMS = {
    "abstract": 0.8,
    "contribution": 0.8,
    "framework": 1.2,
    "method": 1.4,
    "overview": 1.6,
    "propose": 0.8,
    "proposed": 0.8,
}

WORKFLOW_INTENT_TERMS = {
    "finally": 0.7,
    "first": 0.7,
    "next": 0.7,
    "overview": 1.4,
    "process": 1.6,
    "procedure": 1.4,
    "step": 1.0,
    "steps": 1.0,
    "then": 0.7,
    "workflow": 1.6,
}


def make_context(results: list[tuple[Document, float]], max_chars_per_source: int) -> tuple[str, list[dict[str, str]]]:
    # 同一批检索结果要整理成两份：
    # 1. context_blocks 给 LLM 阅读；
    # 2. source_rows 给用户查看来源。
    context_blocks: list[str] = []
    source_rows: list[dict[str, str]] = []

    for source_number, (doc, score) in enumerate(results, start=1):
        source_id = f"S{source_number}"
        metadata = doc.metadata
        snippet = normalize_text(doc.page_content)[:max_chars_per_source]
        row = {
            "id": source_id,
            "source": str(metadata.get("source", "unknown")),
            "page": str(metadata.get("page", "?")),
            "chunk": str(metadata.get("chunk_id", "?")),
            "score": f"{score:.4f}",
            "snippet": snippet,
        }
        source_rows.append(row)
        context_blocks.append(
            f"[{source_id}] {row['source']} | page {row['page']} | chunk {row['chunk']}\n{snippet}"
        )

    return "\n\n".join(context_blocks), source_rows


def hybrid_search_with_score(vectorstore, query: str, k: int) -> list[tuple[Document, float]]:
    vector_results = vectorstore.similarity_search_with_score(query, k=k)
    keyword_results = keyword_search_with_score(vectorstore, query, k=k)

    merged: dict[tuple[str, str, str, str], dict] = {}

    for rank, (doc, score) in enumerate(vector_results, start=1):
        entry = merged.setdefault(_document_key(doc), {"doc": doc})
        entry["vector_rank"] = rank
        entry["vector_score"] = float(score)

    for rank, (doc, score) in enumerate(keyword_results, start=1):
        entry = merged.setdefault(_document_key(doc), {"doc": doc})
        entry["keyword_rank"] = rank
        entry["keyword_score"] = score

    ranked = sorted(
        merged.values(),
        key=lambda entry: (
            -_hybrid_rrf_score(entry),
            entry.get("vector_rank", 10**9),
            entry.get("keyword_rank", 10**9),
        ),
    )
    return [(entry["doc"], _hybrid_rrf_score(entry)) for entry in ranked[:k]]


def keyword_search_with_score(vectorstore, query: str, k: int) -> list[tuple[Document, float]]:
    anchors, weighted_terms = _keyword_terms(query)
    if not anchors:
        return []

    results: list[tuple[Document, float]] = []
    for doc in _iter_vectorstore_documents(vectorstore):
        searchable = _searchable_text(doc)
        if not any(_term_count(searchable, anchor) for anchor in anchors):
            continue

        score = 0.0
        for term, weight in weighted_terms:
            count = _term_count(searchable, term)
            if not count:
                continue
            effective_count = 1 if term in anchors else min(count, 3)
            score += weight * (1.0 + log1p(effective_count))
            first_pos = searchable.find(term)
            if first_pos < 120:
                score += weight
            elif first_pos < 500:
                score += weight * 0.5

        if score:
            results.append((doc, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:k]


def select_retrieval_results(
    results: list[tuple[Document, float]],
    k: int,
    include_references: bool,
) -> list[tuple[Document, float]]:
    if include_references:
        return results[:k]

    # 普通问答里，参考文献列表容易干扰模型，所以默认把它们排到后面。
    # 做 Related Work 检索时，可以用 --include-references 关闭这个过滤。
    selected: list[tuple[Document, float]] = []
    reference_like: list[tuple[Document, float]] = []

    for doc, score in results:
        if is_reference_like(doc.page_content):
            reference_like.append((doc, score))
        else:
            selected.append((doc, score))
        if len(selected) >= k:
            return selected[:k]

    return (selected + reference_like)[:k]


def is_reference_like(text: str) -> bool:
    # 这是一个轻量规则：如果 chunk 里年份、会议/期刊、pp. 等引用特征很多，
    # 就把它当作 References/参考文献片段处理。
    normalized = normalize_text(text).lower()
    head = normalized[:500]
    citation_years = len(re.findall(r"\b(?:19|20)\d{2}\b", normalized))
    reference_markers = [
        "proceedings",
        "transactions",
        "conference",
        "journal",
        "arxiv",
        "acm",
        "ieee",
        "springer",
        " proc.",
        " conf.",
        " pp.",
        " et al.",
        " available:",
        "press.",
        "trans.",
    ]
    marker_hits = sum(1 for term in reference_markers if term in normalized)
    if "references" in head:
        return True
    return citation_years >= 3 and marker_hits >= 2


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _hybrid_rrf_score(entry: dict) -> float:
    score = 0.0
    if "vector_rank" in entry:
        score += VECTOR_RRF_WEIGHT / (RRF_K + entry["vector_rank"])
    if "keyword_rank" in entry:
        score += KEYWORD_RRF_WEIGHT / (RRF_K + entry["keyword_rank"])
    return score


def _keyword_terms(query: str) -> tuple[list[str], list[tuple[str, float]]]:
    normalized_query = normalize_text(query).lower()
    anchors = _query_anchor_terms(normalized_query)
    if not anchors:
        return [], []

    weighted_terms: dict[str, float] = {term: 8.0 for term in anchors}

    if _has_summary_intent(normalized_query):
        _merge_weighted_terms(weighted_terms, SUMMARY_INTENT_TERMS)
    if _has_workflow_intent(normalized_query):
        _merge_weighted_terms(weighted_terms, WORKFLOW_INTENT_TERMS)

    return anchors, list(weighted_terms.items())


def _query_anchor_terms(normalized_query: str) -> list[str]:
    terms: list[str] = []
    for raw_term in re.findall(r"[a-z][a-z0-9_-]*", normalized_query):
        normalized_term = raw_term.strip("-_")
        if len(normalized_term) < 3 or normalized_term in QUERY_STOPWORDS:
            continue
        terms.extend(_term_variants(normalized_term))
    return _dedupe(terms)


def _term_variants(term: str) -> list[str]:
    variants = [term]
    compact = term.replace("-", "").replace("_", "")
    if compact != term:
        variants.append(compact)
    if compact == "migraph":
        variants.append("mi-graph")
    return variants


def _merge_weighted_terms(target: dict[str, float], source: dict[str, float]) -> None:
    for term, weight in source.items():
        target[term] = max(target.get(term, 0.0), weight)


def _has_summary_intent(normalized_query: str) -> bool:
    return any(term in normalized_query for term in ["总结", "概述", "summary", "summarize", "overview"])


def _has_workflow_intent(normalized_query: str) -> bool:
    return any(
        term in normalized_query
        for term in ["流程", "步骤", "过程", "workflow", "process", "procedure", "step"]
    )


def _iter_vectorstore_documents(vectorstore) -> list[Document]:
    docstore_dict = getattr(vectorstore.docstore, "_dict", {})
    if not isinstance(docstore_dict, dict):
        return []
    return [doc for doc in docstore_dict.values() if isinstance(doc, Document)]


def _searchable_text(doc: Document) -> str:
    metadata = doc.metadata
    metadata_text = " ".join(
        str(metadata.get(key, "")) for key in ("source", "source_path", "page", "chunk_id")
    )
    return f"{normalize_text(metadata_text)} {normalize_text(doc.page_content)}".lower()


def _term_count(text: str, term: str) -> int:
    return text.count(term)


def _document_key(doc: Document) -> tuple[str, str, str, str]:
    metadata = doc.metadata
    source = str(metadata.get("source_path") or metadata.get("source") or "")
    page = str(metadata.get("page") or "")
    chunk_id = str(metadata.get("chunk_id") or "")
    start_index = str(metadata.get("start_index") or "")
    return source, page, chunk_id, start_index


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def clean_model_output(text: str) -> str:
    # 本地推理模型有时会输出 <think>...</think>，这里删掉推理过程，只保留回答。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def ensure_answer_citations(answer: str, source_rows: list[dict[str, str]]) -> str:
    # 小模型有时不严格按提示写 [S1] 引用；这里做一个简单兜底。
    if re.search(r"\[S\d+\]", answer) or not source_rows:
        return answer
    source_ids = " ".join(f"[{row['id']}]" for row in source_rows)
    return f"{answer}\n\n主要依据：{source_ids}"
