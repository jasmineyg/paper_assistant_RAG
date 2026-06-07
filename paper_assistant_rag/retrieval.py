"""Retrieval post-processing, reference filtering, context formatting, and answer cleanup."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
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

SECTION_TYPE_WEIGHTS = {
    "definition": {
        "abstract": 0.028,
        "method": 0.018,
        "conclusion": 0.006,
        "related": -0.006,
        "references": -0.035,
    },
    "workflow": {
        "method": 0.032,
        "abstract": 0.012,
        "conclusion": 0.006,
        "related": -0.006,
        "references": -0.035,
    },
    "summary": {
        "abstract": 0.024,
        "method": 0.016,
        "conclusion": 0.014,
        "experiment": 0.006,
        "references": -0.035,
    },
    "comparison": {
        "abstract": 0.016,
        "method": 0.018,
        "conclusion": 0.010,
        "experiment": 0.006,
        "references": -0.035,
    },
    "citation": {
        "references": 0.022,
        "related": 0.014,
        "abstract": 0.006,
    },
    "specific": {
        "abstract": 0.006,
        "method": 0.006,
        "references": -0.030,
    },
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


def hierarchical_search_with_score(
    vectorstore,
    query: str,
    k: int,
    include_references: bool,
) -> list[tuple[Document, float]]:
    """Retrieve chunks with paper/section-aware reranking.

    This keeps the existing vector + keyword hybrid retrieval as the first-stage
    candidate generator, then adds a lightweight ArchRAG-lite layer:
    paper/source scoring, inferred section type scoring, adjacent chunk expansion,
    and source balancing for broad comparison/list questions.
    """
    raw_k = max(k * 6, k + 20)
    raw_results = hybrid_search_with_score(vectorstore, query, k=raw_k)
    return select_retrieval_results(
        raw_results,
        k=k,
        include_references=include_references,
        query=query,
        vectorstore=vectorstore,
    )


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
    query: str = "",
    vectorstore=None,
) -> list[tuple[Document, float]]:
    if query and vectorstore is not None:
        return _select_hierarchical_results(
            results=results,
            query=query,
            k=k,
            include_references=include_references,
            vectorstore=vectorstore,
        )

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


def _select_hierarchical_results(
    results: list[tuple[Document, float]],
    query: str,
    k: int,
    include_references: bool,
    vectorstore,
) -> list[tuple[Document, float]]:
    intent = infer_query_intent(query)
    documents = _iter_vectorstore_documents(vectorstore)
    source_scores = _paper_source_scores(
        documents=documents,
        initial_results=results,
        query=query,
    )
    expanded_results = _expand_with_adjacent_chunks(
        vectorstore=vectorstore,
        results=results,
        intent=intent,
        include_references=include_references,
    )

    entries = []
    for rank, (doc, base_score) in enumerate(expanded_results, start=1):
        section_type = infer_section_type(doc)
        metadata = dict(doc.metadata)
        metadata["section_type"] = section_type
        reranked_doc = Document(page_content=doc.page_content, metadata=metadata)
        score = _hierarchical_score(
            doc=reranked_doc,
            query=query,
            intent=intent,
            base_score=base_score,
            source_scores=source_scores,
            rank=rank,
            include_references=include_references,
        )
        entries.append((reranked_doc, score))

    entries.sort(key=lambda item: item[1], reverse=True)
    entries = _dedupe_scored_documents(entries)
    if _needs_source_coverage(query, intent):
        return _balanced_source_select(entries, k=k)
    return entries[:k]


def infer_query_intent(query: str) -> str:
    normalized = normalize_text(query).lower()
    if any(term in normalized for term in ["引用", "参考文献", "related work", "references", "cite", "cited"]):
        return "citation"
    if _has_workflow_intent(normalized):
        return "workflow"
    if any(
        term in normalized
        for term in [
            "比较",
            "对比",
            "区别",
            "差别",
            "优缺点",
            "哪些",
            "共同",
            "路线",
            "vs",
            "compare",
            "comparison",
            "different",
        ]
    ):
        return "comparison"
    if any(
        term in normalized
        for term in ["是哪篇", "指的是", "是什么", "定义", "what is", "which paper", "stands for"]
    ):
        return "definition"
    if _has_summary_intent(normalized) or any(
        term in normalized for term in ["贡献", "局限", "框架", "总结", "概述", "contribution", "limitation"]
    ):
        return "summary"
    return "specific"


def infer_section_type(doc: Document) -> str:
    metadata = doc.metadata
    explicit = str(metadata.get("section_type", "")).strip().lower()
    if explicit:
        return explicit

    text = normalize_text(doc.page_content).lower()
    head = text[:700]
    page = _int_metadata(metadata.get("page"))
    chunk_id = _int_metadata(metadata.get("chunk_id"))

    if is_reference_like(doc.page_content):
        return "references"
    if re.search(r"\babstract\b", head) or (page == 1 and chunk_id is not None and chunk_id <= 2):
        return "abstract"
    if re.search(r"\b(related work|background|literature review)\b", head):
        return "related"
    if re.search(r"\b(conclusion|future work|limitations?|discussion)\b", head):
        return "conclusion"
    if re.search(r"\b(experiments?|evaluation|results?|ablation|datasets?)\b", head):
        return "experiment"
    if re.search(
        r"\b(proposed|methodology|methods?|framework|algorithm|model|architecture|approach|implementation)\b",
        head,
    ):
        return "method"
    return "body"


def _hierarchical_score(
    doc: Document,
    query: str,
    intent: str,
    base_score: float,
    source_scores: dict[str, float],
    rank: int,
    include_references: bool,
) -> float:
    metadata = doc.metadata
    source = str(metadata.get("source", ""))
    section_type = str(metadata.get("section_type", "body"))
    score = float(base_score)

    # The first-stage RRF score is small (~0.03), so these boosts are
    # intentionally modest and capped.
    score += min(source_scores.get(source, 0.0), 1.0) * 0.030
    score += SECTION_TYPE_WEIGHTS.get(intent, {}).get(section_type, 0.0)
    score += _raw_term_hit_boost(query, doc)
    score += 0.004 / max(rank, 1)

    if section_type == "references" and not include_references:
        score -= 0.035
    return score


def _paper_source_scores(
    documents: list[Document],
    initial_results: list[tuple[Document, float]],
    query: str,
) -> dict[str, float]:
    anchors, _weighted_terms = _keyword_terms(query)
    raw_terms = _raw_query_terms(query)
    source_docs: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        source_docs[str(doc.metadata.get("source", "unknown"))].append(doc)

    scores: dict[str, float] = defaultdict(float)
    for rank, (doc, _score) in enumerate(initial_results, start=1):
        source = str(doc.metadata.get("source", "unknown"))
        scores[source] += 1.0 / (rank + 2)

    for source, docs in source_docs.items():
        ordered = sorted(docs, key=lambda doc: _int_metadata(doc.metadata.get("chunk_id")) or 0)
        early_text = normalize_text(" ".join(doc.page_content for doc in ordered[:6])).lower()
        source_text = normalize_text(source).lower()
        score = 0.0
        for term in anchors:
            if term in source_text:
                score += 1.2
            if term in early_text:
                score += min(0.6, 0.18 * _term_count(early_text, term))
        for term in raw_terms:
            if term in source:
                score += 1.0
            if term in " ".join(doc.page_content for doc in ordered[:4]):
                score += 0.5
        scores[source] += score

    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {source: score / max_score for source, score in scores.items()}


def _expand_with_adjacent_chunks(
    vectorstore,
    results: list[tuple[Document, float]],
    intent: str,
    include_references: bool,
) -> list[tuple[Document, float]]:
    if intent not in {"summary", "workflow", "comparison", "definition"}:
        return results

    by_source_chunk: dict[tuple[str, int], Document] = {}
    for doc in _iter_vectorstore_documents(vectorstore):
        source = str(doc.metadata.get("source", "unknown"))
        chunk_id = _int_metadata(doc.metadata.get("chunk_id"))
        if chunk_id is None:
            continue
        by_source_chunk[(source, chunk_id)] = doc

    expanded: list[tuple[Document, float]] = list(results)
    seen_keys = {_document_key(doc) for doc, _score in expanded}
    offsets = (-2, -1, 1, 2) if intent == "workflow" else (-1, 1)
    for doc, score in results[:12]:
        source = str(doc.metadata.get("source", "unknown"))
        chunk_id = _int_metadata(doc.metadata.get("chunk_id"))
        if chunk_id is None:
            continue
        for offset in offsets:
            neighbor = by_source_chunk.get((source, chunk_id + offset))
            if neighbor is None:
                continue
            if not include_references and is_reference_like(neighbor.page_content):
                continue
            key = _document_key(neighbor)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            expanded.append((neighbor, score * 0.72))
    return expanded


def _balanced_source_select(entries: list[tuple[Document, float]], k: int) -> list[tuple[Document, float]]:
    if k <= 3:
        return entries[:k]

    selected: list[tuple[Document, float]] = []
    selected_keys: set[tuple[str, str, str, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    candidate_sources = _top_sources(entries, max_sources=min(5, k))
    if len(candidate_sources) <= 1:
        return entries[:k]

    max_per_source_first_pass = max(1, k // len(candidate_sources))
    for doc, score in entries:
        source = str(doc.metadata.get("source", "unknown"))
        if source not in candidate_sources:
            continue
        if counts[source] >= max_per_source_first_pass:
            continue
        key = _document_key(doc)
        if key in selected_keys:
            continue
        selected.append((doc, score))
        selected_keys.add(key)
        counts[source] += 1
        if len(selected) >= k:
            return selected

    for doc, score in entries:
        key = _document_key(doc)
        if key in selected_keys:
            continue
        selected.append((doc, score))
        selected_keys.add(key)
        if len(selected) >= k:
            return selected
    return selected


def _top_sources(entries: list[tuple[Document, float]], max_sources: int) -> set[str]:
    scores: dict[str, float] = defaultdict(float)
    for rank, (doc, score) in enumerate(entries[: max_sources * 6], start=1):
        source = str(doc.metadata.get("source", "unknown"))
        scores[source] += score + 0.01 / rank
    return {source for source, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_sources]}


def _needs_source_coverage(query: str, intent: str) -> bool:
    normalized = normalize_text(query).lower()
    return intent == "comparison" or any(
        term in normalized
        for term in ["哪些", "共同", "这几篇", "这些论文", "列表", "分别", "compare", "comparison", "which methods"]
    )


def _raw_term_hit_boost(query: str, doc: Document) -> float:
    raw_terms = _raw_query_terms(query)
    if not raw_terms:
        return 0.0
    text = doc.page_content
    source = str(doc.metadata.get("source", ""))
    boost = 0.0
    for term in raw_terms:
        if term in source:
            boost += 0.012
        if term in text:
            boost += 0.008
    return min(boost, 0.026)


def _raw_query_terms(query: str) -> list[str]:
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query):
        if term.lower() in QUERY_STOPWORDS:
            continue
        terms.append(term)
    return _dedupe(terms)


def _dedupe_scored_documents(items: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[tuple[Document, float]] = []
    for doc, score in items:
        key = _document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, score))
    return deduped


def _int_metadata(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
