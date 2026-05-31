from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document


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

