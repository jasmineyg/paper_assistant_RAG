"""Community-augmented retrieval and lightweight adaptive evidence filtering."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from paper_assistant_rag.indexing import index_exists, load_index
from paper_assistant_rag.retrieval import is_reference_like, normalize_text
from paper_assistant_rag.graph_retrieval import retrieve_graph_chunks_with_score
from paper_assistant_rag.settings import Settings

ARCHRAG_RRF_K = 60
CHUNK_RRF_WEIGHT = 1.0
COMMUNITY_RRF_WEIGHT = 2.0


def retrieve_community_augmented_chunks_with_score(
    vectorstore,
    query: str,
    k: int,
    include_references: bool,
    graph_dir: Path,
    community_index_dir: Path,
    community_k: int,
    include_community_docs: bool,
) -> list[tuple[Document, float]]:
    """Retrieve chunks and community summaries, then fuse high-level and low-level evidence."""
    raw_chunk_k = max(k * 5, k + 30)
    chunk_results = retrieve_graph_chunks_with_score(
        vectorstore=vectorstore,
        query=query,
        k=raw_chunk_k,
        include_references=include_references,
        graph_dir=graph_dir,
    )
    community_results = _community_search(query, community_index_dir, community_k)
    if not community_results:
        return chunk_results[:k]

    doc_index = _build_doc_index(vectorstore)
    merged: dict[str, dict[str, Any]] = {}
    for rank, (doc, _score) in enumerate(chunk_results, start=1):
        chunk_key = _doc_chunk_key(doc)
        if not chunk_key:
            continue
        entry = merged.setdefault(chunk_key, {"doc": doc})
        entry["chunk_rank"] = rank
        entry["chunk_rank_score"] = 1.0 / (ARCHRAG_RRF_K + rank)

    for community_rank, (community_doc, _community_score) in enumerate(community_results, start=1):
        for ref_rank, chunk_key in enumerate(_community_chunk_keys(community_doc), start=1):
            doc = doc_index.get(chunk_key)
            if doc is None:
                continue
            entry = merged.setdefault(chunk_key, {"doc": doc})
            best_rank = entry.get("community_rank", 10**9)
            if community_rank < best_rank:
                entry["community_rank"] = community_rank
                entry["community_id"] = str(community_doc.metadata.get("community_id", ""))
            entry["community_rank_score"] = entry.get("community_rank_score", 0.0) + (
                1.0 / (ARCHRAG_RRF_K + community_rank)
                * 1.0 / (1.0 + 0.04 * max(ref_rank - 1, 0))
            )

    ranked = sorted(
        merged.values(),
        key=lambda entry: (
            -_combined_score(entry),
            entry.get("community_rank", 10**9),
            entry.get("chunk_rank", 10**9),
        ),
    )

    selected_chunks: list[tuple[Document, float]] = []
    reference_like: list[tuple[Document, float]] = []
    for entry in ranked:
        doc = _with_archrag_metadata(entry["doc"], entry)
        score = _combined_score(entry)
        if include_references or not is_reference_like(doc.page_content):
            selected_chunks.append((doc, score))
        else:
            reference_like.append((doc, score))
        if len(selected_chunks) >= k:
            break

    selected = selected_chunks[:k]
    if include_community_docs:
        community_doc_limit = min(len(community_results), max(1, min(2, k // 4)))
        chunk_limit = max(k - community_doc_limit, 0)
        community_docs = [
            (_with_community_score_metadata(doc, rank), 1.0 / (ARCHRAG_RRF_K + rank))
            for rank, (doc, _score) in enumerate(community_results, start=1)
        ][:community_doc_limit]
        return selected[:chunk_limit] + community_docs
    return (selected + reference_like)[:k]


def adaptive_filter_documents(
    llm,
    query: str,
    docs: list[Document],
    max_documents: int,
    max_chars_per_doc: int = 900,
) -> list[Document]:
    """Ask the LLM to analyze retrieved evidence and keep the most useful items."""
    if not docs or max_documents <= 0:
        return docs

    evidence_blocks: list[str] = []
    source_lookup: dict[str, Document] = {}
    for index, doc in enumerate(docs, start=1):
        source = f"E{index}"
        source_lookup[source] = doc
        metadata = doc.metadata
        evidence_blocks.append(
            "\n".join(
                [
                    f"{source}:",
                    f"type: {metadata.get('document_type', 'chunk')}",
                    f"source: {metadata.get('source', 'unknown')}",
                    f"page: {metadata.get('page', '?')}",
                    f"chunk/community: {metadata.get('chunk_id', metadata.get('community_id', '?'))}",
                    f"text: {normalize_text(doc.page_content)[:max_chars_per_doc]}",
                ]
            )
        )

    prompt = f"""
You are an evidence analyzer for academic-paper RAG.
The retriever already found candidate evidence. Your job is not to answer.
Score each item by how useful it is for answering the question.

Question:
{query}

Evidence:
{chr(10).join(evidence_blocks)}

Return valid JSON only:
{{
  "items": [
    {{"source": "E1", "score": 0-100, "description": "why this evidence is useful or not"}}
  ]
}}

Scoring rules:
- 90-100: directly answers the question with concrete evidence.
- 70-89: strongly relevant and likely useful.
- 40-69: topic-related but incomplete.
- 0-39: weak, generic, wrong paper, or only tangentially related.
- Community summaries are useful for high-level framing, but concrete chunk evidence should score higher when it directly supports the answer.
""".strip()

    try:
        response = llm.invoke(prompt)
        parsed = _parse_json_object(_response_text(response))
        scores = _scores_from_response(parsed)
    except Exception:
        return _renumber_documents(docs[:max_documents])

    ranked: list[tuple[float, int, Document]] = []
    for index, doc in enumerate(docs, start=1):
        source = f"E{index}"
        score_row = scores.get(source, {})
        score = float(score_row.get("score", 0.0))
        metadata = dict(doc.metadata)
        metadata["adaptive_score"] = f"{score:.1f}"
        metadata["adaptive_description"] = str(score_row.get("description", ""))
        ranked.append((score, -index, Document(page_content=doc.page_content, metadata=metadata)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return _renumber_documents([doc for _score, _neg_index, doc in ranked[:max_documents]])


def _community_search(query: str, community_index_dir: Path, community_k: int) -> list[tuple[Document, float]]:
    if community_k <= 0 or not index_exists(community_index_dir):
        return []
    vectorstore = _load_community_index(str(community_index_dir.resolve()), _index_signature(community_index_dir))
    return vectorstore.similarity_search_with_score(
        query,
        k=community_k,
        fetch_k=max(community_k * 4, 20),
    )


@lru_cache(maxsize=4)
def _load_community_index(index_dir_text: str, signature: tuple[tuple[str, int, int], ...]):
    settings = Settings.from_env()
    return load_index(Path(index_dir_text), settings)


def _build_doc_index(vectorstore) -> dict[str, Document]:
    doc_index: dict[str, Document] = {}
    docstore_dict = getattr(vectorstore.docstore, "_dict", {})
    if not isinstance(docstore_dict, dict):
        return doc_index
    for doc in docstore_dict.values():
        if not isinstance(doc, Document):
            continue
        chunk_key = _doc_chunk_key(doc)
        if chunk_key:
            doc_index[chunk_key] = doc
    return doc_index


def _doc_chunk_key(doc: Document) -> str:
    return str(doc.metadata.get("stable_chunk_id", "")).strip()


def _community_chunk_keys(community_doc: Document) -> list[str]:
    raw_refs = str(community_doc.metadata.get("source_chunks_json", "[]"))
    try:
        refs = json.loads(raw_refs)
    except json.JSONDecodeError:
        return []
    keys: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        chunk_key = str(ref.get("stable_chunk_id", "") or ref.get("chunk_key", "")).strip()
        if chunk_key and chunk_key not in keys:
            keys.append(chunk_key)
    return keys


def _combined_score(entry: dict[str, Any]) -> float:
    return (
        CHUNK_RRF_WEIGHT * float(entry.get("chunk_rank_score", 0.0))
        + COMMUNITY_RRF_WEIGHT * float(entry.get("community_rank_score", 0.0))
    )


def _with_archrag_metadata(doc: Document, entry: dict[str, Any]) -> Document:
    metadata = dict(doc.metadata)
    metadata["document_type"] = str(metadata.get("document_type", "chunk"))
    if "chunk_rank" in entry:
        metadata["chunk_rank"] = str(entry["chunk_rank"])
        metadata["base_rank"] = str(entry["chunk_rank"])
    if "community_rank" in entry:
        metadata["community_rank"] = str(entry["community_rank"])
        metadata["community_id"] = str(entry.get("community_id", ""))
    return Document(page_content=doc.page_content, metadata=metadata)


def _with_community_score_metadata(doc: Document, rank: int) -> Document:
    metadata = dict(doc.metadata)
    metadata["document_type"] = "community"
    metadata["community_rank"] = str(rank)
    metadata["chunk_id"] = str(metadata.get("chunk_id", "-"))
    metadata["page"] = str(metadata.get("page", "-"))
    return Document(page_content=doc.page_content, metadata=metadata)


def _scores_from_response(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = parsed.get("items", [])
    if not isinstance(rows, list):
        return {}
    scores: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        if not source:
            continue
        try:
            score = max(0.0, min(100.0, float(row.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        scores[source] = {
            "score": score,
            "description": str(row.get("description", "")),
        }
    return scores


def _renumber_documents(docs: list[Document]) -> list[Document]:
    renumbered: list[Document] = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata)
        metadata["source_id"] = f"S{index}"
        renumbered.append(Document(page_content=doc.page_content, metadata=metadata))
    return renumbered


def _response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
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


def _index_signature(index_dir: Path) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for name in ["index.faiss", "index.pkl"]:
        path = index_dir / name
        if not path.exists():
            signature.append((name, 0, 0))
            continue
        stat = path.stat()
        signature.append((name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)
