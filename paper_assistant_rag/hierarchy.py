"""Paper/section document builders for hierarchical retrieval indexes."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from langchain_core.documents import Document


MAX_PAPER_TEXT_CHARS = 5000
MAX_SECTION_TEXT_CHARS = 4500


def build_paper_documents(chunks: list[Document]) -> list[Document]:
    docs_by_source = _group_by_source(chunks)
    paper_docs: list[Document] = []
    for source, docs in sorted(docs_by_source.items()):
        ordered = _sort_chunks(docs)
        first = ordered[0]
        title = _extract_title(ordered, source)
        abstract = _joined_text(
            [
                doc
                for doc in ordered
                if str(doc.metadata.get("section_type", "")).lower() == "abstract"
                or _int_metadata(doc.metadata.get("page")) == 1
            ][:4],
            MAX_PAPER_TEXT_CHARS // 2,
        )
        section_lines = _section_inventory(ordered)
        opening = _joined_text(ordered[:6], MAX_PAPER_TEXT_CHARS // 2)
        page_count = len({str(doc.metadata.get("page", "?")) for doc in ordered})

        content = "\n".join(
            part
            for part in [
                f"Paper: {source}",
                f"Title: {title}",
                f"Pages represented: {page_count}",
                "Sections:",
                section_lines,
                "Abstract or opening content:",
                abstract or opening,
            ]
            if part
        )
        paper_docs.append(
            Document(
                page_content=content[:MAX_PAPER_TEXT_CHARS],
                metadata={
                    "document_type": "paper",
                    "source": source,
                    "source_path": str(first.metadata.get("source_path", "")),
                    "paper_id": str(first.metadata.get("paper_id") or _paper_id_from_source(source)),
                    "paper_title": title,
                    "page": "paper",
                    "chunk_id": "paper",
                },
            )
        )
    return paper_docs


def build_section_documents(chunks: list[Document]) -> list[Document]:
    section_docs: list[Document] = []
    for source, docs in sorted(_group_by_source(chunks).items()):
        for group in _iter_section_groups(_sort_chunks(docs)):
            first = group[0]
            source_path = str(first.metadata.get("source_path", ""))
            paper_id = str(first.metadata.get("paper_id") or _paper_id_from_source(source))
            section_title = str(first.metadata.get("section_title") or "").strip()
            section_type = str(first.metadata.get("section_type") or "body").strip() or "body"
            chunk_ids = [_int_metadata(doc.metadata.get("chunk_id")) for doc in group]
            chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id is not None]
            pages = [str(doc.metadata.get("page", "?")) for doc in group]
            page_start, page_end = pages[0], pages[-1]
            chunk_start = min(chunk_ids) if chunk_ids else ""
            chunk_end = max(chunk_ids) if chunk_ids else ""
            section_id = _section_id(source, section_title, section_type, page_start, chunk_start)
            content = "\n".join(
                [
                    f"Paper: {source}",
                    f"Section: {section_title or section_type}",
                    f"Section type: {section_type}",
                    f"Pages: {page_start}-{page_end}",
                    f"Chunks: {chunk_start}-{chunk_end}",
                    _joined_text(group, MAX_SECTION_TEXT_CHARS),
                ]
            )
            section_docs.append(
                Document(
                    page_content=content[:MAX_SECTION_TEXT_CHARS],
                    metadata={
                        "document_type": "section",
                        "source": source,
                        "source_path": source_path,
                        "paper_id": paper_id,
                        "section_id": section_id,
                        "section_title": section_title,
                        "section_type": section_type,
                        "page": page_start,
                        "page_end": page_end,
                        "chunk_id": f"section:{section_id}",
                        "chunk_start": str(chunk_start),
                        "chunk_end": str(chunk_end),
                    },
                )
            )
    return section_docs


def _group_by_source(chunks: list[Document]) -> dict[str, list[Document]]:
    docs_by_source: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        docs_by_source[str(chunk.metadata.get("source", "unknown"))].append(chunk)
    return docs_by_source


def _sort_chunks(chunks: list[Document]) -> list[Document]:
    return sorted(
        chunks,
        key=lambda doc: (
            _int_metadata(doc.metadata.get("page")) or 0,
            _int_metadata(doc.metadata.get("chunk_id")) or 0,
            _int_metadata(doc.metadata.get("start_index")) or 0,
        ),
    )


def _iter_section_groups(chunks: list[Document]):
    group: list[Document] = []
    current_key: tuple[str, str] | None = None
    for chunk in chunks:
        section_title = str(chunk.metadata.get("section_title") or "").strip()
        section_type = str(chunk.metadata.get("section_type") or "body").strip() or "body"
        key = (section_title, section_type)
        if group and key != current_key:
            yield group
            group = []
        current_key = key
        group.append(chunk)
    if group:
        yield group


def _extract_title(chunks: list[Document], source: str) -> str:
    for chunk in chunks[:3]:
        for raw_line in chunk.page_content.splitlines()[:20]:
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not _looks_like_title(line):
                continue
            return line
    return _paper_id_from_source(source)


def _looks_like_title(line: str) -> bool:
    if len(line) < 8 or len(line) > 180:
        return False
    lowered = line.lower()
    blocked = ["abstract", "introduction", "copyright", "arxiv", "proceedings", "keywords"]
    if any(term in lowered for term in blocked):
        return False
    if re.fullmatch(r"[\d\s.:-]+", line):
        return False
    return bool(re.search(r"[A-Za-z]", line))


def _section_inventory(chunks: list[Document]) -> str:
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for chunk in chunks:
        section_title = str(chunk.metadata.get("section_title") or "").strip()
        section_type = str(chunk.metadata.get("section_type") or "body").strip() or "body"
        key = (section_title, section_type)
        if key in seen:
            continue
        seen.add(key)
        if section_title:
            lines.append(f"- {section_title} ({section_type})")
        else:
            lines.append(f"- {section_type}")
        if len(lines) >= 20:
            break
    return "\n".join(lines)


def _joined_text(docs: list[Document], max_chars: int) -> str:
    text = "\n".join(_normalize_text(doc.page_content) for doc in docs)
    return text[:max_chars]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _section_id(
    source: str,
    section_title: str,
    section_type: str,
    page_start: str,
    chunk_start: int | str,
) -> str:
    raw = f"{source}|{section_title}|{section_type}|{page_start}|{chunk_start}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _paper_id_from_source(source: str) -> str:
    return re.sub(r"\.pdf$", "", source, flags=re.IGNORECASE)


def _int_metadata(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
