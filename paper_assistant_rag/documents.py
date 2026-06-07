"""PDF loading, metadata normalization, and document chunk splitting."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import typer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_assistant_rag.ui import console, create_progress

SECTION_TYPE_TERMS = {
    "abstract": ["abstract"],
    "related": ["related work", "background", "literature review"],
    "method": [
        "method",
        "methods",
        "methodology",
        "proposed method",
        "proposed framework",
        "framework",
        "algorithm",
        "model",
        "architecture",
        "approach",
    ],
    "experiment": ["experiment", "experiments", "evaluation", "results", "ablation", "dataset"],
    "conclusion": ["conclusion", "discussion", "limitations", "future work"],
    "references": ["references", "bibliography"],
}


def find_pdf_paths(paper_dir: Path) -> list[Path]:
    pdf_paths = sorted(paper_dir.glob("*.pdf"))
    if not pdf_paths:
        raise typer.BadParameter(f"No PDF files found in {paper_dir}")
    return pdf_paths


def load_pdf_pages(paper_dir: Path) -> list[Document]:
    return load_pdf_files(find_pdf_paths(paper_dir))


def load_pdf_files(pdf_paths: list[Path]) -> list[Document]:
    pages: list[Document] = []
    with create_progress() as progress:
        task = progress.add_task("Reading PDFs", total=len(pdf_paths))
        for pdf_path in pdf_paths:
            progress.update(task, description=f"Reading PDF: {pdf_path.name}")
            pages.extend(load_one_pdf(pdf_path))
            progress.advance(task)
    return pages


def load_one_pdf(pdf_path: Path) -> list[Document]:
    try:
        loaded_pages = PyPDFLoader(str(pdf_path)).load()
        pages: list[Document] = []
        for page in loaded_pages:
            page_number = int(page.metadata.get("page", 0)) + 1
            page.metadata.update(pdf_metadata(pdf_path, page_number))
            pages.append(page)
        return pages
    except Exception as exc:
        console.print(
            f"[yellow]PyPDF failed for {pdf_path.name}; falling back to PyMuPDF: {exc}[/yellow]"
        )
        return load_one_pdf_with_pymupdf(pdf_path)


def load_one_pdf_with_pymupdf(pdf_path: Path) -> list[Document]:
    import pymupdf

    pages: list[Document] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue
            pages.append(
                Document(
                    page_content=text,
                    metadata=pdf_metadata(pdf_path, page_index),
                )
            )
    return pages


def pdf_metadata(pdf_path: Path, page_number: int) -> dict[str, str | int]:
    return {
        "source": pdf_path.name,
        "source_path": str(pdf_path),
        "paper_id": pdf_path.stem,
        "page": page_number,
    }


def split_pages(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["stable_chunk_id"] = stable_chunk_id(chunk)
        chunk.metadata["document_type"] = "chunk"
    annotate_chunk_sections(chunks)
    return chunks


def stable_chunk_id(chunk: Document) -> str:
    metadata = chunk.metadata
    source = str(metadata.get("source", "unknown"))
    page = str(metadata.get("page", "?"))
    start_index = str(metadata.get("start_index", "?"))
    text_hash = hashlib.sha1(normalize_for_id(chunk.page_content).encode("utf-8")).hexdigest()[:12]
    raw_id = f"{source}|p{page}|s{start_index}|{text_hash}"
    return hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]


def normalize_for_id(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def annotate_chunk_sections(chunks: list[Document]) -> None:
    current_section_by_source: dict[str, str] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        detected_section = detect_section_title(chunk.page_content)
        if detected_section:
            current_section_by_source[source] = detected_section

        section_title = current_section_by_source.get(source, "")
        section_type = classify_section_type(
            section_title=section_title,
            text=chunk.page_content,
            page_number=_int_metadata(chunk.metadata.get("page")),
            chunk_id=_int_metadata(chunk.metadata.get("chunk_id")),
        )
        chunk.metadata["section_title"] = section_title
        chunk.metadata["section_type"] = section_type


def detect_section_title(text: str) -> str:
    for raw_line in text.splitlines()[:20]:
        line = re.sub(r"\s+", " ", raw_line).strip(" .\t")
        if not line or len(line) > 120:
            continue
        normalized = line.lower()
        normalized = re.sub(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\).:-]?\s+", "", normalized)
        for terms in SECTION_TYPE_TERMS.values():
            for term in terms:
                if normalized == term or normalized.startswith(f"{term} "):
                    return line
    return ""


def classify_section_type(
    section_title: str,
    text: str,
    page_number: int | None,
    chunk_id: int | None,
) -> str:
    searchable = f"{section_title} {text[:700]}".lower()
    for section_type, terms in SECTION_TYPE_TERMS.items():
        if any(term in searchable for term in terms):
            return section_type
    if page_number == 1 and chunk_id is not None and chunk_id <= 2:
        return "abstract"
    return "body"


def _int_metadata(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
