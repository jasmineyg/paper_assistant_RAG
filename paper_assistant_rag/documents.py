"""PDF loading, metadata normalization, and document chunk splitting."""

from __future__ import annotations

from pathlib import Path

import typer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_assistant_rag.ui import console, create_progress


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
    return chunks
