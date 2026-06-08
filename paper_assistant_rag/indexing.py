"""FAISS index lifecycle: build, append, save, existence check, and load."""

from __future__ import annotations

from pathlib import Path

import typer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from paper_assistant_rag.documents import find_pdf_paths, load_pdf_files, load_pdf_pages, split_pages
from paper_assistant_rag.hierarchy import build_paper_documents, build_section_documents
from paper_assistant_rag.models import build_embeddings
from paper_assistant_rag.paths import DEFAULT_EMBED_BATCH_SIZE
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, create_progress


def index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def auxiliary_index_exists(index_dir: Path, name: str) -> bool:
    path = index_dir / name
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def build_index(
    paper_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
) -> None:
    settings = Settings.from_env()

    if index_exists(index_dir) and not force:
        console.print(f"Index already exists: [cyan]{index_dir}[/cyan]")
        console.print("Use --force to rebuild it.")
        return

    index_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Building paper index[/bold]")
    pages = load_pdf_pages(paper_dir)
    with console.status("[cyan]Splitting paper text...[/cyan]", spinner="dots"):
        chunks = split_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    console.print(f"Loaded {len(pages)} pages, split into {len(chunks)} chunks.")

    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore_with_progress(chunks, embeddings)
    with console.status("[cyan]Saving FAISS index...[/cyan]", spinner="dots"):
        vectorstore.save_local(str(index_dir))
    save_hierarchical_indexes(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
    console.print(f"Saved FAISS index to [cyan]{index_dir}[/cyan]")


def append_to_index(
    paper_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    skip_existing: bool,
) -> None:
    settings = Settings.from_env()
    embeddings = build_embeddings(settings)

    with console.status("[cyan]Loading existing FAISS index...[/cyan]", spinner="dots"):
        vectorstore = _load_index_with_embeddings(index_dir, embeddings)

    pdf_paths = find_pdf_paths(paper_dir)
    if skip_existing:
        indexed_sources = _existing_source_names(vectorstore)
        before_count = len(pdf_paths)
        pdf_paths = [pdf_path for pdf_path in pdf_paths if pdf_path.name not in indexed_sources]
        skipped_count = before_count - len(pdf_paths)
        if skipped_count:
            console.print(f"Skipped {skipped_count} PDF(s) already present in the index.")

    if not pdf_paths:
        console.print("No new PDF files to append.")
        return

    console.print("[bold]Appending papers to existing index[/bold]")
    for pdf_path in pdf_paths:
        console.print(f"- {pdf_path.name}")

    pages = load_pdf_files(pdf_paths)
    with console.status("[cyan]Splitting new paper text...[/cyan]", spinner="dots"):
        chunks = split_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        _renumber_chunks(chunks, start=_max_existing_chunk_id(vectorstore) + 1)
    if not chunks:
        console.print("No text chunks found in the new PDF files.")
        return

    chunk_counts = _chunk_counts_by_source(chunks)
    added_count = add_documents_with_progress(vectorstore, chunks, embeddings)
    with console.status("[cyan]Saving updated FAISS index...[/cyan]", spinner="dots"):
        vectorstore.save_local(str(index_dir))
    with console.status("[cyan]Refreshing paper/section indexes...[/cyan]", spinner="dots"):
        save_hierarchical_indexes(
            index_dir=index_dir,
            chunks=_iter_index_documents(vectorstore),
            embeddings=embeddings,
        )
    console.print("[bold]Added paper indexes[/bold]")
    for pdf_path in pdf_paths:
        chunk_count = chunk_counts.get(pdf_path.name, 0)
        if chunk_count:
            console.print(f"- {pdf_path.name} ({chunk_count} chunks)")
    console.print(f"Appended {added_count} chunks to [cyan]{index_dir}[/cyan]")


def refresh_hierarchical_indexes(index_dir: Path) -> None:
    settings = Settings.from_env()
    embeddings = build_embeddings(settings)
    with console.status("[cyan]Loading existing FAISS index...[/cyan]", spinner="dots"):
        vectorstore = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    chunks = _iter_index_documents(vectorstore)
    if not chunks:
        console.print("[yellow]No chunk documents found in the existing index.[/yellow]")
        return
    save_hierarchical_indexes(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
    console.print(f"Saved paper/section indexes under [cyan]{index_dir}[/cyan]")


def build_vectorstore_with_progress(chunks: list[Document], embeddings) -> FAISS:
    text_embeddings: list[tuple[str, list[float]]] = []
    metadatas: list[dict] = []

    with create_progress() as progress:
        task = progress.add_task("Generating embeddings", total=len(chunks))
        for start in range(0, len(chunks), DEFAULT_EMBED_BATCH_SIZE):
            batch = chunks[start : start + DEFAULT_EMBED_BATCH_SIZE]
            texts = [chunk.page_content for chunk in batch]
            batch_embeddings = embeddings.embed_documents(texts)
            text_embeddings.extend(zip(texts, batch_embeddings))
            metadatas.extend(chunk.metadata for chunk in batch)
            progress.advance(task, advance=len(batch))

    with console.status("[cyan]Writing FAISS vectorstore...[/cyan]", spinner="dots"):
        return FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)


def add_documents_with_progress(vectorstore: FAISS, chunks: list[Document], embeddings) -> int:
    added_count = 0
    with create_progress() as progress:
        task = progress.add_task("Embedding new chunks", total=len(chunks))
        for start in range(0, len(chunks), DEFAULT_EMBED_BATCH_SIZE):
            batch = chunks[start : start + DEFAULT_EMBED_BATCH_SIZE]
            texts = [chunk.page_content for chunk in batch]
            batch_embeddings = embeddings.embed_documents(texts)
            text_embeddings = list(zip(texts, batch_embeddings))
            metadatas = [chunk.metadata for chunk in batch]
            vectorstore.add_embeddings(text_embeddings, metadatas=metadatas)
            added_count += len(batch)
            progress.advance(task, advance=len(batch))
    return added_count


def load_index(index_dir: Path, settings: Settings) -> FAISS:
    return _load_index_with_embeddings(index_dir, build_embeddings(settings))


def _load_index_with_embeddings(index_dir: Path, embeddings) -> FAISS:
    if not index_exists(index_dir):
        raise typer.BadParameter(
            f"Index not found at {index_dir}. Run `uv run python main.py index` first."
        )
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    attach_hierarchical_indexes(vectorstore, index_dir=index_dir, embeddings=embeddings)
    return vectorstore


def save_hierarchical_indexes(index_dir: Path, chunks: list[Document], embeddings) -> None:
    paper_docs = build_paper_documents(chunks)
    section_docs = build_section_documents(chunks)
    if paper_docs:
        console.print(f"Building paper-level index ({len(paper_docs)} papers).")
        paper_index = build_vectorstore_with_progress(paper_docs, embeddings)
        paper_index.save_local(str(index_dir / "paper_index"))
    if section_docs:
        console.print(f"Building section-level index ({len(section_docs)} sections).")
        section_index = build_vectorstore_with_progress(section_docs, embeddings)
        section_index.save_local(str(index_dir / "section_index"))


def attach_hierarchical_indexes(vectorstore: FAISS, index_dir: Path, embeddings) -> None:
    if auxiliary_index_exists(index_dir, "paper_index"):
        vectorstore.paper_vectorstore = FAISS.load_local(
            str(index_dir / "paper_index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore.paper_vectorstore = None

    if auxiliary_index_exists(index_dir, "section_index"):
        vectorstore.section_vectorstore = FAISS.load_local(
            str(index_dir / "section_index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore.section_vectorstore = None


def _existing_source_names(vectorstore: FAISS) -> set[str]:
    sources: set[str] = set()
    for doc in _iter_index_documents(vectorstore):
        source = doc.metadata.get("source")
        source_path = doc.metadata.get("source_path")
        if source:
            sources.add(str(source))
        if source_path:
            sources.add(Path(str(source_path)).name)
    return sources


def _max_existing_chunk_id(vectorstore: FAISS) -> int:
    max_chunk_id = 0
    for doc in _iter_index_documents(vectorstore):
        try:
            max_chunk_id = max(max_chunk_id, int(doc.metadata.get("chunk_id", 0)))
        except (TypeError, ValueError):
            continue
    return max_chunk_id


def _chunk_counts_by_source(chunks: list[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _iter_index_documents(vectorstore: FAISS) -> list[Document]:
    docstore_dict = getattr(vectorstore.docstore, "_dict", {})
    if not isinstance(docstore_dict, dict):
        return []
    return [doc for doc in docstore_dict.values() if isinstance(doc, Document)]


def _renumber_chunks(chunks: list[Document], start: int) -> None:
    for chunk_id, chunk in enumerate(chunks, start=start):
        chunk.metadata["chunk_id"] = chunk_id
