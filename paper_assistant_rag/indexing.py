from __future__ import annotations

from pathlib import Path

import typer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from paper_assistant_rag.documents import load_pdf_pages, split_pages
from paper_assistant_rag.models import build_embeddings
from paper_assistant_rag.paths import DEFAULT_EMBED_BATCH_SIZE
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, create_progress


def index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


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

    console.print("[bold]开始构建论文索引[/bold]")
    # 建库流程：读 PDF -> 切 chunk -> 算 embedding -> 写入 FAISS。
    pages = load_pdf_pages(paper_dir)
    with console.status("[cyan]正在切分论文文本...[/cyan]", spinner="dots"):
        chunks = split_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    console.print(f"Loaded {len(pages)} pages, split into {len(chunks)} chunks.")

    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore_with_progress(chunks, embeddings)
    with console.status("[cyan]正在保存 FAISS 索引...[/cyan]", spinner="dots"):
        vectorstore.save_local(str(index_dir))
    console.print(f"Saved FAISS index to [cyan]{index_dir}[/cyan]")


def build_vectorstore_with_progress(chunks: list[Document], embeddings) -> FAISS:
    # LangChain 的 FAISS.from_documents 会一次性生成所有 embedding，看不到进度。
    # 这里手动分批生成 embedding，就能显示“已经处理了多少个 chunk”。
    text_embeddings: list[tuple[str, list[float]]] = []
    metadatas: list[dict] = []

    with create_progress() as progress:
        task = progress.add_task("生成 embedding", total=len(chunks))
        for start in range(0, len(chunks), DEFAULT_EMBED_BATCH_SIZE):
            batch = chunks[start : start + DEFAULT_EMBED_BATCH_SIZE]
            texts = [chunk.page_content for chunk in batch]
            batch_embeddings = embeddings.embed_documents(texts)
            text_embeddings.extend(zip(texts, batch_embeddings))
            metadatas.extend(chunk.metadata for chunk in batch)
            progress.advance(task, advance=len(batch))

    with console.status("[cyan]正在写入 FAISS 向量库...[/cyan]", spinner="dots"):
        return FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)


def load_index(index_dir: Path, settings: Settings) -> FAISS:
    if not index_exists(index_dir):
        raise typer.BadParameter(
            f"Index not found at {index_dir}. Run `uv run python main.py index` first."
        )
    return FAISS.load_local(
        str(index_dir),
        build_embeddings(settings),
        # FAISS 的本地索引包含 pickle 文件；这里只加载自己生成的本地文件。
        allow_dangerous_deserialization=True,
    )

