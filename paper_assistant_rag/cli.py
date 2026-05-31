from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from paper_assistant_rag.indexing import build_index
from paper_assistant_rag.paths import DEFAULT_INDEX_DIR, DEFAULT_PAPER_DIR
from paper_assistant_rag.qa import ask_question
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console


app = typer.Typer(
    help="Minimal domain paper RAG assistant: index PDFs, ask questions, and show source snippets."
)


@app.command(name="index")
def index_command(
    paper_dir: Annotated[Path, typer.Option(help="Directory containing PDF papers.")] = DEFAULT_PAPER_DIR,
    index_dir: Annotated[Path, typer.Option(help="Directory used to save the FAISS index.")] = DEFAULT_INDEX_DIR,
    chunk_size: Annotated[int, typer.Option(help="Characters per chunk.")] = 1000,
    chunk_overlap: Annotated[int, typer.Option(help="Overlapping characters between chunks.")] = 180,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if an index already exists.")] = False,
) -> None:
    """Build a FAISS vector index from PDFs in data/paper."""
    build_index(
        paper_dir=paper_dir,
        index_dir=index_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force=force,
    )


@app.command(name="ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Question to answer from the paper knowledge base.")],
    paper_dir: Annotated[Path, typer.Option(help="Directory containing PDF papers.")] = DEFAULT_PAPER_DIR,
    index_dir: Annotated[Path, typer.Option(help="Directory used to load/save the FAISS index.")] = DEFAULT_INDEX_DIR,
    k: Annotated[int, typer.Option(help="Number of source chunks to retrieve.")] = 5,
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Rebuild the index before asking.")] = False,
    show_snippets: Annotated[
        bool,
        typer.Option("--show-snippets/--hide-snippets", help="Print retrieved source snippets after the answer."),
    ] = True,
    include_references: Annotated[
        bool,
        typer.Option("--include-references", help="Allow bibliography/reference-list chunks in retrieval results."),
    ] = False,
) -> None:
    """Ask one question and return an answer with source snippets."""
    ask_question(
        question=question,
        paper_dir=paper_dir,
        index_dir=index_dir,
        k=k,
        rebuild=rebuild,
        show_snippets=show_snippets,
        include_references=include_references,
    )


@app.command(name="models")
def models_command() -> None:
    """Show the currently configured model providers."""
    settings = Settings.from_env()
    table = Table(title="Model Settings")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("LLM_PROVIDER", settings.llm_provider)
    table.add_row("EMBEDDING_PROVIDER", settings.embedding_provider)
    table.add_row("OLLAMA_BASE_URL", settings.ollama_base_url)
    table.add_row("OLLAMA_CHAT_MODEL", settings.ollama_chat_model)
    table.add_row("OLLAMA_EMBED_MODEL", settings.ollama_embed_model)
    table.add_row("DEEPSEEK_BASE_URL", settings.deepseek_base_url)
    table.add_row("DEEPSEEK_CHAT_MODEL", settings.deepseek_chat_model)
    table.add_row("OPENAI_BASE_URL", settings.openai_base_url or "")
    table.add_row("OPENAI_CHAT_MODEL", settings.openai_chat_model)
    table.add_row("OPENAI_EMBED_MODEL", settings.openai_embed_model)
    console.print(table)

