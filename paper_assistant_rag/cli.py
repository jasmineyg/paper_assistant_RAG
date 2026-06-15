"""Typer command definitions that expose index, append, ask, eval, and models commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from paper_assistant_rag.archrag.pipeline import ArchRAGPipeline
from paper_assistant_rag.archrag_hierarchy import build_archrag_hierarchy_cache
from paper_assistant_rag.archrag_index import build_archrag_index
from paper_assistant_rag.communities import build_community_index
from paper_assistant_rag.evaluation import run_evaluation
from paper_assistant_rag.indexing import append_to_index, build_index
from paper_assistant_rag.kg import build_kg_cache
from paper_assistant_rag.models import build_embeddings
from paper_assistant_rag.paths import (
    DEFAULT_ARCHRAG_DIR,
    DEFAULT_COMMUNITY_INDEX_DIR,
    DEFAULT_EVAL_DATASET,
    DEFAULT_EVAL_RUN_DIR,
    DEFAULT_GRAPH_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_MEMORY_DB,
    DEFAULT_PAPER_DIR,
)
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


@app.command(name="append")
def append_command(
    paper_dir: Annotated[Path, typer.Option(help="Directory containing PDF papers.")] = DEFAULT_PAPER_DIR,
    index_dir: Annotated[Path, typer.Option(help="Directory used to load/save the FAISS index.")] = DEFAULT_INDEX_DIR,
    chunk_size: Annotated[int, typer.Option(help="Characters per chunk.")] = 1000,
    chunk_overlap: Annotated[int, typer.Option(help="Overlapping characters between chunks.")] = 180,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--allow-duplicates",
            help="Skip PDFs whose filename is already present in the index.",
        ),
    ] = True,
) -> None:
    """Append new PDFs to an existing FAISS vector index."""
    append_to_index(
        paper_dir=paper_dir,
        index_dir=index_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        skip_existing=skip_existing,
    )


@app.command(name="kg-build")
def kg_build_command(
    index_dir: Annotated[Path, typer.Option(help="Directory containing the existing chunk FAISS index.")] = DEFAULT_INDEX_DIR,
    graph_dir: Annotated[Path, typer.Option(help="Directory used to save KG extraction cache files.")] = DEFAULT_GRAPH_DIR,
    limit: Annotated[
        int | None,
        typer.Option(help="Extract only the first N chunks for a smoke test."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild KG cache files instead of reusing successful chunks.")] = False,
    max_chars_per_chunk: Annotated[
        int,
        typer.Option(help="Maximum chunk text characters sent to the LLM extractor."),
    ] = 2500,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, help="Number of chunks to extract concurrently."),
    ] = 12,
) -> None:
    """Extract entity/relation cache files from indexed chunks."""
    build_kg_cache(
        index_dir=index_dir,
        graph_dir=graph_dir,
        limit=limit,
        force=force,
        max_chars_per_chunk=max_chars_per_chunk,
        concurrency=concurrency,
    )


@app.command(name="community-build")
def community_build_command(
    graph_dir: Annotated[Path, typer.Option(help="Directory containing KG cache files.")] = DEFAULT_GRAPH_DIR,
    community_index_dir: Annotated[
        Path,
        typer.Option(help="Directory used to save the community FAISS index."),
    ] = DEFAULT_COMMUNITY_INDEX_DIR,
    algorithm: Annotated[
        str,
        typer.Option(help="Community detection algorithm: louvain, greedy, or label."),
    ] = "louvain",
    resolution: Annotated[
        float,
        typer.Option(help="Louvain resolution. Larger values produce more communities."),
    ] = 1.0,
    max_summary_entities: Annotated[
        int,
        typer.Option(help="Maximum key entities kept in each community summary."),
    ] = 24,
    max_summary_relations: Annotated[
        int,
        typer.Option(help="Maximum key relations kept in each community summary."),
    ] = 24,
    llm_summaries: Annotated[
        bool,
        typer.Option("--llm-summaries", help="Use the chat LLM to refine deterministic community summaries."),
    ] = False,
    summary_concurrency: Annotated[
        int,
        typer.Option("--summary-concurrency", min=1, help="Number of community summaries to refine concurrently."),
    ] = 12,
) -> None:
    """Build single-level KG communities and a community-summary FAISS index."""
    build_community_index(
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        algorithm=algorithm,
        resolution=resolution,
        max_summary_entities=max_summary_entities,
        max_summary_relations=max_summary_relations,
        llm_summaries=llm_summaries,
        summary_concurrency=summary_concurrency,
    )


@app.command(name="archrag-build")
def archrag_build_command(
    graph_dir: Annotated[Path, typer.Option(help="Directory containing KG cache files.")] = DEFAULT_GRAPH_DIR,
    archrag_dir: Annotated[
        Path,
        typer.Option(help="Directory used to save the hierarchical ArchRAG index."),
    ] = DEFAULT_ARCHRAG_DIR,
    max_levels: Annotated[
        int,
        typer.Option("--max-levels", min=1, help="Maximum hierarchy levels including level 0 entities."),
    ] = 3,
    min_nodes_per_level: Annotated[
        int,
        typer.Option("--min-nodes-per-level", min=2, help="Stop building upward when a level is smaller than this."),
    ] = 5,
    similarity_top_k: Annotated[
        int,
        typer.Option("--similarity-top-k", min=0, help="Attribute-similarity neighbors per node."),
    ] = 5,
    similarity_threshold: Annotated[
        float,
        typer.Option("--similarity-threshold", help="Minimum cosine similarity for attribute edges."),
    ] = 0.65,
    m_neighbors: Annotated[
        int,
        typer.Option("--m-neighbors", min=1, help="Intra-layer nearest-neighbor links per node."),
    ] = 8,
    community_algorithm: Annotated[
        str,
        typer.Option("--community-algorithm", help="Community algorithm: louvain, greedy, or label."),
    ] = "louvain",
    summary_concurrency: Annotated[
        int,
        typer.Option("--summary-concurrency", min=1, help="Concurrent LLM community summary calls during ArchRAG build."),
    ] = 12,
) -> None:
    """Build hierarchical attributed communities and a C-HNSW-like ArchRAG index."""
    settings = Settings.from_env()
    hierarchy = build_archrag_hierarchy_cache(
        graph_dir=graph_dir,
        archrag_dir=archrag_dir,
        max_levels=max_levels,
        min_nodes_per_level=min_nodes_per_level,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
        community_algorithm=community_algorithm,
        summary_concurrency=summary_concurrency,
    )
    arch_index = build_archrag_index(
        hierarchy=hierarchy,
        embeddings=build_embeddings(settings),
        m_neighbors=m_neighbors,
        archrag_dir=archrag_dir,
    )
    layer_counts = {level: len(layer.nodes) for level, layer in arch_index.layers.items()}
    console.print(f"Saved hierarchical ArchRAG index to [cyan]{archrag_dir}[/cyan]")
    console.print(f"Levels: {len(layer_counts)} | nodes per level: {layer_counts}")


@app.command(name="archrag-index")
def archrag_index_command(
    paper_dir: Annotated[Path, typer.Option(help="Directory containing PDF papers.")] = DEFAULT_PAPER_DIR,
    index_dir: Annotated[Path, typer.Option(help="Directory used to save the chunk FAISS index.")] = DEFAULT_INDEX_DIR,
    graph_dir: Annotated[Path, typer.Option(help="Directory used to save KG cache files.")] = DEFAULT_GRAPH_DIR,
    archrag_dir: Annotated[
        Path,
        typer.Option(help="Directory used to save the hierarchical ArchRAG index."),
    ] = DEFAULT_ARCHRAG_DIR,
    chunk_size: Annotated[int, typer.Option(help="Characters per chunk.")] = 1000,
    chunk_overlap: Annotated[int, typer.Option(help="Overlapping characters between chunks.")] = 180,
    max_levels: Annotated[
        int,
        typer.Option("--max-levels", min=1, help="Maximum hierarchy levels including level 0 entities."),
    ] = 3,
    min_nodes_per_level: Annotated[
        int,
        typer.Option("--min-nodes-per-level", min=2, help="Stop building upward when a level is smaller than this."),
    ] = 5,
    similarity_top_k: Annotated[
        int,
        typer.Option("--similarity-top-k", min=0, help="Attribute-similarity neighbors per node."),
    ] = 5,
    similarity_threshold: Annotated[
        float,
        typer.Option("--similarity-threshold", help="Minimum cosine similarity for attribute edges."),
    ] = 0.65,
    m_neighbors: Annotated[
        int,
        typer.Option("--m-neighbors", min=1, help="Intra-layer nearest-neighbor links per node."),
    ] = 8,
    community_algorithm: Annotated[
        str,
        typer.Option("--community-algorithm", help="Community algorithm: louvain, greedy, or label."),
    ] = "louvain",
    kg_limit: Annotated[
        int | None,
        typer.Option("--kg-limit", help="Extract only the first N chunks for an ArchRAG smoke test."),
    ] = None,
    max_chars_per_chunk: Annotated[
        int,
        typer.Option(help="Maximum chunk text characters sent to the LLM extractor."),
    ] = 2500,
    extraction_concurrency: Annotated[
        int,
        typer.Option("--extraction-concurrency", min=1, help="Concurrent chunk extraction calls."),
    ] = 12,
    summary_concurrency: Annotated[
        int,
        typer.Option("--summary-concurrency", min=1, help="Concurrent community summary calls."),
    ] = 12,
    force: Annotated[bool, typer.Option("--force", help="Rebuild all ArchRAG artifacts.")] = False,
) -> None:
    """Build the full ArchRAG offline pipeline from PDFs to hierarchical index."""
    summary = ArchRAGPipeline(
        paper_dir=paper_dir,
        index_dir=index_dir,
        graph_dir=graph_dir,
        archrag_dir=archrag_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_levels=max_levels,
        min_nodes_per_level=min_nodes_per_level,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
        m_neighbors=m_neighbors,
        community_algorithm=community_algorithm,
        max_chars_per_chunk=max_chars_per_chunk,
        extraction_concurrency=extraction_concurrency,
        summary_concurrency=summary_concurrency,
    ).build(force=force, kg_limit=kg_limit)
    console.print(f"Saved ArchRAG pipeline manifest to [cyan]{archrag_dir / 'pipeline_manifest.json'}[/cyan]")
    console.print(f"Levels: {summary['levels']} | nodes per level: {summary['layer_node_counts']}")


@app.command(name="ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Question to answer from the paper knowledge base.")],
    paper_dir: Annotated[Path, typer.Option(help="Directory containing PDF papers.")] = DEFAULT_PAPER_DIR,
    index_dir: Annotated[Path, typer.Option(help="Directory used to load/save the FAISS index.")] = DEFAULT_INDEX_DIR,
    graph_dir: Annotated[Path, typer.Option(help="Directory containing KG cache files for graph retrieval.")] = DEFAULT_GRAPH_DIR,
    community_index_dir: Annotated[
        Path,
        typer.Option(help="Directory containing community summary FAISS index for archrag retrieval."),
    ] = DEFAULT_COMMUNITY_INDEX_DIR,
    archrag_dir: Annotated[
        Path,
        typer.Option(help="Directory containing hierarchical ArchRAG index files."),
    ] = DEFAULT_ARCHRAG_DIR,
    memory_db: Annotated[Path, typer.Option(help="SQLite database used to persist chat memory.")] = DEFAULT_MEMORY_DB,
    session: Annotated[str, typer.Option(help="Conversation session id used for chat memory.")] = "default",
    reset_memory: Annotated[
        bool,
        typer.Option("--reset-memory", help="Clear this session's chat memory before answering."),
    ] = False,
    k: Annotated[int, typer.Option(help="Number of source chunks to retrieve.")] = 10,
    final_k: Annotated[
        int | None,
        typer.Option("--final-k", help="Final number of source chunks for answer context. Defaults to --k."),
    ] = None,
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Rebuild the index before asking.")] = False,
    show_snippets: Annotated[
        bool,
        typer.Option("--show-snippets/--hide-snippets", help="Print retrieved source snippets after the answer."),
    ] = True,
    include_references: Annotated[
        bool,
        typer.Option("--include-references", help="Allow bibliography/reference-list chunks in retrieval results."),
    ] = False,
    retrieval_mode: Annotated[
        str,
        typer.Option("--retrieval-mode", help="Retrieval mode: hybrid, graph, community, archrag-lite, archrag, or archrag-gated."),
    ] = "archrag",
    top_k_per_level: Annotated[
        int,
        typer.Option(
            "--top-k-per-level",
            min=1,
            help="Minimum ArchRAG candidates kept per level; adaptive beam width may keep more.",
        ),
    ] = 5,
    show_archrag_debug: Annotated[
        bool,
        typer.Option("--show-archrag-debug", help="Print hierarchical ArchRAG search and filtering diagnostics."),
    ] = False,
    max_levels: Annotated[
        int | None,
        typer.Option("--max-levels", min=1, help="Limit hierarchy levels used during ArchRAG query search."),
    ] = None,
    candidate_papers: Annotated[
        int,
        typer.Option("--candidate-papers", min=1, help="Number of candidate papers for archrag-gated mode."),
    ] = 5,
    per_paper_k: Annotated[
        int,
        typer.Option("--per-paper-k", min=1, help="Chunks to retrieve inside each candidate paper."),
    ] = 5,
    community_k: Annotated[
        int,
        typer.Option("--community-k", min=0, help="Number of community summaries to retrieve in archrag mode."),
    ] = 3,
    include_community_docs: Annotated[
        bool,
        typer.Option(
            "--include-community-docs/--no-include-community-docs",
            help="Allow a small quota of community summaries in final context.",
        ),
    ] = False,
    show_retrieval_debug: Annotated[
        bool,
        typer.Option("--show-retrieval-debug", help="Print candidate papers and final chunk scores."),
    ] = False,
    adaptive_filter: Annotated[
        bool,
        typer.Option(
            "--adaptive-filter/--no-adaptive-filter",
            help="Use the chat LLM to score and filter evidence before final answering in archrag mode.",
        ),
    ] = True,
) -> None:
    """Ask with persistent chat memory and return an answer with source snippets."""
    ask_question(
        question=question,
        paper_dir=paper_dir,
        index_dir=index_dir,
        memory_db=memory_db,
        session_id=session,
        reset_memory=reset_memory,
        k=final_k or k,
        rebuild=rebuild,
        show_snippets=show_snippets,
        include_references=include_references,
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        archrag_dir=archrag_dir,
        community_k=community_k,
        candidate_papers=candidate_papers,
        per_paper_k=per_paper_k,
        include_community_docs=include_community_docs,
        show_retrieval_debug=show_retrieval_debug,
        top_k_per_level=top_k_per_level,
        show_archrag_debug=show_archrag_debug,
        max_levels=max_levels,
        adaptive_filter=adaptive_filter,
        retrieval_mode=retrieval_mode,
    )


@app.command(name="eval")
def eval_command(
    dataset: Annotated[
        Path,
        typer.Option(help="Evaluation dataset JSON file."),
    ] = DEFAULT_EVAL_DATASET,
    index_dir: Annotated[
        Path,
        typer.Option(help="Directory used to load the FAISS index."),
    ] = DEFAULT_INDEX_DIR,
    memory_db: Annotated[
        Path,
        typer.Option(help="SQLite database used when answer generation is enabled."),
    ] = DEFAULT_MEMORY_DB,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for JSON/CSV evaluation reports."),
    ] = DEFAULT_EVAL_RUN_DIR,
    graph_dir: Annotated[
        Path,
        typer.Option(help="Directory containing KG cache files for graph retrieval."),
    ] = DEFAULT_GRAPH_DIR,
    community_index_dir: Annotated[
        Path,
        typer.Option(help="Directory containing community summary FAISS index for archrag retrieval."),
    ] = DEFAULT_COMMUNITY_INDEX_DIR,
    archrag_dir: Annotated[
        Path,
        typer.Option(help="Directory containing hierarchical ArchRAG index files."),
    ] = DEFAULT_ARCHRAG_DIR,
    k: Annotated[int, typer.Option(help="Number of source chunks to retrieve per item.")] = 10,
    limit: Annotated[
        int | None,
        typer.Option(help="Run only the first N items for a smoke test."),
    ] = None,
    query_field: Annotated[
        str,
        typer.Option(help="Dataset field used as the RAG query: question or canonical_question."),
    ] = "question",
    include_references: Annotated[
        bool,
        typer.Option("--include-references", help="Allow bibliography/reference-list chunks in retrieval results."),
    ] = False,
    with_answers: Annotated[
        bool,
        typer.Option(
            "--with-answers/--retrieval-only",
            help="Run the full RAG answer chain, or only compute retrieval metrics.",
        ),
    ] = True,
    session_prefix: Annotated[
        str,
        typer.Option(help="Session id prefix used when --with-answers is enabled."),
    ] = "eval",
    retrieval_mode: Annotated[
        str,
        typer.Option("--retrieval-mode", help="Retrieval mode: hybrid, graph, community, archrag-lite, archrag, or archrag-gated."),
    ] = "archrag",
    top_k_per_level: Annotated[
        int,
        typer.Option(
            "--top-k-per-level",
            min=1,
            help="Minimum ArchRAG candidates kept per level; adaptive beam width may keep more.",
        ),
    ] = 5,
    max_levels: Annotated[
        int | None,
        typer.Option("--max-levels", min=1, help="Limit hierarchy levels used during ArchRAG evaluation search."),
    ] = None,
    candidate_papers: Annotated[
        int,
        typer.Option("--candidate-papers", min=1, help="Number of candidate papers for archrag-gated mode."),
    ] = 5,
    per_paper_k: Annotated[
        int,
        typer.Option("--per-paper-k", min=1, help="Chunks to retrieve inside each candidate paper."),
    ] = 5,
    community_k: Annotated[
        int,
        typer.Option("--community-k", min=0, help="Number of community summaries to retrieve in archrag mode."),
    ] = 3,
    adaptive_filter: Annotated[
        bool,
        typer.Option(
            "--adaptive-filter/--no-adaptive-filter",
            help="Use the chat LLM to score and filter evidence before final answering in archrag mode.",
        ),
    ] = True,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            min=1,
            help="Number of single-turn evaluation items to run concurrently. Multi-turn items still run sequentially.",
        ),
    ] = 12,
) -> None:
    """Run a curated RAG evaluation dataset and write metric reports."""
    run_evaluation(
        dataset_path=dataset,
        index_dir=index_dir,
        memory_db=memory_db,
        output_dir=output_dir,
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        archrag_dir=archrag_dir,
        k=k,
        limit=limit,
        query_field=query_field,
        include_references=include_references,
        with_answers=with_answers,
        session_prefix=session_prefix,
        community_k=community_k,
        candidate_papers=candidate_papers,
        per_paper_k=per_paper_k,
        top_k_per_level=top_k_per_level,
        max_levels=max_levels,
        adaptive_filter=adaptive_filter,
        retrieval_mode=retrieval_mode,
        concurrency=concurrency,
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
    if settings.llm_provider == "ollama" or settings.embedding_provider == "ollama":
        table.add_row("OLLAMA_BASE_URL", settings.ollama_base_url)
        table.add_row("OLLAMA_CHAT_MODEL", settings.ollama_chat_model)
        table.add_row("OLLAMA_EMBED_MODEL", settings.ollama_embed_model)
    if settings.llm_provider == "deepseek":
        table.add_row("DEEPSEEK_BASE_URL", settings.deepseek_base_url)
        table.add_row("DEEPSEEK_CHAT_MODEL", settings.deepseek_chat_model)
    if settings.llm_provider == "siliconflow":
        table.add_row("SILICONFLOW_BASE_URL", settings.siliconflow_base_url)
        table.add_row("SILICONFLOW_CHAT_MODEL", settings.siliconflow_chat_model)
    if settings.llm_provider == "openai" or settings.embedding_provider == "openai":
        table.add_row("OPENAI_BASE_URL", settings.openai_base_url or "")
    if settings.llm_provider == "openai":
        table.add_row("OPENAI_CHAT_MODEL", settings.openai_chat_model)
    if settings.embedding_provider == "openai":
        table.add_row("OPENAI_EMBED_MODEL", settings.openai_embed_model)
    console.print(table)
