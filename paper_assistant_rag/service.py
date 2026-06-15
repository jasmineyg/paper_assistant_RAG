"""Programmatic service facade for browser and other non-CLI entrypoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from paper_assistant_rag.archrag.pipeline import ArchRAGPipeline
from paper_assistant_rag.archrag_generation import generate_archrag_answer
from paper_assistant_rag.archrag_index import load_archrag_index
from paper_assistant_rag.indexing import build_index as build_chunk_index
from paper_assistant_rag.indexing import index_exists, load_index
from paper_assistant_rag.memory import clear_session_history
from paper_assistant_rag.memory import get_session_history
from paper_assistant_rag.models import build_embeddings, build_llm
from paper_assistant_rag.paths import (
    DEFAULT_ARCHRAG_DIR,
    DEFAULT_COMMUNITY_INDEX_DIR,
    DEFAULT_GRAPH_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_MEMORY_DB,
    DEFAULT_PAPER_DIR,
)
from paper_assistant_rag.qa import (
    MAX_CHARS_PER_SOURCE,
    build_conversational_rag_chain,
    normalize_retrieval_mode,
    source_rows_from_documents,
)
from paper_assistant_rag.settings import Settings


class PaperAssistantService:
    """Small stable API over the existing ArchRAG and baseline QA pipelines."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.paper_dir = DEFAULT_PAPER_DIR
        self.index_dir = DEFAULT_INDEX_DIR
        self.graph_dir = DEFAULT_GRAPH_DIR
        self.community_index_dir = DEFAULT_COMMUNITY_INDEX_DIR
        self.archrag_dir = DEFAULT_ARCHRAG_DIR
        self.memory_db = DEFAULT_MEMORY_DB

    def status(self) -> dict[str, Any]:
        """Return filesystem-level readiness without calling model services."""
        return {
            "knowledge_base_path": str(self.paper_dir),
            "baseline_index": {
                "path": str(self.index_dir),
                "ready": index_exists(self.index_dir),
            },
            "archrag_index": {
                "path": str(self.archrag_dir),
                "ready": self._archrag_index_exists(),
            },
            "graph_cache": {
                "path": str(self.graph_dir),
                "ready": (self.graph_dir / "entities.jsonl").exists()
                and (self.graph_dir / "relations.jsonl").exists(),
            },
        }

    def model_summary(self) -> dict[str, str]:
        """Return non-secret model settings for display."""
        settings = Settings.from_env()
        return {
            "LLM_PROVIDER": settings.llm_provider,
            "LLM_MODEL": _chat_model(settings),
            "LLM_BASE_URL": _chat_base_url(settings),
            "EMBEDDING_PROVIDER": settings.embedding_provider,
            "EMBEDDING_MODEL": _embedding_model(settings),
            "EMBEDDING_BASE_URL": _embedding_base_url(settings),
            "TEMPERATURE": str(settings.temperature),
        }

    def build_index(self, force: bool = False, retrieval_mode: str = "archrag") -> None:
        """Build the index required by the selected retrieval mode."""
        mode = _ui_mode_to_retrieval_mode(retrieval_mode)
        if mode == "archrag":
            ArchRAGPipeline(
                paper_dir=self.paper_dir,
                index_dir=self.index_dir,
                graph_dir=self.graph_dir,
                archrag_dir=self.archrag_dir,
            ).build(force=force)
            return

        build_chunk_index(
            paper_dir=self.paper_dir,
            index_dir=self.index_dir,
            chunk_size=1000,
            chunk_overlap=180,
            force=force,
        )

    def clear_chat_history(self, session_id: str = "streamlit") -> None:
        """Clear persisted chat memory used by ArchRAG and baseline query rewriting."""
        clear_session_history(session_id=session_id, db_path=self.memory_db)

    def ask(
        self,
        question: str,
        retrieval_mode: str = "archrag",
        session_id: str = "streamlit",
        k: int = 10,
        top_k_per_level: int = 5,
        max_levels: int | None = None,
        adaptive_filter: bool = True,
    ) -> dict[str, Any]:
        """Answer a question and return UI-friendly answer, source, and debug fields."""
        mode = _ui_mode_to_retrieval_mode(retrieval_mode)
        start = time.perf_counter()
        if mode == "archrag":
            result = self._ask_archrag(
                question=question,
                session_id=session_id,
                k=k,
                top_k_per_level=top_k_per_level,
                max_levels=max_levels,
            )
        else:
            result = self._ask_hybrid(
                question=question,
                session_id=session_id,
                k=k,
                adaptive_filter=adaptive_filter,
            )
        result.setdefault("metadata", {})
        result["metadata"].update(
            {
                "mode": mode,
                "latency": round(time.perf_counter() - start, 3),
            }
        )
        return result

    def _ask_archrag(
        self,
        question: str,
        session_id: str,
        k: int,
        top_k_per_level: int,
        max_levels: int | None,
    ) -> dict[str, Any]:
        if not self._archrag_index_exists() or not index_exists(self.index_dir):
            self.build_index(force=False, retrieval_mode="archrag")

        settings = Settings.from_env()
        arch_index = load_archrag_index(self.archrag_dir)
        embeddings = build_embeddings(settings)
        history = get_session_history(
            session_id=session_id,
            db_path=self.memory_db,
            max_messages=12,
        )
        result = generate_archrag_answer(
            query=question,
            arch_index=arch_index,
            llm=build_llm(settings),
            embeddings=embeddings,
            vectorstore=load_index(self.index_dir, settings),
            top_k_per_level=top_k_per_level,
            max_levels=max_levels,
            final_chunk_limit=k,
            chat_history=list(history.messages),
        )
        history.add_messages(
            [
                HumanMessage(content=question),
                AIMessage(content=str(result.get("answer", ""))),
            ]
        )
        debug_info = result.get("debug_info", {})
        level_results = _level_results(debug_info)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "rewritten_query": result.get("rewritten_query", {}),
            "entry_nodes": result.get("entry_nodes", []),
            "retrieval_paths": result.get("retrieval_paths", []),
            "final_chunks": result.get("final_chunks", []),
            "chunk_scores": result.get("chunk_scores", []),
            "query_type": result.get("query_type", "fact"),
            "retrieval": _retrieval_summary_from_archrag(result.get("sources", []), level_results),
            "filter_reports": debug_info.get("reports", []),
            "metadata": {
                "debug_info": debug_info,
            },
        }

    def _ask_hybrid(
        self,
        question: str,
        session_id: str,
        k: int,
        adaptive_filter: bool,
    ) -> dict[str, Any]:
        if not index_exists(self.index_dir):
            self.build_index(force=False, retrieval_mode="hybrid")

        settings = Settings.from_env()
        vectorstore = load_index(self.index_dir, settings)
        chain = build_conversational_rag_chain(
            vectorstore=vectorstore,
            settings=settings,
            memory_db=self.memory_db,
            k=k,
            include_references=False,
            graph_dir=self.graph_dir,
            community_index_dir=self.community_index_dir,
            archrag_dir=self.archrag_dir,
            adaptive_filter=adaptive_filter,
            retrieval_mode="hybrid",
        )
        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        docs = result.get("context", [])
        sources = source_rows_from_documents(
            [doc for doc in docs if isinstance(doc, Document)],
            max_chars_per_source=MAX_CHARS_PER_SOURCE,
        )
        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "retrieval": _retrieval_summary_from_sources(sources),
            "filter_reports": [],
            "metadata": {},
        }

    def _archrag_index_exists(self) -> bool:
        return (self.archrag_dir / "hierarchy.json").exists() and (self.archrag_dir / "nodes.jsonl").exists()


def _ui_mode_to_retrieval_mode(retrieval_mode: str) -> str:
    mode = retrieval_mode.strip().lower()
    if mode in {"baseline hybrid rag", "baseline", "hybrid"}:
        return "hybrid"
    if mode in {"archrag", "archrag rag", "archrag-style rag"}:
        return "archrag"
    return normalize_retrieval_mode(retrieval_mode)


def _level_results(debug_info: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    search = debug_info.get("search", {}) if isinstance(debug_info, dict) else {}
    raw = search.get("level_results", {}) if isinstance(search, dict) else {}
    level_results: dict[int, list[dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return level_results
    for level, nodes in raw.items():
        try:
            level_key = int(level)
        except (TypeError, ValueError):
            continue
        if isinstance(nodes, list):
            level_results[level_key] = [node for node in nodes if isinstance(node, dict)]
    return level_results


def _retrieval_summary_from_archrag(
    sources: list[dict[str, Any]],
    level_results: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    communities_by_level = {
        str(level): [_node_summary(node) for node in nodes]
        for level, nodes in sorted(level_results.items())
    }
    entities = [
        _node_summary(node)
        for node in level_results.get(0, [])
        if str(node.get("node_type", "")).lower() in {"entity", "entities", ""}
    ]
    return {
        **_retrieval_summary_from_sources(sources),
        "entities": entities,
        "communities_by_level": communities_by_level,
    }


def _retrieval_summary_from_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    papers = sorted({str(row.get("source", "")) for row in sources if row.get("source")})
    chunks = [
        {
            "source": row.get("source", "unknown"),
            "page": row.get("page", "?"),
            "chunk": row.get("chunk", "?"),
            "stable_chunk_id": row.get("stable_chunk_id", ""),
            "score": row.get("score", "?"),
        }
        for row in sources
    ]
    return {
        "papers": papers,
        "chunks": chunks,
        "entities": [],
        "communities_by_level": {},
    }


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    text = str(node.get("summary") or node.get("text") or "")
    return {
        "node_id": node.get("node_id", ""),
        "name": node.get("name", ""),
        "type": node.get("node_type", ""),
        "level": node.get("level", ""),
        "score": round(float(node.get("score", 0.0)), 4),
        "source_chunks": node.get("source_chunks", []),
        "text": text[:500],
    }


def _chat_base_url(settings: Settings) -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_base_url
    if settings.llm_provider == "deepseek":
        return settings.deepseek_base_url
    if settings.llm_provider == "siliconflow":
        return settings.siliconflow_base_url
    return settings.openai_base_url or ""


def _chat_model(settings: Settings) -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_chat_model
    if settings.llm_provider == "deepseek":
        return settings.deepseek_chat_model
    if settings.llm_provider == "siliconflow":
        return settings.siliconflow_chat_model
    return settings.openai_chat_model


def _embedding_base_url(settings: Settings) -> str:
    if settings.embedding_provider == "ollama":
        return settings.ollama_base_url
    return settings.openai_base_url or ""


def _embedding_model(settings: Settings) -> str:
    if settings.embedding_provider == "ollama":
        return settings.ollama_embed_model
    return settings.openai_embed_model
