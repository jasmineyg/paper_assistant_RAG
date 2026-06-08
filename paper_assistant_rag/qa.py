"""End-to-end conversational RAG flow: retrieve chunks, call LLM, persist memory."""

from __future__ import annotations

import warnings
from pathlib import Path

import typer
from httpx import HTTPError
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from openai import OpenAIError

from paper_assistant_rag.community_retrieval import (
    adaptive_filter_documents,
    retrieve_community_augmented_chunks_with_score,
)
from paper_assistant_rag.graph_retrieval import retrieve_graph_chunks_with_score
from paper_assistant_rag.indexing import build_index, index_exists, load_index
from paper_assistant_rag.memory import clear_session_history, get_session_history
from paper_assistant_rag.models import build_llm
from paper_assistant_rag.retrieval import (
    clean_model_output,
    ensure_answer_citations,
    normalize_text,
    retrieve_chunks_with_score,
)
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, print_answer, print_sources, safe_for_console


CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你会根据历史对话和用户最新问题，改写出一个可以独立检索论文知识库的问题。"
            "如果最新问题本身已经完整，就原样输出。"
            "只输出改写后的检索问题，不要回答问题。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是领域论文知识库助手。只能基于给定论文片段回答，不能编造。"
            "不要把 References/参考文献列表中的被引用论文误当作当前论文的方法。"
            "历史对话只用于理解用户追问和代词，事实依据必须来自本轮检索到的论文片段。"
            "必须用中文回答，并在关键结论后引用来源编号，例如 [S1] 或 [S2]。"
            "如果片段不足以回答，要明确说明不足。",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "问题：\n{input}\n\n检索到的论文片段：\n{context}\n\n"
            "请用中文给出结构化回答。若问题涉及方法流程，请按 5-8 个步骤展开，"
            "每一步说明输入、处理方式、输出或作用，并在关键结论后引用来源编号。"
            "最后单独说明不确定性。"
            "回答不要少于 500 字，除非检索片段确实不足。",
        ),
    ]
)


DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[{source_id}] {source} | page {page} | chunk {chunk_id} | score {score}\n{page_content}"
)

MAX_HISTORY_MESSAGES = 12
MAX_CHARS_PER_SOURCE = 1200


class RetrievalServiceError(RuntimeError):
    """Raised when the embedding-backed retrieval call fails."""


def ask_question(
    question: str,
    paper_dir: Path,
    index_dir: Path,
    memory_db: Path,
    session_id: str,
    reset_memory: bool,
    k: int,
    rebuild: bool,
    show_snippets: bool,
    include_references: bool,
    graph_dir: Path,
    community_index_dir: Path,
    community_k: int,
    adaptive_filter: bool,
    retrieval_mode: str,
) -> None:
    settings = Settings.from_env()
    if reset_memory:
        clear_session_history(session_id=session_id, db_path=memory_db)
        console.print(f"[yellow]已清空会话记忆：{session_id}[/yellow]")

    # 如果用户还没建索引，第一次提问时自动建一个最小索引。
    if rebuild or not index_exists(index_dir):
        console.print("[yellow]未发现索引，先自动构建索引。第一次会比较慢。[/yellow]")
        build_index(
            paper_dir=paper_dir,
            index_dir=index_dir,
            chunk_size=1000,
            chunk_overlap=180,
            force=rebuild,
        )

    with console.status("[cyan]正在加载 FAISS 索引...[/cyan]", spinner="dots"):
        vectorstore = load_index(index_dir, settings)

    chain = build_conversational_rag_chain(
        vectorstore=vectorstore,
        settings=settings,
        memory_db=memory_db,
        k=k,
        include_references=include_references,
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        community_k=community_k,
        adaptive_filter=adaptive_filter,
        retrieval_mode=retrieval_mode,
    )
    try:
        with console.status("[cyan]正在结合对话记忆检索并生成回答...[/cyan]", spinner="dots"):
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )
    except RetrievalServiceError as error:
        print_retrieval_error(error, settings)
        raise typer.Exit(1) from error
    except (OpenAIError, HTTPError) as error:
        print_model_error(error, settings)
        raise typer.Exit(1) from error

    source_rows = source_rows_from_documents(result["context"], max_chars_per_source=MAX_CHARS_PER_SOURCE)
    answer = str(result["answer"])

    console.print("\n[bold]Answer[/bold]")
    print_answer(answer)
    console.print()
    print_sources(source_rows, show_snippets=show_snippets)


def build_conversational_rag_chain(
    vectorstore,
    settings: Settings,
    memory_db: Path,
    k: int,
    include_references: bool,
    graph_dir: Path | None = None,
    community_index_dir: Path | None = None,
    community_k: int = 3,
    adaptive_filter: bool = True,
    retrieval_mode: str = "hybrid",
):
    llm = build_llm(settings)
    retriever = build_hybrid_retriever(
        vectorstore,
        k=k,
        include_references=include_references,
        graph_dir=graph_dir,
        community_index_dir=community_index_dir,
        community_k=community_k,
        adaptive_filter=adaptive_filter,
        retrieval_mode=retrieval_mode,
        llm=llm,
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONTEXTUALIZE_QUESTION_PROMPT,
    )
    answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ANSWER_PROMPT,
        document_prompt=DOCUMENT_PROMPT,
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_chain) | RunnableLambda(finalize_chain_result)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*RunnableWithMessageHistory.*",
            category=LangChainDeprecationWarning,
        )
        return RunnableWithMessageHistory(
            rag_chain,
            lambda current_session_id: get_session_history(
                session_id=current_session_id,
                db_path=memory_db,
                max_messages=MAX_HISTORY_MESSAGES,
            ),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )


def build_hybrid_retriever(
    vectorstore,
    k: int,
    include_references: bool,
    graph_dir: Path | None = None,
    community_index_dir: Path | None = None,
    community_k: int = 3,
    adaptive_filter: bool = True,
    retrieval_mode: str = "hybrid",
    llm=None,
):
    mode = normalize_retrieval_mode(retrieval_mode)

    def retrieve(query: str) -> list[Document]:
        print_retrieval_query(query)
        # 先多取一些候选，再做参考文献过滤，避免最相关的正文片段被挤掉。
        try:
            if mode == "graph":
                if graph_dir is None:
                    raise RetrievalServiceError("graph retrieval requires graph_dir")
                selected_results = retrieve_graph_chunks_with_score(
                    vectorstore,
                    query=query,
                    k=k,
                    include_references=include_references,
                    graph_dir=graph_dir,
                )
            elif mode == "archrag":
                if graph_dir is None or community_index_dir is None:
                    raise RetrievalServiceError("archrag retrieval requires graph_dir and community_index_dir")
                selected_results = retrieve_community_augmented_chunks_with_score(
                    vectorstore=vectorstore,
                    query=query,
                    k=k,
                    include_references=include_references,
                    graph_dir=graph_dir,
                    community_index_dir=community_index_dir,
                    community_k=community_k,
                    include_community_docs=True,
                )
            else:
                selected_results = retrieve_chunks_with_score(
                    vectorstore,
                    query=query,
                    k=k,
                    include_references=include_references,
                )
        except (OpenAIError, HTTPError) as error:
            raise RetrievalServiceError(str(error)) from error
        documents = documents_with_source_metadata(selected_results)
        if mode == "archrag" and adaptive_filter and llm is not None:
            return adaptive_filter_documents(
                llm=llm,
                query=query,
                docs=documents,
                max_documents=max(k, 1),
            )
        return documents

    return RunnableLambda(retrieve)


def normalize_retrieval_mode(retrieval_mode: str) -> str:
    mode = retrieval_mode.strip().lower().replace("_", "-")
    aliases = {
        "hybrid": "hybrid",
        "chunk": "hybrid",
        "chunks": "hybrid",
        "graph": "graph",
        "kg": "graph",
        "graph-assisted": "graph",
        "community": "archrag",
        "communities": "archrag",
        "archrag": "archrag",
        "archrag-lite": "archrag",
    }
    if mode not in aliases:
        raise typer.BadParameter("retrieval-mode must be one of: hybrid, graph, archrag")
    return aliases[mode]


def print_retrieval_query(query: str) -> None:
    console.print("\n[bold]本轮检索问题[/bold]")
    console.print(safe_for_console(query), markup=False)


def print_retrieval_error(error: RetrievalServiceError, settings: Settings) -> None:
    console.print("\n[bold red]检索失败：embedding 服务没有成功返回向量。[/bold red]")
    console.print(
        safe_for_console(
            f"当前 embedding 配置：provider={settings.embedding_provider}, "
            f"base_url={_embedding_base_url(settings)}, model={_embedding_model(settings)}"
        )
    )
    console.print(safe_for_console(f"服务返回：{error}"))
    console.print("这通常是 embedding 服务端 5xx、模型名不被该服务支持、额度/鉴权异常，或服务临时不可用。")


def print_model_error(error: OpenAIError | HTTPError, settings: Settings) -> None:
    console.print("\n[bold red]模型服务调用失败。[/bold red]")
    console.print(
        safe_for_console(
            f"当前 chat 配置：provider={settings.llm_provider}, "
            f"base_url={_chat_base_url(settings)}, model={_chat_model(settings)}"
        )
    )
    console.print(safe_for_console(f"服务返回：{error}"))


def _embedding_base_url(settings: Settings) -> str:
    if settings.embedding_provider == "ollama":
        return settings.ollama_base_url
    return settings.openai_base_url or ""


def _embedding_model(settings: Settings) -> str:
    if settings.embedding_provider == "ollama":
        return settings.ollama_embed_model
    return settings.openai_embed_model


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


def finalize_chain_result(result: dict) -> dict:
    source_rows = source_rows_from_documents(result["context"], max_chars_per_source=MAX_CHARS_PER_SOURCE)
    finalized_result = dict(result)
    finalized_result["answer"] = ensure_answer_citations(
        clean_model_output(str(result["answer"])),
        source_rows,
    )
    return finalized_result


def documents_with_source_metadata(results: list[tuple[Document, float]]) -> list[Document]:
    documents: list[Document] = []
    for source_number, (doc, score) in enumerate(results, start=1):
        metadata = dict(doc.metadata)
        metadata["source_id"] = f"S{source_number}"
        metadata["source"] = str(metadata.get("source", "unknown"))
        metadata["page"] = str(metadata.get("page", "?"))
        metadata["chunk_id"] = str(metadata.get("chunk_id", "?"))
        metadata["score"] = f"{score:.4f}"
        documents.append(Document(page_content=normalize_text(doc.page_content), metadata=metadata))
    return documents


def source_rows_from_documents(docs: list[Document], max_chars_per_source: int) -> list[dict[str, str]]:
    source_rows: list[dict[str, str]] = []
    for doc in docs:
        metadata = doc.metadata
        source_rows.append(
            {
                "id": str(metadata.get("source_id", "?")),
                "source": str(metadata.get("source", "unknown")),
                "page": str(metadata.get("page", "?")),
                "chunk": str(metadata.get("chunk_id", "?")),
                "score": str(metadata.get("score", "?")),
                "snippet": normalize_text(doc.page_content)[:max_chars_per_source],
            }
        )
    return source_rows
