from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from rich.markdown import Markdown

from paper_assistant_rag.indexing import build_index, index_exists, load_index
from paper_assistant_rag.models import build_llm
from paper_assistant_rag.retrieval import (
    clean_model_output,
    ensure_answer_citations,
    make_context,
    select_retrieval_results,
)
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, print_sources, safe_for_console


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是领域论文知识库助手。只能基于给定论文片段回答，不能编造。"
            "不要把 References/参考文献列表中的被引用论文误当作当前论文的方法。"
            "必须用中文回答，并在关键结论后引用来源编号，例如 [S1] 或 [S2]。"
            "如果片段不足以回答，要明确说明不足。",
        ),
        (
            "human",
            "问题：\n{question}\n\n检索到的论文片段：\n{context}\n\n"
            "请给出简洁但有用的回答，最后说明不确定性。",
        ),
    ]
)


def ask_question(
    question: str,
    paper_dir: Path,
    index_dir: Path,
    k: int,
    rebuild: bool,
    show_snippets: bool,
    include_references: bool,
) -> None:
    settings = Settings.from_env()
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
    # 先多取一些候选，再做参考文献过滤，避免最相关的正文片段被挤掉。
    with console.status("[cyan]正在检索相关论文片段...[/cyan]", spinner="dots"):
        raw_results = vectorstore.similarity_search_with_score(question, k=max(k * 5, k + 10))
    results = select_retrieval_results(raw_results, k=k, include_references=include_references)
    context, source_rows = make_context(results, max_chars_per_source=1200)

    # LangChain 的 LCEL 写法：prompt 的输出直接传给 LLM。
    with console.status("[cyan]正在调用模型生成回答...[/cyan]", spinner="dots"):
        response = (ANSWER_PROMPT | build_llm(settings)).invoke({"question": question, "context": context})
    answer = ensure_answer_citations(clean_model_output(str(response.content)), source_rows)

    console.print("\n[bold]Answer[/bold]")
    console.print(Markdown(safe_for_console(answer)))
    console.print()
    print_sources(source_rows, show_snippets=show_snippets)

