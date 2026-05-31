from __future__ import annotations

import os
import re
import sys
import warnings
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PAPER_DIR = ROOT_DIR / "data" / "paper"
DEFAULT_INDEX_DIR = ROOT_DIR / "vectorstore" / "faiss_index"

console = Console()
app = typer.Typer(
    help="Minimal domain paper RAG assistant: index PDFs, ask questions, and show source snippets."
)


@dataclass(frozen=True)
class Settings:
    # 这里集中保存模型和 API 配置，避免这些配置散落在代码各处。
    llm_provider: Literal["ollama", "deepseek", "openai"] = "ollama"
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "deepseek-r1:1.5b"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        # 每次运行命令时读取 .env，这样改模型或 API key 不需要改代码。
        load_dotenv(ROOT_DIR / ".env")
        return cls(
            llm_provider=_env_choice("LLM_PROVIDER", "ollama", {"ollama", "deepseek", "openai"}),
            embedding_provider=_env_choice("EMBEDDING_PROVIDER", "ollama", {"ollama", "openai"}),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:1.5b"),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
            deepseek_api_key=_empty_to_none(os.getenv("DEEPSEEK_API_KEY")),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            openai_api_key=_empty_to_none(os.getenv("OPENAI_API_KEY")),
            openai_base_url=_empty_to_none(os.getenv("OPENAI_BASE_URL")),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
        )


def _empty_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise typer.BadParameter(f"{name} must be one of: {allowed}")
    return value


def build_embeddings(settings: Settings):
    # Embedding 模型负责把论文片段和用户问题都转换成向量。
    # 后面 FAISS 会用这些向量计算“问题”和“论文片段”的相似度。
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    if not settings.openai_api_key:
        raise typer.BadParameter(
            "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai. "
            "DeepSeek is kept for chat; use Ollama or an OpenAI-compatible embedding API for embeddings."
        )
    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def build_llm(settings: Settings):
    # LLM/chat 模型负责根据检索到的论文片段生成最终回答。
    # 默认走本机 Ollama，也保留 DeepSeek 和 OpenAI-compatible 接口。
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise typer.BadParameter("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek.")
        return ChatOpenAI(
            model=settings.deepseek_chat_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.temperature,
        )

    if not settings.openai_api_key:
        raise typer.BadParameter("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.temperature,
    )


def load_pdf_pages(paper_dir: Path) -> list[Document]:
    # 把 data/paper 目录下的每篇 PDF 按页读成 LangChain 的 Document。
    pdf_paths = sorted(paper_dir.glob("*.pdf"))
    if not pdf_paths:
        raise typer.BadParameter(f"No PDF files found in {paper_dir}")

    pages: list[Document] = []
    for pdf_path in pdf_paths:
        console.print(f"Loading [bold]{pdf_path.name}[/bold]")
        pages.extend(load_one_pdf(pdf_path))
    return pages


def load_one_pdf(pdf_path: Path) -> list[Document]:
    try:
        # 优先用 LangChain 自带的 PyPDFLoader，简单、和 LangChain 生态兼容。
        loaded_pages = PyPDFLoader(str(pdf_path)).load()
        pages: list[Document] = []
        for page in loaded_pages:
            page_number = int(page.metadata.get("page", 0)) + 1
            page.metadata.update(pdf_metadata(pdf_path, page_number))
            pages.append(page)
        return pages
    except Exception as exc:
        # 有些 PDF 的内部结构不标准，pypdf 会读失败；这时换 PyMuPDF 兜底。
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
    # metadata 会一路跟着 chunk 进入向量库，最后用于展示来源论文和页码。
    return {
        "source": pdf_path.name,
        "source_path": str(pdf_path),
        "page": page_number,
    }


def split_pages(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    # RAG 通常不把整篇论文直接塞给模型，而是先切成小片段再检索。
    # overlap 能让相邻 chunk 保留一点上下文，减少句子被切断带来的信息损失。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks, start=1):
        # chunk_id 只是一个方便人类查看来源的编号，不参与模型推理。
        chunk.metadata["chunk_id"] = index
    return chunks


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

    # 建库流程：读 PDF -> 切 chunk -> 算 embedding -> 写入 FAISS。
    pages = load_pdf_pages(paper_dir)
    chunks = split_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    console.print(f"Loaded {len(pages)} pages, split into {len(chunks)} chunks.")

    embeddings = build_embeddings(settings)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))
    console.print(f"Saved FAISS index to [cyan]{index_dir}[/cyan]")


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


def make_context(results: list[tuple[Document, float]], max_chars_per_source: int) -> tuple[str, list[dict[str, str]]]:
    # 同一批检索结果要整理成两份：
    # 1. context_blocks 给 LLM 阅读；
    # 2. source_rows 给用户查看来源。
    context_blocks: list[str] = []
    source_rows: list[dict[str, str]] = []

    for source_number, (doc, score) in enumerate(results, start=1):
        source_id = f"S{source_number}"
        metadata = doc.metadata
        snippet = normalize_text(doc.page_content)[:max_chars_per_source]
        row = {
            "id": source_id,
            "source": str(metadata.get("source", "unknown")),
            "page": str(metadata.get("page", "?")),
            "chunk": str(metadata.get("chunk_id", "?")),
            "score": f"{score:.4f}",
            "snippet": snippet,
        }
        source_rows.append(row)
        context_blocks.append(
            f"[{source_id}] {row['source']} | page {row['page']} | chunk {row['chunk']}\n{snippet}"
        )

    return "\n\n".join(context_blocks), source_rows


def select_retrieval_results(
    results: list[tuple[Document, float]],
    k: int,
    include_references: bool,
) -> list[tuple[Document, float]]:
    if include_references:
        return results[:k]

    # 普通问答里，参考文献列表容易干扰模型，所以默认把它们排到后面。
    # 做 Related Work 检索时，可以用 --include-references 关闭这个过滤。
    selected: list[tuple[Document, float]] = []
    reference_like: list[tuple[Document, float]] = []

    for doc, score in results:
        if is_reference_like(doc.page_content):
            reference_like.append((doc, score))
        else:
            selected.append((doc, score))
        if len(selected) >= k:
            return selected[:k]

    return (selected + reference_like)[:k]


def is_reference_like(text: str) -> bool:
    # 这是一个轻量规则：如果 chunk 里年份、会议/期刊、pp. 等引用特征很多，
    # 就把它当作 References/参考文献片段处理。
    normalized = normalize_text(text).lower()
    head = normalized[:500]
    citation_years = len(re.findall(r"\b(?:19|20)\d{2}\b", normalized))
    reference_markers = [
        "proceedings",
        "transactions",
        "conference",
        "journal",
        "arxiv",
        "acm",
        "ieee",
        "springer",
        " proc.",
        " conf.",
        " pp.",
        " et al.",
        " available:",
        "press.",
        "trans.",
    ]
    marker_hits = sum(1 for term in reference_markers if term in normalized)
    if "references" in head:
        return True
    return citation_years >= 3 and marker_hits >= 2


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_for_console(text: str) -> str:
    # Windows 终端有时不是 UTF-8，PDF 里又常有特殊字符；这里避免打印时报编码错误。
    encoding = console.file.encoding or sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def clean_model_output(text: str) -> str:
    # 本地推理模型有时会输出 <think>...</think>，这里删掉推理过程，只保留回答。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def ensure_answer_citations(answer: str, source_rows: list[dict[str, str]]) -> str:
    # 小模型有时不严格按提示写 [S1] 引用；这里做一个简单兜底。
    if re.search(r"\[S\d+\]", answer) or not source_rows:
        return answer
    source_ids = " ".join(f"[{row['id']}]" for row in source_rows)
    return f"{answer}\n\n主要依据：{source_ids}"


def print_sources(source_rows: list[dict[str, str]], show_snippets: bool) -> None:
    # 把检索到的片段来源打印出来，这是 RAG 比普通聊天更可信的关键。
    table = Table(title="Retrieved Sources")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Paper")
    table.add_column("Page", justify="right")
    table.add_column("Chunk", justify="right")
    table.add_column("Score", justify="right")
    for row in source_rows:
        table.add_row(
            row["id"],
            safe_for_console(row["source"]),
            row["page"],
            row["chunk"],
            row["score"],
        )
    console.print(table)

    if show_snippets:
        console.print("\n[bold]Source snippets[/bold]")
        for row in source_rows:
            console.print(
                safe_for_console(
                    f"\n[cyan][{row['id']}][/cyan] {row['source']} | page {row['page']} | chunk {row['chunk']}"
                )
            )
            console.print(safe_for_console(row["snippet"]))


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
    settings = Settings.from_env()
    # 如果用户还没建索引，第一次提问时自动建一个最小索引。
    if rebuild or not index_exists(index_dir):
        build_index(
            paper_dir=paper_dir,
            index_dir=index_dir,
            chunk_size=1000,
            chunk_overlap=180,
            force=rebuild,
        )

    vectorstore = load_index(index_dir, settings)
    # 先多取一些候选，再做参考文献过滤，避免最相关的正文片段被挤掉。
    raw_results = vectorstore.similarity_search_with_score(question, k=max(k * 5, k + 10))
    results = select_retrieval_results(raw_results, k=k, include_references=include_references)
    context, source_rows = make_context(results, max_chars_per_source=1200)

    # prompt 明确要求模型只能根据检索片段回答，并且用 [S1] 这类编号引用来源。
    prompt = ChatPromptTemplate.from_messages(
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
    # LangChain 的 LCEL 写法：prompt 的输出直接传给 LLM。
    response = (prompt | build_llm(settings)).invoke({"question": question, "context": context})
    answer = ensure_answer_citations(clean_model_output(str(response.content)), source_rows)

    console.print("\n[bold]Answer[/bold]")
    console.print(Markdown(safe_for_console(answer)))
    console.print()
    print_sources(source_rows, show_snippets=show_snippets)


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


if __name__ == "__main__":
    app()
