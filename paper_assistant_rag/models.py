"""Factory functions that build chat LLMs and embedding models from Settings."""

from __future__ import annotations

import typer
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from paper_assistant_rag.settings import Settings


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
            "Use an OpenAI-compatible embedding API and set OPENAI_BASE_URL/OPENAI_EMBED_MODEL if needed."
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
