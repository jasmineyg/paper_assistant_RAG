from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import typer
from dotenv import load_dotenv

from paper_assistant_rag.paths import ROOT_DIR


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

