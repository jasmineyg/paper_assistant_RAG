"""Minimal Streamlit UI for the local Paper Assistant RAG service."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from paper_assistant_rag.service import PaperAssistantService  # noqa: E402


MISSING_FIELD = "当前 pipeline 未返回该字段"
SESSION_ID = "streamlit"


@st.cache_resource
def get_service() -> PaperAssistantService:
    return PaperAssistantService()


def main() -> None:
    st.set_page_config(page_title="Paper Assistant RAG", layout="wide")
    st.title("Paper Assistant RAG")

    service = get_service()
    mode_label = render_sidebar(service)
    ensure_chat_state()
    render_chat_history()

    prompt = st.chat_input("输入你的论文问题")
    if prompt:
        ask_question(service, prompt, mode_label)


def render_sidebar(service: PaperAssistantService) -> str:
    with st.sidebar:
        st.header("Knowledge Base")
        status = service.status()
        st.caption("当前知识库路径")
        st.code(status.get("knowledge_base_path", MISSING_FIELD), language=None)

        st.caption("当前索引状态")
        render_index_status(status)

        mode_label = st.radio(
            "当前使用的检索模式",
            ["ArchRAG", "Baseline Hybrid RAG"],
            index=0,
        )

        st.caption("LLM 配置摘要")
        try:
            st.json(service.model_summary())
        except Exception as exc:
            st.error(f"无法读取模型配置：{exc}")

        if st.button("重新构建索引", use_container_width=True):
            with st.spinner("正在重新构建索引，这可能需要较长时间..."):
                try:
                    service.build_index(force=True, retrieval_mode=mode_label)
                except Exception as exc:
                    st.error(f"索引构建失败：{exc}")
                else:
                    st.success("索引构建完成")
                    st.rerun()

        if st.button("清空聊天记录", use_container_width=True):
            st.session_state.messages = []
            try:
                service.clear_chat_history(session_id=SESSION_ID)
            except Exception as exc:
                st.warning(f"本地页面记录已清空，但持久化会话清理失败：{exc}")
            st.rerun()

    return mode_label


def render_index_status(status: dict[str, Any]) -> None:
    rows = [
        ("FAISS baseline", status.get("baseline_index", {})),
        ("ArchRAG hierarchy", status.get("archrag_index", {})),
        ("KG cache", status.get("graph_cache", {})),
    ]
    for label, row in rows:
        ready = bool(row.get("ready"))
        marker = "Ready" if ready else "Missing"
        st.write(f"{label}: {marker}")
        path = row.get("path")
        if path:
            st.caption(path)


def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_result_details(message.get("result", {}))


def ask_question(service: PaperAssistantService, prompt: str, mode_label: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索证据并生成回答..."):
            try:
                result = service.ask(
                    question=prompt,
                    retrieval_mode=mode_label,
                    session_id=SESSION_ID,
                )
            except Exception as exc:
                answer = f"问答失败：{exc}"
                st.error(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "result": {}}
                )
                return

        answer = str(result.get("answer") or MISSING_FIELD)
        st.markdown(answer)
        render_result_details(result)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "result": result}
        )


def render_result_details(result: dict[str, Any]) -> None:
    metadata = result.get("metadata") if isinstance(result, dict) else None
    if isinstance(metadata, dict):
        latency = metadata.get("latency")
        mode = metadata.get("mode")
        if mode or latency is not None:
            st.caption(f"mode={mode or '?'} | latency={latency if latency is not None else '?'}s")

    with st.expander("引用证据 / Source Chunks", expanded=True):
        render_sources(result.get("sources") if isinstance(result, dict) else None)

    with st.expander("命中的 Paper / Chunk / Entity / Community"):
        retrieval = result.get("retrieval") if isinstance(result, dict) else None
        render_retrieval(retrieval if isinstance(retrieval, dict) else {})

    with st.expander("每层检索结果"):
        retrieval = result.get("retrieval") if isinstance(result, dict) else None
        render_levels(retrieval if isinstance(retrieval, dict) else {})

    with st.expander("Adaptive Filtering Report"):
        render_filter_reports(result.get("filter_reports") if isinstance(result, dict) else None)


def render_sources(sources: Any) -> None:
    if not sources:
        st.info(MISSING_FIELD)
        return
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = (
            f"{source.get('id', '?')} | {source.get('source', 'unknown')} "
            f"| page {source.get('page', '?')} | chunk {source.get('chunk', '?')}"
        )
        with st.expander(title):
            st.write(
                {
                    "score": source.get("score", "?"),
                    "stable_chunk_id": source.get("stable_chunk_id", ""),
                }
            )
            st.markdown(str(source.get("snippet") or MISSING_FIELD))


def render_retrieval(retrieval: dict[str, Any]) -> None:
    papers = retrieval.get("papers")
    chunks = retrieval.get("chunks")
    entities = retrieval.get("entities")

    st.subheader("Papers")
    render_table_or_missing(papers)

    st.subheader("Chunks")
    render_table_or_missing(chunks)

    st.subheader("Entities")
    render_table_or_missing(entities)


def render_levels(retrieval: dict[str, Any]) -> None:
    communities = retrieval.get("communities_by_level")
    if not communities:
        st.info(MISSING_FIELD)
        return
    for level, rows in sorted(communities.items(), key=lambda item: int(item[0])):
        label = "Level 0 Entities" if str(level) == "0" else f"Level {level} Communities"
        st.subheader(label)
        render_table_or_missing(rows)


def render_filter_reports(reports: Any) -> None:
    if not reports:
        st.info(MISSING_FIELD)
        return
    for report in reports:
        if not isinstance(report, dict):
            st.json(report)
            continue
        st.subheader(f"Level {report.get('level', '?')}")
        points = report.get("points")
        render_table_or_missing(points)


def render_table_or_missing(value: Any) -> None:
    if not value:
        st.info(MISSING_FIELD)
        return
    if isinstance(value, list):
        st.dataframe(value, use_container_width=True)
        return
    st.json(value)


if __name__ == "__main__":
    main()

