"""Online query rewriting and query-conditioned retrieval policies."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

QUERY_TYPES = {"fact", "multi-hop", "abstract", "procedural"}


def rewrite_query(
    query: str,
    llm=None,
    chat_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Rewrite a user question into a validated structure for online retrieval."""
    original_query = str(query).strip()
    if not original_query:
        raise ValueError("query must not be empty")

    history_text = _format_chat_history(chat_history)
    parsed: dict[str, Any] = {}
    if llm is not None:
        prompt = f"""
You are an academic RAG query planner. Convert the user question into a structured retrieval query.
Do not answer the question and do not add paper-specific facts that are absent from the question.
Use previous conversation only to resolve pronouns, omitted subjects, abbreviations, and follow-up context.
The standalone query must preserve the latest user intent and must not treat prior assistant claims as verified evidence.

Previous conversation:
{history_text or "No previous conversation."}

Latest user question:
{original_query}

Return valid JSON only:
{{
  "original_query": "{_json_prompt_value(original_query)}",
  "standalone_query": "self-contained retrieval question with references resolved",
  "entities": ["named methods, models, datasets, authors, or concepts"],
  "keywords": ["retrieval terms and technical synonyms"],
  "sub_questions": ["independently retrievable question"],
  "query_type": "fact | multi-hop | abstract | procedural"
}}

Rules:
- multi-hop must contain 2 to 4 sub_questions covering the required reasoning hops.
- abstract must add broad field, global context, related-method, or research-landscape keywords.
- fact should emphasize exact entities and identifying terms.
- procedural should emphasize steps, inputs, operations, outputs, and method terms.
- Keep every list concise, deduplicated, and grounded in the latest question plus conversational context.
""".strip()
        try:
            parsed = _parse_json_object(_response_text(llm.invoke(prompt)))
        except Exception:
            parsed = {}

    return _normalize_rewrite(original_query, parsed)


def build_retrieval_query(rewritten_query: dict[str, Any]) -> str:
    """Build one embedding-ready query string from a structured rewrite."""
    standalone_query = str(
        rewritten_query.get("standalone_query")
        or rewritten_query.get("original_query", "")
    ).strip()
    parts = [standalone_query]
    entities = _string_list(rewritten_query.get("entities"))
    keywords = _string_list(rewritten_query.get("keywords"))
    sub_questions = _string_list(rewritten_query.get("sub_questions"))
    if entities:
        parts.append("Entities: " + "; ".join(entities))
    if keywords:
        parts.append("Keywords: " + "; ".join(keywords))
    if sub_questions:
        parts.append("Sub-questions: " + " | ".join(sub_questions))
    return "\n".join(part for part in parts if part)


def get_beam_width(query_type: str, depth: int) -> int:
    """Return the beam width used at a hierarchy depth."""
    normalized = normalize_query_type(query_type)
    depth = max(0, int(depth))
    if normalized == "fact":
        return 2
    if normalized == "multi-hop":
        return 4 + depth
    if normalized == "abstract":
        return 6 + depth
    return 5


def get_rerank_weights(query_type: str) -> dict[str, float]:
    """Return normalized semantic, hierarchy, and keyword weights."""
    normalized = normalize_query_type(query_type)
    if normalized == "fact":
        return {"semantic": 0.45, "hierarchy": 0.20, "keyword": 0.35}
    if normalized == "multi-hop":
        return {"semantic": 0.45, "hierarchy": 0.40, "keyword": 0.15}
    if normalized == "abstract":
        return {"semantic": 0.65, "hierarchy": 0.25, "keyword": 0.10}
    return {"semantic": 0.45, "hierarchy": 0.45, "keyword": 0.10}


def normalize_query_type(query_type: str) -> str:
    """Normalize aliases while keeping the public schema's `fact` value."""
    normalized = str(query_type).strip().lower().replace("_", "-")
    aliases = {
        "fact": "fact",
        "factual": "fact",
        "multi-hop": "multi-hop",
        "multihop": "multi-hop",
        "abstract": "abstract",
        "global": "abstract",
        "procedural": "procedural",
        "procedure": "procedural",
        "method": "procedural",
    }
    return aliases.get(normalized, "fact")


def _normalize_rewrite(original_query: str, parsed: dict[str, Any]) -> dict[str, Any]:
    standalone_query = str(parsed.get("standalone_query", "")).strip() or original_query
    query_type = normalize_query_type(parsed.get("query_type", _infer_query_type(standalone_query)))
    entities = _string_list(parsed.get("entities"))
    keywords = _string_list(parsed.get("keywords"))
    sub_questions = _string_list(parsed.get("sub_questions"))

    if not entities:
        entities = _heuristic_entities(original_query)
    if not keywords:
        keywords = _heuristic_keywords(original_query)
    if query_type == "multi-hop":
        sub_questions = _ensure_multi_hop_questions(original_query, sub_questions)
    elif not sub_questions:
        sub_questions = [original_query]
    if query_type == "abstract":
        keywords = _dedupe(
            keywords
            + [
                "global context",
                "research landscape",
                "related methods",
            ]
        )
    if query_type == "procedural":
        keywords = _dedupe(keywords + ["steps", "method", "input", "output"])

    return {
        "original_query": original_query,
        "standalone_query": standalone_query,
        "entities": entities,
        "keywords": keywords,
        "sub_questions": sub_questions[:4],
        "query_type": query_type,
    }


def _format_chat_history(
    chat_history: Sequence[Any] | None,
    max_messages: int = 8,
    max_chars_per_message: int = 1200,
) -> str:
    """Format recent conversation turns without treating them as retrieval evidence."""
    if not chat_history:
        return ""
    lines: list[str] = []
    for message in list(chat_history)[-max_messages:]:
        role, content = _history_message_parts(message)
        if not content:
            continue
        lines.append(f"{role}: {content[:max_chars_per_message]}")
    return "\n".join(lines)


def _history_message_parts(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "message")
        content = message.get("content", "")
    else:
        role = str(getattr(message, "type", message.__class__.__name__))
        content = getattr(message, "content", message)
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    role_aliases = {"human": "user", "ai": "assistant"}
    return role_aliases.get(role.lower(), role.lower()), str(content).strip()


def _infer_query_type(query: str) -> str:
    lowered = query.lower()
    if re.search(r"\b(how to|steps?|procedure|pipeline|workflow|algorithm)\b", lowered) or any(
        token in query for token in ("如何", "步骤", "流程", "过程")
    ):
        return "procedural"
    if re.search(r"\b(overview|survey|landscape|trend|broadly|overall)\b", lowered) or any(
        token in query for token in ("综述", "全局", "整体", "趋势")
    ):
        return "abstract"
    if re.search(r"\b(and|versus|vs\.?|compare|why|relationship|difference)\b", lowered) or any(
        token in query for token in ("以及", "并且", "为什么", "区别", "关系", "对比")
    ):
        return "multi-hop"
    return "fact"


def _ensure_multi_hop_questions(query: str, questions: list[str]) -> list[str]:
    normalized = _dedupe(questions)
    if len(normalized) >= 2:
        return normalized[:4]
    return _dedupe(
        normalized
        + [
            f"What are the key entities and claims in: {query}",
            f"How are those entities or claims connected in: {query}",
        ]
    )[:4]


def _heuristic_entities(query: str) -> list[str]:
    quoted = re.findall(r'["“](.+?)["”]', query)
    technical = re.findall(r"\b[A-Z][A-Za-z0-9+_.-]*(?:\s+[A-Z][A-Za-z0-9+_.-]*)*\b", query)
    return _dedupe(quoted + technical)[:8]


def _heuristic_keywords(query: str) -> list[str]:
    latin = re.findall(r"[A-Za-z0-9][A-Za-z0-9_+\-]{2,}", query)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    return _dedupe(latin + chinese)[:12]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe(str(item).strip() for item in value if str(item).strip())


def _dedupe(values) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return rows


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("LLM did not return a JSON object")
    parsed = json.loads(cleaned[start : end + 1], strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object")
    return parsed


def _response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _json_prompt_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


__all__ = [
    "build_retrieval_query",
    "get_beam_width",
    "get_rerank_weights",
    "normalize_query_type",
    "rewrite_query",
]
