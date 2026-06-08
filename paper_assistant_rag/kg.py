"""Offline knowledge-graph extraction and cache generation."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from paper_assistant_rag.documents import stable_chunk_id
from paper_assistant_rag.indexing import load_index
from paper_assistant_rag.models import build_llm
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console, create_progress

CHUNK_EXTRACTIONS_FILE = "chunk_extractions.jsonl"
ENTITIES_FILE = "entities.jsonl"
RELATIONS_FILE = "relations.jsonl"
ENTITY_CHUNK_LINKS_FILE = "entity_chunk_links.jsonl"
MANIFEST_FILE = "manifest.json"

ENTITY_TYPES = [
    "Paper",
    "Method",
    "Model",
    "Dataset",
    "Metric",
    "Task",
    "Problem",
    "Concept",
    "Experiment",
    "Result",
    "Limitation",
    "Application",
    "Other",
]

RELATION_TYPES = [
    "proposes",
    "uses",
    "improves",
    "compares_with",
    "evaluated_on",
    "reports_metric",
    "addresses_problem",
    "has_limitation",
    "extends",
    "mentions",
    "other",
]


def build_kg_cache(
    index_dir: Path,
    graph_dir: Path,
    limit: int | None,
    force: bool,
    max_chars_per_chunk: int,
    concurrency: int,
) -> None:
    settings = Settings.from_env()
    graph_dir.mkdir(parents=True, exist_ok=True)
    if force:
        _remove_graph_outputs(graph_dir)

    vectorstore = load_index(index_dir, settings)
    chunks = _sorted_chunks(_iter_index_documents(vectorstore))
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        console.print("[yellow]No chunk documents found in the index.[/yellow]")
        return

    llm = build_llm(settings)
    extraction_path = graph_dir / CHUNK_EXTRACTIONS_FILE
    extraction_records = _load_extraction_records(extraction_path)
    cached = {
        key: record
        for key, record in extraction_records.items()
        if record.get("status") == "ok"
    }

    reused_count = 0
    extracted_count = 0
    failed_count = 0
    chunks_to_extract: list[tuple[Document, str, str]] = []

    with create_progress() as progress:
        task = progress.add_task("Extracting KG facts", total=len(chunks))
        for chunk in chunks:
            chunk_key = _chunk_key(chunk)
            legacy_chunk_key = _legacy_chunk_key(chunk)
            text_hash = _text_hash(chunk.page_content)
            cached_record = cached.get(chunk_key) or cached.get(legacy_chunk_key)
            if cached_record and cached_record.get("text_hash") == text_hash:
                migrated_record = _migrate_cached_record(cached_record, chunk, text_hash)
                extraction_records.pop(str(cached_record.get("chunk_key", "")), None)
                extraction_records[chunk_key] = migrated_record
                _write_jsonl(extraction_path, _sorted_records(extraction_records.values()))
                reused_count += 1
                progress.advance(task)
                continue
            chunks_to_extract.append((chunk, chunk_key, text_hash))

        if chunks_to_extract:
            max_workers = max(1, concurrency)
            if max_workers == 1:
                for chunk, chunk_key, text_hash in chunks_to_extract:
                    record = _extract_chunk_record(
                        llm=llm,
                        chunk=chunk,
                        text_hash=text_hash,
                        max_chars=max_chars_per_chunk,
                    )
                    if record.get("status") == "ok":
                        extracted_count += 1
                    else:
                        failed_count += 1
                    extraction_records[chunk_key] = record
                    _write_jsonl(extraction_path, _sorted_records(extraction_records.values()))
                    progress.advance(task)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_key = {
                        executor.submit(
                            _extract_chunk_record,
                            llm,
                            chunk,
                            text_hash,
                            max_chars_per_chunk,
                        ): chunk_key
                        for chunk, chunk_key, text_hash in chunks_to_extract
                    }
                    for future in as_completed(future_to_key):
                        chunk_key = future_to_key[future]
                        record = future.result()
                        if record.get("status") == "ok":
                            extracted_count += 1
                        else:
                            failed_count += 1
                        extraction_records[chunk_key] = record
                        _write_jsonl(extraction_path, _sorted_records(extraction_records.values()))
                        progress.advance(task)

    all_records = _sorted_records(extraction_records.values())
    entities, relations, links = _merge_graph_records(all_records)
    _write_jsonl(graph_dir / ENTITIES_FILE, entities)
    _write_jsonl(graph_dir / RELATIONS_FILE, relations)
    _write_jsonl(graph_dir / ENTITY_CHUNK_LINKS_FILE, links)
    _write_manifest(
        graph_dir=graph_dir,
        index_dir=index_dir,
        selected_chunks=len(chunks),
        cached_chunks=len(all_records),
        reused=reused_count,
        extracted=extracted_count,
        failed=failed_count,
        entities=len(entities),
        relations=len(relations),
        links=len(links),
    )

    console.print(f"Saved KG cache to [cyan]{graph_dir}[/cyan]")
    console.print(
        f"Chunks: {len(chunks)} | reused: {reused_count} | extracted: {extracted_count} | "
        f"failed: {failed_count} | concurrency: {max(1, concurrency)}"
    )
    console.print(f"Entities: {len(entities)} | relations: {len(relations)} | links: {len(links)}")


def _extract_chunk_record(
    llm,
    chunk: Document,
    text_hash: str,
    max_chars: int,
) -> dict[str, Any]:
    try:
        extraction = _extract_chunk_graph(
            llm=llm,
            chunk=chunk,
            max_chars=max_chars,
        )
        return _chunk_record(chunk, text_hash=text_hash, extraction=extraction)
    except Exception as exc:  # Keep long jobs resumable even when one chunk fails.
        return _chunk_record(
            chunk,
            text_hash=text_hash,
            extraction={"entities": [], "relations": []},
            error=str(exc),
        )


def _extract_chunk_graph(llm, chunk: Document, max_chars: int) -> dict[str, Any]:
    metadata = chunk.metadata
    entity_types = ", ".join(ENTITY_TYPES)
    relation_types = ", ".join(RELATION_TYPES)
    prompt = f"""
You extract a small knowledge graph from one academic paper chunk.
Use only the provided chunk text. Do not infer facts from outside knowledge.

Return valid JSON only, with this exact shape:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "one of: {entity_types}",
      "aliases": ["optional aliases"],
      "description": "short evidence-grounded description",
      "attributes": {{"key": "value"}}
    }}
  ],
  "relations": [
    {{
      "subject": "entity name from entities",
      "predicate": "one of: {relation_types}",
      "object": "entity name from entities",
      "description": "short evidence-grounded relation description",
      "evidence": "short quote or close paraphrase from the chunk"
    }}
  ]
}}

Guidelines:
- Prefer high-signal academic entities: methods, models, datasets, metrics, tasks, problems, experiments, results, limitations, applications, and paper names.
- Keep at most 12 entities and 16 relations.
- If the chunk only contains references or unusable text, return empty arrays.
- Preserve original acronyms and method names exactly when visible.

Chunk metadata:
source: {metadata.get("source", "unknown")}
page: {metadata.get("page", "?")}
chunk_id: {metadata.get("chunk_id", "?")}
section_title: {metadata.get("section_title", "")}
section_type: {metadata.get("section_type", "")}

Chunk text:
{chunk.page_content[:max_chars]}
""".strip()
    response = llm.invoke(prompt)
    content = _response_text(response)
    parsed = _parse_json_object(content)
    return {
        "entities": _normalize_entities(parsed.get("entities", [])),
        "relations": _normalize_relations(parsed.get("relations", [])),
    }


def _chunk_record(
    chunk: Document,
    text_hash: str,
    extraction: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    metadata = chunk.metadata
    record = {
        "chunk_key": _chunk_key(chunk),
        "text_hash": text_hash,
        "source": str(metadata.get("source", "unknown")),
        "source_path": str(metadata.get("source_path", "")),
        "paper_id": str(metadata.get("paper_id", "")),
        "page": str(metadata.get("page", "?")),
        "chunk_id": str(metadata.get("chunk_id", "?")),
        "stable_chunk_id": _stable_chunk_id(chunk),
        "section_title": str(metadata.get("section_title", "")),
        "section_type": str(metadata.get("section_type", "")),
        "entities": extraction.get("entities", []),
        "relations": extraction.get("relations", []),
        "status": "error" if error else "ok",
    }
    if error:
        record["error"] = error
    return record


def _merge_graph_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    entity_map: dict[str, dict[str, Any]] = {}
    entity_name_index: dict[str, str] = {}
    relation_map: dict[str, dict[str, Any]] = {}
    link_map: dict[str, dict[str, Any]] = {}

    for record in records:
        if record.get("status") != "ok":
            continue
        chunk_ref = _chunk_ref(record)
        local_entity_keys: dict[str, str] = {}
        for entity in record.get("entities", []):
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_type = _clean_type(str(entity.get("type", "Other")), ENTITY_TYPES)
            entity_names = _entity_names(entity)
            entity_key = _find_entity_key(entity_names, entity_name_index) or _entity_key(name)
            for entity_name in entity_names:
                normalized_entity_name = _canonical_name(entity_name)
                entity_name_index[normalized_entity_name] = entity_key
                local_entity_keys[normalized_entity_name] = entity_key

            merged = entity_map.setdefault(
                entity_key,
                {
                    "entity_id": entity_key,
                    "canonical_name": _canonical_name(name),
                    "name": name,
                    "type": entity_type,
                    "observed_types": [],
                    "aliases": [],
                    "description": "",
                    "attributes": {},
                    "source_chunks": [],
                },
            )
            _merge_list(merged["observed_types"], [entity_type])
            merged["type"] = _primary_entity_type(merged["observed_types"])
            _merge_list(merged["aliases"], entity.get("aliases", []))
            _merge_list(merged["aliases"], [candidate for candidate in entity_names if _canonical_name(candidate) != merged["canonical_name"]])
            if len(str(entity.get("description", ""))) > len(str(merged.get("description", ""))):
                merged["description"] = str(entity.get("description", "")).strip()
            if isinstance(entity.get("attributes"), dict):
                merged["attributes"].update({str(k): str(v) for k, v in entity["attributes"].items()})
            _merge_list(merged["source_chunks"], [chunk_ref])

            link_key = f"{entity_key}|{chunk_ref['chunk_key']}"
            link_map[link_key] = {
                "entity_id": entity_key,
                "chunk_key": chunk_ref["chunk_key"],
                "source": chunk_ref["source"],
                "page": chunk_ref["page"],
                "chunk_id": chunk_ref["chunk_id"],
                "stable_chunk_id": chunk_ref["stable_chunk_id"],
            }

        for relation in record.get("relations", []):
            subject_key = _resolve_entity_key(relation.get("subject", ""), local_entity_keys, entity_map)
            object_key = _resolve_entity_key(relation.get("object", ""), local_entity_keys, entity_map)
            if not subject_key or not object_key or subject_key == object_key:
                continue
            predicate = _clean_type(str(relation.get("predicate", "other")), RELATION_TYPES)
            relation_key = f"{subject_key}|{predicate}|{object_key}"
            merged = relation_map.setdefault(
                relation_key,
                {
                    "relation_id": _hash_id(relation_key),
                    "subject_id": subject_key,
                    "predicate": predicate,
                    "object_id": object_key,
                    "description": "",
                    "evidence": [],
                    "source_chunks": [],
                },
            )
            if len(str(relation.get("description", ""))) > len(str(merged.get("description", ""))):
                merged["description"] = str(relation.get("description", "")).strip()
            _merge_list(merged["evidence"], [str(relation.get("evidence", "")).strip()])
            _merge_list(merged["source_chunks"], [chunk_ref])

    entities = sorted(entity_map.values(), key=lambda item: (item["type"], item["name"].lower()))
    relations = sorted(relation_map.values(), key=lambda item: (item["predicate"], item["subject_id"], item["object_id"]))
    links = sorted(link_map.values(), key=lambda item: (item["source"], int_or_zero(item["chunk_id"]), item["entity_id"]))
    return entities, relations, links


def _normalize_entities(raw_entities: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []
    entities: list[dict[str, Any]] = []
    for raw in raw_entities[:12]:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        attributes = raw.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        entities.append(
            {
                "name": name[:160],
                "type": _clean_type(str(raw.get("type", "Other")), ENTITY_TYPES),
                "aliases": [str(alias).strip()[:160] for alias in aliases if str(alias).strip()][:8],
                "description": str(raw.get("description", "")).strip()[:600],
                "attributes": {str(k)[:80]: str(v)[:300] for k, v in attributes.items()},
            }
        )
    return entities


def _normalize_relations(raw_relations: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_relations, list):
        return []
    relations: list[dict[str, Any]] = []
    for raw in raw_relations[:16]:
        if not isinstance(raw, dict):
            continue
        subject = str(raw.get("subject", "")).strip()
        obj = str(raw.get("object", "")).strip()
        if not subject or not obj:
            continue
        relations.append(
            {
                "subject": subject[:160],
                "predicate": _clean_type(str(raw.get("predicate", "other")), RELATION_TYPES),
                "object": obj[:160],
                "description": str(raw.get("description", "")).strip()[:600],
                "evidence": str(raw.get("evidence", "")).strip()[:600],
            }
        )
    return relations


def _response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM did not return a JSON object")
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object")
    return parsed


def _load_extraction_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[str(record.get("chunk_key", ""))] = record
    return records


def _sorted_records(rows: Any) -> list[dict[str, Any]]:
    return sorted(list(rows), key=lambda row: (str(row.get("source", "")), int_or_zero(row.get("chunk_id")), str(row.get("chunk_key", ""))))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_manifest(
    graph_dir: Path,
    index_dir: Path,
    selected_chunks: int,
    cached_chunks: int,
    reused: int,
    extracted: int,
    failed: int,
    entities: int,
    relations: int,
    links: int,
) -> None:
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "index_dir": str(index_dir),
        "files": {
            "chunk_extractions": CHUNK_EXTRACTIONS_FILE,
            "entities": ENTITIES_FILE,
            "relations": RELATIONS_FILE,
            "entity_chunk_links": ENTITY_CHUNK_LINKS_FILE,
        },
        "counts": {
            "selected_chunks": selected_chunks,
            "cached_chunks": cached_chunks,
            "reused_chunks": reused,
            "extracted_chunks": extracted,
            "failed_chunks": failed,
            "entities": entities,
            "relations": relations,
            "entity_chunk_links": links,
        },
        "status": "partial" if failed else "ok",
    }
    (graph_dir / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _remove_graph_outputs(graph_dir: Path) -> None:
    for name in [CHUNK_EXTRACTIONS_FILE, ENTITIES_FILE, RELATIONS_FILE, ENTITY_CHUNK_LINKS_FILE, MANIFEST_FILE]:
        path = graph_dir / name
        if path.exists():
            path.unlink()


def _sorted_chunks(chunks: list[Document]) -> list[Document]:
    return sorted(
        chunks,
        key=lambda doc: (
            str(doc.metadata.get("source", "")),
            int_or_zero(doc.metadata.get("page")),
            int_or_zero(doc.metadata.get("chunk_id")),
            int_or_zero(doc.metadata.get("start_index")),
        ),
    )


def _iter_index_documents(vectorstore) -> list[Document]:
    docstore_dict = getattr(vectorstore.docstore, "_dict", {})
    if not isinstance(docstore_dict, dict):
        return []
    return [doc for doc in docstore_dict.values() if isinstance(doc, Document)]


def _chunk_key(chunk: Document) -> str:
    return _stable_chunk_id(chunk)


def _stable_chunk_id(chunk: Document) -> str:
    metadata = chunk.metadata
    stable_id = str(metadata.get("stable_chunk_id", "")).strip()
    if stable_id:
        return stable_id
    stable_id = stable_chunk_id(chunk)
    metadata["stable_chunk_id"] = stable_id
    return stable_id


def _legacy_chunk_key(chunk: Document) -> str:
    metadata = chunk.metadata
    return "|".join(
        [
            str(metadata.get("source", "unknown")),
            str(metadata.get("page", "?")),
            str(metadata.get("chunk_id", "?")),
            str(metadata.get("start_index", "?")),
        ]
    )


def _migrate_cached_record(record: dict[str, Any], chunk: Document, text_hash: str) -> dict[str, Any]:
    migrated = dict(record)
    metadata = chunk.metadata
    migrated.update(
        {
            "chunk_key": _chunk_key(chunk),
            "text_hash": text_hash,
            "source": str(metadata.get("source", "unknown")),
            "source_path": str(metadata.get("source_path", "")),
            "paper_id": str(metadata.get("paper_id", "")),
            "page": str(metadata.get("page", "?")),
            "chunk_id": str(metadata.get("chunk_id", "?")),
            "stable_chunk_id": _stable_chunk_id(chunk),
            "section_title": str(metadata.get("section_title", "")),
            "section_type": str(metadata.get("section_type", "")),
        }
    )
    return migrated


def _chunk_ref(record: dict[str, Any]) -> dict[str, str]:
    return {
        "chunk_key": str(record.get("chunk_key", "")),
        "source": str(record.get("source", "")),
        "page": str(record.get("page", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "stable_chunk_id": str(record.get("stable_chunk_id", "")),
    }


def _entity_names(entity: dict[str, Any]) -> list[str]:
    names = [str(entity.get("name", "")).strip()]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        names.extend(str(alias).strip() for alias in aliases)
    return [name for name in names if name]


def _find_entity_key(names: list[str], entity_name_index: dict[str, str]) -> str:
    for name in names:
        entity_key = entity_name_index.get(_canonical_name(name))
        if entity_key:
            return entity_key
    return ""


def _entity_key(name: str) -> str:
    return _hash_id(_canonical_name(name))


def _resolve_entity_key(name: Any, local_entity_keys: dict[str, str], entity_map: dict[str, dict[str, Any]]) -> str:
    normalized = _canonical_name(str(name))
    if normalized in local_entity_keys:
        return local_entity_keys[normalized]
    for entity_key, entity in entity_map.items():
        names = [_canonical_name(str(entity.get("name", "")))]
        names.extend(_canonical_name(alias) for alias in entity.get("aliases", []))
        if normalized in names:
            return entity_key
    return ""


def _canonical_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name).strip()
    normalized = re.sub(r"\s*\(([A-Za-z0-9][A-Za-z0-9_-]{1,20})\)\s*$", "", normalized)
    normalized = re.sub(r"[\s_-]+", " ", normalized).strip().lower()
    return normalized


def _primary_entity_type(observed_types: list[str]) -> str:
    priority = [
        "Paper",
        "Method",
        "Model",
        "Dataset",
        "Metric",
        "Task",
        "Problem",
        "Concept",
        "Experiment",
        "Result",
        "Limitation",
        "Application",
        "Other",
    ]
    for entity_type in priority:
        if entity_type in observed_types:
            return entity_type
    return "Other"


def _clean_type(value: str, allowed: list[str]) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    lookup = {re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_"): item for item in allowed}
    return lookup.get(normalized, allowed[-1])


def _merge_list(target: list[Any], values: list[Any]) -> None:
    for value in values:
        if value in ("", None):
            continue
        if value not in target:
            target.append(value)


def _text_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()[:16]


def _hash_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
