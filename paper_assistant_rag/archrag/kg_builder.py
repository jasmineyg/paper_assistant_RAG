"""ArchRAG KG construction helpers with explicit textual attributes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_assistant_rag.kg import ENTITIES_FILE, RELATIONS_FILE, build_kg_cache
from paper_assistant_rag.paths import DEFAULT_EMBED_BATCH_SIZE
from paper_assistant_rag.ui import create_progress

ATTRIBUTED_KG_FILE = "archrag_attributed_kg.json"


@dataclass
class ArchRAGEntity:
    """Entity node with ArchRAG textual attributes and source provenance."""

    entity_id: str
    name: str
    type: str
    description: str
    source_chunk_ids: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return the textual attribute embedded by ArchRAG."""
        return "\n".join(
            [
                f"Name: {self.name}",
                f"Type: {self.type}",
                f"Description: {self.description}",
                f"Attributes: {json.dumps(self.attributes, ensure_ascii=False, sort_keys=True)}",
            ]
        )


@dataclass
class ArchRAGRelation:
    """Relation edge with textual attributes and source provenance."""

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    description: str
    source_chunk_ids: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Return the textual attribute embedded by ArchRAG."""
        return "\n".join(
            [
                f"Predicate: {self.predicate}",
                f"Description: {self.description}",
                f"Evidence: {' | '.join(self.evidence[:5])}",
            ]
        )


def ensure_kg_cache(
    index_dir: Path,
    graph_dir: Path,
    limit: int | None,
    force: bool,
    max_chars_per_chunk: int,
    concurrency: int,
) -> None:
    """Build or refresh the LLM entity/relation KG cache."""
    build_kg_cache(
        index_dir=index_dir,
        graph_dir=graph_dir,
        limit=limit,
        force=force,
        max_chars_per_chunk=max_chars_per_chunk,
        concurrency=concurrency,
    )


def load_attributed_kg(graph_dir: Path) -> tuple[list[ArchRAGEntity], list[ArchRAGRelation]]:
    """Load KG JSONL files into ArchRAG entity and relation records."""
    entities_payload = _read_jsonl(graph_dir / ENTITIES_FILE)
    relations_payload = _read_jsonl(graph_dir / RELATIONS_FILE)
    if not entities_payload:
        raise ValueError(f"No KG entities found at {graph_dir / ENTITIES_FILE}. Run kg-build or archrag-index first.")

    relation_ids_by_entity: dict[str, list[str]] = {}
    relations: list[ArchRAGRelation] = []
    for relation in relations_payload:
        relation_id = str(relation.get("relation_id", ""))
        subject_id = str(relation.get("subject_id", ""))
        object_id = str(relation.get("object_id", ""))
        relation_ids_by_entity.setdefault(subject_id, []).append(relation_id)
        relation_ids_by_entity.setdefault(object_id, []).append(relation_id)
        relations.append(
            ArchRAGRelation(
                relation_id=relation_id,
                subject_id=subject_id,
                predicate=str(relation.get("predicate", "other")),
                object_id=object_id,
                description=str(relation.get("description", "")),
                source_chunk_ids=_source_chunk_ids(relation.get("source_chunks", [])),
                evidence=[str(item) for item in relation.get("evidence", []) if str(item).strip()],
            )
        )

    entities = [
        ArchRAGEntity(
            entity_id=str(entity.get("entity_id", "")),
            name=str(entity.get("name", "")),
            type=str(entity.get("type", "Other")),
            description=str(entity.get("description", "")),
            source_chunk_ids=_source_chunk_ids(entity.get("source_chunks", [])),
            relations=relation_ids_by_entity.get(str(entity.get("entity_id", "")), []),
            attributes=dict(entity.get("attributes", {})) if isinstance(entity.get("attributes", {}), dict) else {},
        )
        for entity in entities_payload
        if str(entity.get("entity_id", "")).strip()
    ]
    return entities, relations


def persist_attributed_kg(graph_dir: Path, embeddings) -> Path:
    """Embed entity/relation textual attributes and persist an ArchRAG KG snapshot."""
    entities, relations = load_attributed_kg(graph_dir)
    entity_vectors = _embed_texts(embeddings, [entity.text for entity in entities], "Embedding ArchRAG entity attributes")
    relation_vectors = _embed_texts(
        embeddings,
        [relation.text for relation in relations],
        "Embedding ArchRAG relation attributes",
    )
    for entity, vector in zip(entities, entity_vectors, strict=False):
        entity.embedding = vector
    for relation, vector in zip(relations, relation_vectors, strict=False):
        relation.embedding = vector

    payload = {
        "schema": "paper_assistant_rag.archrag.attributed_kg.v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "entities": [asdict(entity) for entity in entities],
        "relations": [asdict(relation) for relation in relations],
    }
    path = graph_dir / ATTRIBUTED_KG_FILE
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _source_chunk_ids(raw_refs: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(raw_refs, list):
        return ids
    for ref in raw_refs:
        if not isinstance(ref, dict):
            continue
        chunk_id = str(ref.get("stable_chunk_id") or ref.get("chunk_key") or "").strip()
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def _embed_texts(embeddings, texts: list[str], description: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    if not texts:
        return vectors
    with create_progress() as progress:
        task = progress.add_task(description, total=len(texts))
        for start in range(0, len(texts), DEFAULT_EMBED_BATCH_SIZE):
            batch = texts[start : start + DEFAULT_EMBED_BATCH_SIZE]
            vectors.extend([[float(value) for value in vector] for vector in embeddings.embed_documents(batch)])
            progress.advance(task, advance=len(batch))
    return vectors


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, strict=False)
        if isinstance(row, dict):
            rows.append(row)
    return rows
