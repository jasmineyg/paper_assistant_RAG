"""Core data structures and JSON persistence helpers for ArchRAG hierarchy indexes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ArchNode:
    """A node in an ArchRAG hierarchy layer."""

    node_id: str
    level: int
    node_type: str
    name: str
    text: str
    summary: str
    embedding: list[float] | np.ndarray = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert this node to a JSON-serializable dictionary."""
        return {
            "node_id": self.node_id,
            "level": self.level,
            "node_type": self.node_type,
            "name": self.name,
            "text": self.text,
            "summary": self.summary,
            "embedding": self.embedding.tolist()
            if isinstance(self.embedding, np.ndarray)
            else list(self.embedding),
            "source_chunks": list(self.source_chunks),
            "children": list(self.children),
            "parents": list(self.parents),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], compact_embedding: bool = False) -> "ArchNode":
        """Create a node from a JSON dictionary, tolerating older missing keys."""
        raw_embedding = payload.get("embedding", [])
        embedding = (
            np.asarray(raw_embedding, dtype=np.float32)
            if compact_embedding
            else [float(value) for value in raw_embedding]
        )
        return cls(
            node_id=str(payload.get("node_id", "")),
            level=int(payload.get("level", 0)),
            node_type=str(payload.get("node_type", "entity")),
            name=str(payload.get("name", "")),
            text=str(payload.get("text", "")),
            summary=str(payload.get("summary", "")),
            embedding=embedding,
            source_chunks=[str(value) for value in payload.get("source_chunks", [])],
            children=[str(value) for value in payload.get("children", [])],
            parents=[str(value) for value in payload.get("parents", [])],
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )


@dataclass
class ArchLayer:
    """A hierarchy layer plus its intra-layer nearest-neighbor links."""

    level: int
    nodes: dict[str, ArchNode] = field(default_factory=dict)
    intra_links: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self, include_nodes: bool = True) -> dict[str, Any]:
        """Convert this layer to a JSON-serializable dictionary."""
        payload: dict[str, Any] = {
            "level": self.level,
            "node_ids": sorted(self.nodes),
            "node_count": len(self.nodes),
            "intra_links": self.intra_links,
        }
        if include_nodes:
            payload["nodes"] = {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchLayer":
        """Create a layer from a JSON dictionary."""
        nodes_payload = payload.get("nodes", {})
        nodes = {
            str(node_id): ArchNode.from_dict(node_payload)
            for node_id, node_payload in nodes_payload.items()
            if isinstance(node_payload, dict)
        }
        intra_links = {
            str(node_id): [str(target) for target in targets]
            for node_id, targets in payload.get("intra_links", {}).items()
            if isinstance(targets, list)
        }
        return cls(level=int(payload.get("level", 0)), nodes=nodes, intra_links=intra_links)


@dataclass
class ArchIndex:
    """A C-HNSW-like hierarchy index used for top-down ArchRAG search."""

    layers: dict[int, ArchLayer] = field(default_factory=dict)
    inter_links: dict[str, str] = field(default_factory=dict)
    entry_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def max_level(self) -> int:
        """Return the highest available hierarchy level."""
        return max(self.layers) if self.layers else 0

    def find_node(self, node_id: str) -> ArchNode | None:
        """Find a node by id across all layers."""
        for layer in self.layers.values():
            node = layer.nodes.get(node_id)
            if node is not None:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert this index to a JSON-serializable dictionary."""
        return {
            "entry_node_id": self.entry_node_id,
            "inter_links": self.inter_links,
            "metadata": self.metadata,
            "layers": {str(level): layer.to_dict(include_nodes=True) for level, layer in self.layers.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchIndex":
        """Create an index from a JSON dictionary."""
        layers = {
            int(level): ArchLayer.from_dict(layer_payload)
            for level, layer_payload in payload.get("layers", {}).items()
            if isinstance(layer_payload, dict)
        }
        return cls(
            layers=layers,
            inter_links={str(k): str(v) for k, v in payload.get("inter_links", {}).items()},
            entry_node_id=str(payload["entry_node_id"]) if payload.get("entry_node_id") else None,
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )


def save_arch_index(index: ArchIndex, archrag_dir: Path, build_config: dict[str, Any] | None = None) -> None:
    """Persist the hierarchy/index files under data/index/archrag-style storage."""
    archrag_dir.mkdir(parents=True, exist_ok=True)
    (archrag_dir / "hierarchy.json").write_text(
        json.dumps(index.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (archrag_dir / "nodes.jsonl").open("w", encoding="utf-8") as file:
        for level in sorted(index.layers):
            for node in sorted(index.layers[level].nodes.values(), key=lambda item: item.node_id):
                file.write(json.dumps(node.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    layers_payload = {
        str(level): {
            "level": layer.level,
            "node_count": len(layer.nodes),
            "node_ids": sorted(layer.nodes),
        }
        for level, layer in sorted(index.layers.items())
    }
    (archrag_dir / "layers.json").write_text(
        json.dumps(layers_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    intra_links = {str(level): layer.intra_links for level, layer in sorted(index.layers.items())}
    (archrag_dir / "intra_links.json").write_text(
        json.dumps(intra_links, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (archrag_dir / "inter_links.json").write_text(
        json.dumps(index.inter_links, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if build_config is not None:
        build_config = dict(build_config)
        build_config.setdefault("entry_node_id", index.entry_node_id)
        (archrag_dir / "build_config.json").write_text(
            json.dumps(build_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_arch_index(archrag_dir: Path) -> ArchIndex:
    """Load an ArchRAG hierarchy/index from disk."""
    sharded_paths = [
        archrag_dir / "nodes.jsonl",
        archrag_dir / "layers.json",
        archrag_dir / "intra_links.json",
        archrag_dir / "inter_links.json",
    ]
    if all(path.exists() for path in sharded_paths):
        return _load_sharded_arch_index(archrag_dir)

    hierarchy_path = archrag_dir / "hierarchy.json"
    if not hierarchy_path.exists():
        raise FileNotFoundError(
            f"ArchRAG index not found at {archrag_dir}. Run `uv run python main.py archrag-build` first."
        )
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    return ArchIndex.from_dict(payload)


def _load_sharded_arch_index(archrag_dir: Path) -> ArchIndex:
    layers_payload = _read_json(archrag_dir / "layers.json")
    intra_links_payload = _read_json(archrag_dir / "intra_links.json")
    inter_links_payload = _read_json(archrag_dir / "inter_links.json")
    build_config_path = archrag_dir / "build_config.json"
    metadata = _read_json(build_config_path) if build_config_path.exists() else {}

    layers = {
        int(level): ArchLayer(level=int(layer.get("level", level)))
        for level, layer in layers_payload.items()
        if isinstance(layer, dict)
    }
    with (archrag_dir / "nodes.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            node_payload = json.loads(line)
            node = ArchNode.from_dict(node_payload, compact_embedding=True)
            layers.setdefault(node.level, ArchLayer(level=node.level)).nodes[node.node_id] = node

    for level, links in intra_links_payload.items():
        level_number = int(level)
        layer = layers.setdefault(level_number, ArchLayer(level=level_number))
        if isinstance(links, dict):
            layer.intra_links = {
                str(node_id): [str(target) for target in targets]
                for node_id, targets in links.items()
                if isinstance(targets, list)
            }

    entry_node_id = str(metadata.get("entry_node_id") or "") or _read_entry_node_id(
        archrag_dir / "hierarchy.json"
    )
    if not entry_node_id and layers:
        top_layer = layers[max(layers)]
        entry_node_id = _fallback_entry_node(top_layer)

    return ArchIndex(
        layers=layers,
        inter_links={str(key): str(value) for key, value in inter_links_payload.items()},
        entry_node_id=entry_node_id or None,
        metadata=metadata,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_entry_node_id(hierarchy_path: Path) -> str:
    if not hierarchy_path.exists():
        return ""
    with hierarchy_path.open("r", encoding="utf-8") as file:
        header = file.read(16_384)
    match = re.search(r'"entry_node_id"\s*:\s*"([^"]+)"', header)
    return match.group(1) if match else ""


def _fallback_entry_node(layer: ArchLayer) -> str:
    if not layer.nodes:
        return ""
    return sorted(
        layer.nodes.values(),
        key=lambda node: (-len(node.children), -len(node.source_chunks), node.node_id),
    )[0].node_id
