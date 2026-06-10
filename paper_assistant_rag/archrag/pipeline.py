"""End-to-end ArchRAG offline indexing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paper_assistant_rag.archrag.kg_builder import ATTRIBUTED_KG_FILE, ensure_kg_cache, persist_attributed_kg
from paper_assistant_rag.archrag_hierarchy import build_archrag_hierarchy_cache
from paper_assistant_rag.archrag_index import build_archrag_index
from paper_assistant_rag.indexing import build_index, index_exists
from paper_assistant_rag.models import build_embeddings
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console


@dataclass
class ArchRAGPipeline:
    """Coordinate ArchRAG offline indexing from PDFs to hierarchical index."""

    paper_dir: Path
    index_dir: Path
    graph_dir: Path
    archrag_dir: Path
    chunk_size: int = 1000
    chunk_overlap: int = 180
    max_levels: int = 3
    min_nodes_per_level: int = 5
    similarity_top_k: int = 5
    similarity_threshold: float = 0.65
    m_neighbors: int = 8
    community_algorithm: str = "louvain"
    max_chars_per_chunk: int = 2500
    extraction_concurrency: int = 12
    summary_concurrency: int = 12

    def build(self, force: bool = False, kg_limit: int | None = None) -> dict[str, object]:
        """Run chunking, KG extraction, hierarchy construction, and C-HNSW-like indexing."""
        settings = Settings.from_env()
        if force or not index_exists(self.index_dir):
            console.print("[bold]ArchRAG offline step 1/5: PDF chunking and chunk index[/bold]")
            build_index(
                paper_dir=self.paper_dir,
                index_dir=self.index_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                force=force,
            )
        else:
            console.print(f"[cyan]Reusing chunk index:[/cyan] {self.index_dir}")

        console.print("[bold]ArchRAG offline step 2/5: LLM entity/relation KG construction[/bold]")
        ensure_kg_cache(
            index_dir=self.index_dir,
            graph_dir=self.graph_dir,
            limit=kg_limit,
            force=force,
            max_chars_per_chunk=self.max_chars_per_chunk,
            concurrency=self.extraction_concurrency,
        )

        console.print("[bold]ArchRAG offline step 3/5: entity/relation attribute embedding snapshot[/bold]")
        attributed_kg_path = persist_attributed_kg(self.graph_dir, build_embeddings(settings))

        console.print("[bold]ArchRAG offline step 4/5: iterative hierarchical attributed communities[/bold]")
        hierarchy = build_archrag_hierarchy_cache(
            graph_dir=self.graph_dir,
            archrag_dir=self.archrag_dir,
            max_levels=self.max_levels,
            min_nodes_per_level=self.min_nodes_per_level,
            similarity_top_k=self.similarity_top_k,
            similarity_threshold=self.similarity_threshold,
            community_algorithm=self.community_algorithm,
            summary_concurrency=self.summary_concurrency,
        )

        console.print("[bold]ArchRAG offline step 5/5: C-HNSW-like hierarchical index[/bold]")
        arch_index = build_archrag_index(
            hierarchy=hierarchy,
            m_neighbors=self.m_neighbors,
            archrag_dir=self.archrag_dir,
        )
        summary = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "index_dir": str(self.index_dir),
            "graph_dir": str(self.graph_dir),
            "archrag_dir": str(self.archrag_dir),
            "attributed_kg": str(attributed_kg_path),
            "attributed_kg_file": ATTRIBUTED_KG_FILE,
            "levels": len(arch_index.layers),
            "layer_node_counts": {str(level): len(layer.nodes) for level, layer in arch_index.layers.items()},
            "m_neighbors": self.m_neighbors,
            "default_online_flow": "query_embedding -> hierarchical_search -> adaptive_filtering_reports -> final_answer",
        }
        self.archrag_dir.mkdir(parents=True, exist_ok=True)
        (self.archrag_dir / "pipeline_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


def build_archrag_pipeline(**kwargs) -> dict[str, object]:
    """Build the complete ArchRAG offline index with keyword arguments."""
    return ArchRAGPipeline(**kwargs).build()


__all__ = ["ArchRAGPipeline", "build_archrag_pipeline"]
