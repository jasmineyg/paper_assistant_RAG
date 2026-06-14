from __future__ import annotations

import unittest

import numpy as np
from langchain_core.documents import Document

from paper_assistant_rag.archrag_generation import rerank_archrag_chunks
from paper_assistant_rag.archrag_index import hierarchical_search
from paper_assistant_rag.archrag_types import ArchIndex, ArchLayer, ArchNode


class _FakeEmbeddings:
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector

    def embed_query(self, _query: str) -> list[float]:
        return list(self.query_vector)


class _FakeFaissIndex:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def reconstruct(self, position: int) -> np.ndarray:
        return self.vectors[position]


class _FakeDocstore:
    def __init__(self, documents: dict[str, Document]) -> None:
        self._dict = documents


class _FakeVectorstore:
    def __init__(self, documents: list[Document], vectors: list[list[float]]) -> None:
        ids = [f"doc-{index}" for index in range(len(documents))]
        self.index_to_docstore_id = dict(enumerate(ids))
        self.docstore = _FakeDocstore(dict(zip(ids, documents, strict=True)))
        self.index = _FakeFaissIndex(vectors)


def _node(
    node_id: str,
    level: int,
    embedding: list[float],
    *,
    children: list[str] | None = None,
    source_chunks: list[str] | None = None,
) -> ArchNode:
    return ArchNode(
        node_id=node_id,
        level=level,
        node_type="community" if level else "entity",
        name=node_id,
        text=node_id,
        summary=node_id,
        embedding=embedding,
        children=children or [],
        source_chunks=source_chunks or [],
    )


class ArchRAGRetrievalTests(unittest.TestCase):
    def test_hierarchical_search_only_descends_through_selected_parent_children(self) -> None:
        parent_a = _node("parent-a", 1, [1.0, 0.0], children=["child-a"])
        parent_b = _node("parent-b", 1, [0.0, 1.0], children=["child-b"])
        child_a = _node("child-a", 0, [0.8, 0.2])
        child_b = _node("child-b", 0, [1.0, 0.0])
        index = ArchIndex(
            layers={
                1: ArchLayer(level=1, nodes={"parent-a": parent_a, "parent-b": parent_b}),
                0: ArchLayer(level=0, nodes={"child-a": child_a, "child-b": child_b}),
            },
            entry_node_id="parent-a",
        )

        result = hierarchical_search(
            arch_index=index,
            query="query",
            embeddings=_FakeEmbeddings([1.0, 0.0]),
            top_k_per_level=2,
            beam_width=1,
        )

        self.assertEqual(["parent-a", "parent-b"], [row["node_id"] for row in result["level_results"][1]])
        self.assertEqual(["child-a"], [row["node_id"] for row in result["level_results"][0]])
        self.assertEqual(["parent-a"], result["beam_trace"][1])

    def test_chunk_reranker_prefers_query_semantics_and_limits_one_node_dominance(self) -> None:
        documents = [
            Document(
                page_content="unrelated background",
                metadata={"stable_chunk_id": "c1", "source": "paper-a.pdf", "page": 1, "chunk_id": 1},
            ),
            Document(
                page_content="graph pooling result",
                metadata={"stable_chunk_id": "c2", "source": "paper-a.pdf", "page": 2, "chunk_id": 2},
            ),
            Document(
                page_content="graph pooling ablation",
                metadata={"stable_chunk_id": "c3", "source": "paper-a.pdf", "page": 3, "chunk_id": 3},
            ),
            Document(
                page_content="graph pooling comparison",
                metadata={"stable_chunk_id": "c4", "source": "paper-b.pdf", "page": 4, "chunk_id": 4},
            ),
        ]
        vectorstore = _FakeVectorstore(
            documents,
            vectors=[
                [0.0, 1.0],
                [1.0, 0.0],
                [0.95, 0.05],
                [0.9, 0.1],
            ],
        )
        level_results = {
            1: [
                {
                    "node_id": "node-a",
                    "level": 1,
                    "node_type": "community",
                    "name": "node-a",
                    "score": 0.9,
                    "source_chunks": ["c1", "c2", "c3"],
                    "metadata": {},
                },
                {
                    "node_id": "node-b",
                    "level": 1,
                    "node_type": "community",
                    "name": "node-b",
                    "score": 0.8,
                    "source_chunks": ["c4"],
                    "metadata": {},
                },
            ]
        }

        ranked = rerank_archrag_chunks(
            query="graph pooling",
            level_results=level_results,
            vectorstore=vectorstore,
            embeddings=_FakeEmbeddings([1.0, 0.0]),
            limit=3,
            max_chunks_per_node=2,
            max_chunks_per_paper=4,
        )

        self.assertEqual("c2", ranked[0][0].metadata["stable_chunk_id"])
        self.assertIn("c4", [doc.metadata["stable_chunk_id"] for doc, _score in ranked])
        self.assertLessEqual(
            sum(doc.metadata["archrag_node_id"] == "node-a" for doc, _score in ranked),
            2,
        )

    def test_chunk_reranker_prioritizes_case_sensitive_technical_terms(self) -> None:
        documents = [
            Document(
                page_content="The mi-Graph algorithm is introduced here.",
                metadata={"stable_chunk_id": "generic", "source": "other.pdf", "page": 1, "chunk_id": 1},
            ),
            Document(
                page_content="miGraph and MIGraph use different graph construction strategies.",
                metadata={"stable_chunk_id": "comparison", "source": "target.pdf", "page": 2, "chunk_id": 2},
            ),
        ]
        vectorstore = _FakeVectorstore(documents, vectors=[[1.0, 0.0], [0.8, 0.6]])
        level_results = {
            0: [
                {
                    "node_id": "node",
                    "level": 0,
                    "node_type": "entity",
                    "name": "graph methods",
                    "score": 0.8,
                    "source_chunks": ["generic", "comparison"],
                    "metadata": {},
                }
            ]
        }

        ranked = rerank_archrag_chunks(
            query="miGraph 和 MIGraph 有什么区别？",
            level_results=level_results,
            vectorstore=vectorstore,
            embeddings=_FakeEmbeddings([1.0, 0.0]),
            limit=2,
        )

        self.assertEqual("comparison", ranked[0][0].metadata["stable_chunk_id"])


if __name__ == "__main__":
    unittest.main()
