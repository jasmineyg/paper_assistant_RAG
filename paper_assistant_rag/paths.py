"""Project path constants, default data directories, and indexing batch sizes."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_DIR = ROOT_DIR / "data" / "paper"
DEFAULT_INDEX_DIR = ROOT_DIR / "vectorstore" / "faiss_index"
DEFAULT_MEMORY_DB = ROOT_DIR / "data" / "chat_history.sqlite"
DEFAULT_EVAL_DATASET = ROOT_DIR / "data" / "eval" / "graph_mil_core_qa_v1.json"
DEFAULT_EVAL_RUN_DIR = ROOT_DIR / "data" / "eval" / "runs"
DEFAULT_EMBED_BATCH_SIZE = 16
