"""Project path constants, default data directories, and indexing batch sizes."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_DIR = ROOT_DIR / "data" / "paper"
DEFAULT_INDEX_DIR = ROOT_DIR / "vectorstore" / "faiss_index"
DEFAULT_EMBED_BATCH_SIZE = 16
