from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from paper_assistant_rag.evaluation import _score_retrieval, _summarize, _write_csv


class EvaluationMetricTests(unittest.TestCase):
    def test_rebuilt_index_metrics_do_not_depend_on_old_chunk_ids(self) -> None:
        item = {
            "target_papers": ["paper-1"],
            "evidence": [
                {
                    "source": "paper.pdf",
                    "page": 2,
                    "chunk_id": 10,
                    "must_include_terms": ["graph", "pooling"],
                }
            ],
        }
        retrieved = [
            {
                "source": "paper.pdf",
                "page": "2",
                "chunk_id": "999",
                "stable_chunk_id": "",
                "text": "The graph pooling module builds the bag representation.",
            }
        ]

        metrics = _score_retrieval(item, retrieved, {"paper-1": "paper.pdf"})

        self.assertTrue(metrics["evidence_hit"])
        self.assertNotIn("exact_evidence_hit", metrics)
        self.assertNotIn("mrr_exact_evidence", metrics)

    def test_summary_and_csv_omit_exact_chunk_metrics(self) -> None:
        metrics = {
            "target_paper_hit": True,
            "all_target_papers_hit": True,
            "evidence_hit": True,
            "target_paper_recall": 1.0,
            "evidence_recall": 1.0,
            "first_target_paper_rank": 1,
            "first_evidence_rank": 1,
            "mrr_target_paper": 1.0,
            "mrr_evidence": 1.0,
        }
        row = {
            "id": "item-1",
            "question": "question",
            "canonical_question": "canonical",
            "type": "single_turn",
            "category": "test",
            "difficulty": "easy",
            "retrieval_metrics": metrics,
            "retrieved": [],
        }

        summary = _summarize([row], k=10, query_field="question", with_answers=False)
        self.assertNotIn("exact_evidence_hit_rate", summary)
        self.assertNotIn("chunk_hit@k", summary)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "metrics.csv"
            _write_csv(csv_path, [row])
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                headers = next(csv.reader(file))

        self.assertFalse(any("exact" in header for header in headers))


if __name__ == "__main__":
    unittest.main()
