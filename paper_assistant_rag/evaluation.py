"""Evaluation runner for curated RAG QA datasets."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from httpx import HTTPError
from openai import OpenAIError
from rich.table import Table

from paper_assistant_rag.indexing import load_index
from paper_assistant_rag.memory import clear_session_history
from paper_assistant_rag.paths import DEFAULT_EVAL_RUN_DIR
from paper_assistant_rag.qa import (
    MAX_CHARS_PER_SOURCE,
    build_conversational_rag_chain,
    source_rows_from_documents,
)
from paper_assistant_rag.retrieval import (
    hybrid_search_with_score,
    normalize_text,
    select_retrieval_results,
)
from paper_assistant_rag.settings import Settings
from paper_assistant_rag.ui import console


def run_evaluation(
    dataset_path: Path,
    index_dir: Path,
    memory_db: Path,
    output_dir: Path,
    k: int,
    limit: int | None,
    query_field: str,
    include_references: bool,
    with_answers: bool,
    session_prefix: str,
) -> None:
    dataset = _load_dataset(dataset_path)
    items = list(dataset["items"])
    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError("Evaluation dataset has no items to run.")

    settings = Settings.from_env()
    with console.status("[cyan]Loading FAISS index...[/cyan]", spinner="dots"):
        vectorstore = load_index(index_dir, settings)

    answer_chain = None
    if with_answers:
        answer_chain = build_conversational_rag_chain(
            vectorstore=vectorstore,
            settings=settings,
            memory_db=memory_db,
            k=k,
            include_references=include_references,
        )

    paper_sources = {
        str(paper["paper_id"]): str(paper["source"])
        for paper in dataset.get("selected_papers", [])
    }

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if _is_multi_turn_item(item):
            console.print(f"[cyan]({index}/{len(items)})[/cyan] {item['id']}: multi-turn")
            row = _evaluate_multi_turn_item(
                item=item,
                vectorstore=vectorstore,
                paper_sources=paper_sources,
                answer_chain=answer_chain,
                memory_db=memory_db,
                session_id=f"{session_prefix}-{item['id']}",
                query_field=query_field,
                k=k,
                include_references=include_references,
            )
        else:
            query = _item_query(item, query_field)
            console.print(f"[cyan]({index}/{len(items)})[/cyan] {item['id']}: {query}")
            row = _evaluate_single_turn_item(
                item=item,
                query=query,
                vectorstore=vectorstore,
                paper_sources=paper_sources,
                answer_chain=answer_chain,
                memory_db=memory_db,
                session_id=f"{session_prefix}-{item['id']}",
                k=k,
                include_references=include_references,
            )
        rows.append(row)

    summary = _summarize(rows, k=k, query_field=query_field, with_answers=with_answers)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(dataset, query_field=query_field, k=k)
    json_path = output_dir / f"{run_id}.json"
    csv_path = output_dir / f"{run_id}.csv"
    markdown_path = output_dir / f"{run_id}_review.md"
    jsonl_path = output_dir / f"{run_id}_review.jsonl"
    payload = {
        "run_id": run_id,
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("version"),
        "dataset_path": str(dataset_path),
        "index_dir": str(index_dir),
        "query_field": query_field,
        "k": k,
        "include_references": include_references,
        "with_answers": with_answers,
        "summary": summary,
        "items": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown_report(markdown_path, payload)
    _write_review_jsonl(jsonl_path, payload)
    _print_summary(summary, json_path, csv_path, markdown_path, jsonl_path)


def _load_dataset(dataset_path: Path) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError(f"Invalid evaluation dataset: {dataset_path}")
    return data


def _item_query(item: dict[str, Any], query_field: str) -> str:
    value = item.get(query_field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Item {item.get('id', '<unknown>')} has no query field: {query_field}")
    return value.strip()


def _is_multi_turn_item(item: dict[str, Any]) -> bool:
    turns = item.get("turns")
    return isinstance(turns, list) and bool(turns)


def _evaluate_single_turn_item(
    item: dict[str, Any],
    query: str,
    vectorstore,
    paper_sources: dict[str, str],
    answer_chain,
    memory_db: Path,
    session_id: str,
    k: int,
    include_references: bool,
    reset_answer_memory: bool = True,
) -> dict[str, Any]:
    retrieval_rows = _retrieve_rows(
        vectorstore=vectorstore,
        query=query,
        k=k,
        include_references=include_references,
    )
    row = {
        "id": str(item["id"]),
        "question": query,
        "canonical_question": str(item.get("canonical_question", "")),
        "alias_terms": list(item.get("alias_terms", [])),
        "type": str(item.get("type", "")),
        "category": str(item.get("category", "")),
        "difficulty": str(item.get("difficulty", "")),
        "target_papers": list(item.get("target_papers", [])),
        "expected_answer_points": list(item.get("expected_answer_points", [])),
        "evidence": list(item.get("evidence", [])),
        "negative_checks": list(item.get("negative_checks", [])),
        "retrieved": retrieval_rows,
        "retrieval_metrics": _score_retrieval(item, retrieval_rows, paper_sources),
    }

    if answer_chain is not None:
        row.update(
            _answer_item(
                chain=answer_chain,
                query=query,
                session_id=session_id,
                memory_db=memory_db,
                reset_memory=reset_answer_memory,
            )
        )

    return row


def _evaluate_multi_turn_item(
    item: dict[str, Any],
    vectorstore,
    paper_sources: dict[str, str],
    answer_chain,
    memory_db: Path,
    session_id: str,
    query_field: str,
    k: int,
    include_references: bool,
) -> dict[str, Any]:
    if answer_chain is not None:
        clear_session_history(session_id=session_id, db_path=memory_db)

    turn_rows: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(item["turns"], start=1):
        turn_item = _turn_item(item, turn, turn_index)
        query = _item_query(turn_item, query_field)
        console.print(f"  [dim]turn {turn_index}:[/dim] {query}")
        turn_rows.append(
            _evaluate_single_turn_item(
                item=turn_item,
                query=query,
                vectorstore=vectorstore,
                paper_sources=paper_sources,
                answer_chain=answer_chain,
                memory_db=memory_db,
                session_id=session_id,
                k=k,
                include_references=include_references,
                reset_answer_memory=False,
            )
        )

    return {
        "id": str(item["id"]),
        "type": str(item.get("type", "multi_turn")),
        "category": str(item.get("category", "")),
        "difficulty": str(item.get("difficulty", "")),
        "conversation_expectation": str(item.get("conversation_expectation", "")),
        "turns": turn_rows,
    }


def _turn_item(parent: dict[str, Any], turn: dict[str, Any], turn_index: int) -> dict[str, Any]:
    turn_id = str(turn.get("turn_id", turn_index))
    return {
        "id": f"{parent['id']}:{turn_id}",
        "parent_id": str(parent["id"]),
        "turn_id": turn_id,
        "turn_index": turn_index,
        "question": str(turn.get("question", "")),
        "canonical_question": str(turn.get("canonical_question", "")),
        "alias_terms": _merge_lists(parent.get("alias_terms", []), turn.get("alias_terms", [])),
        "type": str(turn.get("type", parent.get("type", ""))),
        "category": str(turn.get("category", parent.get("category", ""))),
        "difficulty": str(turn.get("difficulty", parent.get("difficulty", ""))),
        "target_papers": list(turn.get("target_papers", [])),
        "expected_answer_points": list(turn.get("expected_answer_points", [])),
        "evidence": list(turn.get("evidence", [])),
        "negative_checks": list(turn.get("negative_checks", [])),
    }


def _merge_lists(*values: Any) -> list[Any]:
    merged: list[Any] = []
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            if item not in merged:
                merged.append(item)
    return merged


def _retrieve_rows(vectorstore, query: str, k: int, include_references: bool) -> list[dict[str, Any]]:
    raw_results = hybrid_search_with_score(vectorstore, query, k=max(k * 5, k + 10))
    selected_results = select_retrieval_results(
        raw_results,
        k=k,
        include_references=include_references,
    )
    rows: list[dict[str, Any]] = []
    for rank, (doc, score) in enumerate(selected_results, start=1):
        metadata = doc.metadata
        rows.append(
            {
                "rank": rank,
                "source": str(metadata.get("source", "unknown")),
                "page": str(metadata.get("page", "?")),
                "chunk_id": str(metadata.get("chunk_id", "?")),
                "score": float(score),
                "snippet": normalize_text(doc.page_content)[:280],
            }
        )
    return rows


def _score_retrieval(
    item: dict[str, Any],
    retrieved_rows: list[dict[str, Any]],
    paper_sources: dict[str, str],
) -> dict[str, Any]:
    evidence_keys = {_evidence_key(evidence) for evidence in item.get("evidence", [])}
    target_sources = {
        paper_sources[paper_id]
        for paper_id in item.get("target_papers", [])
        if paper_id in paper_sources
    }
    retrieved_keys = [_row_key(row) for row in retrieved_rows]
    retrieved_sources = [str(row["source"]) for row in retrieved_rows]

    if not target_sources and not evidence_keys:
        return {
            "target_paper_hit": None,
            "all_target_papers_hit": None,
            "exact_evidence_hit": None,
            "target_paper_recall": None,
            "evidence_recall": None,
            "first_target_paper_rank": None,
            "first_evidence_rank": None,
            "mrr_target_paper": None,
            "mrr_evidence": None,
            "target_sources": [],
            "retrieved_target_sources": [],
            "expected_evidence_count": 0,
            "retrieved_evidence_count": 0,
            "expected_no_answer": True,
        }

    evidence_ranks = [
        rank
        for rank, key in enumerate(retrieved_keys, start=1)
        if key in evidence_keys
    ]
    target_paper_ranks = [
        rank
        for rank, source in enumerate(retrieved_sources, start=1)
        if source in target_sources
    ]
    retrieved_evidence = set(retrieved_keys).intersection(evidence_keys)
    retrieved_target_sources = set(retrieved_sources).intersection(target_sources)

    return {
        "target_paper_hit": bool(target_paper_ranks),
        "all_target_papers_hit": target_sources.issubset(retrieved_target_sources)
        if target_sources
        else False,
        "exact_evidence_hit": bool(evidence_ranks),
        "target_paper_recall": _safe_ratio(len(retrieved_target_sources), len(target_sources)),
        "evidence_recall": _safe_ratio(len(retrieved_evidence), len(evidence_keys)),
        "first_target_paper_rank": min(target_paper_ranks) if target_paper_ranks else None,
        "first_evidence_rank": min(evidence_ranks) if evidence_ranks else None,
        "mrr_target_paper": 1.0 / min(target_paper_ranks) if target_paper_ranks else 0.0,
        "mrr_evidence": 1.0 / min(evidence_ranks) if evidence_ranks else 0.0,
        "target_sources": sorted(target_sources),
        "retrieved_target_sources": sorted(retrieved_target_sources),
        "expected_evidence_count": len(evidence_keys),
        "retrieved_evidence_count": len(retrieved_evidence),
    }


def _answer_item(
    chain,
    query: str,
    session_id: str,
    memory_db: Path,
    reset_memory: bool,
) -> dict[str, Any]:
    if reset_memory:
        clear_session_history(session_id=session_id, db_path=memory_db)
    try:
        result = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
    except (OpenAIError, HTTPError) as error:
        return {
            "answer_error": str(error),
            "answer": "",
            "answer_sources": [],
            "answer_metrics": {
                "answer_present": False,
                "citation_present": False,
                "answer_length": 0,
            },
        }

    answer = str(result.get("answer", ""))
    answer_sources = source_rows_from_documents(
        result.get("context", []),
        max_chars_per_source=MAX_CHARS_PER_SOURCE,
    )
    return {
        "answer": answer,
        "answer_sources": answer_sources,
        "answer_metrics": {
            "answer_present": bool(answer.strip()),
            "citation_present": bool(re.search(r"\[S\d+\]", answer)),
            "answer_length": len(answer),
        },
    }


def _summarize(
    rows: list[dict[str, Any]],
    k: int,
    query_field: str,
    with_answers: bool,
) -> dict[str, Any]:
    scored_rows = list(_iter_scored_rows(rows))
    retrieval_metrics = [row["retrieval_metrics"] for row in scored_rows]
    summary = {
        "items": len(rows),
        "turns": len(scored_rows),
        "k": k,
        "query_field": query_field,
        "target_paper_hit_rate": _mean_bool(retrieval_metrics, "target_paper_hit"),
        "all_target_papers_hit_rate": _mean_bool(retrieval_metrics, "all_target_papers_hit"),
        "exact_evidence_hit_rate": _mean_bool(retrieval_metrics, "exact_evidence_hit"),
        "avg_target_paper_recall": _mean_float(retrieval_metrics, "target_paper_recall"),
        "avg_evidence_recall": _mean_float(retrieval_metrics, "evidence_recall"),
        "mrr_target_paper": _mean_float(retrieval_metrics, "mrr_target_paper"),
        "mrr_evidence": _mean_float(retrieval_metrics, "mrr_evidence"),
    }
    if with_answers:
        answer_metrics = [row.get("answer_metrics", {}) for row in scored_rows]
        summary.update(
            {
                "answer_present_rate": _mean_bool(answer_metrics, "answer_present"),
                "citation_present_rate": _mean_bool(answer_metrics, "citation_present"),
                "avg_answer_length": _mean_float(answer_metrics, "answer_length"),
            }
        )
    return summary


def _iter_scored_rows(rows: list[dict[str, Any]]):
    for row in rows:
        if "turns" in row:
            yield from row["turns"]
        else:
            yield row


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "canonical_question",
        "type",
        "category",
        "difficulty",
        "target_paper_hit",
        "all_target_papers_hit",
        "exact_evidence_hit",
        "target_paper_recall",
        "evidence_recall",
        "first_target_paper_rank",
        "first_evidence_rank",
        "mrr_target_paper",
        "mrr_evidence",
        "citation_present",
        "answer_length",
        "retrieved_sources",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in _iter_scored_rows(rows):
            metrics = row["retrieval_metrics"]
            answer_metrics = row.get("answer_metrics", {})
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "canonical_question": row["canonical_question"],
                    "type": row["type"],
                    "category": row["category"],
                    "difficulty": row["difficulty"],
                    "target_paper_hit": metrics["target_paper_hit"],
                    "all_target_papers_hit": metrics["all_target_papers_hit"],
                    "exact_evidence_hit": metrics["exact_evidence_hit"],
                    "target_paper_recall": metrics["target_paper_recall"],
                    "evidence_recall": metrics["evidence_recall"],
                    "first_target_paper_rank": metrics["first_target_paper_rank"],
                    "first_evidence_rank": metrics["first_evidence_rank"],
                    "mrr_target_paper": metrics["mrr_target_paper"],
                    "mrr_evidence": metrics["mrr_evidence"],
                    "citation_present": answer_metrics.get("citation_present"),
                    "answer_length": answer_metrics.get("answer_length"),
                    "retrieved_sources": " | ".join(
                        f"{item['rank']}:{item['source']}#p{item['page']}c{item['chunk_id']}"
                        for item in row["retrieved"]
                    ),
                }
            )


def _write_markdown_report(markdown_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# RAG Evaluation Review: {payload['run_id']}",
        "",
        "## Run",
        "",
        f"- Dataset: `{payload.get('dataset_id')}` v`{payload.get('dataset_version')}`",
        f"- Query field: `{payload.get('query_field')}`",
        f"- Top-k: `{payload.get('k')}`",
        f"- Answers generated: `{payload.get('with_answers')}`",
        f"- Include references: `{payload.get('include_references')}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"| `{_md_escape(key)}` | {_md_escape(_format_metric(value))} |")

    lines.extend(
        [
            "",
            "## Review Guide",
            "",
            "For each item, compare the model answer with the expected answer points and evidence.",
            "Use the checkboxes as a manual review worksheet. The retrieved-source table is truncated for readability; the JSON report keeps the full record.",
            "",
            "Suggested manual score per item:",
            "",
            "- answer correctness: 0-45",
            "- evidence support: 0-30",
            "- academic reading quality: 0-15",
            "- citation format: 0-10",
            "",
            "## Items",
            "",
        ]
    )

    for index, row in enumerate(payload["items"], start=1):
        if "turns" in row:
            lines.extend(
                [
                    f"### {index}. {row['id']} ({row['difficulty']} | {row['category']})",
                    "",
                    "**Conversation expectation:**",
                    "",
                    row.get("conversation_expectation", "_none_") or "_none_",
                    "",
                ]
            )
            for turn_index, turn in enumerate(row["turns"], start=1):
                lines.extend(
                    _review_item_markdown(
                        turn,
                        f"#### Turn {turn_index}. {turn['id']} ({turn['difficulty']} | {turn['category']})",
                    )
                )
            continue

        lines.extend(
            _review_item_markdown(
                row,
                f"### {index}. {row['id']} ({row['difficulty']} | {row['category']})",
            )
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _review_item_markdown(row: dict[str, Any], heading: str) -> list[str]:
    metrics = row["retrieval_metrics"]
    answer_metrics = row.get("answer_metrics", {})
    lines = [
        heading,
        "",
        f"**Question:** {row['question']}",
        "",
        f"**Canonical intent:** {row['canonical_question']}",
        "",
        f"**Aliases:** {_comma_join(row.get('alias_terms', []))}",
        "",
        "**Automatic retrieval checks:**",
        "",
        f"- target paper hit: `{metrics['target_paper_hit']}`",
        f"- all target papers hit: `{metrics['all_target_papers_hit']}`",
        f"- exact evidence hit: `{metrics['exact_evidence_hit']}`",
        f"- target paper recall: `{_format_metric(metrics['target_paper_recall'])}`",
        f"- evidence recall: `{_format_metric(metrics['evidence_recall'])}`",
        f"- first evidence rank: `{metrics['first_evidence_rank']}`",
        "",
        "**Model answer:**",
        "",
        _answer_block(row),
        "",
    ]
    if answer_metrics:
        lines.extend(
            [
                "**Answer structure checks:**",
                "",
                f"- answer present: `{answer_metrics.get('answer_present')}`",
                f"- citation present: `{answer_metrics.get('citation_present')}`",
                f"- answer length: `{answer_metrics.get('answer_length')}`",
                "",
            ]
        )

    lines.extend(["**Expected answer points:**", ""])
    for point in row.get("expected_answer_points", []):
        lines.append(f"- [ ] {point}")
    lines.extend(["", "**Negative checks:**", ""])
    for check in row.get("negative_checks", []):
        lines.append(f"- [ ] {check}")
    lines.extend(["", "**Gold evidence:**", ""])
    for evidence in row.get("evidence", []):
        terms = _comma_join(evidence.get("must_include_terms", []))
        lines.append(
            f"- `{evidence.get('source')}` p.{evidence.get('page')} "
            f"chunk `{evidence.get('chunk_id')}`; terms: {terms}"
        )
    lines.extend(["", "**Retrieved sources (top 5):**", ""])
    lines.extend(_retrieved_sources_table(row))
    lines.append("")
    return lines


def _write_review_jsonl(jsonl_path: Path, payload: dict[str, Any]) -> None:
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in payload["items"]:
            if "turns" in row:
                record = {
                    "schema": "paper_assistant_rag.review_conversation.v1",
                    "run_id": payload["run_id"],
                    "dataset_id": payload.get("dataset_id"),
                    "query_field": payload.get("query_field"),
                    "k": payload.get("k"),
                    "id": row["id"],
                    "type": row["type"],
                    "category": row["category"],
                    "difficulty": row["difficulty"],
                    "conversation_expectation": row.get("conversation_expectation", ""),
                    "turns": [_review_jsonl_turn(turn) for turn in row["turns"]],
                }
            else:
                record = _review_jsonl_turn(row)
                record.update(
                    {
                        "schema": "paper_assistant_rag.review_item.v1",
                        "run_id": payload["run_id"],
                        "dataset_id": payload.get("dataset_id"),
                        "query_field": payload.get("query_field"),
                        "k": payload.get("k"),
                    }
                )
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _review_jsonl_turn(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question": row["question"],
        "canonical_question": row["canonical_question"],
        "alias_terms": row.get("alias_terms", []),
        "type": row["type"],
        "category": row["category"],
        "difficulty": row["difficulty"],
        "answer": row.get("answer", ""),
        "answer_error": row.get("answer_error", ""),
        "answer_metrics": row.get("answer_metrics", {}),
        "expected_answer_points": row.get("expected_answer_points", []),
        "negative_checks": row.get("negative_checks", []),
        "gold_evidence": row.get("evidence", []),
        "retrieval_metrics": row["retrieval_metrics"],
        "retrieved": row["retrieved"],
        "manual_review": {
            "answer_correctness_0_to_45": None,
            "evidence_support_0_to_30": None,
            "academic_reading_quality_0_to_15": None,
            "citation_format_0_to_10": None,
            "notes": "",
        },
    }


def _print_summary(
    summary: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    markdown_path: Path,
    jsonl_path: Path,
) -> None:
    table = Table(title="Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, _format_metric(value))
    console.print(table)
    console.print(f"Human review MD: [cyan]{markdown_path}[/cyan]")
    console.print(f"Agent review JSONL: [cyan]{jsonl_path}[/cyan]")
    console.print(f"Full JSON report: [cyan]{json_path}[/cyan]")
    console.print(f"CSV metrics: [cyan]{csv_path}[/cyan]")
    console.print(
        "[yellow]Answer correctness still needs human review or a separate judge model; "
        "automatic metrics here focus on retrieval and citation structure.[/yellow]"
    )


def _answer_block(row: dict[str, Any]) -> str:
    if row.get("answer_error"):
        return f"> Answer generation failed: `{_md_escape(str(row['answer_error']))}`"
    answer = str(row.get("answer", "")).strip()
    if not answer:
        return "_Answer was not generated. Re-run without `--retrieval-only` to include model answers._"
    return answer


def _retrieved_sources_table(row: dict[str, Any]) -> list[str]:
    gold_keys = {_evidence_key(evidence) for evidence in row.get("evidence", [])}
    lines = [
        "| Rank | Gold? | Source | Page | Chunk | Snippet |",
        "| ---: | :---: | --- | ---: | ---: | --- |",
    ]
    for source in row.get("retrieved", [])[:5]:
        row_key = _row_key(source)
        is_gold = "yes" if row_key in gold_keys else ""
        lines.append(
            "| "
            f"{source.get('rank')} | "
            f"{is_gold} | "
            f"{_md_escape(str(source.get('source', '')))} | "
            f"{_md_escape(str(source.get('page', '')))} | "
            f"{_md_escape(str(source.get('chunk_id', '')))} | "
            f"{_md_escape(str(source.get('snippet', '')))} |"
        )
    return lines


def _comma_join(values: list[Any]) -> str:
    if not values:
        return "_none_"
    return ", ".join(f"`{_md_escape(str(value))}`" for value in values)


def _md_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", "<br>")
    )


def _run_id(dataset: dict[str, Any], query_field: str, k: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = str(dataset.get("dataset_id", "eval"))
    return f"{dataset_id}_{query_field}_k{k}_{timestamp}"


def _evidence_key(evidence: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(evidence.get("source", "")),
        str(evidence.get("page", "")),
        str(evidence.get("chunk_id", "")),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source", "")),
        str(row.get("page", "")),
        str(row.get("chunk_id", "")),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return _mean([1.0 if value else 0.0 for value in values])


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return _mean([float(row.get(key)) for row in rows if row.get(key) is not None])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


__all__ = ["DEFAULT_EVAL_RUN_DIR", "run_evaluation"]
