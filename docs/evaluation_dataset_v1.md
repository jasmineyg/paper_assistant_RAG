# Graph MIL Core QA V1

This document explains the first evaluation set for the local Paper Assistant RAG project.

## Goal

The dataset is designed to evaluate final answer quality, paper-reading ability, and fuzzy retrieval under natural user wording. Each item provides:

- a natural `question` for RAG input;
- a precise `canonical_question` for scoring/debugging only;
- alias terms that the system should be able to connect to the target paper;
- target core papers;
- expected answer points;
- source/page/chunk evidence from the current FAISS docstore;
- negative checks for common hallucinations or shallow answers.

The dataset file is:

```text
data/eval/graph_mil_core_qa_v1.json
```

## Query Design

Use `question` as the actual evaluation input. It intentionally looks like a real user query: abbreviations, partial method names, mixed Chinese/English, missing author/year, and informal wording are allowed.

Do not use `canonical_question` as model input. That field preserves the precise intent so a human evaluator can understand what the item is testing.

Example:

- natural: `GAMIL 为什么能拿来做结直肠癌分期？`
- canonical: `Raju 2020 的 Graph Attention MIL 为什么用于 colorectal cancer staging？`

## Running Evaluation

Run the full review evaluation:

```powershell
uv run python main.py eval
```

By default this generates final model answers and writes four files to `data/eval/runs/`:

- `*_review.md`: human-readable Markdown report for manual review.
- `*_review.jsonl`: one JSON object per item, suitable for Codex or scripts to inspect.
- `*.json`: full structured run record.
- `*.csv`: compact metrics table.

For a quick smoke test:

```powershell
uv run python main.py eval --limit 3
```

To compute retrieval metrics only, without calling the chat model:

```powershell
uv run python main.py eval --retrieval-only
```

To compare natural fuzzy questions with precise questions:

```powershell
uv run python main.py eval --query-field question
uv run python main.py eval --query-field canonical_question
```

Use a larger `k` for cross-paper questions:

```powershell
uv run python main.py eval --k 12
```

## How To Review The Markdown

Open the generated `*_review.md`. Each item contains:

- the natural user question;
- the canonical intent;
- automatic retrieval checks;
- the model's final answer;
- expected answer points as checkboxes;
- negative checks as checkboxes;
- gold evidence;
- top retrieved sources.

For each item, compare the model answer against the expected answer points and tick off what is covered. Penalize unsupported claims, wrong paper attribution, missing citations, and answers that only restate the method title.

## Automatic Metrics

The evaluator reports these retrieval metrics:

- `target_paper_hit_rate`: whether at least one target paper appears in top-k.
- `all_target_papers_hit_rate`: whether all target papers appear in top-k, mainly useful for cross-paper questions.
- `exact_evidence_hit_rate`: whether at least one gold source/page/chunk appears in top-k.
- `avg_target_paper_recall`: fraction of target papers retrieved on average.
- `avg_evidence_recall`: fraction of gold evidence chunks retrieved on average.
- `mrr_target_paper`: mean reciprocal rank of the first target-paper hit.
- `mrr_evidence`: mean reciprocal rank of the first exact evidence hit.

When answer generation is enabled, it also reports:

- `answer_present_rate`: whether the answer text is non-empty.
- `citation_present_rate`: whether the answer contains `[S1]`, `[S2]`, etc.
- `avg_answer_length`: average answer length.

These automatic answer metrics only check structure. Answer correctness still needs human review or a separate judge model.

## Expansion Rules

When adding V2 items:

- keep one question per row/item;
- prefer questions that require reasoning over title lookup;
- include at least one evidence chunk for every target paper;
- add negative checks for likely confusions;
- avoid requiring facts that are only in the survey PDF unless the survey is added to `data/paper`.

Good question types:

- single-paper method summary;
- method workflow;
- method comparison;
- application reasoning;
- critical reading or limitation analysis;
- cross-paper taxonomy.
