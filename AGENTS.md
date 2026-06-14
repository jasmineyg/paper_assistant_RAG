# AGENTS.md

## 项目目的

paper_assistant_RAG 已从普通 Hybrid RAG 重构为 ArchRAG-style 学术论文 RAG 助手。核心目标：

1. 构建本地论文知识库：PDF -> chunks -> KG -> attributed communities -> hierarchical index。
2. 默认问答流程使用 ArchRAG：多层 entity/community 检索 + adaptive filtering-based generation。
3. 输出可追溯引用：paper / page / chunk / stable_chunk_id。
4. 保留 hybrid / graph / community-gated 作为 baseline，但不作为默认主流程。
5. 优先保证 chunk 命中率、citation 可追溯性，以及 entity/community 命中调试信息。

开发原则：

- 功能优先，架构服务于论文流程。
- 每次修改必须考虑对 paper hit / chunk hit / citation / entity hit / community hit 的影响。
- 不要把新功能塞进 `main.py`；CLI 只放命令编排，核心逻辑放模块。
- memory 仅辅助问题改写，不可替代本轮 ArchRAG 检索。
- `stable_chunk_id` 不可随意改动。
- 查询改写、术语识别和 rerank 必须采用可迁移的通用方法；禁止把评测集中的论文名、作者、方法简称或答案映射硬编码进运行时检索逻辑。
- 修改 ArchRAG 流程、数据结构或命令时，必须同步维护本文件。

## ArchRAG 对齐流程

### Offline Indexing

当前默认离线阶段：

```text
PDF / Paper Corpus
-> Chunking
-> LLM Entity & Relation Extraction
-> Knowledge Graph Construction
-> Entity / Relation Attribute Embedding
-> Attribute-aware Graph Augmentation
-> Weighted Community Detection
-> Community Summary Generation
-> Iterative Hierarchical Community Construction
-> Hierarchical Index Construction
```

对应入口：

- `uv run python main.py archrag-index`：完整离线构建，从 PDF 到层级索引。
- `uv run python main.py index`：仅构建 chunk FAISS baseline。
- `uv run python main.py kg-build`：仅构建 KG cache。
- `uv run python main.py archrag-build`：基于已有 KG 构建 hierarchical attributed communities 和 C-HNSW-like index。

### Online Retrieval

当前默认在线阶段：

```text
Query Embedding
-> Parent-constrained top-down beam search over entities + communities
-> Query-aware source chunk reranking
-> Per-level adaptive filtering reports
-> Ranked evidence selection
-> Final Answer Generation
```

默认在线检索参数：

- 每层展示 `top_k_per_level=5` 个节点，向下扩展 beam width 为 `3`。
- 子层候选仅来自当前 beam 父节点的 `children` / inter-layer link；仅在层级链接缺失时回退到层内搜索。
- 子节点路径分数使用 `0.85 * local semantic score + 0.15 * parent route score`。
- 最终 chunk 使用 `0.65 * chunk semantic + 0.25 * hierarchy route + 0.10 * keyword` 重排。
- 多样性约束默认为每个层级节点最多 `3` 个 chunk、每篇论文最多 `6` 个 chunk；候选不足时再放宽约束补齐。
- 以上均为在线逻辑修改，可直接复用已有 FAISS、KG、hierarchy 和 node embedding，不要求重建离线索引。

对应入口：

- `uv run python main.py ask "问题"`：默认 `--retrieval-mode archrag`。
- `uv run python main.py eval`：默认 `--retrieval-mode archrag`。
- `uv run streamlit run app/streamlit_app.py`：本机浏览器 UI，默认调用 ArchRAG service，可切换 Baseline Hybrid RAG。
- baseline 需要显式指定：`--retrieval-mode hybrid`、`graph`、`archrag-lite` 或 `archrag-gated`。

## 文件导航与关键入口

### 核心命令入口

- `main.py`：Typer 入口，不放功能逻辑。
- `paper_assistant_rag/cli.py`：命令定义：`index` / `append` / `kg-build` / `community-build` / `archrag-index` / `archrag-build` / `ask` / `eval` / `models`。

### PDF、chunk 与 baseline index

- `paper_assistant_rag/documents.py`：PDF 解析、chunk 切分、metadata 与 `stable_chunk_id`。
- `paper_assistant_rag/indexing.py`：chunk FAISS baseline index 构建、追加、加载。
- `paper_assistant_rag/paths.py`：数据路径管理。

### ArchRAG package

- `paper_assistant_rag/archrag/pipeline.py`：完整离线 ArchRAG pipeline 编排。
- `paper_assistant_rag/archrag/kg_builder.py`：ArchRAG entity/relation textual attributes，`archrag_attributed_kg.json` 持久化。
- `paper_assistant_rag/archrag/attributed_graph.py`：attribute-aware graph augmentation。
- `paper_assistant_rag/archrag/community_detection.py`：weighted community detection facade。
- `paper_assistant_rag/archrag/community_summary.py`：LLM community summary facade。
- `paper_assistant_rag/archrag/hierarchy_builder.py`：Algorithm 1 风格迭代层级 community 构建。
- `paper_assistant_rag/archrag/hierarchical_index.py`：C-HNSW-like hierarchical index facade。
- `paper_assistant_rag/archrag/hierarchical_retriever.py`：多层 entity/community 检索。
- `paper_assistant_rag/archrag/adaptive_filter.py`：adaptive filtering report + final merge generation。

### ArchRAG 底层兼容模块

- `paper_assistant_rag/archrag_types.py`：`ArchNode`、`ArchLayer`、`ArchIndex` 及 JSON persistence。
- `paper_assistant_rag/archrag_hierarchy.py`：层级 attributed community 构建实现。
- `paper_assistant_rag/archrag_index.py`：C-HNSW-like intra/inter links 与 hierarchical search。
- `paper_assistant_rag/archrag_generation.py`：adaptive filtering-based generation 和 source backprojection。

### KG 与旧 community baseline

- `paper_assistant_rag/kg.py`：LLM 实体/关系抽取，KG cache。
- `paper_assistant_rag/communities.py`：single-level KG community baseline。

### Baseline retrieval

- `paper_assistant_rag/retrieval.py`：hybrid 检索、keyword 检索、RRF 融合。
- `paper_assistant_rag/graph_retrieval.py`：graph 扩展检索 baseline。
- `paper_assistant_rag/community_retrieval.py`：single-level community augmented baseline。
- `paper_assistant_rag/archrag_gated.py`：paper gate baseline。

### QA、记忆、评测

- `paper_assistant_rag/qa.py`：问答流程；默认 ArchRAG，旧 conversational RAG 仅用于 baseline。
- `paper_assistant_rag/service.py`：非 CLI 调用门面；Streamlit UI 通过它调用 ArchRAG / Hybrid pipeline，不在页面中重写 RAG 逻辑。
- `paper_assistant_rag/memory.py`：会话历史管理，只用于追问理解。
- `paper_assistant_rag/evaluation.py`：评测 paper / chunk hit rate，并输出 ArchRAG level/entity/community/debug 字段。
- `paper_assistant_rag/ui.py`：Rich 终端输出。
- `app/streamlit_app.py`：最小本机网络访问 UI，展示回答、source chunks、entity/community 层级结果和 adaptive filtering report。

### 模型与配置

- `paper_assistant_rag/settings.py`：LLM / embedding provider 配置。
- `paper_assistant_rag/models.py`：LLM / embedding 构建。
- `.env.example`：API key 和环境变量示例。

## 重要类和函数

- `ArchRAGPipeline.build()`：完整离线构建。
- `build_kg_cache()`：chunk -> entity/relation extraction。
- `persist_attributed_kg()`：entity/relation textual attributes + embeddings snapshot。
- `build_hierarchical_communities()`：迭代 attributed community hierarchy。
- `build_archrag_index()`：C-HNSW-like layer links。
- `hierarchical_search()`：父节点约束的 top-down beam retrieval。
- `rerank_archrag_chunks()`：回到原始 FAISS chunk，执行 query-aware 语义 / 路径 / 关键词重排和多样性约束。
- `generate_archrag_answer()`：hierarchical beam search -> chunk rerank -> adaptive filtering -> final answer。
- `PaperAssistantService.ask()`：UI / 非 CLI 入口的稳定问答接口，返回 answer、sources、retrieval、filter_reports 和 metadata。
- `run_evaluation()`：评测并输出 retrieval/debug/report artifacts。

## 数据产物

- `vectorstore/faiss_index/`：chunk FAISS baseline index，仍用于 chunk store 和 baseline。
- `data/graph/chunk_extractions.jsonl`：每个 chunk 的 LLM extraction cache。
- `data/graph/entities.jsonl`：KG entities。
- `data/graph/relations.jsonl`：KG relations。
- `data/graph/archrag_attributed_kg.json`：ArchRAG textual attributes + embeddings snapshot。
- `data/index/archrag/hierarchy.json`：层级 entity/community tree。
- `data/index/archrag/nodes.jsonl`：所有层级节点。
- `data/index/archrag/intra_links.json`：C-HNSW-like intra-layer links。
- `data/index/archrag/inter_links.json`：C-HNSW-like inter-layer links。
- `data/index/archrag/pipeline_manifest.json`：完整离线构建 manifest。
- `data/eval/runs/`：评测 JSON/CSV/Markdown/JSONL。

## 修改建议

1. 先判断任务属于哪一层：chunk、KG、attribute graph、community、hierarchy、index、retrieval、generation、evaluation。
2. 修改检索或生成前，明确它影响的是 paper hit、chunk hit、entity/community hit、adaptive filtering，还是 citation。
3. 不要把普通 hybrid retrieval 重新设为默认主流程；它只能作为 baseline。
4. 层级节点必须保留 source chunk 回溯信息，最终答案引用必须能落回 paper/page/chunk。
5. 若新增中间产物，必须可持久化并在缺失时有清晰错误或自动构建路径。
