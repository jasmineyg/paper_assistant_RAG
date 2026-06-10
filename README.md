# 科研知识库与文献问答平台

面向实验室论文资料和科研阅读场景的本地 RAG 助手。你可以把一批 PDF 论文放到本地目录中，系统会自动解析正文、切分 chunk、生成 embedding，并构建 FAISS 向量索引。提问时，它会从论文原文片段中检索证据，生成中文回答，并给出可核验的 `paper / page / chunk` 引用。

项目当前不只是基础向量检索，还加入了 GraphRAG / ArchRAG-like 思路：先从 chunk 中抽取实体和关系，构建论文知识图谱，再通过 graph、community 和层级索引辅助检索。目标不是让图谱替代原文证据，而是帮助系统更好地找到相关论文、方法、任务、数据集、指标和结论，最终回答仍尽量回到原始 chunk 引用，优先保证 chunk 命中率和引用可追溯性。

## 核心功能

- **本地论文知识库**：读取 `data/paper/` 下的 PDF，抽取正文并保留论文名、页码、chunk 编号、稳定 chunk id 等 metadata。
- **精准溯源问答**：回答中使用 `[S1]`、`[S2]` 等来源编号，并在终端展示来源表格和原文片段，方便回到论文核验。
- **混合检索**：融合 FAISS 向量相似度和关键词检索，并使用 RRF 排序，兼顾语义问题、论文名、方法名、缩写和专有术语。
- **GraphRAG 增强检索**：从 chunk 中抽取实体和关系，构建 KG cache，让检索能利用“方法-任务-数据集-指标-结论”等结构化信号。
- **ArchRAG-like community 检索**：基于知识图谱做 community detection 和 community summary 索引，从论文群组、方法簇和研究主题层面辅助召回。
- **层级 ArchRAG 检索**：支持层级 attributed community 构建、Python C-HNSW-like 层级索引、从高层社区到低层实体的 top-down search。
- **候选论文门控 + 精确 chunk 回查**：`archrag-gated` 模式先用 hybrid / graph / community 找候选论文，再在候选论文内部做 chunk 级检索和 rerank，避免只找到对的论文却引用错的片段。
- **连续追问**：使用 SQLite 保存多轮会话历史，支持“它的方法流程是什么？”这类依赖上下文的问题；历史只用于问题改写，不替代本轮检索证据。
- **评测报告**：内置 `eval` 命令，可统计 paper hit、chunk/evidence hit、MRR、answer present、citation present 等指标，并输出 JSON、CSV、Markdown review 和 JSONL review 文件。
- **多模型配置**：支持 SiliconFlow、OpenAI-compatible API、DeepSeek chat 和本地 Ollama，可分别配置 chat 与 embedding 服务。

## 适合解决的问题

- “这篇论文主要解决什么问题，核心贡献是什么？”
- “某个方法的完整流程是什么？输入、处理步骤和输出分别是什么？”
- “哪些论文都在处理同一个任务或场景？它们方法有什么差异？”
- “某组论文用了哪些数据集、指标和 baseline？”
- “某篇论文的实验结论、局限性和未来工作在哪里？”
- “跨论文比较某个方法方向的发展脉络。”

## 工作流程

```text
PDF papers
  -> PDF 解析
  -> chunk 切分与 metadata 生成
  -> embedding
  -> FAISS chunk index

可选 GraphRAG / ArchRAG 构建
  -> chunk 实体/关系抽取
  -> KG cache
  -> graph-assisted retrieval
  -> community detection + summary index
  -> hierarchical attributed community index

用户问题 + 会话历史
  -> history-aware question rewrite
  -> hybrid / graph / community / archrag 检索
  -> source chunk 组织与引用编号
  -> LLM 生成中文答案
  -> SQLite 保存会话历史
```

## 快速开始

安装依赖：

```powershell
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中配置模型服务。默认可使用 SiliconFlow chat + OpenAI-compatible embedding：

```env
LLM_PROVIDER=siliconflow
EMBEDDING_PROVIDER=openai

SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_CHAT_MODEL=Pro/zai-org/GLM-4.7

OPENAI_API_KEY=your_embedding_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBED_MODEL=Pro/BAAI/bge-m3

TEMPERATURE=0.2
```

把论文 PDF 放入：

```text
data/paper/
```

构建基础 FAISS 索引：

```powershell
uv run python main.py index --force
```

开始提问：

```powershell
uv run python main.py ask "MI-GNN 这篇论文主要解决什么问题？"
```

如果索引不存在，`ask` 会自动先构建索引。第一次运行通常较慢，因为需要读取 PDF、切分文本并调用 embedding API；索引构建完成后，后续提问会直接加载本地 FAISS 索引。

## 常用命令

查看当前模型配置：

```powershell
uv run python main.py models
```

重建全部论文索引：

```powershell
uv run python main.py index --force
```

向已有索引追加新论文：

```powershell
uv run python main.py append
```

连续追问：

```powershell
uv run python main.py ask "MI-GNN 这篇论文主要解决什么问题？"
uv run python main.py ask "它的方法流程是什么？"
```

使用独立会话隔离不同主题：

```powershell
uv run python main.py ask "这组论文的共同问题是什么？" --session graph-mil
```

清空当前会话后重新提问：

```powershell
uv run python main.py ask "重新总结这些论文" --reset-memory
```

隐藏原文片段，只显示答案和来源表格：

```powershell
uv run python main.py ask "这篇论文的核心贡献是什么？" --hide-snippets
```

允许参考文献列表参与检索：

```powershell
uv run python main.py ask "这些论文引用了哪些图结构多实例学习相关工作？" --include-references
```

默认会降低 References / 参考文献片段的优先级，避免把被引用论文误当作当前论文的方法；只有当你明确要分析引用关系时，才建议开启 `--include-references`。

## GraphRAG / ArchRAG 使用方式

基础问答只需要 `index`。如果要使用 graph、community 或 ArchRAG-like 检索，需要先构建离线缓存。

### 1. 构建知识图谱缓存

```powershell
uv run python main.py kg-build
```

该命令会读取 FAISS 索引中的 chunk，调用 LLM 抽取实体和关系，并保存到 `data/graph/`。缓存内容包括：

- `chunk_extractions.jsonl`
- `entities.jsonl`
- `relations.jsonl`
- `entity_chunk_links.jsonl`
- `manifest.json`

构建完成后可以使用 KG 辅助检索：

```powershell
uv run python main.py ask "这篇论文用到了哪些关键模块？" --retrieval-mode graph
```

`graph` 模式会把基础 hybrid 检索结果与 KG 实体/关系信号融合，适合方法组件、任务、数据集、指标、结论、局限等结构化问题。

### 2. 构建 community summary 索引

```powershell
uv run python main.py community-build
```

该命令会基于 KG 做单层 community detection，生成 community summary，并为 summary 建立 FAISS 索引。适合高层总结、跨论文比较、研究主题聚类和候选论文召回。

使用 community / archrag-lite 检索：

```powershell
uv run python main.py ask "这些论文主要可以分成哪些研究方向？" --retrieval-mode community
```

`community` 是 `archrag-lite` 的别名。它会检索 community summary，再映射回相关 source chunk，并与 chunk 检索结果融合。默认仍以原文 chunk 作为最终引用证据。

### 3. 使用候选论文门控模式

```powershell
uv run python main.py ask "哪些论文关注图结构多实例学习？" --retrieval-mode archrag-gated --candidate-papers 5 --per-paper-k 5 --show-retrieval-debug
```

`archrag-gated` 是当前更稳的 GraphRAG 使用路径：

```text
query
  -> 判断问题类型
  -> 用 hybrid + graph + community 召回候选论文
  -> 在候选论文内部重新做 chunk 级检索
  -> 对候选 chunk 做轻量 rerank
  -> 使用原文 chunk 生成带引用答案
```

这种模式适合“先找对论文，再找准证据”的问题，能缓解 paper hit 提升但 chunk hit 下降的情况。默认不会把 community summary 直接放进最终上下文；只有高层总结或跨论文对比确实需要摘要证据时，才建议加：

```powershell
uv run python main.py ask "这些论文的共同研究脉络是什么？" --retrieval-mode archrag-gated --include-community-docs
```

### 4. 构建层级 ArchRAG 索引

```powershell
uv run python main.py kg-build
uv run python main.py archrag-build --max-levels 3 --min-nodes-per-level 5 --similarity-top-k 5 --similarity-threshold 0.65 --m-neighbors 8
```

`archrag-build` 会从 KG 出发构建层级 attributed communities，并保存 Python C-HNSW-like 层级索引到 `data/index/archrag/`：

- `hierarchy.json`
- `nodes.jsonl`
- `layers.json`
- `intra_links.json`
- `inter_links.json`
- `build_config.json`

使用层级 ArchRAG 检索：

```powershell
uv run python main.py ask "这组论文围绕哪些核心方法簇展开？" --retrieval-mode archrag --top-k-per-level 5 --show-archrag-debug
```

`archrag` 模式会从最高层 community 开始向下检索到低层实体，经过 adaptive filtering 后合并生成答案。它更接近 ArchRAG 论文中的层级思想，但当前实现仍是工程近似版本：C-HNSW 是 Python 实现，community detection 使用 NetworkX 的 Louvain / greedy / label propagation，效果依赖 KG 抽取质量、embedding 模型和 LLM。

## 检索模式怎么选

| 模式 | 适合问题 | 是否需要额外构建 |
| --- | --- | --- |
| `hybrid` | 单篇论文问答、具体方法流程、快速检索 | 只需要 `index` |
| `graph` | 方法组件、任务、数据集、指标、结论、局限等结构化问题 | 需要 `kg-build` |
| `community` / `archrag-lite` | 跨论文总结、研究方向、主题聚类、论文群组召回 | 需要 `kg-build` + `community-build` |
| `archrag-gated` | 先定位候选论文，再在论文内找精确 chunk | 建议 `kg-build` + `community-build` |
| `archrag` | 层级社区视角、从高层主题到低层实体的 top-down 检索 | 需要 `kg-build` + `archrag-build` |

默认建议：

```powershell
uv run python main.py ask "问题" --retrieval-mode hybrid
```

如果问题明显是跨论文、跨方法或需要结构化关系，再切到：

```powershell
uv run python main.py ask "问题" --retrieval-mode archrag-gated
```

## 评测

运行内置评测集：

```powershell
uv run python main.py eval --retrieval-mode hybrid
```

评测 GraphRAG / ArchRAG-like 模式：

```powershell
uv run python main.py eval --retrieval-mode graph
uv run python main.py eval --retrieval-mode archrag-gated
uv run python main.py eval --retrieval-mode archrag
```

只看检索，不生成答案：

```powershell
uv run python main.py eval --retrieval-mode archrag-gated --retrieval-only
```

评测输出默认写入 `data/eval/runs/`，包括：

- 完整 JSON 报告
- CSV 指标表
- Markdown 人工 review 文件
- JSONL agent review 文件

重点关注的指标：

- `target_paper_hit_rate`：是否找到了目标论文。
- `all_target_papers_hit_rate`：多目标论文是否全部命中。
- `evidence_hit_rate` / `avg_evidence_recall`：是否找到了证据页或证据片段。
- `exact_evidence_hit_rate` / `chunk_hit@k`：是否命中精确 chunk。
- `mrr_evidence`：证据 chunk 排名是否靠前。
- `citation_present_rate`：答案中是否带 `[S#]` 引用。

注意：回答存在和引用格式正确不等于检索质量好。对这个项目来说，paper hit、chunk/evidence hit 和 citation traceability 需要分开看。

## 配置说明

SiliconFlow chat：

```env
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_CHAT_MODEL=Pro/zai-org/GLM-4.7
TEMPERATURE=0.2
```

OpenAI-compatible chat / embedding：

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_CHAT_MODEL=your-chat-model
OPENAI_EMBED_MODEL=your-embedding-model
TEMPERATURE=0.2
```

本地 Ollama：

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:1.5b
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
```

切换 embedding 模型或 embedding 服务后，旧 FAISS 索引不能继续复用，需要重建：

```powershell
uv run python main.py index --force
```

## 技术栈

- Python
- LangChain
- FAISS
- SQLite
- NetworkX
- Typer
- Rich
- PyPDF / PyMuPDF
- SiliconFlow / OpenAI-compatible API / DeepSeek / Ollama

## 代码结构

```text
main.py                         CLI 入口
paper_assistant_rag/
  cli.py                        Typer 命令定义
  documents.py                  PDF 读取、metadata 生成、文本切分
  indexing.py                   FAISS 索引构建、追加、保存、加载
  retrieval.py                  hybrid 检索、keyword 检索、RRF 融合、引用清理
  graph_retrieval.py            KG 辅助 chunk 检索
  kg.py                         实体/关系抽取，KG cache 生成
  communities.py                community detection，summary 与 community FAISS index
  community_retrieval.py        community summary 检索和 evidence filtering
  archrag_gated.py              候选论文门控与二阶段 chunk rerank
  archrag_types.py              ArchRAG 层级索引数据结构
  archrag_hierarchy.py          层级 attributed community 构建
  archrag_index.py              Python C-HNSW-like 索引与 top-down search
  archrag_generation.py         adaptive filtering 与最终答案合并
  qa.py                         带会话记忆的 RAG 问答主流程
  memory.py                     SQLite 对话历史
  evaluation.py                 paper / chunk / citation 评测与报告
  settings.py                   环境变量和模型配置
  models.py                     LLM / embedding 模型工厂
  ui.py                         Rich 终端输出和来源表格
  paths.py                      默认目录和路径配置
```

## 当前边界

- 当前版本以命令行交互为主，尚未提供 Web UI。
- PDF 文本抽取质量依赖原始论文格式；扫描版 PDF 需要额外 OCR 支持。
- KG 和 community 质量依赖 LLM 抽取、embedding 模型和论文 chunk 质量。
- `archrag` 是对 ArchRAG 思想的工程化近似，不是论文原实现的完全复现。
- 对最终答案正确性仍建议结合人工 review；自动评测主要覆盖检索、证据命中和引用结构。
