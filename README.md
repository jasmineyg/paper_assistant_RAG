# Paper Assistant RAG

面向领域论文阅读的命令行 RAG 系统。项目可以把本地 PDF 论文构建为可检索知识库，基于论文片段回答问题，并返回可追溯的来源论文、页码、chunk 编号和原文片段。

这个项目重点展示的是一个完整 RAG 应用从 0 到 1 的工程实现：PDF 文档解析、文本切分、embedding 生成、FAISS 向量索引、混合检索、对话记忆、Prompt 编排、多模型配置和命令行交互。

## 项目亮点

- **本地论文知识库**：读取 `data/paper` 下的 PDF 文件，抽取正文并保留论文名、页码、chunk 编号等 metadata。
- **向量索引构建**：使用 embedding 模型将论文片段向量化，并将 FAISS 索引持久化到本地，后续提问无需重复建库。
- **增量更新索引**：支持向已有 FAISS 索引追加新论文，默认跳过已索引 PDF，减少重复 embedding 成本。
- **混合检索策略**：结合向量相似度检索和关键词检索，并用 RRF 排序融合，提高论文名、方法名、缩写等查询的命中率。
- **参考文献干扰控制**：默认降低 References / 参考文献片段的优先级，避免模型把“被引用论文”误当作“当前论文方法”。
- **可追溯回答**：回答要求带 `[S1]`、`[S2]` 等来源引用，并在终端展示来源表格和原文片段，方便核验模型输出。
- **连续追问能力**：使用 SQLite 保存会话历史，支持“它的方法流程是什么？”这类依赖上下文的追问。
- **多模型兼容**：支持 SiliconFlow、OpenAI-compatible chat / embedding API，也保留 DeepSeek chat 和本地 Ollama 的可选接口。

## 系统流程

```text
PDF papers
  -> PDF loader
  -> text chunks with metadata
  -> embedding model
  -> FAISS vector index

user question + chat history
  -> history-aware question rewrite
  -> vector + keyword hybrid retrieval
  -> reference filtering and source formatting
  -> LLM answer generation
  -> answer with citations + source snippets
  -> SQLite chat memory
```

### Graph-enhanced retrieval modes

The project now keeps two graph-enhanced retrieval families separate:

1. `community` / `archrag-lite`
   - Single-level KG community detection.
   - Community summary index.
   - Community source-chunk expansion.
   - RRF-style fusion with chunk retrieval.
   - Useful as an engineering baseline, but not the full ArchRAG paper structure.

2. `archrag`
   - Hierarchical attributed communities.
   - Python C-HNSW-like hierarchical index.
   - Top-down hierarchical search from the highest community layer to level 0 entities.
   - Adaptive filtering over each layer's retrieved nodes.
   - Final answer merging with source chunk citations.
   - Closer to the ArchRAG paper's structure than `archrag-lite`.

Build the full hierarchy after `kg-build`:

```powershell
uv run python main.py kg-build
uv run python main.py archrag-build --max-levels 3 --min-nodes-per-level 5 --similarity-top-k 5 --similarity-threshold 0.65 --m-neighbors 8
```

Ask with the paper-style hierarchical implementation:

```powershell
uv run python main.py ask "问题" --retrieval-mode archrag --top-k-per-level 5 --show-archrag-debug
```

Known approximations versus the original paper:

- C-HNSW is a simplified Python implementation, not a high-performance C++/FAISS C-HNSW implementation.
- Community detection uses NetworkX weighted Louvain/greedy/label propagation as a practical approximation to weighted Leiden-style clustering.
- Embedding and LLM behavior depends on the configured providers and models.
- Retrieval and answer quality still need validation through `eval --retrieval-mode archrag`.

Intermediate files are saved under `data/index/archrag/`:

- `hierarchy.json`
- `nodes.jsonl`
- `layers.json`
- `intra_links.json`
- `inter_links.json`
- `build_config.json`

### ArchRAG-gated legacy experiment

当前项目不是完整复现 ArchRAG，而是新增了一个更稳的
`archrag-gated` / community-gated 检索模式：

```text
query
  -> detect query type
  -> retrieve candidate papers by hybrid + graph + community
  -> retrieve precise chunks inside candidate papers
  -> rerank candidate chunks
  -> final answer with original chunk citations
```

KG/community 的作用是提高 paper-level recall：它们帮助判断哪些论文值得进入候选集合，但不会默认把 community summary 或 community source chunks 直接塞进最终 answer context。进入候选论文后，系统会在这些论文内部重新做 chunk-level 精确检索和轻量 rerank，最终优先使用原文 chunk 作为 `[S1]`、`[S2]` 引用证据。这能缓解“paper hit 上升但 chunk hit 下降”的问题。

示例：

```powershell
uv run python main.py kg-build
uv run python main.py community-build
uv run python main.py ask "问题" --retrieval-mode archrag-gated --candidate-papers 5 --per-paper-k 5
```

默认不会把 community summary 放进最终上下文。只有在高层总结/对比类问题中确实需要少量社区摘要时，才建议显式加上 `--include-community-docs`。

## 技术栈

- LangChain
- FAISS
- Typer
- Rich
- SQLite
- PyPDF / PyMuPDF
- SiliconFlow / OpenAI-compatible API / DeepSeek / Ollama

## 快速开始

安装依赖：

```powershell
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中配置模型服务。默认配置使用 SiliconFlow chat API 和 OpenAI-compatible embedding API：

```env
LLM_PROVIDER=siliconflow
EMBEDDING_PROVIDER=openai
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_CHAT_MODEL=Pro/zai-org/GLM-4.7
OPENAI_API_KEY=your_embedding_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBED_MODEL=Pro/BAAI/bge-m3
```

把 PDF 论文放入：

```text
data/paper/
```

构建索引：

```powershell
uv run python main.py index --force
```

开始提问：

```powershell
uv run python main.py ask "MI-GNN 这篇论文主要解决什么问题？"
```

如果索引不存在，`ask` 命令会自动先构建索引。第一次运行通常较慢，因为需要读取 PDF、切分文本并调用 embedding API；索引构建完成后，后续提问会直接加载本地 FAISS 索引。

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

清空当前会话记忆后再提问：

```powershell
uv run python main.py ask "重新总结这些论文" --reset-memory
```

保留参考文献片段参与检索：

```powershell
uv run python main.py ask "这些论文引用了哪些图结构多实例学习相关工作？" --include-references
```

隐藏原文片段，只显示答案和来源表格：

```powershell
uv run python main.py ask "这篇论文的核心贡献是什么？" --hide-snippets
```

## 配置说明

如果使用 SiliconFlow chat API：

```env
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_CHAT_MODEL=Pro/zai-org/GLM-4.7
TEMPERATURE=0.2
```

如果服务同时支持 chat 和 embedding，也可以统一使用 OpenAI-compatible 配置：

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_CHAT_MODEL=your-chat-model
OPENAI_EMBED_MODEL=your-embedding-model
TEMPERATURE=0.2
```

如果使用本地 Ollama：

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

## 代码结构

```text
main.py                         CLI 入口
paper_assistant_rag/
  cli.py                        Typer 命令定义
  settings.py                   环境变量和模型配置
  models.py                     LLM / embedding 模型工厂
  documents.py                  PDF 读取、metadata 生成、文本切分
  indexing.py                   FAISS 索引构建、追加、保存、加载
  retrieval.py                  混合检索、参考文献过滤、回答清理
  graph_retrieval.py            KG 辅助 chunk 检索
  community_retrieval.py        community summary 检索和可选证据过滤
  archrag_gated.py              ArchRAG-lite 候选论文门控和二阶段 chunk rerank
  archrag_types.py              ArchRAG hierarchy node/layer/index data structures
  archrag_hierarchy.py          hierarchical attributed community construction
  archrag_index.py              Python C-HNSW-like index and top-down search
  archrag_generation.py         adaptive filtering and final answer merge
  memory.py                     SQLite 对话历史
  qa.py                         带记忆的 RAG 问答主流程
  ui.py                         Rich 终端输出和来源表格
  paths.py                      默认目录和批处理参数
```

## 工程能力体现

- 将 RAG 流程拆成文档处理、索引管理、模型适配、检索后处理、对话记忆和 UI 输出等独立模块，便于维护和扩展。
- 在检索阶段加入关键词召回、RRF 融合和参考文献过滤，而不是只依赖基础向量相似度。
- 对回答结果做引用约束和兜底处理，让用户可以从答案直接回到原始论文片段。
- 使用 `.env` 管理模型服务配置，避免把 provider、API key、模型名硬编码在业务逻辑中。
- 使用 SQLite 持久化会话状态，使命令行应用也能支持多轮问答。

## 当前边界

- 当前版本以命令行交互为主，尚未提供 Web UI。
- PDF 文本抽取质量依赖原始论文格式；扫描版 PDF 需要额外 OCR 支持。
- 目前还没有系统化评测集，检索质量主要通过人工问题和来源核验验证。
