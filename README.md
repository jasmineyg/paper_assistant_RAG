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
- **多模型兼容**：支持 OpenAI-compatible chat / embedding API，也保留 DeepSeek chat 和本地 Ollama 的可选接口。
- **命令行体验**：基于 Typer 和 Rich 实现清晰的 CLI、进度条、状态提示和来源结果表格。

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

## 技术栈

- Python 3.12
- LangChain
- FAISS
- Typer
- Rich
- SQLite
- PyPDF / PyMuPDF
- OpenAI-compatible API / DeepSeek / Ollama

## 快速开始

安装依赖：

```powershell
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中配置模型服务。默认配置使用 OpenAI-compatible API：

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
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

如果服务同时支持 chat 和 embedding，推荐统一使用 OpenAI-compatible 配置：

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_CHAT_MODEL=your-chat-model
OPENAI_EMBED_MODEL=your-embedding-model
TEMPERATURE=0.2
```

如果只想把聊天模型切换到 DeepSeek，可以配置：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
```

此时 embedding 仍需要单独配置，例如继续使用 OpenAI-compatible embedding：

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_EMBED_MODEL=your-embedding-model
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
