# 领域论文知识库助手

这是一个最小可运行的论文 RAG 项目，用 `data/paper` 里的 PDF 论文作为知识库。

当前版本已经支持：

- 读取 `data/paper` 中的 PDF 论文；
- 将论文文本切分成 chunk；
- 调用 OpenAI-compatible embedding API 生成向量；
- 用 FAISS 在本地保存向量索引；
- 根据问题检索相关论文片段；
- 调用 OpenAI-compatible chat API 回答问题；
- 返回答案对应的来源论文、页码、chunk 编号和原文片段；
- 用 SQLite 保存 `ask` 的会话历史，支持连续追问和上下文代词；
- 保留 DeepSeek chat 和本机 Ollama 的可选接口。

## 当前默认模型

当前 `.env` 默认使用 OpenAI-compatible API：

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=你的_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

如果你用的是兼容 OpenAI 协议的第三方服务，把 `OPENAI_BASE_URL` 改成服务商给的 `/v1` 地址，并把 `OPENAI_CHAT_MODEL`、`OPENAI_EMBED_MODEL` 改成该服务支持的模型名即可。

## 环境说明

项目使用：

- Python 3.12
- `uv`
- LangChain
- OpenAI-compatible API
- FAISS
- PyPDF / PyMuPDF

关键环境文件：

- `.python-version`：告诉 `uv` 使用 Python 3.12；
- `pyproject.toml`：记录项目依赖；
- `uv.lock`：锁定依赖版本；
- `.venv/`：当前已经安装好的虚拟环境。

如果要重新同步依赖，可以运行：

```powershell
uv sync
```

说明：Codex 沙箱里不能直接写工作区外的 `D:\download\uv` 缓存目录，所以我在排查时临时用过 `--no-cache`。你在自己的 PowerShell 里运行时不需要这个参数。

## 代码结构

`main.py` 现在只保留命令行入口，具体逻辑拆到 `paper_assistant_rag/` 包中：

- `cli.py`：Typer 命令定义；
- `settings.py`：`.env` 配置读取；
- `models.py`：LLM 和 embedding 模型创建；
- `documents.py`：PDF 读取、metadata 生成和文本切分；
- `indexing.py`：FAISS 索引构建、保存和加载；
- `retrieval.py`：检索结果筛选、上下文拼接和回答清理；
- `memory.py`：SQLite 对话历史读写；
- `qa.py`：带对话记忆的问题改写、检索和回答流程；
- `ui.py`：Rich 终端输出、进度条和来源表格。

建议阅读顺序：

1. 先看 `main.py`，理解程序入口只负责启动 Typer CLI；
2. 再看 `paper_assistant_rag/cli.py`，理解 `index`、`ask`、`models` 三个命令分别调用哪些函数；
3. 重点看 `paper_assistant_rag/qa.py`，这里是提问主流程：检查索引、加载向量库、结合历史改写问题、检索片段、组装 prompt、调用模型、打印答案；
4. 然后看 `paper_assistant_rag/indexing.py`，理解建索引流程：读取 PDF、切 chunk、生成 embedding、保存 FAISS；
5. 再分别看 `documents.py`、`retrieval.py`、`models.py`、`settings.py`，它们是主流程调用的支撑模块；
6. 最后看 `ui.py` 和 `paths.py`，它们主要是终端展示和默认路径配置。

如果只想抓住主线，优先读：

```text
main.py -> paper_assistant_rag/cli.py -> paper_assistant_rag/qa.py
```

## 常用命令

查看当前模型配置：

```powershell
uv run python main.py models
```

重建论文索引：

```powershell
uv run python main.py index --force
```

提问：

```powershell
uv run python main.py ask "2019年的图神经网络多实例学习论文主要方法是什么？"
```

`ask` 默认会使用 `default` 会话保存上下文，因此可以直接连续追问：

```powershell
uv run python main.py ask "MI-GNN 这篇论文主要解决什么问题？"
uv run python main.py ask "它的方法流程是什么？"
```

如果要隔离不同主题，可以换一个会话名：

```powershell
uv run python main.py ask "这组论文的共同问题是什么？" --session graph-mil
```

如果要重新开始当前会话：

```powershell
uv run python main.py ask "重新总结这些论文" --reset-memory
```

如果索引还不存在，`ask` 会自动先建立索引。

## 运行速度说明

第一次运行 `ask` 可能比较慢，因为程序会先做一遍建索引：

1. 读取 PDF；
2. 切分论文文本；
3. 调用 embedding API 给每个 chunk 生成向量；
4. 保存 FAISS 索引；
5. 再检索并调用 chat 模型回答。

其中最慢的一般是第 3 步，也就是生成 embedding。索引建好以后，再次提问会跳过建索引，速度会快很多。

当前命令行里已经加入了进度提示：

- 读取 PDF：显示 PDF 读取进度；
- 生成 embedding：显示 chunk 处理进度；
- 加载索引、检索片段、生成回答：显示运行状态提示。

注意：模型服务生成回答时没有准确百分比，所以这里只显示“正在调用模型生成回答”的状态提示，而不是百分比进度条。

默认情况下，检索会尽量把论文 References / 参考文献列表排到后面，避免模型把“被引用论文”误当成“当前论文的方法”。如果你在做 Related Work 检索，需要包含参考文献片段，可以加：

```powershell
uv run python main.py ask "这些论文引用了哪些图结构多实例学习相关工作？" --include-references
```

## 模型 API 配置

如果你的 API 同时支持 chat 和 embedding，推荐统一走 OpenAI-compatible 配置：

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=你的_api_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_CHAT_MODEL=your-chat-model
OPENAI_EMBED_MODEL=your-embedding-model
```

如果只想把聊天模型切换到 DeepSeek，在 `.env` 里改成：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
```

DeepSeek 主要用于 chat；如果不再使用本机 Ollama，embedding 仍需要另配一个 OpenAI-compatible embedding 接口：

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=你的_api_key
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
OPENAI_EMBED_MODEL=your-embedding-model
```

切换 embedding 模型或 embedding 服务后，旧 FAISS 索引不能继续复用，需要重建：

```powershell
uv run python main.py index --force
```

如果以后还想临时切回本机 Ollama，可以把 `.env` 改回：

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:1.5b
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
```

## 关于 uv cache

项目目录中可能看到这些目录：

- `.uv-cache/`
- `.uv-cache-local/`
- `.uv-cache-run/`

它们都是排查 Codex 沙箱写缓存受限时生成的临时缓存目录，不是项目运行必需品。当前项目真正使用的是 `.venv/` 里的虚拟环境，以及 `uv.lock` 里的依赖锁定信息。

如果只想保持项目干净，可以删除这些 `.uv-cache*` 目录；以后 `uv` 需要缓存时，会继续使用你用户环境变量里的 `D:\download\uv`。
