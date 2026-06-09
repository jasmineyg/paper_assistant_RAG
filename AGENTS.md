# AGENTS.md

## 项目目的

paper_assistant_RAG 是学术论文 RAG 助手，核心目标：
1. 构建本地论文知识库（PDF → chunks → vector/FAISS index）。
2. 支持问答、方法检索、连续追问。
3. 输出可追溯引用（paper / page / chunk）。
4. 支持混合检索、KG + graph、ArchRAG-like community 检索。
5. 优先保证 chunk 命中率和引用可追溯性。

**开发原则**：
- 功能优先，架构次之。
- 每次修改必须考虑对 paper hit / chunk hit / citation 的影响。
- 不要把新功能塞进 main.py。
- memory 仅辅助问题改写，不可替代检索。
- stable_chunk_id 不可随意改动。

---

## 文件导航与关键入口

### 核心命令入口
- `main.py` → 入口，不改功能。
- `cli.py` → 命令定义：index / append / kg-build / community-build / ask / eval / models。

### 数据与索引
- `documents.py` → PDF 解析、chunk 切分、metadata 生成。
- `indexing.py` → FAISS 索引构建与追加。
- `paths.py` → 数据路径管理。

### 检索模块
- `retrieval.py` → hybrid 检索、keyword 检索、RRF 融合。
- `graph_retrieval.py` → graph 扩展检索（依赖 KG）。
- `community_retrieval.py` → ArchRAG-like community 检索。

### KG 与 Community
- `kg.py` → 实体/关系抽取，KG cache。
- `communities.py` → community detection，community summary，community FAISS index。

### 问答与记忆
- `qa.py` → 问答流程，Prompt 定义，retriever 调用。
- `memory.py` → 会话历史管理。

### 模型与配置
- `settings.py` → LLM / embedding provider 配置。
- `models.py` → LLM / embedding 构建。
- `.env.example` → API key 和环境变量示例。

### UI 与评测
- `ui.py` → Rich 终端输出。
- `evaluation.py` → 评测 paper / chunk hit rate，生成报告。

---

## 使用建议

1. 先读本文件，明确任务属于哪一层：
   - PDF/chunk → documents.py / indexing.py
   - 检索逻辑 → retrieval.py / graph_retrieval.py / community_retrieval.py
   - KG 构建 → kg.py
   - Community 构建 → communities.py
   - QA/Prompt → qa.py
   - 会话记忆 → memory.py
   - 模型配置 → settings.py / models.py
   - CLI → cli.py
   - UI/评测 → ui.py / evaluation.py

2. 修改前考虑对检索、chunk 命中率、引用的影响。
3. 优先定位相关模块，避免全仓库扫描。