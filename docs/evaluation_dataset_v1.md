# Graph MIL RAG 测试集设计说明

对应运行文件：`data/eval/graph_mil_core_qa_v1.json`

当前版本：`0.2.0`

本版已删除旧问题，改为 12 个单轮问题 + 1 个三轮对话，共 15 个真实用户问题。设计重点是：缩写定位、大小写区分、GNN 与 kernel 方法边界、总结质量、跨论文优缺点对比、知识库外拒答、多轮上下文记忆，以及误召回控制。

## 核心论文

| ID | 简称 | 文件 |
| --- | --- | --- |
| p01 | miGraph/MIGraph, 2009 | `zhou2009.pdf.pdf` |
| p02 | GNN-MIL, 2019 | `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` |
| p03 | RGMIL, 2024 | `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` |
| p04 | DSMIL, 2024 | `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` |
| p05 | BGMIL/BGNN-MIL, 2022 | `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` |
| p06 | Patch-GCN, 2021 | `Chen 等 - 2021 - Whole Slide Images are 2D Point Clouds Context-Aware Survival Prediction Using Patch-Based Graph Co.pdf` |
| p07 | NAGCN, 2022 | `Guan 等 - 2022 - Node-aligned graph convolutional network for whole-slide image representation and classification.pdf` |

## 问题清单

### 1. miGraph 和 MIGraph 是一个东西吗？

**测试点：** 大小写缩写区分、方法定位、graph-kernel 与 GNN 边界。

**目标论文：** p01

**得分点：**

- 二者都来自 Zhou 2009，但不是同一个实现细节。
- MIGraph 是显式 bag graph + graph kernel。
- miGraph 是 affinity matrix 隐式建图 + 更高效 graph kernel。
- MIGraph 结构表达直接但计算代价高。
- miGraph 更省计算，适合更大规模数据。

**应召回片段：**

- `zhou2009.pdf.pdf` p.2 chunk `165`
- `zhou2009.pdf.pdf` p.2 chunk `166`
- `zhou2009.pdf.pdf` p.4 chunk `176`
- `zhou2009.pdf.pdf` p.5 chunk `180`
- `zhou2009.pdf.pdf` p.6 chunk `190`

**不应出现：** 把 miGraph/MIGraph 当成 GNN；忽略大小写差异。

### 2. GNN-MIL 指的是哪篇？流程是什么？

**测试点：** 缩写定位、流程总结。

**目标论文：** p02

**得分点：**

- 定位到 Tu 等 2019。
- bag 中实例作为 graph 节点。
- 用启发式距离/阈值构造邻接矩阵。
- GNN 进行 message passing。
- graph aggregation/differentiable pooling 得到 bag embedding 后分类。

**应召回片段：**

- `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` p.1 chunk `1`
- 同上 p.2 chunk `10`
- 同上 p.2 chunk `11`
- 同上 p.2 chunk `12`

**不应出现：** 把 GNN-MIL 归到 Zhou 2009；只说“用了 GNN”。

### 3. RGMIL 是哪篇？解决 GNN-MIL 的什么问题？

**测试点：** 缩写定位、后续改进识别。

**目标论文：** p03

**得分点：**

- 定位到 Zhao 等 2024。
- 关注 GNN-based MIL 中 bag graph 和 GNN 架构因素。
- 边过滤阈值影响 edge density。
- GNN 层数影响 aggregation range 和过平滑风险。
- 用 MADRL 两个 agent 同步搜索 edge threshold 与 GNN layers。

**应召回片段：**

- `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` p.1 chunk `52`
- 同上 p.2 chunk `60`
- 同上 p.2 chunk `62`

**不应出现：** 只说“强化学习提升性能”；说成普通 attention MIL。

### 4. DSMIL 是 double similarities 那篇吗？

**测试点：** 缩写歧义消解，避免把 DSMIL 误认为别的 dual-stream MIL。

**目标论文：** p04

**得分点：**

- 本知识库里用户指定的 DSMIL 是 Double Similarities weighted MIL kernel。
- 核心融合 Bag-to-Bag similarity 和 Instance-to-Bag similarity。
- 构造 kernel function，并接入 SVM。
- 强调同时利用 bag 级和 instance 级信息。

**应召回片段：**

- `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` p.1 chunk `262`
- 同上 p.2 chunk `270`
- 同上 p.3 chunk `282`
- 同上 p.2 chunk `272`

**不应出现：** 把它说成 GNN、attention 或 dual-stream DSMIL。

### 5. BGMIL/BGNN-MIL 连的是哪些对象？

**测试点：** 缩写定位、bag graph 与 bag 内 instance graph 的边界。

**目标论文：** p05

**得分点：**

- 定位到 Pal 等 2022 Bag Graph。
- 关键不是只在一个 bag 内连 instances。
- 它建模 bags 之间的 interactions/dependencies。
- 有意义 bag graph 可能缺失或有噪声。
- Bayesian GNN 用来处理 observed graph 的不确定性。

**应召回片段：**

- `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` p.1 chunk `1128`
- 同上 p.1 chunk `1131`
- 同上 p.3 chunk `1143`

**不应出现：** 把 BGMIL 简化成 Tu 2019 式 bag 内构图。

### 6. 核心论文里哪些方法用了图神经网络？

**测试点：** 方法族检索；GNN 方法应召回，graph kernel 不应混入。

**目标论文：** p02, p03, p05, p06, p07

**得分点：**

- GNN-MIL/Tu 2019。
- RGMIL/Zhao 2024。
- BGMIL/BGNN-MIL/Pal 2022。
- Patch-GCN/Chen 2021。
- NAGCN/Guan 2022。
- 可以补充 corpus 中其他真实 GNN 论文，但 miGraph/MIGraph 和 DSMIL 不能出现为 GNN。

**应召回片段：**

- `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` p.1 chunk `1`
- `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` p.1 chunk `52`
- `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` p.1 chunk `1128`
- `Chen 等 - 2021 - Whole Slide Images are 2D Point Clouds Context-Aware Survival Prediction Using Patch-Based Graph Co.pdf` p.1 chunk `589`
- `Guan 等 - 2022 - Node-aligned graph convolutional network for whole-slide image representation and classification.pdf` p.1 chunk `723`

**不应出现：** 只因标题含 graph 就列入 GNN。

### 7. 哪些是图核或核方法，不是 GNN？

**测试点：** 误召回控制，与第 6 题互为反向测试。

**目标论文：** p01, p04

**得分点：**

- Zhou 2009 的 MIGraph/miGraph 是 graph-kernel 路线。
- DSMIL 2024 是 Double Similarities weighted MIL kernel/SVM。
- 应明确排除 GNN-MIL、RGMIL、BGMIL、Patch-GCN、NAGCN。

**应召回片段：**

- `zhou2009.pdf.pdf` p.2 chunk `166`
- `zhou2009.pdf.pdf` p.4 chunk `176`
- `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` p.1 chunk `262`
- 同上 p.2 chunk `272`

**不应出现：** 把 GNN 方法放入 kernel 列表。

### 8. 总结 DSMIL 2024 的贡献和局限

**测试点：** 单篇论文总结，不只翻译标题。

**目标论文：** p04

**得分点：**

- 背景：已有 MIL kernel 往往只利用部分 bag/instance 信息。
- 贡献：融合 I2B 和 B2B。
- 技术路径：构造有效 kernel 并用于 SVM。
- 实验：消融显示组合模块优于单独模块。
- 局限：仍是 kernel/SVM，测试时相似度计算更慢，未来希望融合 deep learning。

**应召回片段：**

- `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` p.1 chunk `262`
- 同上 p.2 chunk `270`
- 同上 p.6 chunk `299`
- 同上 p.9 chunk `315`
- 同上 p.11 chunk `321`

**不应出现：** 总结成 GNN 或 attention 模型。

### 9. miGraph/MIGraph 和 GNN-MIL 怎么比？

**测试点：** 跨论文优缺点比较。

**目标论文：** p01, p02

**得分点：**

- miGraph/MIGraph 是 kernel/SVM 范式，改造 bag similarity。
- MIGraph 显式构图但计算代价高；miGraph 更高效。
- GNN-MIL 是 neural message passing 范式，端到端学习 graph representation。
- GNN-MIL 受启发式构图、聚合策略和图质量影响。
- 不能把二者归成同一种 GNN 方法。

**应召回片段：**

- `zhou2009.pdf.pdf` p.2 chunk `165`
- `zhou2009.pdf.pdf` p.4 chunk `176`
- `zhou2009.pdf.pdf` p.5 chunk `180`
- `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` p.1 chunk `1`
- 同上 p.2 chunk `12`

**不应出现：** 只按年份比较，不比较方法范式。

### 10. RGMIL 和 BGMIL 如何处理图不确定性？

**测试点：** 相近概念精读比较。

**目标论文：** p03, p05

**得分点：**

- RGMIL 关注 edge threshold 与 GNN layers 的同步选择。
- RGMIL 用 MADRL 两个 agent 搜索结构与架构动作。
- BGMIL 关注 bag 之间依赖图可能缺失、有噪声或不可靠。
- BGMIL 用 Bayesian GNN 处理 observed graph 与真实关系之间的不确定性。
- 二者都不把初始图当固定真理，但问题定义不同。

**应召回片段：**

- `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` p.1 chunk `52`
- 同上 p.2 chunk `62`
- `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` p.1 chunk `1128`
- 同上 p.3 chunk `1143`

**不应出现：** 把二者都泛泛说成“自动学习图结构”。

### 11. RGMIL 和 DSMIL 都是 2024，优缺点差别是什么？

**测试点：** 同年份不同范式比较，防止方法混淆。

**目标论文：** p03, p04

**得分点：**

- RGMIL 是 reinforced GNN framework。
- DSMIL 是 Double Similarities weighted MIL kernel/SVM。
- RGMIL 优点是结构-架构同步控制，代价是强化学习搜索复杂度。
- DSMIL 优点是 I2B+B2B kernel 设计清晰。
- DSMIL 局限是相似度计算和测试时间，未来方向是融合 deep learning。

**应召回片段：**

- `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` p.1 chunk `52`
- 同上 p.13 chunk `135`
- `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` p.1 chunk `262`
- 同上 p.9 chunk `315`
- 同上 p.11 chunk `321`

**不应出现：** 把 DSMIL 写成 RGMIL，或把 RGMIL 写成 kernel SVM。

### 12. GraphMIL-Transformer++ 2026 在知识库里讲了什么？

**测试点：** 知识库外拒答，防止乱编。

**目标论文：** 无

**得分点：**

- 应说明当前知识库没有找到这篇论文或无法确认。
- 可以请求用户提供 PDF、标题或 DOI。
- 可以说明检索到的相近 graph MIL 论文不能作为该论文证据。
- 不得编造作者、方法模块、数据集或实验数字。

**应召回片段：** 无。该题的核心不是命中证据，而是正确拒答。

**不应出现：** 将 RGMIL、Patch-GCN、TAD-Graph 等相近论文冒充为该论文。

## 多轮上下文测试

`gmil-v2-013` 必须按顺序在同一个 session 中运行。第 2 轮“其”和第 3 轮“这个思路”都应该指代第 1 轮识别出的 GNN-MIL/Tu 2019。

### 13.1 MIL-GNN 的流程是什么？

**测试点：** 用户把 GNN-MIL 写成 MIL-GNN，模型是否能模糊匹配。

**目标论文：** p02

**得分点：**

- 将 MIL-GNN 关联到 GNN-MIL/Tu 2019。
- bag-to-graph：实例作为节点。
- 启发式构造 adjacency matrix。
- GNN message passing。
- graph aggregation/differentiable pooling 后分类。

**应召回片段：**

- `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` p.1 chunk `1`
- 同上 p.2 chunk `10`
- 同上 p.2 chunk `11`
- 同上 p.2 chunk `12`

### 13.2 总结其优缺点。

**测试点：** 上下文代词“其”。

**目标论文：** p02

**得分点：**

- 明确“其”指 GNN-MIL。
- 优点：显式建模 instance 关系。
- 优点：GNN message passing 和端到端 bag representation。
- 优点：比传统 graph-kernel 表示学习更灵活。
- 局限：启发式构图、图质量、聚合策略会影响结果。

**应召回片段：**

- `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` p.2 chunk `11`
- 同上 p.2 chunk `12`
- 同上 p.3 chunk `14`
- 同上 p.7 chunk `36`

### 13.3 有哪些论文是在这个思路上继续改进的？

**测试点：** 上下文指代“这个思路”，以及后续相关论文检索。

**目标论文：** p03, p05, p06, p07

**得分点：**

- “这个思路”应指 GNN-MIL 的 bag graph + GNN 表示学习路线。
- RGMIL：调节 edge threshold 和 GNN layers。
- BGMIL/BGNN-MIL：把 GNN 用于 bag 之间的 graph，并处理图不确定性。
- Patch-GCN：WSI patch 空间 kNN 图 + GCN。
- NAGCN：global-to-local clustering 和 node alignment。
- 不应把 miGraph/MIGraph 或 DSMIL 当成 GNN-MIL 的神经网络后续。

**应召回片段：**

- `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` p.1 chunk `52`
- 同上 p.3 chunk `66`
- `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` p.1 chunk `1131`
- `Chen 等 - 2021 - Whole Slide Images are 2D Point Clouds Context-Aware Survival Prediction Using Patch-Based Graph Co.pdf` p.1 chunk `589`
- 同上 p.4 chunk `599`
- `Guan 等 - 2022 - Node-aligned graph convolutional network for whole-slide image representation and classification.pdf` p.2 chunk `730`
- 同上 p.2 chunk `731`

## 运行方式

生成带最终答案的人工 review 报告：

```powershell
uv run python main.py eval
```

只看召回指标，不调用生成模型：

```powershell
uv run python main.py eval --retrieval-only
```

快速 smoke test：

```powershell
uv run python main.py eval --limit 3
```

运行后重点看：

- `data/eval/runs/*_review.md`：给人看的逐题回答与证据检查表。
- `data/eval/runs/*_review.jsonl`：给脚本或我继续评估用的结构化 review 数据。
- `data/eval/runs/*.csv`：召回指标简表。
- `data/eval/runs/*.json`：完整运行记录。
