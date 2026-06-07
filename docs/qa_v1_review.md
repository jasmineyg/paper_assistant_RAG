# Graph MIL QA v1 Gold Chunk Review

- Dataset: `graph_mil_core_qa_v1`
- Version: `0.2.0`
- Questions / turns: `15`
- Gold evidence entries: `62`
- Unique gold chunks: `31`
- Missing chunks in current FAISS docstore: `0`

用途：把每个测试集问题与它应该召回的 gold chunks 放在一起，便于人工核对检索结果、答案证据和负面检查项。

> 说明：`Source / Page / Chunk` 使用测试集和当前 FAISS docstore 中的原始 metadata；中文翻译为逐 chunk 意译，尽量忠实保留论文原意，不补充外部事实。

## Questions

### 1. gmil-v2-001

- Type: `single_turn`
- Category: `abbreviation_case_sensitivity`
- Difficulty: `hard`

**问题**

miGraph 和 MIGraph 是一个东西吗？它们区别在哪，各自优缺点是什么？

**评分意图**

比较 Zhou 2009 中小写 miGraph 与大写 MIGraph 的方法差异、计算代价和优缺点。

**应覆盖答案点**

- 应明确二者都来自 Zhou 2009，但不是同一个实现细节。
- MIGraph 显式把每个 bag 构造成 graph，并基于节点/边设计 graph kernel。
- miGraph 通过 affinity matrix 隐式建图，并定义更高效的 graph kernel，考虑 clique 信息。
- MIGraph 的优势是结构表达直接，缺点是计算复杂度随边数增加，部分任务上不能在合理时间内返回结果。
- miGraph 的优势是计算更省，复杂度接近 O(n_i n_j)，适合更大规模数据；潜在代价是结构表达更隐式。

**负面检查项**

- 不要把 miGraph 和 MIGraph 混成完全相同的方法。
- 不要把二者说成 GNN 方法；它们是 graph-kernel 路线。
- 不要忽略大小写差异，因为本题专门测试缩写大小写。

**应召回 Chunks**

#### Chunk 1: `zhou2009.pdf.pdf` / page `2` / chunk `165`

- Must include terms: `MIGraph`, `bag`, `graph`

**原文**

```text
should not be treated as i.i.d. samples, and this paper provides a solution. Our basic idea is to regard every bag as an entity to be processed as a whole. There are alternative ways to realize the idea, while in this paper we work by regarding each bag as a graph. McGovern and Jensen (2003) have taken multi-instance learning as a tool to handle relational data where each instance is given as a graph. Here, we are working on proposi- tional data and there is no natural graph. In contrast to having instances as graphs, we regard every bag as a graph and each instance as a node in the graph. 3. The Proposed Methods In this section we propose the MIGraph and miGraph methods. The MIGraph method explicitly maps every bag to an undirected graph and uses a new graph ker- nel to distinguish the positive and negative bags. The miGraph method implicitly constructs graphs by de- riving aﬃnity matrices and deﬁnes an eﬃcient graph kernel considering the clique information.
```

**中文翻译**

这段说明，多实例学习中的实例不应被当作独立同分布样本处理，论文的基本想法是把每个 bag 作为一个整体实体来处理。作者选择把每个 bag 看作一张图：不同于已有工作中“每个 instance 本身就是一张图”的关系数据设定，这里处理的是命题型数据，本来没有天然图结构。因此，论文把每个 bag 映射为图，并把其中每个 instance 作为图节点。本节随后提出两种方法：MIGraph 显式地把每个 bag 映射为无向图，并使用新的 graph kernel 区分正负 bag；miGraph 则通过推导 affinity matrix 隐式构图，并定义一种考虑 clique 信息的高效 graph kernel。

#### Chunk 2: `zhou2009.pdf.pdf` / page `2` / chunk `166`

- Must include terms: `miGraph`, `affinity matrices`, `clique`

**原文**

```text
miGraph method implicitly constructs graphs by de- riving aﬃnity matrices and deﬁnes an eﬃcient graph kernel considering the clique information. Before presenting the details, we give the formal def- inition of multi-instance learning as following. Let X denote the instance space. Given a data set 1250
```

**中文翻译**

这段继续强调 miGraph 的核心：它不是显式构造 bag graph，而是通过 affinity matrix 隐式构图，并定义一个考虑 clique 信息的高效 graph kernel。随后作者准备给出多实例学习的形式化定义：令 X 表示 instance space，并基于给定数据集定义后续问题。

#### Chunk 3: `zhou2009.pdf.pdf` / page `4` / chunk `176`

- Must include terms: `MIGraph`, `computational`

**原文**

```text
tional complexity of kG(Xi, X j) is O(ninj + mimj). The kG clearly satisﬁes all the four major properties that should be considered for a graph kernel deﬁnition (Borgwardt & Kriegel, 2005). 1 Our above design is very simple, but in the next section we can see that the proposed MIGraph method is quite eﬀective. A deﬁciency of MIGraph is that the computational complexity of kG is O(ninj + mimj), dominated by the number of edges. For bags containing a lot of in- stances, there will exist a large number of edges and MIGraph will be hard to execute. So, it is desired to have a method with smaller computational cost. For this purpose, we propose the miGraph method which is simple, eﬃcient but eﬀective. For bag Xi, we can calculate the distance between its instances and derive an aﬃnity matrix W i by com- paring the distances with a threshold δ. For example, if the distance between the instances xia and xiu is smaller than δ, W i’s element at the ath row and uth column, wi
```

**中文翻译**

这段先给出 MIGraph 中图核 kG(Xi, Xj) 的计算复杂度，为 O(n_i n_j + m_i m_j)，并说明该核满足图核定义应考虑的主要性质。作者认为 MIGraph 设计简单且有效，但它的缺点也很明显：复杂度受边数量主导；当 bag 中 instance 很多时，会产生大量边，MIGraph 会很难执行。因此需要计算成本更低的方法。为此作者提出 miGraph，它简单、高效且有效。对每个 bag Xi，miGraph 计算其中 instance 之间的距离，并通过阈值 δ 得到 affinity matrix W_i。

#### Chunk 4: `zhou2009.pdf.pdf` / page `5` / chunk `180`

- Must include terms: `computational complexity`, `O`

**原文**

```text
ab, Wia and Wib should decrease since they become more similar, while other Wiq (q ̸= a, b) should not be aﬀected. It is evident that the computational complexity of kg is as similar as that of the multi-instance kernel shown in Eq. 4, i.e., O(ninj). Note that once the multi-instance kernel is obtained, the Gaussian distances between ev- ery pair of instances have already been calculated, and it is easy to get the W i’s. 4. Experiments 4.1. Benchmark T asks First, we evaluate the proposed MIGraph and mi- Graph methods on ﬁve benchmark data sets popularly used in studies of multi-instance learning, including Musk1, Musk2, Elephant, Fox and Tiger. Musk1 con- tains 47 positive and 45 negative bags, Musk2 contains 39 positive and 63 negative bags, each of the other three data sets contains 100 positive and 100 negative bags. More details of the data sets can be found in (Dietterich et al., 1997; Andrews et al., 2003). We compare MIGraph, miGraph with MI-Kernel
```

**中文翻译**

这段讨论 miGraph 权重设计和计算复杂度。若两个 instance 变得更相似，对应权重 W_ia 和 W_ib 应降低，而其他 W_iq 不应受影响。作者指出，miGraph 的 kernel kg 的计算复杂度与多实例 kernel 类似，即 O(n_i n_j)。因为一旦多实例 kernel 已经得到，所有 instance 对之间的 Gaussian distance 也已计算好，所以很容易得到各个 W_i。随后论文进入实验部分，在 Musk、Elephant、Fox、Tiger 等标准 MIL 数据集上比较 MIGraph、miGraph 和 MI-Kernel 等方法。

#### Chunk 5: `zhou2009.pdf.pdf` / page `6` / chunk `190`

- Must include terms: `reasonable time`, `miGraph`

**原文**

```text
Graph does not return results in a reasonable time, and so we only present the average accuracy with stan- dard deviations of miGraph and MI-Kernel in Table 3, where the best result on each data set is bolded. Pairwise t-tests at 95% signiﬁcance level indicate that, miGraph is signiﬁcantly better than MI-Kernel on all the text categorization data sets. It is impressive that, by examining the detail results we found that if we consider each time of the ten times 10-fold cross val- idation, the number of win/tie/lose of miGraph ver- sus MI-Kernel is 10/0/0 on 16 out of the 20 data sets, 9/0/1 on two data sets ( talk.politics.guns and talk.religion.misc), and 7/2/1 on the other two data sets (alt.atheism and misc.forsale). 4.4. Multi-Instance Regression We also compare MIGraph, miGraph and MI-Kernel on four multi-instance regression data sets, includ- 1254
```

**中文翻译**

这段说明在某些文本分类数据集上，MIGraph 无法在合理时间内返回结果，因此作者只报告 miGraph 和 MI-Kernel 的平均准确率及标准差。95% 显著性水平下的 pairwise t-test 表明，miGraph 在所有文本分类数据集上都显著优于 MI-Kernel。更细的交叉验证结果也显示，miGraph 相对 MI-Kernel 在大多数数据集上几乎全部获胜。随后论文还计划在四个多实例回归数据集上比较 MIGraph、miGraph 和 MI-Kernel。

### 2. gmil-v2-002

- Type: `single_turn`
- Category: `abbreviation_entity_linking`
- Difficulty: `medium`

**问题**

GNN-MIL 指的是哪篇？它的基本流程是什么？

**评分意图**

定位 Tu 2019 GNN-MIL，并说明 bag-to-graph、GNN message passing、graph aggregation 和分类流程。

**应覆盖答案点**

- 应定位到 Tu 等 2019 Multiple instance learning with graph neural networks。
- 流程从 MIL bag 开始，把实例作为节点，将 bag 转换为图。
- 邻接关系可用启发式距离/阈值策略构建。
- 在图上运行 GNN 进行节点间信息传递。
- 通过 graph aggregation 或 differentiable pooling 得到 bag-level embedding，再用 MLP/分类器预测 bag 标签。

**负面检查项**

- 不要把 GNN-MIL 归到 Zhou 2009 的 graph-kernel 方法。
- 不要只回答“用了 GNN”，必须说明 bag 如何变成 graph 以及如何得到 bag 表示。

**应召回 Chunks**

#### Chunk 1: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `1` / chunk `1`

- Must include terms: `GNN`, `MIL`, `each bag as a graph`

**原文**

```text
Multiple instance learning with graph neural networks Ming Tu1 Jing Huang 1 Xiaodong He 1 Bowen Zhou 1 Abstract Multiple instance learning (MIL) aims to learn the mapping between a bag of instances and the bag-level label. In this paper, we propose a new end-to-end graph neural network (GNN) based al- gorithm for MIL: we treat each bag as a graph and use GNN to learn the bag embedding, in order to explore the useful structural information among instances in bags. The ﬁnal graph representation is fed into a classiﬁer for label prediction. Our algorithm is the ﬁrst attempt to use GNN for MIL. We empirically show that the proposed algorithm achieves the state of the art performance on sev- eral popular MIL data sets without losing model interpretability. 1. Introduction Multiple instance learning (MIL) as a weakly-supervised learning algorithm deals with weakly-labeled data, where each data sample (often named as a bag) has multiple in-
```

**中文翻译**

这段是 Tu 等 2019 论文的标题和摘要。论文提出一种用于多实例学习的端到端图神经网络算法：把每个 bag 看作一张图，并用 GNN 学习 bag embedding，从而探索 bag 内 instance 之间有用的结构信息。最终图表示会输入分类器进行标签预测。作者称这是首次尝试将 GNN 用于 MIL，并在多个常用 MIL 数据集上取得当时最优表现，同时保持模型可解释性。

#### Chunk 2: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `10`

- Must include terms: `Graph building`, `GNN`, `Graph aggregation`

**原文**

```text
dimension of node feature. While the mapping from bag space to graph space can be done heuristically (will be introduced in next subsection), the key of graph based MIL is how to learn the mapping from graph space to label space. Graph-level classiﬁcation 1Our code will be published after review. Graph building GNNembd Graph aggregation Graph embeddingMLPPrediction Input bags Figure 1. GNN based MIL framework overview. usually involves deriving a good representation of graphs given variant number of nodes and different graph struc- tures, which requires to reduce the input graph to a ﬁxed- dimensional feature vector. In this paper, we focus on GNN based graph representation learning for MIL, and propose a new angle to solve the MIL problem in the current study. 2.2. Proposed algorithm Figure 1 illustrate the diagram of our proposed framework on GNN based MIL. First, to convert input bags of instances to graphs, we adopt a heuristic strategy similar with the one
```

**中文翻译**

这段说明 GNN-based MIL 的整体框架。bag space 到 graph space 的映射可以用启发式策略完成，但 graph-based MIL 的关键是如何学习从 graph space 到 label space 的映射。图级分类需要把节点数量和图结构都可能变化的输入图，压缩成固定维度的特征向量。图 1 的流程是：Input bags -> Graph building -> GNN embedding -> Graph aggregation -> Graph embedding -> MLP prediction。论文聚焦于用 GNN 做 MIL 的 graph representation learning，并提出一个解决 MIL 的新视角。

#### Chunk 3: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `11`

- Must include terms: `convert input bags`, `adjacency matrix`

**原文**

```text
Figure 1 illustrate the diagram of our proposed framework on GNN based MIL. First, to convert input bags of instances to graphs, we adopt a heuristic strategy similar with the one used in (Zhou et al., 2009). Given a bag with instances [x(i) 1 , x(i) 2 , · · ·, x(i) K ], the adjacency matrixA can be derived with the following formula: Amn = { 1 ifdist(x(i) m, x(i) n )<η 0 otherwise (1) wheredist(x(i) m, x(i) n ) is the distance betweenm-th andn-th instance in bag i. In this study, Euclidean distance is em- ployed for simplicity.η is the threshold to decide whether there is an edge between two instances based on their dis- tance.η = 0 means there is no edge in the input graph while η = + ∞ means the input is a complete graph. η can be tuned for speciﬁc tasks. After converting bags of instances to graphs, we propose an end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈
```

**中文翻译**

这段给出 GNN-MIL 的图构建方法。为了把输入 bag 中的 instance 转成图，作者采用类似 Zhou 2009 的启发式策略。给定一个包含多个 instance 的 bag，通过距离阈值 η 构造邻接矩阵 A：若第 m 个和第 n 个 instance 的距离小于 η，则 A_mn = 1，否则为 0。本文为简单起见使用欧氏距离。η 用来决定两个 instance 之间是否连边；η = 0 表示输入图没有边，η = +∞ 表示完全图，η 可以针对具体任务调节。完成 bag-to-graph 后，作者提出基于 GNN 的端到端 graph representation learning 算法。

#### Chunk 4: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `12`

- Must include terms: `information passing`, `graph representation`

**原文**

```text
end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈ RK×D constructed from a bag ofXi, a GNN is ﬁrst applied to the input graph to conduct information passing over the graph. The output graph has the same number of nodes as the input graph, and the computation can be formulated as Zi =GNN embd(Ai,V i), (2) whereZi ∈ RK×D′ is the node embedding of graph output. D′ is the dimension of output node embedding, and can be different from input feature dimensionD. In order to obtain a ﬁxed-dimensional representation of the graph, we need a strategy to aggregate information over the whole graph with adjacency matrix Ai and updated node
```

**中文翻译**

这段描述 GNN-MIL 的信息传递阶段。给定由某个 bag Xi 构造的输入图 Gi，它包含邻接矩阵 Ai 和节点特征矩阵 Vi。首先在输入图上应用 GNN，在图中进行节点间信息传递。输出图与输入图有相同数量的节点，计算形式为 Zi = GNN_embd(Ai, Vi)。其中 Zi 是输出图的节点 embedding，输出维度 D′ 可以不同于输入特征维度 D。为了得到固定维度的图表示，还需要一个策略在整个图上聚合更新后的节点信息。

### 3. gmil-v2-003

- Type: `single_turn`
- Category: `abbreviation_entity_linking`
- Difficulty: `medium`

**问题**

RGMIL 是哪篇的方法？它主要想解决 GNN-MIL 里的什么问题？

**评分意图**

定位 Zhao 2024 RGMIL，并说明它针对 bag graph 边过滤阈值和 GNN 层数需要联合调节的问题。

**应覆盖答案点**

- 应定位到 Zhao 等 2024 Reinforced GNNs for Multiple Instance Learning。
- RGMIL 不是重新定义 MIL，而是在 GNN-based MIL 中自动调节影响 bag graph 和 GNN 的因素。
- 它关注边过滤阈值/边密度：阈值太高会丢失有用边，阈值太低会引入无意义边。
- 它也关注 GNN 层数/聚合范围：层数影响信息传播范围和过平滑风险。
- 核心方案是用多智能体深度强化学习同步控制 edge filtering threshold 和 GNN layers。

**负面检查项**

- 不要把 RGMIL 解释成普通 attention MIL。
- 不要只说“强化学习提升性能”，必须指出同步调节的两个对象。

**应召回 Chunks**

#### Chunk 1: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `1` / chunk `52`

- Must include terms: `RGMIL`, `MADRL`, `synchronous control`

**原文**

```text
these issues, we propose a reinforced GNN framework for MIL (RGMIL), pioneering the exploitation of multiagent deep rein- forcement learning (MADRL) in MIL tasks. MADRL enables the flexible definition or extension of factors that influence bag graphs or GNNs and provides synchronous control over them. Moreover, MADRL explores structure-to-architecture correlations while automating adjustments. Experimental results on multiple MIL datasets demonstrate that RGMIL achieves the best performance with excellent explainability. The code and data are available at https://github.com/RingBDStack/RGMIL. Index Terms— Deep reinforcement learning (RL), graph neu- ral network (GNN), multiple instance learning (MIL), neural architecture search. NOMENCLATURE B Set of bag samples. G Set of bag graphs corresponding to B. Y Set of bag-level labels corresponding to G. M Seven-tuple of the Markov game. S State space of M. O Observation space of M. Manuscript received 6 April 2023; revised 14 November
```

**中文翻译**

这段是 RGMIL 的摘要核心。作者针对前文问题提出 reinforced GNN framework for MIL，即 RGMIL，率先在 MIL 任务中利用多智能体深度强化学习（MADRL）。MADRL 允许灵活定义或扩展影响 bag graph 或 GNN 的因素，并对这些因素进行同步控制；同时，它能在自动调节过程中探索图结构与 GNN 架构之间的相关性。多个 MIL 数据集上的实验显示，RGMIL 获得最佳性能并具有较好的可解释性。

#### Chunk 2: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `2` / chunk `60`

- Must include terms: `filtering threshold`, `edges`

**原文**

```text
higher filtering threshold represents that fewer but more robust edges are preserved, while potentially losing some meaningful but modestly reliable information. Conversely, a lower filtering threshold may introduce too many meaningless edges in a bag. Given that GNNs rely on graph structures for aggregation, such uncertain filtration may result in unstable results. As shown in Fig. 3, more aggregation iterations (reflected in the number of GNN layers) prompt nodes to adopt farther hop features when fewer edges are retained. With a moderate threshold reduction, edge density increases, and more GNN layers may cause nodes to fuse information from too many other nodes, increasing the risk of oversmoothing [17], [18]. As factors influencing graph structures and GNN architectures are numerous and correlated, manually adjusting them in an asynchronous manner is tedious and inflexible. To deal with the above challenges, we propose a reinforced
```

**中文翻译**

这段解释 RGMIL 关注的边过滤阈值和 GNN 层数问题。较高的 filtering threshold 会保留更少但更可靠的边，同时可能丢失一些有意义但置信度中等的信息；较低阈值则可能在 bag 中引入太多无意义边。由于 GNN 依赖图结构做聚合，不确定的边过滤会造成不稳定结果。当保留边较少时，更多 GNN 层会让节点采用更远 hop 的特征；若阈值适度降低、边密度上升，过多 GNN 层又可能让节点融合太多其他节点信息，增加 oversmoothing 风险。影响图结构和 GNN 架构的因素多且相关，手动异步调参繁琐且不灵活。

#### Chunk 3: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `2` / chunk `62`

- Must include terms: `two agents`, `edge filtering thresholds`, `GNN layers`

**原文**

```text
we divide the training set into equal-sized blocks, one of which serves as the validation set, and the others are used to construct the MG state space. Then, two agents search for edge filtering thresholds and GNN layers, both of which have discrete action spaces. At a time step, each agent picks out an action according to the corresponding partial observations of the current global state, thereby guiding the construction of bag graphs and GNN layers. Since the purpose of GNNs is to improve representation learning, we regard the difference in adjacent performance on the validation set as the current reward. In other words, agents receive a positive reward if the model trained with the current action combination performs better on the validation data than the previous one, and vice versa. Finally, we introduce a novel heuristic state transition function to determine the next global state based on current actions. When the game reaches a Nash
```

**中文翻译**

这段描述 RGMIL 的多智能体搜索过程。作者把训练集划分为等大小 block，其中一个作为验证集，其余用于构建 Markov game 的状态空间。两个 agent 分别搜索 edge filtering threshold 和 GNN layers，这两个对象都有离散动作空间。在每个时间步，每个 agent 根据当前全局状态下自己的局部观察选择动作，从而指导 bag graph 构建和 GNN 层数设置。由于 GNN 的目标是改进表示学习，作者把验证集上相邻性能差异作为当前 reward：若当前动作组合训练出的模型优于上一次，则给正 reward，反之给负 reward。最后，作者引入启发式状态转移函数，根据当前动作决定下一个全局状态。

### 4. gmil-v2-004

- Type: `single_turn`
- Category: `abbreviation_entity_linking`
- Difficulty: `hard`

**问题**

这里说的 DSMIL 是 double similarities 那篇吗？它到底融合了哪两类相似度？

**评分意图**

定位 2024 Double Similarities weighted MIL kernel，并说明 DSMIL 的 Bag-to-Bag 与 Instance-to-Bag 相似度。

**应覆盖答案点**

- 应明确本知识库中用户指定的 DSMIL 是 2024 Double Similarities weighted Multi-Instance Learning kernel。
- 它不是常见语境里可能出现的 dual-stream DSMIL；回答应避免混淆。
- 核心融合 Bag-to-Bag similarity 和 Instance-to-Bag similarity。
- 方法构造 kernel function，并用于 SVM 解决 MIL 任务。
- 它强调同时利用 bag 级和 instance 级信息，而不是只用其中一类。

**负面检查项**

- 不要把 DSMIL 说成 GNN 方法。
- 不要把本题的 DSMIL 混同为 dual-stream MIL 或 attention-based WSI DSMIL。

**应召回 Chunks**

#### Chunk 1: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `1` / chunk `262`

- Must include terms: `Double Similarities`, `Bag-to-Bag`, `Instance-to-Bag`

**原文**

```text
applications. However, most existing MIL methods just utilize partial information (bags or instances) of MIL data to construct the kernel function, resulting in deteriorated classification performance of MIL. In this paper, we propose a Double Similarities weighted Multi-Instance Learning (DSMIL) kernel framework, which utilizes the similarities of Bag-to-Bag (B2B) and Instance-to-Bag (I2B). In the proposed kernel framework, the similarities of B2B and I2B could be derived from the prototypes distance of inter-bag and similarity matrix of intra-bag, respectively, based on the affinity propagation (AP) clustering of the bag. Meanwhile, we give theoretical proof of the validity of the designed kernel function. Experimental results on benchmark and semi- synthetic datasets show that our proposed method obtains competitive classification performance and achieves robustness to parameters and noise. 1. Introduction Multi-instance learning (MIL), which originated from drug activity
```

**中文翻译**

这段是 DSMIL 摘要核心。作者指出，多数已有 MIL 方法只利用 MIL 数据中的部分信息（bag 或 instance）来构造 kernel function，导致分类性能下降。论文提出 Double Similarities weighted Multi-Instance Learning kernel framework，简称 DSMIL，同时利用 Bag-to-Bag（B2B）和 Instance-to-Bag（I2B）两类相似度。在该 kernel 框架中，B2B 和 I2B 分别可由 bag 间 prototype distance 和 bag 内 similarity matrix 得到，并基于 bag 的 affinity propagation clustering。作者还给出设计的 kernel function 有效性的理论证明。基准和半合成数据集实验显示，该方法具有竞争性分类性能，并对参数和噪声有鲁棒性。

#### Chunk 2: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `2` / chunk `270`

- Must include terms: `bag`, `instance`, `Double Similarities`

**原文**

```text
and identical distribution (i.i.d.) condition, ignoring the fact that the relationship among the intra-bag instances implies important structure information (Zhou et al., 2009). In this paper, we consider both the bag and instance information of multi-instance data and propose a Double Similarities weighted Multi-instance Learning (DSMIL) kernel to alleviate the aforementioned problems. We regard each bag as an entity and the intra-bag instances as inter-correlated components of the entity. Inspired by Carbonneau et al. (2018), we divide the information expressed by multi-instance data into three types: the instance co-occurrence information in the bag (relationship among the intra-bag instances), the importance of an instance to the label of the corresponding bag (instance-to-bag, I2B), and the similarity between bags (bag-to-bag, B2B). Then we get the three types of information separately and integrate them. To be specific,
```

**中文翻译**

这段说明 DSMIL 的动机和信息划分。已有方法常忽略 bag 内 instance 关系中包含的重要结构信息。作者同时考虑 multi-instance 数据的 bag 信息和 instance 信息，提出 Double Similarities weighted MIL kernel 来缓解这些问题。他们把每个 bag 视为一个整体实体，把 bag 内 instance 视为该实体中相互关联的组成部分。受 Carbonneau 等工作的启发，作者把 MIL 数据表达的信息分为三类：bag 内 instance 共现信息，即 instance 间关系；instance 对其所属 bag 标签的重要性，即 Instance-to-Bag（I2B）；bag 与 bag 之间的相似性，即 Bag-to-Bag（B2B）。随后分别获取并整合这些信息。

#### Chunk 3: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `3` / chunk `282`

- Must include terms: `kernel function`, `s`, `d`

**原文**

```text
bag is positive, and otherwise 𝑌𝑖 = −1 means that the bag is negative. Our goal is to develop a kernel function that makes the most of the information available in the data. Definition 3.1 (DSMIL). For any two bags 𝐁𝑖 and 𝐁𝑗, the kernel function is defined as follows: 𝐹𝜙(𝐁𝑖, 𝐁𝑗 ) = 𝑠(𝐁𝑖, 𝐁𝑗 ) ∑ 𝑘 ∑ 𝑝 𝑘(𝑑(𝑋(𝑖) 𝑘 , 𝐁𝑖)𝑋(𝑖) 𝑘 , 𝑑(𝑋(𝑗) 𝑝 , 𝐁𝑗 )𝑋(𝑗) 𝑝 ), (1) where 𝑠(𝐁𝑖, 𝐁𝑗 ) denotes the similarity of bag-to-bag, 𝑑(𝑋(𝑖) 𝑘 , 𝐁𝑖) and 𝑑(𝑋(𝑗) 𝑝 , 𝐁𝑗 ) represent the importance of instance-to-bag, which is mea- sured by the similarity between the instance and its corresponding bag, 𝜙 is a feature mapping related to kernel 𝑘(⋅, ⋅). In the next, we would manifest that the defined function (1) is a valid kernel function. Let 𝑣𝑘(𝑋(𝑖) 𝑘 ) = 𝑑(𝑋(𝑖) 𝑘 , 𝐁𝑖)𝑋(𝑖) 𝑘 , the kernel function (1) can be represented as: 𝐹𝜙(𝐁𝑖, 𝐁𝑗 ) = 𝑠(𝐁𝑖, 𝐁𝑗 ) ∑ 𝑘 ∑ 𝑝 𝑘(𝑣𝑘(𝑋(𝑖) 𝑘 ), 𝑣𝑝(𝑋(𝑗) 𝑝 )). (2) Definition 3.2 (Feature Mapping of DSMIL ). For any bag 𝐁𝑖 ∈ R𝑚×𝑑, the
```

**中文翻译**

这段给出 DSMIL kernel 的形式化定义。目标是构造一个能充分利用数据可用信息的 kernel function。对任意两个 bag Bi 和 Bj，DSMIL kernel 定义为：用 s(Bi, Bj) 表示 bag-to-bag 相似度，再乘上 instance 级 kernel 的双重求和；其中 d(X_k^(i), Bi) 和 d(X_p^(j), Bj) 表示 instance-to-bag 重要性，即某个 instance 与其所属 bag 的相似度；φ 是与 kernel k 相关的 feature mapping。接下来作者说明该定义函数是有效 kernel，并把加权后的 instance 表示写成 v_k(X_k^(i))，从而重写 kernel function。

#### Chunk 4: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `2` / chunk `272`

- Must include terms: `kernel function`, `SVM`

**原文**

```text
instances, and the theoretical proof with respect to the validity of the designed kernel function is given accordingly. Finally, the designed kernel function is applied to support vector machine (SVM) for solving the MIL task, achieving competitive classification performance. The main contributions of this paper are summarized as follows: • A novel double similarities weighted kernel function is designed and given analysis, which organically fuses the information of bag and instance levels. • We directly explore the information of intra-bag instances through the clustering method without considering the basic i.i.d. assump- tion of the MIL data, and thus the proposed framework can be applied to real-world classification tasks. • We apply the proposed kernel framework to the SVM classifier, and experimental results on MIL benchmarks and semi-synthetic Newsgroups datasets confirm that the designed framework pro- vides competitive results.
```

**中文翻译**

这段总结 DSMIL 的贡献：作者给出所设计 kernel function 有效性的理论证明，并将该 kernel function 应用于支持向量机（SVM）以解决 MIL 任务，取得有竞争力的分类表现。主要贡献包括：设计并分析一种新的 double similarities weighted kernel function，有机融合 bag level 和 instance level 信息；通过 clustering 方法直接探索 bag 内 instance 信息，而不依赖 MIL 数据的基本 i.i.d. 假设；把该 kernel framework 应用于 SVM 分类器，并在 MIL benchmark 和半合成 Newsgroups 数据集上验证竞争性结果。

### 5. gmil-v2-005

- Type: `single_turn`
- Category: `abbreviation_entity_linking`
- Difficulty: `hard`

**问题**

BGNN-MIL 是在 bag 里面连 instance 吗？这篇贝叶斯图方法到底连的是哪些对象？

**评分意图**

定位 Pal 2022 Bag Graph MIL，并说明它主要建模 bag 之间的 graph，而不是只做 bag 内实例图。

**应覆盖答案点**

- 应定位到 Pal 等 2022 Bag Graph: Multiple Instance Learning Using Bayesian Graph Neural Networks。
- 这篇的关键不是像 GNN-MIL 那样只把一个 bag 内的 instances 连成图。
- 它建模的是 bags 之间的 interactions/dependencies，并用 GNN 进行端到端学习。
- 论文指出有意义的 bag graph 往往不可得或有噪声，因此用 Bayesian GNN 处理图结构不确定性。
- 回答可以提到它与 Tu 2019 的区别：Tu 2019 主要是 bag 内 instance graph，而 Pal 2022 强调 bag graph。

**负面检查项**

- 不要把 BGMIL 简化成普通 bag 内部构图的 GNN-MIL。
- 不要遗漏 Bayesian/uncertainty，因为这是这篇的核心特征。

**应召回 Chunks**

#### Chunk 1: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `1` / chunk `1128`

- Must include terms: `interactions between bags`, `Graph Neural Networks`, `meaningful graph`

**原文**

```text
to learn effective bag-level representations by suitably com- bining permutation invariant pooling techniques with neural architectures. In this paper, we consider modelling the inter- actions between bags using a graph and employ Graph Neu- ral Networks (GNNs) to facilitate end-to-end learning. Since a meaningful graph representing dependencies between bags is rarely available, we propose to use a Bayesian GNN frame- work that can generate a likely graph structure for scenarios where there is uncertainty in the graph or when no graph is available. Empirical results demonstrate the efﬁcacy of the proposed technique for several MIL benchmark tasks and a distribution regression task. Introduction In numerous supervised learning settings, our aim is to as- sign a label to a group (or bag) of instances as opposed to assigning labels to the individual instances. Example appli- cations include drug activity prediction (Dietterich, Lathrop,
```

**中文翻译**

这段说明 BGMIL/BGNN-MIL 的核心设定。已有方法通过 permutation-invariant pooling 和神经网络架构组合来学习有效的 bag-level representation。本文考虑用一张图来建模 bag 之间的交互，并使用 GNN 进行端到端学习。由于表示 bag 之间依赖关系的有意义图通常很少现成可用，作者提出 Bayesian GNN framework，用于在图结构存在不确定性或没有图时生成可能的图结构。实验证明该技术在多个 MIL benchmark 和一个 distribution regression 任务上有效。

#### Chunk 2: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `1` / chunk `1131`

- Must include terms: `structure of instances within a bag`, `key observation`, `relationships`

**原文**

```text
In (Zhang et al. 2011), a relational graph was used to spec- ify similarities between instances. With the recent advances in graph neural networks (GNNs), there have been efforts to use these to represent the structure of instances within a bag (Tu et al. 2019; Yin et al. 2019). Our key observation is that while graphs have been used to model relationships between instances, they have not been employed to specify relationships between bags. In some applications, side-information provides a clear mechanism for constructing a graph. For example, in a real estate ap- plication when the goal is to predict mean neighborhood rental prices, we may assume that nearby neighborhoods have similar pricing (Valkanas, Regol, and Coates 2020). A graph can then be constructed with edges representing geo- graphic proximity. The identiﬁed dependencies are valuable in a graph-based learning framework, leading to improved predictive performance. In other cases, there is no graph
```

**中文翻译**

这段明确区分 bag 内 instance graph 和 bag 间 graph。已有工作使用 relational graph 表示 instance 之间的相似性；随着 GNN 发展，也有人用 GNN 表示一个 bag 内 instance 的结构。作者的关键观察是：虽然图已经被用于建模 instance 之间的关系，但还没有被用于指定 bag 之间的关系。在某些应用中，side-information 可以清晰地构造图，例如房地产任务中可用地理邻近性连接社区；这些依赖关系对 graph-based learning 有价值并能提升预测表现。但其他情况下并没有现成图。

#### Chunk 3: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `3` / chunk `1143`

- Must include terms: `Bayesian GNN`, `observed graph`, `uncertainty`

**原文**

```text
Bayesian GNN Framework In many graph based learning problems, the observed graph is constructed from noisy data or derived based on heuris- tics and/or imperfect modelling assumptions. As a result, the observed graph might not represent the true underlying rela- tionship among the data on its nodes; it might contain spu- rious links and important links might be unobserved. How- ever, most existing GNNs do not account for the uncertainty of the graph structure during training. Several recent works such as (Ma et al. 2019; Jiang et al. 2019; Zhang et al. 2019; Pal et al. 2020; Elinas, Bonilla, and Tiao 2020; Wan et al. 2021) address this issue by incorporat- ing probabilistic modelling or joint optimization of the graph during model training. In particular, Zhang et al. (2019) in- troduce a general Bayesian framework, where the observed graph is assumed to be a random sample from a paramet- ric random graph family and posterior inference of the true
```

**中文翻译**

这段解释 Bayesian GNN 的必要性。许多 graph-based learning 问题中的 observed graph 来自噪声数据、启发式方法或不完美建模假设，因此 observed graph 可能无法代表节点数据之间真实的底层关系：它可能包含伪边，也可能缺失重要边。然而多数现有 GNN 在训练时不考虑图结构不确定性。近期一些工作通过概率建模或训练时联合优化图来处理这一问题；其中 Zhang 等提出通用 Bayesian framework，把 observed graph 看作来自某个参数化 random graph family 的随机样本，并对真实图进行后验推断。

### 6. gmil-v2-006

- Type: `single_turn`
- Category: `method_family_retrieval`
- Difficulty: `hard`

**问题**

核心论文里哪些方法真的用了图神经网络？图核那类不要混进来。

**评分意图**

从核心论文中筛选 GNN-based MIL 方法，并排除 miGraph/MIGraph 和 DSMIL 这类 kernel 方法。

**应覆盖答案点**

- 至少应列出 GNN-MIL/Tu 2019：每个 bag 建图并用 GNN 学习 graph representation。
- 应列出 RGMIL/Zhao 2024：reinforced GNN framework，用 MADRL 调节 graph 和 GNN 层数。
- 应列出 BGMIL/BGNN-MIL/Pal 2022：用 Bayesian GNN 建模 bag graph。
- 可列出 Patch-GCN/Chen 2021：WSI patch 图上的 GCN。
- 可列出 NAGCN/Guan 2022：node-aligned GCN 用于 WSI 表示。
- 如果检索到其他 corpus 中的有效 GNN 论文，可作为补充，但 miGraph/MIGraph 和 DSMIL 不能作为 GNN 方法出现。

**负面检查项**

- 不要把 miGraph/MIGraph 列为 GNN；它们是 graph-kernel。
- 不要把 Double Similarities DSMIL 列为 GNN；它是 kernel/SVM 路线。
- 不要只按标题包含 graph 就判定为 GNN。

**应召回 Chunks**

#### Chunk 1: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `1` / chunk `1`

- Must include terms: `graph neural network`, `each bag as a graph`

**原文**

```text
Multiple instance learning with graph neural networks Ming Tu1 Jing Huang 1 Xiaodong He 1 Bowen Zhou 1 Abstract Multiple instance learning (MIL) aims to learn the mapping between a bag of instances and the bag-level label. In this paper, we propose a new end-to-end graph neural network (GNN) based al- gorithm for MIL: we treat each bag as a graph and use GNN to learn the bag embedding, in order to explore the useful structural information among instances in bags. The ﬁnal graph representation is fed into a classiﬁer for label prediction. Our algorithm is the ﬁrst attempt to use GNN for MIL. We empirically show that the proposed algorithm achieves the state of the art performance on sev- eral popular MIL data sets without losing model interpretability. 1. Introduction Multiple instance learning (MIL) as a weakly-supervised learning algorithm deals with weakly-labeled data, where each data sample (often named as a bag) has multiple in-
```

**中文翻译**

这段是 Tu 等 2019 论文的标题和摘要。论文提出一种用于多实例学习的端到端图神经网络算法：把每个 bag 看作一张图，并用 GNN 学习 bag embedding，从而探索 bag 内 instance 之间有用的结构信息。最终图表示会输入分类器进行标签预测。作者称这是首次尝试将 GNN 用于 MIL，并在多个常用 MIL 数据集上取得当时最优表现，同时保持模型可解释性。

#### Chunk 2: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `1` / chunk `52`

- Must include terms: `reinforced GNN`, `RGMIL`

**原文**

```text
these issues, we propose a reinforced GNN framework for MIL (RGMIL), pioneering the exploitation of multiagent deep rein- forcement learning (MADRL) in MIL tasks. MADRL enables the flexible definition or extension of factors that influence bag graphs or GNNs and provides synchronous control over them. Moreover, MADRL explores structure-to-architecture correlations while automating adjustments. Experimental results on multiple MIL datasets demonstrate that RGMIL achieves the best performance with excellent explainability. The code and data are available at https://github.com/RingBDStack/RGMIL. Index Terms— Deep reinforcement learning (RL), graph neu- ral network (GNN), multiple instance learning (MIL), neural architecture search. NOMENCLATURE B Set of bag samples. G Set of bag graphs corresponding to B. Y Set of bag-level labels corresponding to G. M Seven-tuple of the Markov game. S State space of M. O Observation space of M. Manuscript received 6 April 2023; revised 14 November
```

**中文翻译**

这段是 RGMIL 的摘要核心。作者针对前文问题提出 reinforced GNN framework for MIL，即 RGMIL，率先在 MIL 任务中利用多智能体深度强化学习（MADRL）。MADRL 允许灵活定义或扩展影响 bag graph 或 GNN 的因素，并对这些因素进行同步控制；同时，它能在自动调节过程中探索图结构与 GNN 架构之间的相关性。多个 MIL 数据集上的实验显示，RGMIL 获得最佳性能并具有较好的可解释性。

#### Chunk 3: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `1` / chunk `1128`

- Must include terms: `Graph Neural Networks`, `interactions between bags`

**原文**

```text
to learn effective bag-level representations by suitably com- bining permutation invariant pooling techniques with neural architectures. In this paper, we consider modelling the inter- actions between bags using a graph and employ Graph Neu- ral Networks (GNNs) to facilitate end-to-end learning. Since a meaningful graph representing dependencies between bags is rarely available, we propose to use a Bayesian GNN frame- work that can generate a likely graph structure for scenarios where there is uncertainty in the graph or when no graph is available. Empirical results demonstrate the efﬁcacy of the proposed technique for several MIL benchmark tasks and a distribution regression task. Introduction In numerous supervised learning settings, our aim is to as- sign a label to a group (or bag) of instances as opposed to assigning labels to the individual instances. Example appli- cations include drug activity prediction (Dietterich, Lathrop,
```

**中文翻译**

这段说明 BGMIL/BGNN-MIL 的核心设定。已有方法通过 permutation-invariant pooling 和神经网络架构组合来学习有效的 bag-level representation。本文考虑用一张图来建模 bag 之间的交互，并使用 GNN 进行端到端学习。由于表示 bag 之间依赖关系的有意义图通常很少现成可用，作者提出 Bayesian GNN framework，用于在图结构存在不确定性或没有图时生成可能的图结构。实验证明该技术在多个 MIL benchmark 和一个 distribution regression 任务上有效。

#### Chunk 4: `Chen 等 - 2021 - Whole Slide Images are 2D Point Clouds Context-Aware Survival Prediction Using Patch-Based Graph Co.pdf` / page `1` / chunk `589`

- Must include terms: `Patch-GCN`, `graph convolutional network`

**原文**

```text
not context-aware and are unable to model important morphological fea- ture interactions between cell identities and tissue types that are prognos- tic for patient survival. In this work, we present Patch-GCN, a context- aware, spatially-resolved patch-based graph convolutional network that hierarchically aggregates instance-level histology features to model local- and global-level topological structures in the tumor microenvironment. We validate Patch-GCN with 4,370 gigapixel WSIs across ﬁve diﬀerent cancer types from the Cancer Genome Atlas (TCGA), and demonstrate that Patch-GCN outperforms all prior weakly-supervised approaches by 3.58-9.46%. Our code and corresponding models are publicly available at https://github.com/mahmoodlab/Patch-GCN. Keywords: Computer Vision · Computational Pathology · Weakly-Supervised Learning · Graph Convolutional Networks · Interpretability 1 Introduction Weakly-supervised deep learning has made remarkable progress in computational
```

**中文翻译**

这段介绍 Patch-GCN。作者指出已有弱监督方法不具备 context-aware 能力，无法建模细胞身份和组织类型之间对患者生存预后重要的形态特征交互。本文提出 Patch-GCN：一种 context-aware、spatially-resolved、patch-based graph convolutional network，能够层次化聚合 instance-level 组织学特征，以建模肿瘤微环境中的局部和全局拓扑结构。作者在 TCGA 的五种癌症类型、4370 张 gigapixel WSI 上验证，Patch-GCN 比此前弱监督方法提升 3.58% 到 9.46%。

#### Chunk 5: `Guan 等 - 2022 - Node-aligned graph convolutional network for whole-slide image representation and classification.pdf` / page `1` / chunk `723`

- Must include terms: `Node-aligned`, `GCN`

**原文**

```text
spondence across patches from different WSIs. Therefore, most methods have to perform non-ordered node pooling to generate the bag-level representation. Direct non-ordered pooling will lose much structural and contextual informa- tion, such as patch distribution and heterogeneous patterns, which is critical for WSI representation. In this paper, we propose a hierarchical global-to-local clustering strategy to build a Node-Aligned GCN (NAGCN) to represent WSI with rich local structural information as well as global distribu- tion. We first deploy a global clustering operation based on the instance features in the dataset to build the corre- spondence across different WSIs. Then, we perform a lo- cal clustering-based sampling strategy to select typical in- stances belonging to each cluster within the WSI. Finally, we employ the graph convolution to obtain the represen- tation. Since our graph construction strategy ensures the alignment among different WSIs, WSI-level representation
```

**中文翻译**

这段介绍 NAGCN 的动机和方法。由于不同 WSI 的 patch 之间缺乏节点对应关系，许多方法只能用无序 node pooling 得到 bag-level representation；直接无序 pooling 会丢失 patch 分布和异质模式等结构与上下文信息，而这些对 WSI 表示很关键。作者提出 hierarchical global-to-local clustering strategy 来构建 Node-Aligned GCN（NAGCN），使 WSI 表示同时包含丰富局部结构信息和全局分布。方法先基于数据集中的 instance features 做 global clustering，以建立不同 WSI 之间的对应关系；再用 local clustering-based sampling 在每个 WSI 中选择典型 instance；最后用 graph convolution 得到表示。由于图构建策略保证了不同 WSI 间的 alignment，因此易于生成 WSI-level representation。

### 7. gmil-v2-007

- Type: `single_turn`
- Category: `false_recall_guard`
- Difficulty: `hard`

**问题**

在这几篇核心里，哪些是图核或核方法，不是 GNN？

**评分意图**

识别核心论文中的 kernel-based 方法，并避免误召回 GNN-based 方法。

**应覆盖答案点**

- 应列出 Zhou 2009 的 MIGraph/miGraph：基于 graph kernel 或隐式 graph kernel 的 MIL。
- 应列出 2024 Double Similarities DSMIL：构造 Double Similarities weighted MIL kernel，并用于 SVM。
- 可以说明二者都利用结构或相似度，但不是神经网络 message passing。
- 应明确排除 GNN-MIL、RGMIL、BGMIL/BGNN-MIL、Patch-GCN、NAGCN。

**负面检查项**

- 不要因为标题或任务中有 graph 就把 GNN 方法放入 kernel 列表。
- 不要把 DSMIL 写成 deep neural network。

**应召回 Chunks**

#### Chunk 1: `zhou2009.pdf.pdf` / page `2` / chunk `166`

- Must include terms: `miGraph`, `graph kernel`

**原文**

```text
miGraph method implicitly constructs graphs by de- riving aﬃnity matrices and deﬁnes an eﬃcient graph kernel considering the clique information. Before presenting the details, we give the formal def- inition of multi-instance learning as following. Let X denote the instance space. Given a data set 1250
```

**中文翻译**

这段继续强调 miGraph 的核心：它不是显式构造 bag graph，而是通过 affinity matrix 隐式构图，并定义一个考虑 clique 信息的高效 graph kernel。随后作者准备给出多实例学习的形式化定义：令 X 表示 instance space，并基于给定数据集定义后续问题。

#### Chunk 2: `zhou2009.pdf.pdf` / page `4` / chunk `176`

- Must include terms: `MIGraph`, `graph kernel`

**原文**

```text
tional complexity of kG(Xi, X j) is O(ninj + mimj). The kG clearly satisﬁes all the four major properties that should be considered for a graph kernel deﬁnition (Borgwardt & Kriegel, 2005). 1 Our above design is very simple, but in the next section we can see that the proposed MIGraph method is quite eﬀective. A deﬁciency of MIGraph is that the computational complexity of kG is O(ninj + mimj), dominated by the number of edges. For bags containing a lot of in- stances, there will exist a large number of edges and MIGraph will be hard to execute. So, it is desired to have a method with smaller computational cost. For this purpose, we propose the miGraph method which is simple, eﬃcient but eﬀective. For bag Xi, we can calculate the distance between its instances and derive an aﬃnity matrix W i by com- paring the distances with a threshold δ. For example, if the distance between the instances xia and xiu is smaller than δ, W i’s element at the ath row and uth column, wi
```

**中文翻译**

这段先给出 MIGraph 中图核 kG(Xi, Xj) 的计算复杂度，为 O(n_i n_j + m_i m_j)，并说明该核满足图核定义应考虑的主要性质。作者认为 MIGraph 设计简单且有效，但它的缺点也很明显：复杂度受边数量主导；当 bag 中 instance 很多时，会产生大量边，MIGraph 会很难执行。因此需要计算成本更低的方法。为此作者提出 miGraph，它简单、高效且有效。对每个 bag Xi，miGraph 计算其中 instance 之间的距离，并通过阈值 δ 得到 affinity matrix W_i。

#### Chunk 3: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `1` / chunk `262`

- Must include terms: `DSMIL`, `kernel`

**原文**

```text
applications. However, most existing MIL methods just utilize partial information (bags or instances) of MIL data to construct the kernel function, resulting in deteriorated classification performance of MIL. In this paper, we propose a Double Similarities weighted Multi-Instance Learning (DSMIL) kernel framework, which utilizes the similarities of Bag-to-Bag (B2B) and Instance-to-Bag (I2B). In the proposed kernel framework, the similarities of B2B and I2B could be derived from the prototypes distance of inter-bag and similarity matrix of intra-bag, respectively, based on the affinity propagation (AP) clustering of the bag. Meanwhile, we give theoretical proof of the validity of the designed kernel function. Experimental results on benchmark and semi- synthetic datasets show that our proposed method obtains competitive classification performance and achieves robustness to parameters and noise. 1. Introduction Multi-instance learning (MIL), which originated from drug activity
```

**中文翻译**

这段是 DSMIL 摘要核心。作者指出，多数已有 MIL 方法只利用 MIL 数据中的部分信息（bag 或 instance）来构造 kernel function，导致分类性能下降。论文提出 Double Similarities weighted Multi-Instance Learning kernel framework，简称 DSMIL，同时利用 Bag-to-Bag（B2B）和 Instance-to-Bag（I2B）两类相似度。在该 kernel 框架中，B2B 和 I2B 分别可由 bag 间 prototype distance 和 bag 内 similarity matrix 得到，并基于 bag 的 affinity propagation clustering。作者还给出设计的 kernel function 有效性的理论证明。基准和半合成数据集实验显示，该方法具有竞争性分类性能，并对参数和噪声有鲁棒性。

#### Chunk 4: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `2` / chunk `272`

- Must include terms: `kernel function`, `SVM`

**原文**

```text
instances, and the theoretical proof with respect to the validity of the designed kernel function is given accordingly. Finally, the designed kernel function is applied to support vector machine (SVM) for solving the MIL task, achieving competitive classification performance. The main contributions of this paper are summarized as follows: • A novel double similarities weighted kernel function is designed and given analysis, which organically fuses the information of bag and instance levels. • We directly explore the information of intra-bag instances through the clustering method without considering the basic i.i.d. assump- tion of the MIL data, and thus the proposed framework can be applied to real-world classification tasks. • We apply the proposed kernel framework to the SVM classifier, and experimental results on MIL benchmarks and semi-synthetic Newsgroups datasets confirm that the designed framework pro- vides competitive results.
```

**中文翻译**

这段总结 DSMIL 的贡献：作者给出所设计 kernel function 有效性的理论证明，并将该 kernel function 应用于支持向量机（SVM）以解决 MIL 任务，取得有竞争力的分类表现。主要贡献包括：设计并分析一种新的 double similarities weighted kernel function，有机融合 bag level 和 instance level 信息；通过 clustering 方法直接探索 bag 内 instance 信息，而不依赖 MIL 数据的基本 i.i.d. 假设；把该 kernel framework 应用于 SVM 分类器，并在 MIL benchmark 和半合成 Newsgroups 数据集上验证竞争性结果。

### 8. gmil-v2-008

- Type: `single_turn`
- Category: `paper_summary`
- Difficulty: `medium`

**问题**

总结一下 DSMIL 2024 这篇，重点说贡献和局限，不要只翻译标题。

**评分意图**

总结 2024 Double Similarities weighted MIL kernel 的问题设定、方法贡献、实验结论和局限。

**应覆盖答案点**

- 问题背景：已有 MIL kernel 往往只利用 bag 或 instance 的部分信息，忽略二者结合。
- 方法贡献：提出 Double Similarities weighted MIL kernel，融合 I2B 和 B2B 两类相似度。
- 技术路径：考虑 instance co-occurrence/instance-to-bag/bag-to-bag 信息，构造有效 kernel 并接入 SVM。
- 实验结论：消融显示 I2B 和 B2B 组合优于单独模块，整体表现有竞争力。
- 局限或代价：仍是 kernel/SVM 框架，测试时需计算两类相似度且可能更慢；作者未来希望融合 deep learning 提升特征表示。

**负面检查项**

- 不要把 DSMIL 总结成 GNN 或 attention 模型。
- 不要只说“性能好”，必须说明为什么 I2B+B2B 是贡献。

**应召回 Chunks**

#### Chunk 1: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `1` / chunk `262`

- Must include terms: `partial information`, `Double Similarities`

**原文**

```text
applications. However, most existing MIL methods just utilize partial information (bags or instances) of MIL data to construct the kernel function, resulting in deteriorated classification performance of MIL. In this paper, we propose a Double Similarities weighted Multi-Instance Learning (DSMIL) kernel framework, which utilizes the similarities of Bag-to-Bag (B2B) and Instance-to-Bag (I2B). In the proposed kernel framework, the similarities of B2B and I2B could be derived from the prototypes distance of inter-bag and similarity matrix of intra-bag, respectively, based on the affinity propagation (AP) clustering of the bag. Meanwhile, we give theoretical proof of the validity of the designed kernel function. Experimental results on benchmark and semi- synthetic datasets show that our proposed method obtains competitive classification performance and achieves robustness to parameters and noise. 1. Introduction Multi-instance learning (MIL), which originated from drug activity
```

**中文翻译**

这段是 DSMIL 摘要核心。作者指出，多数已有 MIL 方法只利用 MIL 数据中的部分信息（bag 或 instance）来构造 kernel function，导致分类性能下降。论文提出 Double Similarities weighted Multi-Instance Learning kernel framework，简称 DSMIL，同时利用 Bag-to-Bag（B2B）和 Instance-to-Bag（I2B）两类相似度。在该 kernel 框架中，B2B 和 I2B 分别可由 bag 间 prototype distance 和 bag 内 similarity matrix 得到，并基于 bag 的 affinity propagation clustering。作者还给出设计的 kernel function 有效性的理论证明。基准和半合成数据集实验显示，该方法具有竞争性分类性能，并对参数和噪声有鲁棒性。

#### Chunk 2: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `2` / chunk `270`

- Must include terms: `bag`, `instance`, `co-occurrence`

**原文**

```text
and identical distribution (i.i.d.) condition, ignoring the fact that the relationship among the intra-bag instances implies important structure information (Zhou et al., 2009). In this paper, we consider both the bag and instance information of multi-instance data and propose a Double Similarities weighted Multi-instance Learning (DSMIL) kernel to alleviate the aforementioned problems. We regard each bag as an entity and the intra-bag instances as inter-correlated components of the entity. Inspired by Carbonneau et al. (2018), we divide the information expressed by multi-instance data into three types: the instance co-occurrence information in the bag (relationship among the intra-bag instances), the importance of an instance to the label of the corresponding bag (instance-to-bag, I2B), and the similarity between bags (bag-to-bag, B2B). Then we get the three types of information separately and integrate them. To be specific,
```

**中文翻译**

这段说明 DSMIL 的动机和信息划分。已有方法常忽略 bag 内 instance 关系中包含的重要结构信息。作者同时考虑 multi-instance 数据的 bag 信息和 instance 信息，提出 Double Similarities weighted MIL kernel 来缓解这些问题。他们把每个 bag 视为一个整体实体，把 bag 内 instance 视为该实体中相互关联的组成部分。受 Carbonneau 等工作的启发，作者把 MIL 数据表达的信息分为三类：bag 内 instance 共现信息，即 instance 间关系；instance 对其所属 bag 标签的重要性，即 Instance-to-Bag（I2B）；bag 与 bag 之间的相似性，即 Bag-to-Bag（B2B）。随后分别获取并整合这些信息。

#### Chunk 3: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `6` / chunk `299`

- Must include terms: `I2BS`, `B2BS`, `ablation`

**原文**

```text
to-bag and the similarity of bag-to-bag. We verify the influence by comparing the accuracy of using each component in DSMIL individu- ally. Table 6 shows the ablation experimental results on five benchmark datasets, where I2BS refers to the method that uses only the similarity of instance-to-bag, B2BS-T and B2BS-H refer to the methods that use only the similarity of bag-to-bag measured with Hausdorff distance and Tanimoto coefficient, respectively. DSMIL-T and DSMIL-H refer to the DSMIL using Hausdorff distance and Tanimoto coefficient, respectively. As can be seen, we obtain that the results of DSMIL-T and DSMIL-H are better than either module alone on benchmark datasets, regardless of whether the B2BS-T, the B2BS-H, or the I2BS is used. Besides, it can be seen from Table 6 that the DSMIL-H has achieved better results on most of the benchmark datasets than the DSMIL-T. This may be because the Tanimoto coefficient is more suitable for binary data, while multiple
```

**中文翻译**

这段是 DSMIL 的消融实验。作者通过分别使用 DSMIL 的各个组成部分来比较准确率，以验证 instance-to-bag similarity 和 bag-to-bag similarity 的影响。表 6 展示五个 benchmark 数据集上的消融结果：I2BS 表示只使用 instance-to-bag similarity；B2BS-T 和 B2BS-H 分别表示只用 Hausdorff distance 和 Tanimoto coefficient 衡量的 bag-to-bag similarity；DSMIL-T 和 DSMIL-H 则分别表示使用 Hausdorff distance 和 Tanimoto coefficient 的完整 DSMIL。结果显示，无论与 B2BS-T、B2BS-H 还是 I2BS 单独模块相比，DSMIL-T 和 DSMIL-H 都更好；DSMIL-H 在多数 benchmark 上优于 DSMIL-T，可能因为 Tanimoto coefficient 更适合二值数据。

#### Chunk 4: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `9` / chunk `315`

- Must include terms: `testing time`, `I2B`, `B2B`

**原文**

```text
In the case of DSMIL-H and DSMIL-T, the bag is mapped into a new feature space after computing the two similarities (I2B and B2B), which is equivalent to inputting one feature vector for each bag to train. In addition, we can find from the table that the testing time for MI-SVM and mi-SVM is similar, while the testing time for DSMIL-H and DSMIL-T is longer because DSMIL-H and DSMIL-T require the calculation of two similarities (I2B and B2B) in the test phase. In the computation of two similarities (I2B and B2B), AP clustering is performed for each bag. The time complexity of AP clustering for each bag is (𝑛2 log 𝑛), and thus it costs (𝑁𝑛 2 log 𝑛) to compute the I2B for all bags, where 𝑁 is the number of bags and 𝑛 is the number of instances in a bag that is variable in each bag. For DSMIL-H, the complexity of calculating the B2B is (𝑁 2𝑐1𝑐2), and in the case of DSMIL-T, it is (4𝑁 2𝑐1𝑐2), where 𝑐1 and 𝑐2 refer to the number of
```

**中文翻译**

这段讨论 DSMIL 的测试时间和复杂度。对 DSMIL-H 和 DSMIL-T 来说，bag 在计算 I2B 和 B2B 两类相似度后会被映射到新的特征空间，相当于每个 bag 输入一个特征向量进行训练。表中可以看到 MI-SVM 和 mi-SVM 的测试时间相近，而 DSMIL-H 和 DSMIL-T 测试更慢，因为测试阶段需要计算 I2B 和 B2B 两类相似度。计算这两类相似度时，每个 bag 都要执行 AP clustering；每个 bag 的 AP clustering 复杂度为 O(n^2 log n)，因此为所有 bag 计算 I2B 的复杂度为 O(N n^2 log n)。DSMIL-H 和 DSMIL-T 的 B2B 计算复杂度也随 bag 数和 cluster 数增加。

#### Chunk 5: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `11` / chunk `321`

- Must include terms: `future work`, `deep learning`

**原文**

```text
TensMIL2 ( Papastergiou et al. , 2019 ) re-designs a feature extraction algorithm for the color image datasets, which performs tensor decom- position of 3D instance matrices and achieves superior performance. In future work, we would like to integrate deep learning into the MIL framework to extract enhanced feature representation of multi-instance data and thus improve classification accuracy. 6. Conclusion Previous studies have demonstrated that the kernel methods usu- ally yield better results compared to other traditional machine learn- ing methods for binary MIL problems. Besides, it has been found that utilizing the multiple data information could improve the per- formance. Thus, a novel method named Double Similarities weighted Multi-Instance Learning kernel that simultaneously integrates the simi- larities of instance-to-bag and bag-to-bag from multi-instance metadata is proposed in this paper. The proposed method utilizes AP clus-
```

**中文翻译**

这段包括 DSMIL 的未来工作和结论。作者提到未来希望把 deep learning 融入 MIL framework，以提取更强的 multi-instance 数据特征表示，从而提升分类准确率。结论中作者指出，已有研究显示 kernel methods 在二分类 MIL 问题上通常优于其他传统机器学习方法；同时，利用多种数据信息能提升性能。因此，本文提出 Double Similarities weighted Multi-Instance Learning kernel，同时从 multi-instance metadata 中整合 instance-to-bag 和 bag-to-bag 相似度。

### 9. gmil-v2-009

- Type: `single_turn`
- Category: `cross_paper_comparison`
- Difficulty: `hard`

**问题**

miGraph/MIGraph 和 GNN-MIL 优缺点怎么比？

**评分意图**

比较 Zhou 2009 graph-kernel MIL 与 Tu 2019 GNN-MIL 在结构建模、学习方式、可扩展性和局限上的差异。

**应覆盖答案点**

- miGraph/MIGraph 通过 graph kernel 改造 bag 相似度，优点是清晰地把 non-i.i.d. 实例关系纳入 kernel 框架。
- MIGraph 显式构图更直观但计算代价高；miGraph 更高效但结构表达更隐式。
- GNN-MIL 将 bag 转图后用 GNN 端到端学习表示，优点是能通过 message passing 学习关系表征。
- GNN-MIL 的局限包括图构建启发式、聚合策略和结构超参数选择会影响结果。
- 二者不是同一技术范式：前者是 kernel/SVM 思路，后者是 neural message passing 思路。

**负面检查项**

- 不要只按年份比较，必须比较方法范式。
- 不要把 graph kernel 和 graph neural network 混为一类。

**应召回 Chunks**

#### Chunk 1: `zhou2009.pdf.pdf` / page `2` / chunk `165`

- Must include terms: `bag`, `graph`

**原文**

```text
should not be treated as i.i.d. samples, and this paper provides a solution. Our basic idea is to regard every bag as an entity to be processed as a whole. There are alternative ways to realize the idea, while in this paper we work by regarding each bag as a graph. McGovern and Jensen (2003) have taken multi-instance learning as a tool to handle relational data where each instance is given as a graph. Here, we are working on proposi- tional data and there is no natural graph. In contrast to having instances as graphs, we regard every bag as a graph and each instance as a node in the graph. 3. The Proposed Methods In this section we propose the MIGraph and miGraph methods. The MIGraph method explicitly maps every bag to an undirected graph and uses a new graph ker- nel to distinguish the positive and negative bags. The miGraph method implicitly constructs graphs by de- riving aﬃnity matrices and deﬁnes an eﬃcient graph kernel considering the clique information.
```

**中文翻译**

这段说明，多实例学习中的实例不应被当作独立同分布样本处理，论文的基本想法是把每个 bag 作为一个整体实体来处理。作者选择把每个 bag 看作一张图：不同于已有工作中“每个 instance 本身就是一张图”的关系数据设定，这里处理的是命题型数据，本来没有天然图结构。因此，论文把每个 bag 映射为图，并把其中每个 instance 作为图节点。本节随后提出两种方法：MIGraph 显式地把每个 bag 映射为无向图，并使用新的 graph kernel 区分正负 bag；miGraph 则通过推导 affinity matrix 隐式构图，并定义一种考虑 clique 信息的高效 graph kernel。

#### Chunk 2: `zhou2009.pdf.pdf` / page `4` / chunk `176`

- Must include terms: `computational complexity`, `MIGraph`

**原文**

```text
tional complexity of kG(Xi, X j) is O(ninj + mimj). The kG clearly satisﬁes all the four major properties that should be considered for a graph kernel deﬁnition (Borgwardt & Kriegel, 2005). 1 Our above design is very simple, but in the next section we can see that the proposed MIGraph method is quite eﬀective. A deﬁciency of MIGraph is that the computational complexity of kG is O(ninj + mimj), dominated by the number of edges. For bags containing a lot of in- stances, there will exist a large number of edges and MIGraph will be hard to execute. So, it is desired to have a method with smaller computational cost. For this purpose, we propose the miGraph method which is simple, eﬃcient but eﬀective. For bag Xi, we can calculate the distance between its instances and derive an aﬃnity matrix W i by com- paring the distances with a threshold δ. For example, if the distance between the instances xia and xiu is smaller than δ, W i’s element at the ath row and uth column, wi
```

**中文翻译**

这段先给出 MIGraph 中图核 kG(Xi, Xj) 的计算复杂度，为 O(n_i n_j + m_i m_j)，并说明该核满足图核定义应考虑的主要性质。作者认为 MIGraph 设计简单且有效，但它的缺点也很明显：复杂度受边数量主导；当 bag 中 instance 很多时，会产生大量边，MIGraph 会很难执行。因此需要计算成本更低的方法。为此作者提出 miGraph，它简单、高效且有效。对每个 bag Xi，miGraph 计算其中 instance 之间的距离，并通过阈值 δ 得到 affinity matrix W_i。

#### Chunk 3: `zhou2009.pdf.pdf` / page `5` / chunk `180`

- Must include terms: `O`, `multi-instance kernel`

**原文**

```text
ab, Wia and Wib should decrease since they become more similar, while other Wiq (q ̸= a, b) should not be aﬀected. It is evident that the computational complexity of kg is as similar as that of the multi-instance kernel shown in Eq. 4, i.e., O(ninj). Note that once the multi-instance kernel is obtained, the Gaussian distances between ev- ery pair of instances have already been calculated, and it is easy to get the W i’s. 4. Experiments 4.1. Benchmark T asks First, we evaluate the proposed MIGraph and mi- Graph methods on ﬁve benchmark data sets popularly used in studies of multi-instance learning, including Musk1, Musk2, Elephant, Fox and Tiger. Musk1 con- tains 47 positive and 45 negative bags, Musk2 contains 39 positive and 63 negative bags, each of the other three data sets contains 100 positive and 100 negative bags. More details of the data sets can be found in (Dietterich et al., 1997; Andrews et al., 2003). We compare MIGraph, miGraph with MI-Kernel
```

**中文翻译**

这段讨论 miGraph 权重设计和计算复杂度。若两个 instance 变得更相似，对应权重 W_ia 和 W_ib 应降低，而其他 W_iq 不应受影响。作者指出，miGraph 的 kernel kg 的计算复杂度与多实例 kernel 类似，即 O(n_i n_j)。因为一旦多实例 kernel 已经得到，所有 instance 对之间的 Gaussian distance 也已计算好，所以很容易得到各个 W_i。随后论文进入实验部分，在 Musk、Elephant、Fox、Tiger 等标准 MIL 数据集上比较 MIGraph、miGraph 和 MI-Kernel 等方法。

#### Chunk 4: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `1` / chunk `1`

- Must include terms: `end-to-end`, `GNN`, `each bag as a graph`

**原文**

```text
Multiple instance learning with graph neural networks Ming Tu1 Jing Huang 1 Xiaodong He 1 Bowen Zhou 1 Abstract Multiple instance learning (MIL) aims to learn the mapping between a bag of instances and the bag-level label. In this paper, we propose a new end-to-end graph neural network (GNN) based al- gorithm for MIL: we treat each bag as a graph and use GNN to learn the bag embedding, in order to explore the useful structural information among instances in bags. The ﬁnal graph representation is fed into a classiﬁer for label prediction. Our algorithm is the ﬁrst attempt to use GNN for MIL. We empirically show that the proposed algorithm achieves the state of the art performance on sev- eral popular MIL data sets without losing model interpretability. 1. Introduction Multiple instance learning (MIL) as a weakly-supervised learning algorithm deals with weakly-labeled data, where each data sample (often named as a bag) has multiple in-
```

**中文翻译**

这段是 Tu 等 2019 论文的标题和摘要。论文提出一种用于多实例学习的端到端图神经网络算法：把每个 bag 看作一张图，并用 GNN 学习 bag embedding，从而探索 bag 内 instance 之间有用的结构信息。最终图表示会输入分类器进行标签预测。作者称这是首次尝试将 GNN 用于 MIL，并在多个常用 MIL 数据集上取得当时最优表现，同时保持模型可解释性。

#### Chunk 5: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `12`

- Must include terms: `information passing`, `graph representation`

**原文**

```text
end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈ RK×D constructed from a bag ofXi, a GNN is ﬁrst applied to the input graph to conduct information passing over the graph. The output graph has the same number of nodes as the input graph, and the computation can be formulated as Zi =GNN embd(Ai,V i), (2) whereZi ∈ RK×D′ is the node embedding of graph output. D′ is the dimension of output node embedding, and can be different from input feature dimensionD. In order to obtain a ﬁxed-dimensional representation of the graph, we need a strategy to aggregate information over the whole graph with adjacency matrix Ai and updated node
```

**中文翻译**

这段描述 GNN-MIL 的信息传递阶段。给定由某个 bag Xi 构造的输入图 Gi，它包含邻接矩阵 Ai 和节点特征矩阵 Vi。首先在输入图上应用 GNN，在图中进行节点间信息传递。输出图与输入图有相同数量的节点，计算形式为 Zi = GNN_embd(Ai, Vi)。其中 Zi 是输出图的节点 embedding，输出维度 D′ 可以不同于输入特征维度 D。为了得到固定维度的图表示，还需要一个策略在整个图上聚合更新后的节点信息。

### 10. gmil-v2-010

- Type: `single_turn`
- Category: `cross_paper_comparison`
- Difficulty: `hard`

**问题**

RGMIL 和 BGNN-MIL 都说图不固定或有不确定性，它们处理方式有什么不同？

**评分意图**

比较 Zhao 2024 RGMIL 与 Pal 2022 BGNN-MIL 对 graph uncertainty/graph design 的不同处理方式。

**应覆盖答案点**

- RGMIL 处理的是 GNN-based MIL 中 bag graph 的边过滤阈值和 GNN 层数如何同步选择的问题。
- RGMIL 用 MADRL 两个 agent 搜索 edge filtering thresholds 和 GNN layers。
- BGMIL/BGNN-MIL 处理的是 bag 之间依赖图可能缺失、有噪声或由启发式构造而不可靠的问题。
- BGMIL 使用 Bayesian GNN 建模 observed graph 与真实关系之间的不确定性。
- 二者都不把初始图当作固定真理，但 RGMIL 偏向结构-架构控制，BGMIL 偏向 Bayesian uncertainty over bag graph。

**负面检查项**

- 不要把二者都泛泛说成“自动学习图结构”。
- 不要把 BGMIL 的 bag-graph 关系误写成只在一个 bag 内连 instance。

**应召回 Chunks**

#### Chunk 1: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `1` / chunk `52`

- Must include terms: `MADRL`, `synchronous control`

**原文**

```text
these issues, we propose a reinforced GNN framework for MIL (RGMIL), pioneering the exploitation of multiagent deep rein- forcement learning (MADRL) in MIL tasks. MADRL enables the flexible definition or extension of factors that influence bag graphs or GNNs and provides synchronous control over them. Moreover, MADRL explores structure-to-architecture correlations while automating adjustments. Experimental results on multiple MIL datasets demonstrate that RGMIL achieves the best performance with excellent explainability. The code and data are available at https://github.com/RingBDStack/RGMIL. Index Terms— Deep reinforcement learning (RL), graph neu- ral network (GNN), multiple instance learning (MIL), neural architecture search. NOMENCLATURE B Set of bag samples. G Set of bag graphs corresponding to B. Y Set of bag-level labels corresponding to G. M Seven-tuple of the Markov game. S State space of M. O Observation space of M. Manuscript received 6 April 2023; revised 14 November
```

**中文翻译**

这段是 RGMIL 的摘要核心。作者针对前文问题提出 reinforced GNN framework for MIL，即 RGMIL，率先在 MIL 任务中利用多智能体深度强化学习（MADRL）。MADRL 允许灵活定义或扩展影响 bag graph 或 GNN 的因素，并对这些因素进行同步控制；同时，它能在自动调节过程中探索图结构与 GNN 架构之间的相关性。多个 MIL 数据集上的实验显示，RGMIL 获得最佳性能并具有较好的可解释性。

#### Chunk 2: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `2` / chunk `62`

- Must include terms: `two agents`, `edge filtering thresholds`, `GNN layers`

**原文**

```text
we divide the training set into equal-sized blocks, one of which serves as the validation set, and the others are used to construct the MG state space. Then, two agents search for edge filtering thresholds and GNN layers, both of which have discrete action spaces. At a time step, each agent picks out an action according to the corresponding partial observations of the current global state, thereby guiding the construction of bag graphs and GNN layers. Since the purpose of GNNs is to improve representation learning, we regard the difference in adjacent performance on the validation set as the current reward. In other words, agents receive a positive reward if the model trained with the current action combination performs better on the validation data than the previous one, and vice versa. Finally, we introduce a novel heuristic state transition function to determine the next global state based on current actions. When the game reaches a Nash
```

**中文翻译**

这段描述 RGMIL 的多智能体搜索过程。作者把训练集划分为等大小 block，其中一个作为验证集，其余用于构建 Markov game 的状态空间。两个 agent 分别搜索 edge filtering threshold 和 GNN layers，这两个对象都有离散动作空间。在每个时间步，每个 agent 根据当前全局状态下自己的局部观察选择动作，从而指导 bag graph 构建和 GNN 层数设置。由于 GNN 的目标是改进表示学习，作者把验证集上相邻性能差异作为当前 reward：若当前动作组合训练出的模型优于上一次，则给正 reward，反之给负 reward。最后，作者引入启发式状态转移函数，根据当前动作决定下一个全局状态。

#### Chunk 3: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `1` / chunk `1128`

- Must include terms: `interactions between bags`, `meaningful graph`

**原文**

```text
to learn effective bag-level representations by suitably com- bining permutation invariant pooling techniques with neural architectures. In this paper, we consider modelling the inter- actions between bags using a graph and employ Graph Neu- ral Networks (GNNs) to facilitate end-to-end learning. Since a meaningful graph representing dependencies between bags is rarely available, we propose to use a Bayesian GNN frame- work that can generate a likely graph structure for scenarios where there is uncertainty in the graph or when no graph is available. Empirical results demonstrate the efﬁcacy of the proposed technique for several MIL benchmark tasks and a distribution regression task. Introduction In numerous supervised learning settings, our aim is to as- sign a label to a group (or bag) of instances as opposed to assigning labels to the individual instances. Example appli- cations include drug activity prediction (Dietterich, Lathrop,
```

**中文翻译**

这段说明 BGMIL/BGNN-MIL 的核心设定。已有方法通过 permutation-invariant pooling 和神经网络架构组合来学习有效的 bag-level representation。本文考虑用一张图来建模 bag 之间的交互，并使用 GNN 进行端到端学习。由于表示 bag 之间依赖关系的有意义图通常很少现成可用，作者提出 Bayesian GNN framework，用于在图结构存在不确定性或没有图时生成可能的图结构。实验证明该技术在多个 MIL benchmark 和一个 distribution regression 任务上有效。

#### Chunk 4: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `3` / chunk `1143`

- Must include terms: `Bayesian GNN`, `observed graph`, `spurious links`

**原文**

```text
Bayesian GNN Framework In many graph based learning problems, the observed graph is constructed from noisy data or derived based on heuris- tics and/or imperfect modelling assumptions. As a result, the observed graph might not represent the true underlying rela- tionship among the data on its nodes; it might contain spu- rious links and important links might be unobserved. How- ever, most existing GNNs do not account for the uncertainty of the graph structure during training. Several recent works such as (Ma et al. 2019; Jiang et al. 2019; Zhang et al. 2019; Pal et al. 2020; Elinas, Bonilla, and Tiao 2020; Wan et al. 2021) address this issue by incorporat- ing probabilistic modelling or joint optimization of the graph during model training. In particular, Zhang et al. (2019) in- troduce a general Bayesian framework, where the observed graph is assumed to be a random sample from a paramet- ric random graph family and posterior inference of the true
```

**中文翻译**

这段解释 Bayesian GNN 的必要性。许多 graph-based learning 问题中的 observed graph 来自噪声数据、启发式方法或不完美建模假设，因此 observed graph 可能无法代表节点数据之间真实的底层关系：它可能包含伪边，也可能缺失重要边。然而多数现有 GNN 在训练时不考虑图结构不确定性。近期一些工作通过概率建模或训练时联合优化图来处理这一问题；其中 Zhang 等提出通用 Bayesian framework，把 observed graph 看作来自某个参数化 random graph family 的随机样本，并对真实图进行后验推断。

### 11. gmil-v2-011

- Type: `single_turn`
- Category: `cross_paper_comparison`
- Difficulty: `hard`

**问题**

RGMIL 和 DSMIL 都是 2024 的 MIL 改进，一个是 GNN 一个是 kernel 吗？优缺点差别是什么？

**评分意图**

比较 Zhao 2024 RGMIL 与 2024 DSMIL 的方法范式、贡献和局限。

**应覆盖答案点**

- 应确认 RGMIL 是 reinforced GNN framework，DSMIL 是 Double Similarities weighted MIL kernel/SVM 框架。
- RGMIL 优点是能同步控制 bag graph 边阈值和 GNN 层数，适应不同图密度和聚合范围。
- RGMIL 代价是引入强化学习搜索和更多训练控制复杂度。
- DSMIL 优点是同时融合 I2B 和 B2B 相似度，kernel 设计清晰且有 SVM 路径。
- DSMIL 局限是仍依赖 kernel/相似度计算，测试时间可能因两类相似度和 AP clustering 等步骤变长，作者未来希望融合深度学习。

**负面检查项**

- 不要因为二者都是 2024 就把贡献混在一起。
- 不要把 DSMIL 写成 reinforced GNN，也不要把 RGMIL 写成 kernel SVM。

**应召回 Chunks**

#### Chunk 1: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `1` / chunk `52`

- Must include terms: `reinforced GNN`, `MADRL`

**原文**

```text
these issues, we propose a reinforced GNN framework for MIL (RGMIL), pioneering the exploitation of multiagent deep rein- forcement learning (MADRL) in MIL tasks. MADRL enables the flexible definition or extension of factors that influence bag graphs or GNNs and provides synchronous control over them. Moreover, MADRL explores structure-to-architecture correlations while automating adjustments. Experimental results on multiple MIL datasets demonstrate that RGMIL achieves the best performance with excellent explainability. The code and data are available at https://github.com/RingBDStack/RGMIL. Index Terms— Deep reinforcement learning (RL), graph neu- ral network (GNN), multiple instance learning (MIL), neural architecture search. NOMENCLATURE B Set of bag samples. G Set of bag graphs corresponding to B. Y Set of bag-level labels corresponding to G. M Seven-tuple of the Markov game. S State space of M. O Observation space of M. Manuscript received 6 April 2023; revised 14 November
```

**中文翻译**

这段是 RGMIL 的摘要核心。作者针对前文问题提出 reinforced GNN framework for MIL，即 RGMIL，率先在 MIL 任务中利用多智能体深度强化学习（MADRL）。MADRL 允许灵活定义或扩展影响 bag graph 或 GNN 的因素，并对这些因素进行同步控制；同时，它能在自动调节过程中探索图结构与 GNN 架构之间的相关性。多个 MIL 数据集上的实验显示，RGMIL 获得最佳性能并具有较好的可解释性。

#### Chunk 2: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `13` / chunk `135`

- Must include terms: `automated`, `synchronized control`, `GNN`

**原文**

```text
augment GNN-based MIL. TABLE VI TIME CONSUMPTION (IN MINUTES) OF A REPETITION OF SIX GNN-BASED MIL ALGORITHMS ON THE ANIMAL DETECTION TASKS VI. CONCLUSION In this article, we propose a new GNN-based MIL frame- work RGMIL, enabling automated and synchronized control of bag structures (the edge filtering threshold) and GNN archi- tectures (the number of GNN layers). RGMIL presents a novel avenue for subsequent graph data mining studies, allowing the use of MADRL to search for nodes beyond the scope of graph neural architectures, an aspect unattainable through traditional GNAS methods. The experimental findings indicate that balancing the edge density and the aggregation scope enhances the untapped potential of GNNs. In the future work, we will explore refining instance-level reinforcement control in MIL, striving to utilize the MADRL while simultaneously reducing time consumption. REFERENCES [1] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep mul-
```

**中文翻译**

这段是 RGMIL 结论。作者提出新的 GNN-based MIL framework RGMIL，能够对 bag structures（edge filtering threshold）和 GNN architectures（GNN layers 数量）进行自动化、同步控制。RGMIL 为后续图数据挖掘研究提供了一条新路径：用 MADRL 搜索 graph neural architecture 范围之外的节点或因素，这是传统 GNAS 方法难以做到的。实验表明，平衡 edge density 和 aggregation scope 能释放 GNN 的潜力。未来工作将探索更细粒度的 instance-level reinforcement control，同时降低 MADRL 带来的时间消耗。

#### Chunk 3: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `1` / chunk `262`

- Must include terms: `Double Similarities`, `kernel`

**原文**

```text
applications. However, most existing MIL methods just utilize partial information (bags or instances) of MIL data to construct the kernel function, resulting in deteriorated classification performance of MIL. In this paper, we propose a Double Similarities weighted Multi-Instance Learning (DSMIL) kernel framework, which utilizes the similarities of Bag-to-Bag (B2B) and Instance-to-Bag (I2B). In the proposed kernel framework, the similarities of B2B and I2B could be derived from the prototypes distance of inter-bag and similarity matrix of intra-bag, respectively, based on the affinity propagation (AP) clustering of the bag. Meanwhile, we give theoretical proof of the validity of the designed kernel function. Experimental results on benchmark and semi- synthetic datasets show that our proposed method obtains competitive classification performance and achieves robustness to parameters and noise. 1. Introduction Multi-instance learning (MIL), which originated from drug activity
```

**中文翻译**

这段是 DSMIL 摘要核心。作者指出，多数已有 MIL 方法只利用 MIL 数据中的部分信息（bag 或 instance）来构造 kernel function，导致分类性能下降。论文提出 Double Similarities weighted Multi-Instance Learning kernel framework，简称 DSMIL，同时利用 Bag-to-Bag（B2B）和 Instance-to-Bag（I2B）两类相似度。在该 kernel 框架中，B2B 和 I2B 分别可由 bag 间 prototype distance 和 bag 内 similarity matrix 得到，并基于 bag 的 affinity propagation clustering。作者还给出设计的 kernel function 有效性的理论证明。基准和半合成数据集实验显示，该方法具有竞争性分类性能，并对参数和噪声有鲁棒性。

#### Chunk 4: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `9` / chunk `315`

- Must include terms: `testing time`, `I2B`, `B2B`

**原文**

```text
In the case of DSMIL-H and DSMIL-T, the bag is mapped into a new feature space after computing the two similarities (I2B and B2B), which is equivalent to inputting one feature vector for each bag to train. In addition, we can find from the table that the testing time for MI-SVM and mi-SVM is similar, while the testing time for DSMIL-H and DSMIL-T is longer because DSMIL-H and DSMIL-T require the calculation of two similarities (I2B and B2B) in the test phase. In the computation of two similarities (I2B and B2B), AP clustering is performed for each bag. The time complexity of AP clustering for each bag is (𝑛2 log 𝑛), and thus it costs (𝑁𝑛 2 log 𝑛) to compute the I2B for all bags, where 𝑁 is the number of bags and 𝑛 is the number of instances in a bag that is variable in each bag. For DSMIL-H, the complexity of calculating the B2B is (𝑁 2𝑐1𝑐2), and in the case of DSMIL-T, it is (4𝑁 2𝑐1𝑐2), where 𝑐1 and 𝑐2 refer to the number of
```

**中文翻译**

这段讨论 DSMIL 的测试时间和复杂度。对 DSMIL-H 和 DSMIL-T 来说，bag 在计算 I2B 和 B2B 两类相似度后会被映射到新的特征空间，相当于每个 bag 输入一个特征向量进行训练。表中可以看到 MI-SVM 和 mi-SVM 的测试时间相近，而 DSMIL-H 和 DSMIL-T 测试更慢，因为测试阶段需要计算 I2B 和 B2B 两类相似度。计算这两类相似度时，每个 bag 都要执行 AP clustering；每个 bag 的 AP clustering 复杂度为 O(n^2 log n)，因此为所有 bag 计算 I2B 的复杂度为 O(N n^2 log n)。DSMIL-H 和 DSMIL-T 的 B2B 计算复杂度也随 bag 数和 cluster 数增加。

#### Chunk 5: `2024 - Double similarities weighted multi-instance learning kernel and its application.pdf` / page `11` / chunk `321`

- Must include terms: `future work`, `deep learning`

**原文**

```text
TensMIL2 ( Papastergiou et al. , 2019 ) re-designs a feature extraction algorithm for the color image datasets, which performs tensor decom- position of 3D instance matrices and achieves superior performance. In future work, we would like to integrate deep learning into the MIL framework to extract enhanced feature representation of multi-instance data and thus improve classification accuracy. 6. Conclusion Previous studies have demonstrated that the kernel methods usu- ally yield better results compared to other traditional machine learn- ing methods for binary MIL problems. Besides, it has been found that utilizing the multiple data information could improve the per- formance. Thus, a novel method named Double Similarities weighted Multi-Instance Learning kernel that simultaneously integrates the simi- larities of instance-to-bag and bag-to-bag from multi-instance metadata is proposed in this paper. The proposed method utilizes AP clus-
```

**中文翻译**

这段包括 DSMIL 的未来工作和结论。作者提到未来希望把 deep learning 融入 MIL framework，以提取更强的 multi-instance 数据特征表示，从而提升分类准确率。结论中作者指出，已有研究显示 kernel methods 在二分类 MIL 问题上通常优于其他传统机器学习方法；同时，利用多种数据信息能提升性能。因此，本文提出 Double Similarities weighted Multi-Instance Learning kernel，同时从 multi-instance metadata 中整合 instance-to-bag 和 bag-to-bag 相似度。

### 12. gmil-v2-012

- Type: `single_turn`
- Category: `out_of_knowledge_base`
- Difficulty: `hard`

**问题**

GraphMIL-Transformer++ 2026 这篇在知识库里讲了什么？

**评分意图**

检测知识库外问题：当前语料中没有 GraphMIL-Transformer++ 2026 时，模型应说明未找到证据并避免编造。

**应覆盖答案点**

- 应明确说明当前知识库/检索结果中没有找到这篇论文或无法确认。
- 可以建议用户提供论文 PDF、标题或 DOI 后再总结。
- 如果检索结果只包含其他 graph MIL 论文，应说明这些不能作为该 2026 论文的证据。
- 不得编造作者、方法模块、实验数据或结论。

**负面检查项**

- 如果回答给出具体架构、数据集或性能数字但没有来源，应判为失败。
- 不要把 RGMIL、TAD-Graph、Patch-GCN 等相近论文当成 GraphMIL-Transformer++。

**应召回 Chunks**

无应召回 chunk。该题用于测试知识库外拒答：系统应说明当前知识库没有找到这篇论文或无法确认，不应编造作者、方法模块、数据集或实验数字。

### 13. gmil-v2-013:t1

- Type: `multi_turn`
- Category: `context_memory_summary`
- Difficulty: `medium`

**问题**

MIL-GNN 的流程是什么？

**评分意图**

用户把 GNN-MIL 写成 MIL-GNN 时，系统仍应定位 Tu 2019 并说明方法流程。

**应覆盖答案点**

- 应把 MIL-GNN 纠正或关联到 GNN-MIL/Tu 2019，而不是另找一篇不存在的 MIL-GNN。
- 流程包括 bag-to-graph：实例作为节点，邻接矩阵由启发式策略构造。
- 在构造的 graph 上用 GNN 做 message passing。
- 用 graph aggregation/differentiable pooling 得到 bag embedding。
- 最后用 MLP/分类器预测 bag label。

**负面检查项**

- 不要因为用户写 MIL-GNN 就编造一个新缩写。
- 不要只说“GNN 用于 MIL”，要说明流程。

**应召回 Chunks**

#### Chunk 1: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `1` / chunk `1`

- Must include terms: `GNN`, `each bag as a graph`

**原文**

```text
Multiple instance learning with graph neural networks Ming Tu1 Jing Huang 1 Xiaodong He 1 Bowen Zhou 1 Abstract Multiple instance learning (MIL) aims to learn the mapping between a bag of instances and the bag-level label. In this paper, we propose a new end-to-end graph neural network (GNN) based al- gorithm for MIL: we treat each bag as a graph and use GNN to learn the bag embedding, in order to explore the useful structural information among instances in bags. The ﬁnal graph representation is fed into a classiﬁer for label prediction. Our algorithm is the ﬁrst attempt to use GNN for MIL. We empirically show that the proposed algorithm achieves the state of the art performance on sev- eral popular MIL data sets without losing model interpretability. 1. Introduction Multiple instance learning (MIL) as a weakly-supervised learning algorithm deals with weakly-labeled data, where each data sample (often named as a bag) has multiple in-
```

**中文翻译**

这段是 Tu 等 2019 论文的标题和摘要。论文提出一种用于多实例学习的端到端图神经网络算法：把每个 bag 看作一张图，并用 GNN 学习 bag embedding，从而探索 bag 内 instance 之间有用的结构信息。最终图表示会输入分类器进行标签预测。作者称这是首次尝试将 GNN 用于 MIL，并在多个常用 MIL 数据集上取得当时最优表现，同时保持模型可解释性。

#### Chunk 2: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `10`

- Must include terms: `Graph building`, `GNN`, `Graph aggregation`

**原文**

```text
dimension of node feature. While the mapping from bag space to graph space can be done heuristically (will be introduced in next subsection), the key of graph based MIL is how to learn the mapping from graph space to label space. Graph-level classiﬁcation 1Our code will be published after review. Graph building GNNembd Graph aggregation Graph embeddingMLPPrediction Input bags Figure 1. GNN based MIL framework overview. usually involves deriving a good representation of graphs given variant number of nodes and different graph struc- tures, which requires to reduce the input graph to a ﬁxed- dimensional feature vector. In this paper, we focus on GNN based graph representation learning for MIL, and propose a new angle to solve the MIL problem in the current study. 2.2. Proposed algorithm Figure 1 illustrate the diagram of our proposed framework on GNN based MIL. First, to convert input bags of instances to graphs, we adopt a heuristic strategy similar with the one
```

**中文翻译**

这段说明 GNN-based MIL 的整体框架。bag space 到 graph space 的映射可以用启发式策略完成，但 graph-based MIL 的关键是如何学习从 graph space 到 label space 的映射。图级分类需要把节点数量和图结构都可能变化的输入图，压缩成固定维度的特征向量。图 1 的流程是：Input bags -> Graph building -> GNN embedding -> Graph aggregation -> Graph embedding -> MLP prediction。论文聚焦于用 GNN 做 MIL 的 graph representation learning，并提出一个解决 MIL 的新视角。

#### Chunk 3: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `11`

- Must include terms: `convert input bags`, `adjacency matrix`

**原文**

```text
Figure 1 illustrate the diagram of our proposed framework on GNN based MIL. First, to convert input bags of instances to graphs, we adopt a heuristic strategy similar with the one used in (Zhou et al., 2009). Given a bag with instances [x(i) 1 , x(i) 2 , · · ·, x(i) K ], the adjacency matrixA can be derived with the following formula: Amn = { 1 ifdist(x(i) m, x(i) n )<η 0 otherwise (1) wheredist(x(i) m, x(i) n ) is the distance betweenm-th andn-th instance in bag i. In this study, Euclidean distance is em- ployed for simplicity.η is the threshold to decide whether there is an edge between two instances based on their dis- tance.η = 0 means there is no edge in the input graph while η = + ∞ means the input is a complete graph. η can be tuned for speciﬁc tasks. After converting bags of instances to graphs, we propose an end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈
```

**中文翻译**

这段给出 GNN-MIL 的图构建方法。为了把输入 bag 中的 instance 转成图，作者采用类似 Zhou 2009 的启发式策略。给定一个包含多个 instance 的 bag，通过距离阈值 η 构造邻接矩阵 A：若第 m 个和第 n 个 instance 的距离小于 η，则 A_mn = 1，否则为 0。本文为简单起见使用欧氏距离。η 用来决定两个 instance 之间是否连边；η = 0 表示输入图没有边，η = +∞ 表示完全图，η 可以针对具体任务调节。完成 bag-to-graph 后，作者提出基于 GNN 的端到端 graph representation learning 算法。

#### Chunk 4: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `12`

- Must include terms: `information passing`, `graph representation`

**原文**

```text
end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈ RK×D constructed from a bag ofXi, a GNN is ﬁrst applied to the input graph to conduct information passing over the graph. The output graph has the same number of nodes as the input graph, and the computation can be formulated as Zi =GNN embd(Ai,V i), (2) whereZi ∈ RK×D′ is the node embedding of graph output. D′ is the dimension of output node embedding, and can be different from input feature dimensionD. In order to obtain a ﬁxed-dimensional representation of the graph, we need a strategy to aggregate information over the whole graph with adjacency matrix Ai and updated node
```

**中文翻译**

这段描述 GNN-MIL 的信息传递阶段。给定由某个 bag Xi 构造的输入图 Gi，它包含邻接矩阵 Ai 和节点特征矩阵 Vi。首先在输入图上应用 GNN，在图中进行节点间信息传递。输出图与输入图有相同数量的节点，计算形式为 Zi = GNN_embd(Ai, Vi)。其中 Zi 是输出图的节点 embedding，输出维度 D′ 可以不同于输入特征维度 D。为了得到固定维度的图表示，还需要一个策略在整个图上聚合更新后的节点信息。

### 14. gmil-v2-013:t2

- Type: `multi_turn`
- Category: `context_memory_pros_cons`
- Difficulty: `hard`

**问题**

总结其优缺点。

**评分意图**

在多轮上下文中，“其”应指代 GNN-MIL/Tu 2019；总结该方法优缺点。

**应覆盖答案点**

- 应保持上下文，明确“其”指 GNN-MIL/Tu 2019。
- 优点：把实例关系显式放入 graph，GNN message passing 能建模 instance 间关系。
- 优点：端到端学习 bag representation，并提供 graph aggregation/differentiable pooling 等聚合路径。
- 优点：相比传统 graph-kernel，学习到的 representation 更灵活。
- 局限：bag-to-graph 依赖启发式邻接构造，阈值/图质量会影响结果。
- 局限：graph aggregation 或 pooling 策略会影响表现，后续工作需要处理图密度、层数、WSI 节点对齐等问题。

**负面检查项**

- 不要把“其”误解为上一轮检索到的其他论文。
- 不要只给泛泛优缺点，必须围绕 GNN-MIL 的构图、message passing 和聚合。

**应召回 Chunks**

#### Chunk 1: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `11`

- Must include terms: `heuristic strategy`, `adjacency matrix`

**原文**

```text
Figure 1 illustrate the diagram of our proposed framework on GNN based MIL. First, to convert input bags of instances to graphs, we adopt a heuristic strategy similar with the one used in (Zhou et al., 2009). Given a bag with instances [x(i) 1 , x(i) 2 , · · ·, x(i) K ], the adjacency matrixA can be derived with the following formula: Amn = { 1 ifdist(x(i) m, x(i) n )<η 0 otherwise (1) wheredist(x(i) m, x(i) n ) is the distance betweenm-th andn-th instance in bag i. In this study, Euclidean distance is em- ployed for simplicity.η is the threshold to decide whether there is an edge between two instances based on their dis- tance.η = 0 means there is no edge in the input graph while η = + ∞ means the input is a complete graph. η can be tuned for speciﬁc tasks. After converting bags of instances to graphs, we propose an end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈
```

**中文翻译**

这段给出 GNN-MIL 的图构建方法。为了把输入 bag 中的 instance 转成图，作者采用类似 Zhou 2009 的启发式策略。给定一个包含多个 instance 的 bag，通过距离阈值 η 构造邻接矩阵 A：若第 m 个和第 n 个 instance 的距离小于 η，则 A_mn = 1，否则为 0。本文为简单起见使用欧氏距离。η 用来决定两个 instance 之间是否连边；η = 0 表示输入图没有边，η = +∞ 表示完全图，η 可以针对具体任务调节。完成 bag-to-graph 后，作者提出基于 GNN 的端到端 graph representation learning 算法。

#### Chunk 2: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `2` / chunk `12`

- Must include terms: `information passing`, `graph representation`

**原文**

```text
end-to-end graph representation learning algorithm based on GNN for MIL. Given an input graphGi with adjacency matrix Ai ∈ { 0, 1}K×K and node feature matrix Vi ∈ RK×D constructed from a bag ofXi, a GNN is ﬁrst applied to the input graph to conduct information passing over the graph. The output graph has the same number of nodes as the input graph, and the computation can be formulated as Zi =GNN embd(Ai,V i), (2) whereZi ∈ RK×D′ is the node embedding of graph output. D′ is the dimension of output node embedding, and can be different from input feature dimensionD. In order to obtain a ﬁxed-dimensional representation of the graph, we need a strategy to aggregate information over the whole graph with adjacency matrix Ai and updated node
```

**中文翻译**

这段描述 GNN-MIL 的信息传递阶段。给定由某个 bag Xi 构造的输入图 Gi，它包含邻接矩阵 Ai 和节点特征矩阵 Vi。首先在输入图上应用 GNN，在图中进行节点间信息传递。输出图与输入图有相同数量的节点，计算形式为 Zi = GNN_embd(Ai, Vi)。其中 Zi 是输出图的节点 embedding，输出维度 D′ 可以不同于输入特征维度 D。为了得到固定维度的图表示，还需要一个策略在整个图上聚合更新后的节点信息。

#### Chunk 3: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `3` / chunk `14`

- Must include terms: `Differentiable pooling`, `assignment matrix`

**原文**

```text
number of nodes to a vector representation. Differentiable pooling is composed of two operations: 1) learning an as- signment matrix for the graph which gives the probability of a node belongs to a cluster. 2) collapsing graph nodes to the number of clusters by soft pooling given the learned assign- ment matrix. The number of clusters is predeﬁned and the same for different graphs. The advantage of differentiable pooling is that it is able to learn the graph representation in a hierarchical way by doing graph clustering in multiple steps. Besides differentiable pooling based algorithm, we also implement an attention-based graph aggregation algorithm on top of Zi, which is similar to the attention based MIL in (Ilse et al., 2018), to show that our proposed paradigm is not limited to one speciﬁc graph representation learning algorithm. We will show the implementation details of both graph aggregation algorithms in supplementary materials. 3. Experiments
```

**中文翻译**

这段解释 GNN-MIL 的 differentiable pooling。Differentiable pooling 包含两个操作：第一，学习图的 assignment matrix，表示每个节点属于某个 cluster 的概率；第二，根据学习到的 assignment matrix 用 soft pooling 把图节点折叠到预设数量的 cluster。cluster 数量是预定义的，并且不同图中相同。Differentiable pooling 的优势是能通过多步图聚类，以层次化方式学习 graph representation。此外，作者还在 Zi 上实现了 attention-based graph aggregation，类似 Ilse 等 2018 的 attention MIL，用来说明该范式不限于某一种具体 graph representation learning 算法。

#### Chunk 4: `Tu 等 - 2019 - Multiple instance learning with graph neural netwo.pdf` / page `7` / chunk `36`

- Must include terms: `attention`, `message passing`

**原文**

```text
yield a ﬁxed-dimensional embedding for each bag. How- ever, we apply attention to the output of GNN after message passing among nodes of the input graph ( Zi in equation 2 in main text). Formally, assume j-th node feature zj i of i-th bag feature matrix Zi, the graph embedding can be calculated with the following equations: V∗ i = ∑ αj i zj i, (7) whereαj i is obtained by: αj i =softmax (MLP att(zj i )). (8) softmax () converts the output ofMLP att() to a probabil- ity by normalizing over all instances. With the attention based graph embeddingV∗ i , we can add multiple layers of MLPs to get a prediction of the bag label. Similarly, we use the same DS technique as in the last sub- section. It is reasonable to note that the differential pooling based graph aggregation utilize the graph relation informa- tion during the aggregation process while the attention based algorithm does not. 6. Data sets and more details of experiments The proposed GNN based MIL is evaluated with different
```

**中文翻译**

这段说明 attention-based graph aggregation 的做法。作者对 GNN 信息传递之后的输出 Zi 使用 attention，从而为每个 bag 得到固定维度 embedding。形式上，对第 i 个 bag 的第 j 个节点特征 z_i^j，用 attention 权重 α_i^j 加权求和得到 graph embedding V_i*；α_i^j 由 MLP_att 后接 softmax 得到，softmax 在所有 instance 上归一化概率。得到 attention-based graph embedding 后，可以叠加多层 MLP 预测 bag label。作者也指出，differentiable pooling 在聚合过程中利用了图关系信息，而 attention-based algorithm 本身没有。

### 15. gmil-v2-013:t3

- Type: `multi_turn`
- Category: `context_memory_followup`
- Difficulty: `hard`

**问题**

有哪些论文是在这个思路上继续改进的？

**评分意图**

在多轮上下文中，“这个思路”应指 GNN-MIL 的 bag graph + GNN 表示学习路线；列出后续相关改进论文。

**应覆盖答案点**

- 应保持上下文，知道“这个思路”指 GNN-MIL/Tu 2019 的图神经网络 MIL 路线。
- 应列出 RGMIL/Zhao 2024：继续在 GNN-based MIL 中调节 bag graph 边阈值和 GNN 层数。
- 可列出 BGMIL/BGNN-MIL/Pal 2022：把 GNN 用到 bag 之间的 graph，并用 Bayesian GNN 处理图不确定性；应说明它不是简单的 bag 内构图改进。
- 可列出 Patch-GCN/Chen 2021：把 WSI patch 建成空间 kNN 图并用 GCN 聚合上下文。
- 可列出 NAGCN/Guan 2022：通过 node alignment 和 global-to-local clustering 改善 WSI 图表示。
- 应避免把 miGraph/MIGraph 或 DSMIL 当成 GNN-MIL 的神经网络后续改进。

**负面检查项**

- 不要把“这个思路”误解为 DSMIL 的 double similarities kernel。
- 不要把 graph-kernel 方法列为 GNN-MIL 的神经网络改进。
- 不要只回答“有很多后续论文”，需要给出具体论文和改进点。

**应召回 Chunks**

#### Chunk 1: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `1` / chunk `52`

- Must include terms: `RGMIL`, `GNN`, `synchronous control`

**原文**

```text
these issues, we propose a reinforced GNN framework for MIL (RGMIL), pioneering the exploitation of multiagent deep rein- forcement learning (MADRL) in MIL tasks. MADRL enables the flexible definition or extension of factors that influence bag graphs or GNNs and provides synchronous control over them. Moreover, MADRL explores structure-to-architecture correlations while automating adjustments. Experimental results on multiple MIL datasets demonstrate that RGMIL achieves the best performance with excellent explainability. The code and data are available at https://github.com/RingBDStack/RGMIL. Index Terms— Deep reinforcement learning (RL), graph neu- ral network (GNN), multiple instance learning (MIL), neural architecture search. NOMENCLATURE B Set of bag samples. G Set of bag graphs corresponding to B. Y Set of bag-level labels corresponding to G. M Seven-tuple of the Markov game. S State space of M. O Observation space of M. Manuscript received 6 April 2023; revised 14 November
```

**中文翻译**

这段是 RGMIL 的摘要核心。作者针对前文问题提出 reinforced GNN framework for MIL，即 RGMIL，率先在 MIL 任务中利用多智能体深度强化学习（MADRL）。MADRL 允许灵活定义或扩展影响 bag graph 或 GNN 的因素，并对这些因素进行同步控制；同时，它能在自动调节过程中探索图结构与 GNN 架构之间的相关性。多个 MIL 数据集上的实验显示，RGMIL 获得最佳性能并具有较好的可解释性。

#### Chunk 2: `Zhao 等 - 2024 - Reinforced GNNs for Multiple Instance Learning.pdf` / page `3` / chunk `66`

- Must include terms: `edge density`, `aggregation range`

**原文**

```text
ZHAO et al.: REINFORCED GNNs FOR MULTIPLE INSTANCE LEARNING 3 the correlations between edge density and aggregation range, which have been overlooked in previous MIL studies. 3) Extensive experiment results demonstrate that the RGMIL outperforms the state-of-the-art baselines, espe- cially on benchmark and text datasets, with an average accuracy improvement of 2%–3%. Besides, RGMIL achieves superior result explainability than the existing methods. This article is organized as follows: Section II introduces related works. Section III describes preliminaries. Sections IV and V present the methodology and experiments. Section VI summarizes this work. II. RELATED WORKS In this section, we introduce two categories of related works. They are GNN-based MIL algorithms and graph neural architecture search (GNAS) with reinforcement learning (RL). A. GNN-Based MIL MIL [22], [23], [24] is receiving broad recognition due to its ability to process ambiguous labels in real-world appli-
```

**中文翻译**

这段强调 RGMIL 的贡献之一：以往 MIL 研究忽略了 edge density 与 aggregation range 之间的相关性。大量实验表明，RGMIL 优于当时 state-of-the-art baseline，尤其在 benchmark 和 text datasets 上平均准确率提升 2% 到 3%，并且结果可解释性更好。随后论文结构安排包括相关工作、预备知识、方法、实验和总结。相关工作部分分为两类：GNN-based MIL algorithms，以及使用 reinforcement learning 的 graph neural architecture search（GNAS）。

#### Chunk 3: `Pal 等 - 2022 - Bag Graph Multiple Instance Learning Using Bayesi.pdf` / page `1` / chunk `1131`

- Must include terms: `GNNs`, `instances within a bag`, `key observation`

**原文**

```text
In (Zhang et al. 2011), a relational graph was used to spec- ify similarities between instances. With the recent advances in graph neural networks (GNNs), there have been efforts to use these to represent the structure of instances within a bag (Tu et al. 2019; Yin et al. 2019). Our key observation is that while graphs have been used to model relationships between instances, they have not been employed to specify relationships between bags. In some applications, side-information provides a clear mechanism for constructing a graph. For example, in a real estate ap- plication when the goal is to predict mean neighborhood rental prices, we may assume that nearby neighborhoods have similar pricing (Valkanas, Regol, and Coates 2020). A graph can then be constructed with edges representing geo- graphic proximity. The identiﬁed dependencies are valuable in a graph-based learning framework, leading to improved predictive performance. In other cases, there is no graph
```

**中文翻译**

这段明确区分 bag 内 instance graph 和 bag 间 graph。已有工作使用 relational graph 表示 instance 之间的相似性；随着 GNN 发展，也有人用 GNN 表示一个 bag 内 instance 的结构。作者的关键观察是：虽然图已经被用于建模 instance 之间的关系，但还没有被用于指定 bag 之间的关系。在某些应用中，side-information 可以清晰地构造图，例如房地产任务中可用地理邻近性连接社区；这些依赖关系对 graph-based learning 有价值并能提升预测表现。但其他情况下并没有现成图。

#### Chunk 4: `Chen 等 - 2021 - Whole Slide Images are 2D Point Clouds Context-Aware Survival Prediction Using Patch-Based Graph Co.pdf` / page `1` / chunk `589`

- Must include terms: `Patch-GCN`, `context-aware`, `graph convolutional network`

**原文**

```text
not context-aware and are unable to model important morphological fea- ture interactions between cell identities and tissue types that are prognos- tic for patient survival. In this work, we present Patch-GCN, a context- aware, spatially-resolved patch-based graph convolutional network that hierarchically aggregates instance-level histology features to model local- and global-level topological structures in the tumor microenvironment. We validate Patch-GCN with 4,370 gigapixel WSIs across ﬁve diﬀerent cancer types from the Cancer Genome Atlas (TCGA), and demonstrate that Patch-GCN outperforms all prior weakly-supervised approaches by 3.58-9.46%. Our code and corresponding models are publicly available at https://github.com/mahmoodlab/Patch-GCN. Keywords: Computer Vision · Computational Pathology · Weakly-Supervised Learning · Graph Convolutional Networks · Interpretability 1 Introduction Weakly-supervised deep learning has made remarkable progress in computational
```

**中文翻译**

这段介绍 Patch-GCN。作者指出已有弱监督方法不具备 context-aware 能力，无法建模细胞身份和组织类型之间对患者生存预后重要的形态特征交互。本文提出 Patch-GCN：一种 context-aware、spatially-resolved、patch-based graph convolutional network，能够层次化聚合 instance-level 组织学特征，以建模肿瘤微环境中的局部和全局拓扑结构。作者在 TCGA 的五种癌症类型、4370 张 gigapixel WSI 上验证，Patch-GCN 比此前弱监督方法提升 3.58% 到 9.46%。

#### Chunk 5: `Chen 等 - 2021 - Whole Slide Images are 2D Point Clouds Context-Aware Survival Prediction Using Patch-Based Graph Co.pdf` / page `4` / chunk `599`

- Must include terms: `coordinates`, `adjacency matrix`, `k-NN`

**原文**

```text
matrix Xj ∈ Rm×1024 for Mj total patches in Wj. For each patch, we save (x,y)-coordinates from the tissue segmentation, from which we use to build an adjacency matrixAj for eachWj via fast approximate k-NN (k = 8) that models a 3× 3 image receptive ﬁeld in CNN convolutions. Finally, we build a subgraph Gj = (Xj,A j), with the patient-level graph across all WSIs constructed as G = {Gj}j=1 which we denote as a WSI-Graph. In comparison to previous graph-based approaches that build neighborhoods us- ing nearest neighbors in the embedding space, our approach is distinct in that graphs are constructed in the Euclidean space. As a result, WSI-Graphs are ef- fectively 2D point clouds (e.g. nodes / points connected to other proximal points in a 2D planar grid), which allows us to leverage spatial convolutions that per- form local neighborhood aggregation functions similar to CNNs. In comparison to CNNs, however, Path-GCN is able to tractably perform CNN-like convolution
```

**中文翻译**

这段说明 Patch-GCN 如何从 WSI patch 构图。对每张 WSI，作者得到 patch 特征矩阵 Xj，并为每个 patch 保存组织分割得到的 (x, y) 坐标。然后用快速近似 k-NN（k = 8）为每张 WSI 构造邻接矩阵 Aj，该邻接关系模拟 CNN 卷积中的 3 x 3 感受野。随后构建子图 Gj = (Xj, Aj)，并把病人层面的所有 WSI 图记为 WSI-Graph。与在 embedding space 中构造邻域的既有图方法不同，Patch-GCN 在 Euclidean space 中构图，因此 WSI-Graph 本质上是 2D point cloud，节点连接到二维平面网格中邻近节点，使其能够利用类似 CNN 的空间卷积进行局部邻域聚合。

#### Chunk 6: `Guan 等 - 2022 - Node-aligned graph convolutional network for whole-slide image representation and classification.pdf` / page `2` / chunk `730`

- Must include terms: `global-to-local`, `Node-Aligned GCN`

**原文**

```text
In order to retain local structural information as well as global distribution, we propose a hierarchical global- to-local clustering strategy to build a Node-Aligned GCN (NAGCN§) for whole-slide image representation and classi- fication. First, to filter out redundant information and select discriminative instances, we borrow the idea from BOVW and construct a codebook by leveraging a global clustering operation to instance features in the dataset. The codebook is comprised of amounts of visual words, where each vi- sual word corresponds to a specific tissue type. Through the global clustering, we can divide instances from WSI bags into distinct sub-bags (each sub-bag corresponds to a visual word) and build correspondence across different WSIs at the sub-bag level. Second, we perform a lo- cal clustering-based sampling strategy to select typical in- stances within sub-bags for each WSI and use them as graph nodes. Finally, different from BOVW which only uses a
```

**中文翻译**

这段介绍 NAGCN 的 global-to-local 构图。为了同时保留局部结构信息和全局分布，作者提出 hierarchical global-to-local clustering strategy 来构建 Node-Aligned GCN。首先，为了过滤冗余信息并选择有判别力的 instance，作者借鉴 BOVW，用数据集中 instance features 做 global clustering 构造 codebook；codebook 由许多 visual words 组成，每个 visual word 对应一种特定组织类型。通过 global clustering，可以把 WSI bag 中的 instance 划分为不同 sub-bag，并在 sub-bag 层面建立不同 WSI 之间的对应关系。第二步，作者在每个 WSI 的 sub-bag 内执行 local clustering-based sampling，选择典型 instance 作为图节点。

#### Chunk 7: `Guan 等 - 2022 - Node-aligned graph convolutional network for whole-slide image representation and classification.pdf` / page `2` / chunk `731`

- Must include terms: `node-aligned graph`, `trainable WSI`

**原文**

```text
cal clustering-based sampling strategy to select typical in- stances within sub-bags for each WSI and use them as graph nodes. Finally, different from BOVW which only uses a non-trainable frequency histogram to represent WSIs, we deploy the node-aligned graph to achieve trainable WSI em- beddings. Since our graph construction strategy ensures the alignment among different WSIs, WSI-level representation can be easily generated, which can be used for the subse- quent classification. We summarize our technical contributions as follows: §GitHub repository: https://github.com/YohnGuan/NAGCN 1. We introduce a novel node-aligned GCN for WSI rep- resentation and classification with only slide-level an- notations. Compared with other graph-based MIL methods that have to perform non-ordered pooling to generate slide-level representation, our aligned graphs can establish node correspondence among WSIs, thus having more options to get the global representation, such as flattening the nodes.
```

**中文翻译**

这段继续说明 NAGCN 的 node-aligned graph。不同于 BOVW 只用不可训练的 frequency histogram 表示 WSI，作者部署 node-aligned graph 来获得可训练的 WSI embedding。由于图构建策略保证不同 WSI 之间的 alignment，因此可以容易地产生 WSI-level representation，并用于后续分类。技术贡献包括：提出一种只需 slide-level annotation 的 node-aligned GCN 用于 WSI 表示和分类；相比其他 graph-based MIL 方法必须通过无序 pooling 生成 slide-level representation，NAGCN 的 aligned graph 能在 WSI 间建立节点对应关系，因此可选择更多方式得到全局表示，例如 flattening nodes。
