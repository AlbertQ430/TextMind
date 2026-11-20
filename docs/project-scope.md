# TextMind 当前项目范围与阶段目标

> 最后更新：2026-11-20  
> 当前阶段：课程大作业的最小可用实验程序  
> 本文件是当前阶段的范围依据。与历史产品需求文档冲突时，以本文件为准。

## 1. 项目定位

TextMind 当前不是一个完整的个人知识管理软件，而是一个**文本信息检索算法实验项目**。

项目要回答的问题是：

> 在相同的数据集和查询条件下，不同检索算法与 Embedding 模型的检索效果和运行表现有什么差异？

## 2. 必须完成的最小闭环

1. 获取一个结构完整、来源清楚的公开文本检索数据集。
2. 读取 Corpus、Queries、Qrels，并进行基本格式校验。
3. 实现 BM25 基线检索。
4. 实现至少一种 Dense Retrieval，并能替换 Embedding 模型。
5. 对各实验组使用相同的 Query 和评价规则。
6. 计算 Precision@K、Recall@K、MRR、NDCG@K 和检索耗时。
7. 保存结果，并生成可比较的表格或图表。

如果时间允许，再加入 Hybrid Retrieval（例如 BM25 + Dense + RRF）作为扩展实验。

## 3. 当前不在范围内

当前不实现：

- GUI、桌面应用或网页前端。
- 个人知识库、文件管理和文件同步。
- AI 问答、摘要、知识点提取、笔记生成。
- 多轮聊天、用户账号或云端同步。
- Reranker、模型微调、自动网格搜索。
- 完整产品级配置管理和实验历史管理界面。

这些方向可以保留为未来产品构想，但不得成为当前阶段的前置依赖。

## 4. 技术实现原则

- 先以 Python 脚本 / CLI 完成算法和实验闭环。
- 优先复用公开数据集原有的文档或 Passage 单位，不为了未来知识库功能而擅自重新分块。
- 将数据集加载、检索器、指标计算和实验运行分成清晰模块。
- 对比实验必须固定数据集版本、查询集合、相关性标注和评价规则。
- 记录模型版本、关键参数、运行环境和各阶段耗时。
- 数据集原始文件、转换后的数据和实验结果分开存放。
- 先做可复现、可验证的实验，再考虑界面和产品体验。

## 5. 当前推进顺序

1. 数据集选择与获取。
2. 统一数据集格式和加载校验。
3. BM25 基线。
4. Dense Retrieval 与 Embedding 模型接入。
5. 评价指标实现及单元测试。
6. 对照实验与结果分析。
7. 视时间增加 Hybrid Retrieval 和结果图表整理。

## 6. 当前已有基础

- Apple Silicon MPS 已验证可用。
- Qwen/Qwen3-Embedding-0.6B 已验证可以在 MPS 上加载并输出向量。
- BM25 已实现（倒排索引、文档长度归一化、可注入分词器），并通过单元测试。
- Dense Retrieval 已实现（Embedding 编码 + 余弦相似度检索），Qwen3-Embedding-0.6B 与 bge-small-zh-v1.5（中文）已接入。
- 数据集加载与校验、评价指标（P@K / R@K / MRR / NDCG@K）已完成。
- SciFact 与 DuRetrieval（中文）均已完成实验闭环：加载、BM25 / Dense / Hybrid 检索、评价与结果落盘。

## 7. 阶段进展

- [x] 流程验证数据集选型：SciFact（英文，CC-BY-SA-4.0）已完成加载校验。
- [x] 数据加载与校验：ID 统一、引用完整性、重复标注检测。
- [x] BM25 基线实现与单元测试。
- [x] 评价指标实现与单元测试。
- [x] SciFact test split（300 queries）BM25 实验跑通：NDCG@10 = 0.6605，
      R@100 = 0.8859，结果保存于 `data/evaluation/`。
- [x] Dense Retrieval 实现与 Qwen3-Embedding-0.6B 接入；SciFact test split
      Dense 实验跑通：NDCG@10 = 0.6762，R@100 = 0.9333。
- [x] 不同 Embedding 模型对照实验（Qwen3-Embedding-0.6B、all-MiniLM-L6-v2、BGE-M3），
      汇总表见 `data/evaluation/summary.md`。
- [x] 自定义查询前缀对照实验：Qwen3-Embedding-0.6B + 自定义前缀（NDCG@10 = 0.7003）、
      BGE-M3 + 自定义前缀（NDCG@10 = 0.6502）。
- [x] Hybrid Retrieval（BM25 + Dense + RRF，rrf_k = 60）：无前缀 NDCG@10 = 0.7111，
      自定义前缀 0.7165；BM25 始终使用原始 Query，仅 Dense 使用前缀。
- [x] Hybrid 耗时拆分（Query Encoding / BM25 检索 / Dense 检索 / RRF 融合 / 端到端检索）；
      缓存校验纳入 model_name、revision 与缓存版本号。
- [x] 中文正式实验数据集：DuRetrieval（C-MTEB 子集，授权核验与格式验证完成）。
- [x] 中文 BM25 基线（jieba 分词）：DuRetrieval dev，NDCG@10 = 0.5931，MRR = 0.7420。
- [x] 中文 Dense（bge-small-zh-v1.5 + 自定义前缀）：NDCG@10 = 0.7900，MRR = 0.8787。
- [x] 中文 Hybrid（BM25 + Dense + RRF，jieba 分词）：NDCG@10 = 0.7233（低于
      Dense 单路；此前 simple 分词版 0.7474 已归档，不参与正式结论）。
- [x] 加权 RRF 参数探索（DuRetrieval dev split）：BM25 权重 0.5/0.3/0.2/0.1 扫描，
      0.1 为当前范围内的最佳配置（NDCG@10 = 0.7871），仍略低于 Dense 单路 0.7900；
      该结果为 dev 调参，不代表独立测试集上的泛化表现。
- [x] 结果分析图表：`data/evaluation/figures/`（指标柱状图、质量-耗时散点图）。
- [ ] 下一阶段重点：课程报告、图表与展示材料整理。
