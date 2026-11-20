# 课程展示讲稿大纲（15~18 页）

> 配套 PPT：`docs/export/textmind-slides.pptx`（不入库）
> 时长建议：12~15 分钟，每页 40~60 秒

## 第 1 页：封面

- 标题：TextMind：中文与英文文本检索算法实验
- 副标题：BM25 / Dense / Hybrid 的对比与错误分析
- 姓名、学号、日期、仓库地址

## 第 2 页：研究问题与目标

- 问题：同一数据集上，不同检索算法与 Embedding 模型差异有多大？
- 三个对比维度：BM25、Dense、Hybrid（RRF）
- 交付：可复现实验闭环 + 结果表 + 图表 + 错误分析

## 第 3 页：实验流程

- 数据集 → 检索方法 → 评价指标 → 结果落盘 → 图表
- 关键工程：embedding 缓存、查询缓存、缓存版本校验、可复现记录

## 第 4 页：数据集

- SciFact：英文流程验证，5,183 篇 / 300 queries
- DuRetrieval（C-MTEB 子集）：中文正式，100,001 篇 / 2,000 queries / 9,839 qrels
- 规模对比图或表格

## 第 5 页：方法一——BM25

- 倒排索引 + 文档长度归一化
- 公式（k1、b 参数，idf 非负形式）
- 中文 jieba 分词

## 第 6 页：方法二——Dense Retrieval

- SentenceTransformer 编码文档与查询，L2 归一化 + 余弦相似度
- 模型：Qwen3-Embedding-0.6B（英文）、bge-small-zh-v1.5（中文）
- 查询前缀的作用（instruct / 检索指令）

## 第 7 页：方法三——Hybrid 与 RRF

- RRF 公式与加权版本
- 融合只用排名、不需要分数归一化
- 候选深度与 top-k 的关系

## 第 8 页：实验设置与可复现性

- 指标：P@K / R@K / MRR / NDCG@K
- 耗时拆分：文档缓存、语料编码、查询编码、向量检索
- 缓存元数据记录模型名与 revision
- 复现命令（README）

## 第 9 页：SciFact 结果

- 指标表：BM25 vs 3 个 Dense 模型 vs Hybrid
- 引用图 `metrics_bars.png`（或截图）
- 一句话结论：Qwen3 + 前缀 + RRF 最佳

## 第 10 页：DuRetrieval 结果

- 指标表：BM25（jieba）/ Dense（bge-small-zh）/ Hybrid
- 引用图 `efficiency.png`（质量-延迟散点）
- 结论：Dense 明显优于 BM25；Hybrid 等权反而更差

## 第 11 页：加权 RRF 实验

- 权重扫描表 + `rrf_weights.png` 曲线
- 结论：BM25 权重 0.1 为当前范围最佳，仍略低于 Dense 单路
- 说明：dev 调参，不代表测试集泛化

## 第 12 页：错误分析

- 三分类统计表 + `error_distribution.png`
- 排序损失为主（Dense ranked_low 13.1%）
- 召回失败：术语歧义、领域漂移

## 第 13 页：典型错误案例

- 案例 1：拆机类问题混淆机型（排序）
- 案例 2：「eve 回电」术语歧义（召回）
- 案例 3：Hybrid 收益 vs 损失对比

## 第 14 页：讨论——Hybrid 何时有效

- 两路信号强度接近时 RRF 有效（SciFact）
- 差距大时等权融合稀释（DuRetrieval）
- 加权 RRF 缓解但仍受 RRF 丢失分数信息限制

## 第 15 页：局限与未来工作

- dev split 调参，需测试集复核
- 未做 Reranker / 查询改写
- 规模与统计检验（置信区间）

## 第 16 页：结论

- 三条结论：模型/前缀重要、Hybrid 有条件有效、错误以排序校准为主
- 工程贡献：可复现实验框架与缓存设计

## 第 17 页：Q&A / 谢谢

- 仓库地址与复现命令
- 致谢
