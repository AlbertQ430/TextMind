# TextMind

TextMind 当前阶段是一个**文本信息检索算法实验项目**，用于获取公开检索数据集、运行不同检索算法与 Embedding 模型，并通过统一指标评价和比较实验结果。

当前项目优先验证 NLP / 信息检索技术本身，不以完整知识管理产品为目标。

## 当前目标

围绕同一数据集和查询集合，比较不同检索方法与模型的表现：

1. 获取并整理公开文本检索数据集。
2. 运行不同检索算法与模型。
3. 使用统一评价指标计算检索效果与耗时。
4. 保存实验配置和结果，进行横向比较。
5. 形成可用于课程展示的实验数据、图表和分析。

## 当前实现范围

- **数据集**：Corpus、Queries、Qrels 的读取、校验和统一格式转换。
- **检索方法**：BM25、Dense Retrieval、Hybrid Retrieval（RRF 融合）已完成实现。
- **模型比较**：比较不同 Embedding 模型在相同检索任务上的效果。
- **评价**：Precision@K、Recall@K、MRR、NDCG@K，以及检索耗时。
- **实验记录**：保存运行配置、环境信息、指标结果和必要的查询级结果。
- **运行方式**：优先使用 Python 脚本或 CLI，不要求 GUI。

## 当前不做

以下内容不属于本阶段交付目标：

- 图形用户界面和桌面应用。
- 个人知识库管理、文件浏览和文件同步。
- AI 问答、摘要、笔记生成和聊天历史。
- 完整的知识管理工作台。
- 复杂模型训练、自动网格搜索和 Reranker（除非后续实验确有必要）。

此前讨论的知识管理工作台与 AI 知识加工方向暂时保留为未来可能的产品方向，不作为当前课程项目的实现约束。

## 当前技术基础

- 已验证 Apple Silicon MPS 可用。
- 已验证 `Qwen/Qwen3-Embedding-0.6B` 可以在 MPS 上加载并生成向量。
- BM25 检索器已实现（倒排索引 + 文档长度归一化），并通过单元测试。
- Dense Retrieval 已实现（Embedding 编码 + 余弦相似度检索），编码器可替换。
- 数据加载与校验、评价指标（P@K / R@K / MRR / NDCG@K）均已完成。
- SciFact 已完成流程验证：BM25 与 Dense 均跑通完整实验并落盘结果。

## 运行方式

```bash
# BM25 实验（默认），结果写入 data/evaluation/
python src/scripts/run_experiment.py

# Dense Retrieval 实验（首次需编码 5183 篇文档，约 9~10 分钟）
python src/scripts/run_experiment.py --method dense

# Hybrid 实验：BM25 + Dense + RRF 融合
python src/scripts/run_experiment.py --method hybrid --model Qwen/Qwen3-Embedding-0.6B

# 中文数据集（DuRetrieval，C-MTEB 子集）+ jieba 分词
python src/scripts/run_experiment.py --dataset duretrieval --tokenizer jieba

# 生成汇总表与图表
python src/scripts/compare_experiments.py --output data/evaluation/summary.md
python src/scripts/plot_results.py

# 跑全部单元测试（自动发现 test/ 下的测试）
python -m unittest discover -s test -v

# 环境与模型 smoke test（各自独立运行）
python test/test_device.py
python test/test_embedding.py
python test/test_similarity.py
```

## 文档

- [当前项目范围与阶段目标](docs/project-scope.md)
- [公开数据集候选与调研](docs/datasets.md)
- [检索实验约定](docs/requirements/retrieval-lab.md)
- [已确认的历史需求与决策](docs/requirements/confirmed-decisions.md)
- [待讨论事项](docs/requirements/open-questions.md)
- [技术验证报告](docs/technical-validation.md)
- [DuRetrieval 错误分析报告](docs/error-analysis.md)

## 当前实验结果

| 数据集 | 方法 | P@1 | NDCG@10 | R@100 | MRR |
| --- | --- | --- | --- | --- | --- |
| DuRetrieval dev | BM25（jieba 分词） | 0.6610 | 0.5931 | 0.7769 | 0.7420 |
| DuRetrieval dev | Dense：bge-small-zh-v1.5（无前缀） | 0.8195 | 0.7949 | 0.9429 | 0.8751 |
| DuRetrieval dev | Dense：bge-small-zh-v1.5 + 自定义前缀 | 0.8260 | 0.7900 | 0.9395 | 0.8787 |
| DuRetrieval dev | Hybrid：BM25 + bge-small-zh-v1.5（RRF，jieba） | 0.7825 | 0.7233 | 0.9338 | 0.8431 |
| SciFact test | BM25（k1=1.2, b=0.75） | 0.5333 | 0.6605 | 0.8859 | 0.6312 |
| SciFact test | Dense：Qwen3-Embedding-0.6B（无前缀） | 0.5500 | 0.6766 | 0.9333 | 0.6439 |
| SciFact test | Dense：Qwen3-Embedding-0.6B + 自定义前缀 | 0.5733 | 0.7003 | 0.9567 | 0.6708 |
| SciFact test | Dense：all-MiniLM-L6-v2 | 0.5033 | 0.6451 | 0.9250 | 0.6110 |
| SciFact test | Dense：BGE-M3（无前缀） | 0.5100 | 0.6437 | 0.9037 | 0.6131 |
| SciFact test | Dense：BGE-M3 + 自定义前缀 | 0.5300 | 0.6502 | 0.9077 | 0.6206 |
| SciFact test | Hybrid：BM25 + Qwen3-Embedding-0.6B（RRF） | 0.5933 | 0.7111 | 0.9533 | 0.6842 |
| SciFact test | Hybrid：BM25 + Qwen3-Embedding-0.6B + 自定义前缀（RRF） | 0.6033 | 0.7165 | 0.9567 | 0.6896 |

完整汇总（含索引、查询编码与向量检索耗时）由 `python src/scripts/compare_experiments.py`
生成，见 `data/evaluation/summary.md`。耗时列反映当次运行状态：命中缓存时
对应编码耗时为 0。

说明：不同数据集的指标不可直接横向比较（语料、query 与 qrels 不同）；
DuRetrieval 使用 dev split 的 9,839 条 qrels。

DuRetrieval 上 Hybrid（等权 RRF）低于 Dense 单路：BM25 在该数据集上的检索质量
明显弱于 Dense（NDCG@10 0.5931 vs 0.7900），等权融合稀释了 Dense 的排序；
RRF 的收益依赖两路信号强度接近（SciFact 上两路接近，Hybrid 即优于单路）。

返修记录：此前 DuRetrieval Hybrid 实验误用 simple 分词器（中文分词失效、BM25
近乎空转），该结果（NDCG@10 0.7474）已归档到 `data/evaluation/archive/`；
修正为 jieba 后 BM25 真实参与融合，NDCG@10 = 0.7233、MRR = 0.8431。

查询前缀实验通过 `--query-prefix` 传入自定义前缀（BGE-M3 与
Qwen3-Embedding 采用各自文档推荐的检索指令），与无前缀结果分开记录，
不可直接混用比较；正式结论需注明所用配置。

Hybrid 实验中 BM25 始终使用原始 Query，仅 Dense 使用前缀；汇总表「检索均值」
为单条查询端到端检索耗时（不含 Query 编码），Dense 行仅含向量检索，Hybrid 行
含 BM25 检索 + Dense 检索 + RRF 融合。旧版 Hybrid 结果（前缀误传给 BM25）已
归档到 `data/evaluation/archive/`，不参与正式汇总。

## 加权 RRF 实验（DuRetrieval dev 参数探索）

错误分析显示，DuRetrieval 上 BM25 明显弱于 Dense，等权 RRF 会稀释 Dense 信号。
在 **DuRetrieval dev split（2,000 queries）** 上对 Hybrid 做 BM25 权重扫描
（Dense 权重固定 1.0，rrf_k=60），属于开发集上的参数探索：

| BM25 权重 | P@1 | NDCG@10 | MRR |
| --- | --- | --- | --- |
| 1.0（等权） | 0.7825 | 0.7233 | 0.8431 |
| 0.5 | 0.8010 | 0.7529 | 0.8573 |
| 0.3 | 0.8080 | 0.7696 | 0.8647 |
| 0.2 | 0.8100 | 0.7794 | 0.8678 |
| 0.1 | 0.8145 | 0.7871 | 0.8716 |
| Dense 单路（参考） | 0.8260 | 0.7900 | 0.8787 |

结论：在当前搜索范围（0.1~1.0）内，BM25 权重 0.1 的配置最佳，但仍略低于
Dense 单路——RRF 只保留排名、丢失分数间隔信息，且 BM25 噪声仍进入 top-10。
该结果来自 dev split 调参，**不代表在独立测试集上的泛化表现**，正式结论需在
留出测试集上复核。SciFact 上 BM25 权重 0.5 小幅提升（NDCG@10 0.7165 → 0.7220，
R@100 0.9567 → 0.9700）。

## 项目状态

项目目前已完成 SciFact 上的检索实验闭环：BM25 基线、三种 Embedding 模型对照
（Qwen3-Embedding-0.6B、all-MiniLM-L6-v2、BGE-M3）、自定义查询前缀实验，
以及 Hybrid（BM25 + Dense + RRF）融合实验；并完成 DuRetrieval 中文数据集上的
BM25 基线（jieba 分词）。实验汇总表与图表均已生成。

下一阶段重点：课程报告撰写、图表与展示材料整理。

当前不冻结完整产品 SRS。项目范围以 [project-scope.md](docs/project-scope.md) 为准。

