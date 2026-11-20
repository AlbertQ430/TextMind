# TextMind（exp：实验冻结分支）

> 本分支是 TextMind 课程实验结果的**冻结快照**（冻结日期 2026-11-20，对应 main 分支 commit `317dfaf`）。
> 本分支只保存实验相关文件（代码、数据、结果、实验文档），之后不再更新；后续实验与软件演进在 main 分支进行。

## 实验概览

- 任务：在统一数据集与评价口径下，比较 BM25、Dense Retrieval 与 Hybrid Retrieval（RRF 融合）三种检索方法。
- 数据集：
  - **SciFact**（英文流程验证）：5,183 篇文档 / 300 条 test queries。
  - **DuRetrieval**（C-MTEB 子集，中文正式实验）：100,001 篇文档 / 2,000 条 dev queries / 9,839 条 qrels。
- 模型：Qwen3-Embedding-0.6B、BGE-M3、all-MiniLM-L6-v2（英文）；bge-small-zh-v1.5（中文）。
- 环境：Apple M5，Python 3.11.18，PyTorch 2.15.0，sentence-transformers 6.3.0，MPS。

## 核心结果

### SciFact test（300 queries）

| 方法 | P@1 | NDCG@10 | R@100 | MRR |
| --- | --- | --- | --- | --- |
| BM25（k1=1.2, b=0.75） | 0.5333 | 0.6605 | 0.8859 | 0.6312 |
| Dense：Qwen3-Embedding-0.6B（官方推荐 Instruct 格式） | 0.5733 | 0.7003 | 0.9567 | 0.6708 |
| Dense：BGE-M3 + 自定义查询前缀 | 0.5300 | 0.6502 | 0.9077 | 0.6206 |
| Hybrid：BM25 + Qwen3（RRF，Instruct 指令） | 0.6033 | 0.7165 | 0.9567 | 0.6896 |
| Hybrid（RRF，Instruct 指令，BM25 权重 0.5） | 0.6033 | 0.7220 | 0.9700 | 0.6939 |

### DuRetrieval dev（2,000 queries）

| 方法 | P@1 | NDCG@10 | R@100 | MRR |
| --- | --- | --- | --- | --- |
| BM25（jieba） | 0.6610 | 0.5931 | 0.7769 | 0.7420 |
| Dense：bge-small-zh-v1.5（无前缀） | 0.8195 | 0.7949 | 0.9429 | 0.8751 |
| Dense：bge-small-zh-v1.5（检索指令前缀） | 0.8260 | 0.7900 | 0.9395 | 0.8787 |
| Hybrid：BM25 + bge-small-zh（RRF，等权） | 0.7825 | 0.7233 | 0.9338 | 0.8431 |
| Hybrid（RRF，BM25 权重 0.1） | 0.8145 | 0.7871 | 0.9395 | 0.8716 |

## 目录结构

```
src/                 实验代码：数据加载 / 检索器（BM25、Dense、Hybrid）/ 评价指标 / 实验脚本
test/                单元测试（103 项）
data/raw/scifact/    英文实验数据（DuRetrieval 数据按约定不入库）
data/evaluation/     实验结果 JSON、汇总表 summary.md、图表 figures/、归档 archive/
docs/                实验文档（课程报告、错误分析、数据集说明、技术验证、实验约定、讲稿大纲）
README.md            本文件
```

## 复现

```bash
# SciFact
python src/scripts/run_experiment.py
python src/scripts/run_experiment.py --method dense
python src/scripts/run_experiment.py --method hybrid --model Qwen/Qwen3-Embedding-0.6B

# DuRetrieval（数据需按 docs/datasets.md 记录的 commit 下载）
python src/scripts/run_experiment.py --dataset duretrieval --tokenizer jieba
python src/scripts/run_experiment.py --dataset duretrieval --method dense --model BAAI/bge-small-zh-v1.5
python src/scripts/run_experiment.py --dataset duretrieval --method hybrid --model BAAI/bge-small-zh-v1.5 --tokenizer jieba

# 汇总与图表
python src/scripts/compare_experiments.py --output data/evaluation/summary.md
python src/scripts/plot_results.py

# 测试
python -m unittest discover -s test -v
```

## 文档索引

- [课程实验报告](docs/course-report.md)
- [错误分析报告](docs/error-analysis.md)
- [数据集说明与授权核验](docs/datasets.md)
- [技术验证报告](docs/technical-validation.md)
- [检索实验约定](docs/requirements/retrieval-lab.md)
- [讲稿大纲](docs/slides-outline.md)

## 限制与说明

- DuRetrieval 结果均基于 **dev split**；加权 RRF 的权重为 dev 调参结果，不代表独立测试集泛化表现。
- 错误分析案例按 query 顺序抽取，仅作定性说明，不具有统计代表性。
- 查询前缀收益因模型与数据集而异：Qwen3 Instruct 在 SciFact 有提升，bge-small-zh 的检索指令前缀在 DuRetrieval 未提升。
- 过程性文档（项目范围、历史产品需求、待讨论事项、材料评审意见）保留在 main 分支，不在本分支。
