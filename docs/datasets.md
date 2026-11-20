# 公开文本检索数据集调研

> 调研日期：2026-11-19  
> 文档性质：历史数据集调研记录。当前实验已采用 DuRetrieval（C-MTEB 子集）作为中文正式实验数据集，SciFact 作为英文流程验证数据集。具体实验口径以当前实验结果和代码为准。

## 1. 初步结论

建议先按三个层次考虑：

1. **中文数据集：DuRetrieval（C-MTEB）**。规模适中、标注完整，官方仓库声明 Apache-2.0，授权核验与格式验证已完成（2026-11-20），作为正式中文实验数据集。
2. **英文流程验证：BEIR / SciFact**。数据规模较小，结构清楚，适合快速验证 Corpus、Queries、Qrels 加载、BM25、Dense 和指标计算。
3. **大规模中文扩展：T2Ranking**。数据规模和标注质量更有研究价值，但完整数据体量较大，不适合作为第一份需要立刻跑通的数据。

当前不建议一开始就下载数 GB 甚至十几 GB 的数据。先把小数据集上的完整实验闭环跑通，再扩大规模。

## 2. 候选数据集对比

| 数据集 | 语言 / 任务 | 规模与标注 | 许可证与获取 | 初步判断 |
| --- | --- | --- | --- | --- |
| DuRetrieval（C-MTEB） | 中文文本检索 | Corpus 100,001 条；Queries 2,000 条；独立 Qrels 9,839 条标注（二值） | [Corpus / Queries](https://huggingface.co/datasets/C-MTEB/DuRetrieval)；[Qrels](https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels)。官方仓库（baidu/DuReader）声明 Apache-2.0，HF 镜像未标 license | **已采用**：正式中文实验数据集 |
| BEIR / SciFact | 英文科学事实核查检索 | Corpus 5,183 篇；官方 BEIR test 查询 300 条；另有 train/test 查询与 Qrels | [Hugging Face 数据集](https://huggingface.co/datasets/BeIR/scifact)，页面标注 CC-BY-SA-4.0；也可通过 BEIR 官方加载器下载 | **小规模流程验证首选**，数据较小，容易快速跑通 |
| T2Ranking | 中文 Passage Retrieval / Ranking | Collection 2,303,643 条；Dev/Test Queries 各 24,832 条；提供 retrieval 与 reranking 的不同 Qrels | [项目仓库](https://github.com/THUIR/T2Ranking)；[Hugging Face 数据集](https://huggingface.co/datasets/THUIR/T2Ranking)，标注 Apache-2.0。完整数据仓库页面显示约 16.5 GB | **后续大规模扩展**，首轮不建议完整下载 |

## 3. 具体说明

### 3.1 DuRetrieval

这里指 C-MTEB 在 Hugging Face 上整理的 DuRetrieval 评测版本，不要与百度发布的原始完整 DuReader Retrieval 数据集（约 8.09M 段落）混为一谈。课程实验使用 C-MTEB 子集（100,001 段落），**不下载、不使用百度原始 8.09M 段落版本**。

实测规模（2026-11-20 下载并校验）：

- Corpus：100,001 条（`corpus.jsonl` 约 92 MB）。
- Queries：2,000 条。
- Qrels（dev split）：9,839 条标注，全部为二值（score = 1）；覆盖全部 2,000 条 query，
  平均每 query 4.92 篇相关文档。
- 引用完整性：qrels 中 9,792 个唯一文档全部存在于 corpus，无缺失；全部 query 均有标注。

获取方式（固定 commit，可复现）：

- Corpus / Queries：`C-MTEB/DuRetrieval`，commit `a1a333e290fe30b10f3f56498e3a0d911a693ced`。
- Qrels：`C-MTEB/DuRetrieval-qrels`，commit `497b7bd1bbb25cb3757ff34d95a8be50a3de2279`（dev split）。
- 本地存放：`data/raw/duretrieval/corpus.jsonl`、`queries.jsonl`、`qrels_dev.tsv`；
  原始 parquet 文件不纳入版本库（`.gitignore`），可由上述 commit 重新下载。

授权核验结论：

- 官方 `baidu/DuReader` 仓库 README 声明 **Apache License 2.0**（Copyright 2017 Baidu.com），
  覆盖 DuReader 系列数据集与基线。
- Hugging Face 的 C-MTEB 镜像数据卡未标注 license（“More Information needed”），
  属第三方搬运；使用时以官方仓库许可为准，并在实验报告中注明来源、许可证与下载 commit。
- 原始完整数据需在百度 AI Studio 报名获取；课程实验采用 C-MTEB 子集，规模适中、标注完整。
- 结论：**可作为正式中文实验数据集**，核验与格式验证完成。

获取入口：
- https://huggingface.co/datasets/C-MTEB/DuRetrieval
- https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels
- 百度官方说明：https://github.com/baidu/DuReader/blob/master/DuReader-Retrieval/README.md

### 3.2 BEIR / SciFact

SciFact 是 BEIR 中的科学事实核查检索任务：给定科学 claim，检索支持或反驳该 claim 的科学文档。

- Corpus：5,183 条。
- 官方 BEIR 测试集包含 300 条 Queries。
- Hugging Face 页面显示查询数据总量为 1,109 条，包含不同划分。
- Qrels 提供查询与相关文档的对应关系。
- Hugging Face 数据卡标注许可证为 CC-BY-SA-4.0。

BEIR 官方提供预处理好的数据和加载器。示例：

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")
```

获取入口：
- https://github.com/beir-cellar/beir
- https://github.com/beir-cellar/beir/wiki/Datasets-available
- https://huggingface.co/datasets/BeIR/scifact

适合用途：
- 快速检查数据加载和 ID 关联是否正确。
- 先实现 BM25，再接入 Dense Retrieval。
- 验证 Precision@K、Recall@K、MRR、NDCG@K 的实现。
- 由于它是英文数据集，不应单独作为项目的中文检索效果结论。

### 3.3 T2Ranking

T2Ranking 是面向中文 Passage Ranking 的大规模基准，来自真实搜索场景，提供专家标注的多级相关性判断。

项目 README 公布的主要数据规模：
- Collection：2,303,643 条 Passage。
- Queries Train：258,042 条。
- Queries Dev：24,832 条。
- Queries Test：24,832 条。
- Retrieval Dev Qrels：118,933 条。
- 另有用于 reranking 的 Qrels，以及规模巨大的负样本文件。

数据集采用 Apache License 2.0。Hugging Face 页面显示完整数据约 16.5 GB。

获取入口：
- https://github.com/THUIR/T2Ranking
- https://huggingface.co/datasets/THUIR/T2Ranking
- 论文：https://arxiv.org/abs/2304.03679

注意事项：
- 项目提供了检索和重排序两种不同用途的标注文件，不能混用。
- 完整数据中包含大量训练负样本；当前只做检索评测时，不需要下载或处理所有训练资源。
- 语料超过 230 万条，向量化和索引会带来明显的时间、内存与磁盘开销。
- 如果后续使用，建议先只选定官方支持的一个评测划分，并核对对应的 Corpus、Queries 和 Retrieval Qrels。

## 4. 历史选型建议（供追溯，不代表当前待办）

### 第一阶段：实验闭环（已完成）

当时建议优先使用 SciFact 或 DuRetrieval 完成以下工作；该阶段现已完成：
1. 数据集加载与格式统一。
2. BM25 检索。
3. Dense Retrieval。
4. 指标计算。
5. 结果保存和比较。

当前结果：SciFact 已完成英文流程验证；DuRetrieval（C-MTEB 子集）已完成授权核验、格式校验和中文正式实验。

### 第二阶段：正式中文实验（已完成）

已使用 DuRetrieval 作为中文正式实验数据集，完成以下比较：
- BM25。
- Dense + Embedding 模型 A。
- Dense + Embedding 模型 B。
- 可选：Hybrid（BM25 + Dense + RRF）。

所有方法必须使用同一版本的 Corpus、Queries 和 Qrels。

### 第三阶段：规模扩展（未列入当前 MVP）

T2Ranking 仍是可供未来研究参考的大规模数据集，但当前 TextMind 软件 MVP 不以增加大规模数据集为目标。

## 5. 重要实验原则

- 数据集的公开可下载性不等于许可证已经核实。
- 不同数据集的 Query、Corpus、Qrels 划分必须严格匹配。
- 不得把 reranking 的 Qrels 当成 retrieval 的 Qrels。
- 不应为了减少计算量而随意删掉 Qrels 中的相关文档；如果制作子集，必须同步过滤并记录规则。
- 如果使用抽样 Corpus 做开发测试，应保证抽样规则固定、可复现，并与正式结果分开报告。
- 不同模型必须使用完全相同的测试条件，不能让某个模型使用更小或更容易的语料。