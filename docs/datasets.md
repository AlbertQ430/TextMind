# 公开文本检索数据集调研

> 调研日期：2026-11-19  
> 状态：候选调研。SciFact 已完成流程验证；正式中文实验数据集尚未选定。数据集许可证、版本和实际下载内容应在正式使用前再次核验。

## 1. 初步结论

建议先按三个层次考虑：

1. **中文主候选：DuRetrieval（C-MTEB）**。规模相对适中，适合先搭建中文检索实验流程；但 Hugging Face 数据卡目前信息不完整，许可证需要进一步核实。
2. **英文流程验证：BEIR / SciFact**。数据规模较小，结构清楚，适合快速验证 Corpus、Queries、Qrels 加载、BM25、Dense 和指标计算。
3. **大规模中文扩展：T2Ranking**。数据规模和标注质量更有研究价值，但完整数据体量较大，不适合作为第一份需要立刻跑通的数据。

当前不建议一开始就下载数 GB 甚至十几 GB 的数据。先把小数据集上的完整实验闭环跑通，再扩大规模。

## 2. 候选数据集对比

| 数据集 | 语言 / 任务 | 规模与标注 | 许可证与获取 | 初步判断 |
| --- | --- | --- | --- | --- |
| DuRetrieval（C-MTEB） | 中文文本检索 | Corpus 约 102,001 条；Queries 约 4,000 条（C-MTEB 任务测试样本数）；独立 Qrels 约 9,839 条标注 | [Corpus / Queries](https://huggingface.co/datasets/C-MTEB/DuRetrieval)；[Qrels](https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels)。数据卡未明确给出许可证，需追溯原始数据集授权 | **中文首选候选**，但授权核验是前置事项 |
| BEIR / SciFact | 英文科学事实核查检索 | Corpus 5,183 篇；官方 BEIR test 查询 300 条；另有 train/test 查询与 Qrels | [Hugging Face 数据集](https://huggingface.co/datasets/BeIR/scifact)，页面标注 CC-BY-SA-4.0；也可通过 BEIR 官方加载器下载 | **小规模流程验证首选**，数据较小，容易快速跑通 |
| T2Ranking | 中文 Passage Retrieval / Ranking | Collection 2,303,643 条；Dev/Test Queries 各 24,832 条；提供 retrieval 与 reranking 的不同 Qrels | [项目仓库](https://github.com/THUIR/T2Ranking)；[Hugging Face 数据集](https://huggingface.co/datasets/THUIR/T2Ranking)，标注 Apache-2.0。完整数据仓库页面显示约 16.5 GB | **后续大规模扩展**，首轮不建议完整下载 |

## 3. 具体说明

### 3.1 DuRetrieval

这里指 C-MTEB 在 Hugging Face 上整理的 DuRetrieval 评测版本，不要与百度发布的原始完整 DuReader Retrieval 数据集混为一谈。

- C-MTEB 版本：Corpus 约 102,001 条，C-MTEB 任务列表标注测试样本数为 4,000，Qrels 约 9,839 条。
- 百度原始 DuReader Retrieval：官方说明包含约 8.09M 段落，训练约 86K 个 Query-相关段落样本，开发集约 2K、测试集约 4K；原始数据下载说明指向官方数据集介绍页，可能需要注册获取。

来源：C-MTEB（Chinese Massive Text Embedding Benchmark）在 Hugging Face 上提供的整理版本，以及百度 DuReader 官方项目说明。

- Corpus 和 Queries 位于 `C-MTEB/DuRetrieval`。
- 相关性标注单独放在 `C-MTEB/DuRetrieval-qrels`。
- Hugging Face 页面显示 Corpus 约 102,001 条记录，整个数据集页面约 64.5 MB。
- Qrels 页面显示约 9,839 条标注记录。
- C-MTEB 任务列表将 DuRetrieval 的测试样本数列为 4,000；不同镜像或拆分页面展示的行数可能不同，正式下载后应以实际 Query / Qrels 文件和官方评测划分为准。

注意事项：
- Corpus / Queries 与 Qrels 分开发布，导入时需要按 ID 关联。
- 当前数据卡的说明为 “More Information needed”，没有明确展示许可证字段。
- 因此可以先做格式和技术验证，但正式课程实验前应确认原始数据来源和授权，不能把“公开可下载”直接等同于“许可证明确”。

获取入口：
- https://huggingface.co/datasets/C-MTEB/DuRetrieval
- https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels
- 百度原始数据说明：https://github.com/baidu/DuReader/blob/master/DuReader-Retrieval/README.md

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

## 4. 对当前项目的建议

### 第一阶段：先跑通实验闭环

优先用 SciFact 或 DuRetrieval 的小规模 / 完整可用划分，完成：
1. 数据集加载与格式统一。
2. BM25 检索。
3. Dense Retrieval。
4. 指标计算。
5. 结果保存和比较。

如果希望尽快验证代码链路，SciFact 更省事；如果希望从一开始就围绕中文任务，先核实 DuRetrieval 授权后再采用它。

### 第二阶段：正式中文实验

将 DuRetrieval 作为中文主实验候选，比较：
- BM25。
- Dense + Embedding 模型 A。
- Dense + Embedding 模型 B。
- 可选：Hybrid（BM25 + Dense + RRF）。

所有方法必须使用同一版本的 Corpus、Queries 和 Qrels。

### 第三阶段：规模扩展

如果课程时间和设备资源允许，再使用 T2Ranking 的检索评测划分，观察不同方法在大规模中文语料上的变化。

## 5. 重要实验原则

- 数据集的公开可下载性不等于许可证已经核实。
- 不同数据集的 Query、Corpus、Qrels 划分必须严格匹配。
- 不得把 reranking 的 Qrels 当成 retrieval 的 Qrels。
- 不应为了减少计算量而随意删掉 Qrels 中的相关文档；如果制作子集，必须同步过滤并记录规则。
- 如果使用抽样 Corpus 做开发测试，应保证抽样规则固定、可复现，并与正式结果分开报告。
- 不同模型必须使用完全相同的测试条件，不能让某个模型使用更小或更容易的语料。