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
- **检索方法**：BM25 与 Dense Retrieval 已完成实现；Hybrid Retrieval 待实现。
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

## 当前实验结果

SciFact test split（300 queries），统一流程与评价口径：

| 方法 | P@1 | NDCG@10 | R@100 | MRR |
| --- | --- | --- | --- | --- |
| BM25（k1=1.2, b=0.75） | 0.5333 | 0.6605 | 0.8859 | 0.6312 |
| Dense：Qwen3-Embedding-0.6B | 0.5500 | 0.6762 | 0.9333 | 0.6435 |
| Dense：all-MiniLM-L6-v2 | 0.5033 | 0.6451 | 0.9250 | 0.6110 |
| Dense：BGE-M3 | 0.5100 | 0.6437 | 0.9037 | 0.6131 |

完整汇总（含索引耗时与检索延迟）由 `python src/scripts/compare_experiments.py`
生成，见 `data/evaluation/summary.md`。

说明：Dense 结果均为默认 encode 配置，未对查询加模型特定的指令前缀；
BGE-M3 等模型的官方推荐前缀会带来不同的分数，正式结论前应作为单独实验记录。

## 项目状态

项目目前已完成 SciFact 上的两条检索基线：BM25（NDCG@10 = 0.6605）与
Dense Retrieval（Qwen3-Embedding-0.6B，NDCG@10 = 0.6762），均完成
数据集加载 → 检索 → 指标计算 → 结果落盘。

下一步优先：不同 Embedding 模型对照实验与结果分析；中文正式实验数据集尚未选定。

当前不冻结完整产品 SRS。项目范围以 [project-scope.md](docs/project-scope.md) 为准。
