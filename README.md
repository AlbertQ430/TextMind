# TextMind

TextMind 正从文本检索算法实验项目，演进为一个**可交互的文本检索实验台**。当前阶段优先把已有检索实验能力封装成易于演示、观察和比较的桌面软件；知识库管理等更大的产品方向暂不纳入当前 MVP。

> **当前定位**：以预设基准实验为核心，让用户选择实验方案、运行实验，并查看检索指标、耗时与结果。
>
> **开发策略**：先完成实验演示型 MVP，再逐步开放有限的算法、模型和参数配置能力。项目不以通用检索研究平台为目标。

## 当前产品范围

### MVP：实验演示

- 内置预设实验方案和对应的预置结果，打开软件即可浏览实验结果。
- 支持重新运行预设实验，实际调用检索流程并生成新的实验结果。
- 清楚区分预置结果与本次运行结果，不将历史结果伪装成实时运行结果。
- 展示实验指标、耗时和必要的检索结果 / 可视化。
- 尽量复用现有 Python 检索核心、实验配置和结构化结果，不为 GUI 重写算法实现。

### 后续演进：有限实验配置

在 MVP 稳定后，再逐步考虑：
- 选择已支持的检索算法与 Embedding 模型。
- 调整有限范围内的算法参数，例如 BM25 参数、RRF 权重。
- 保存每次运行记录，并查看、比较同一实验方案的历史结果。
- 视需求增加可选数据集。

以下方向不属于当前产品路线：任意第三方检索器 / 模型插件、自由组合检索流水线、自动调参研究平台、个人知识库管理、AI 问答与笔记生成。

## 已有技术基础

TextMind 已完成一轮文本检索算法实验，相关实验代码与成果是软件化的基础：

- **检索方法**：BM25、Dense Retrieval、Hybrid Retrieval（RRF）。
- **数据集**：英文 SciFact、中文 DuRetrieval（C-MTEB 子集）。
- **Embedding 模型**：Qwen3-Embedding-0.6B、BGE-M3、bge-small-zh-v1.5、all-MiniLM-L6-v2。
- **评价指标**：Precision@K、Recall@K、MRR、NDCG@K，以及分阶段耗时。
- **实验产物**：结构化 JSON 结果、汇总表、图表和错误分析材料。
- **运行环境**：已验证 Apple Silicon MPS 上的模型加载与向量计算。

实验结论、数据集信息和验证记录属于已有技术资料，不代表软件 MVP 已经实现相应 GUI 功能。

## 实验结果概览

以下为已有实验中的代表性结果。不同数据集使用不同语料、查询和标注，**不可跨数据集直接比较**。

| 数据集 | 方法 | P@1 | NDCG@10 | R@100 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| DuRetrieval dev | BM25（jieba） | 0.6610 | 0.5931 | 0.7769 | 0.7420 |
| DuRetrieval dev | Dense：bge-small-zh-v1.5（无前缀） | 0.8195 | 0.7949 | 0.9429 | 0.8751 |
| DuRetrieval dev | Dense：bge-small-zh-v1.5（有前缀） | 0.8260 | 0.7900 | 0.9395 | 0.8787 |
| DuRetrieval dev | Hybrid：BM25 + Dense（等权 RRF） | 0.7825 | 0.7233 | 0.9338 | 0.8431 |
| SciFact test | BM25 | 0.5333 | 0.6605 | 0.8859 | 0.6312 |
| SciFact test | Dense：Qwen3-Embedding-0.6B（无前缀） | 0.5500 | 0.6766 | 0.9333 | 0.6439 |
| SciFact test | Dense：Qwen3-Embedding-0.6B（有前缀） | 0.5733 | 0.7003 | 0.9567 | 0.6708 |
| SciFact test | Hybrid：BM25 + Qwen3（有前缀，RRF） | 0.6033 | 0.7165 | 0.9567 | 0.6896 |

DuRetrieval 使用 dev split。其加权 RRF 参数探索也在 dev split 上完成，不应解读为独立测试集上的泛化结论。更多结果和实验口径见 实验汇总表由 `data/evaluation/summary.md` 生成；详细错误分析见 [DuRetrieval 错误分析](docs/error-analysis.md)。

## 文档导航

- [当前产品范围与 MVP](docs/project-scope.md)
- [已确认的产品决策](docs/requirements/confirmed-decisions.md)
- [当前待讨论事项](docs/requirements/open-questions.md)
- [检索实验与执行约定](docs/requirements/retrieval-lab.md)
- [数据集调研记录](docs/datasets.md)
- [技术验证记录](docs/technical-validation.md)
- [DuRetrieval 错误分析](docs/error-analysis.md)

## 当前阶段说明

实验算法与数据分析阶段已形成可复用成果。当前工作重心转为需求探索和软件化设计，尚未冻结完整 SRS，也不应将本文中的后续方向视为已经实现的功能。
