# TextMind

TextMind 正从文本检索算法实验项目，演进为一个**可交互的文本检索实验台**。当前阶段优先把已有检索实验能力封装成本地 Web 演示软件；知识库管理等更大的产品方向暂不纳入当前 MVP。

> **当前定位**：以预设基准实验为核心，让用户选择方案中的一个实验变体、运行实验，并查看检索指标、耗时与结果。
>
> **开发策略**：先完成实验演示型 MVP，再逐步开放有限的算法、模型和参数配置能力。项目不以通用检索研究平台为目标。

## 当前产品范围

### MVP：实验演示

- 本地 Web 应用，由用户自己的电脑运行服务，通过浏览器访问。
- 可选开启局域网访问，其他设备通过主机局域网地址访问同一实例；不依赖云服务器，不做公网访问和多用户账号系统。
- 采用左侧固定导航栏 + 右侧独立工作区，包含实验中心、运行监控、实验结果、历史记录四个主要页面；资源和环境配置放在次级设置入口。
- 内置预设实验方案和对应的预置结果，打开软件即可浏览实验结果。
- 支持重新运行任一可运行的预设变体，实际调用检索流程并生成新的实验结果。
- 同一时间只运行一个实验，其他任务进入等待队列；任务由后端统一管理。
- 运行时展示当前实验阶段，不要求量化整体进度；支持取消运行，失败 / 取消的运行也保留记录和相关信息。
- 每次运行生成独立 JSON 记录及图表文件；JSON 使用相对路径引用图表，并支持导入单份 JSON 查看结果。
- 导入时将 JSON 和可用图表复制到 `results/imports/<run_id>/`；缺少图表不阻止导入，格式版本不兼容时拒绝导入。
- 展示现有实验指标（P@K、R@K、MRR、NDCG@K）和分阶段耗时；算法、Embedding 模型、参数和耗时对比图由比较页面按所选记录生成。
- 结果按 `results/presets`、`results/runs`、`results/imports` 区分来源，JSON 不保存来源和导入时间。
- 历史比较限定在同一方案及版本内；时间戳区分每次运行，稳定的方案 ID、版本和变体 ID 用于归组与比较。
- 实验记录由用户通过文件管理器管理，应用内不提供删除功能；results/README.md 说明 JSON 与图表的关系及预置结果删除后果。
- 记录实验环境信息，以便理解不同设备和软件环境下的运行差异。
- SciFact 默认随软件提供并附带来源 / CC-BY-SA-4.0 许可说明；DuRetrieval 由用户自行获取，软件提供来源和路径配置；Embedding 模型文件由用户自行准备。
- 用户配置统一模型根目录，TextMind 按预设模型名称查找对应子目录；不自动扫描、下载或整理资源。
- 复用现有 test 脚本进行 Python 环境自检；MVP 不自动安装依赖。
- 尽量复用现有 Python 检索核心、实验配置和结构化结果，不为 Web 界面重写算法实现。

### 后续演进：有限实验配置

在 MVP 稳定后，再逐步考虑：
- 选择已支持的检索算法与 Embedding 模型。
- 调整有限范围内的算法参数，例如 BM25 参数、RRF 权重。
- 继续完善同一方案及版本下的运行历史查看与比较。
- 视需求增加可选数据集。

以下方向不属于当前产品路线：任意第三方检索器 / 模型插件、自由组合检索流水线、自动调参研究平台、个人知识库管理、AI 问答与笔记生成、云端部署和公网访问。

## 已有技术基础

TextMind 已完成一轮文本检索算法实验，相关实验代码与成果是软件化的基础：

- **检索方法**：BM25、Dense Retrieval、Hybrid Retrieval（RRF）。
- **数据集**：英文 SciFact、中文 DuRetrieval（C-MTEB 子集）。
- **Embedding 模型**：Qwen3-Embedding-0.6B、BGE-M3、bge-small-zh-v1.5、all-MiniLM-L6-v2。
- **评价指标**：Precision@K、Recall@K、MRR、NDCG@K，以及分阶段耗时。
- **实验产物**：结构化 JSON 结果、汇总表和图表。
- **开发环境**：以 SDS 和当前环境检查脚本的结果为准。

实验结论和数据集信息属于已有技术资料，不代表软件 MVP 已经实现相应 GUI 功能。

## 安装、检查与启动

需要 Python 3.11（本项目在 3.11.18 上验证）。依赖统一安装在项目根目录 `.venv`，不使用全局环境、不下载模型或数据集。

```bash
# 1. 创建 .venv 并从 requirements.lock 安装锁定依赖
#    当前 python3 不是 3.11 时，用 --python 或 TEXTMIND_PYTHON 指定 3.11 解释器
python3 scripts/install.py --python /path/to/python3.11
# TEXTMIND_PYTHON=/path/to/python3.11 python3 scripts/install.py

# 2. 检查 Python 3.11 与核心依赖
.venv/bin/python scripts/check_env.py

# 3. 启动本地服务（默认 http://127.0.0.1:8765，仅监听回环地址）
.venv/bin/python scripts/start.py

# 4. 健康检查
curl http://127.0.0.1:8765/api/v1/health
```

- 依赖版本以 `requirements.lock` 为唯一来源；安装失败时脚本返回非零退出码。
- 启动前执行核心依赖检查，检查不通过或端口被占用时阻止启动并输出错误。
- 服务只监听 `127.0.0.1`，当前版本不开启局域网访问，也不自动打开浏览器。
- 启动后在浏览器打开 <http://127.0.0.1:8765/>：可浏览实验中心和方案详情、提交单个变体并查看运行监控（含 SSE 实时状态）。实验结果、历史记录和设置目前只是明确的占位入口，具体功能由后续任务实现。

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

DuRetrieval 使用 dev split。其加权 RRF 参数探索也在 dev split 上完成，不应解读为独立测试集上的泛化结论。完整指标与耗时汇总见 [实验汇总表](data/evaluation/summary.md)，开发环境与运行约束见 [SDS](docs/design/SDS.md)。

## 文档导航

- [当前产品范围与 MVP](docs/project-scope.md)
- [已确认的产品决策](docs/requirements/confirmed-decisions.md)
- [软件需求规格说明书（SRS v0.2，冻结）](docs/requirements/SRS.md)
- [SRS / 技术设计待处理事项](docs/requirements/open-questions.md)
- [软件设计说明书（SDS v0.1 评审通过版）](docs/design/SDS.md)
- [SRS v0.2 至 SDS 需求追溯矩阵](docs/design/srs-v0.2-traceability.md)
- [Codex / DSH 开发协作规则](AGENTS.md)
- [开发计划](docs/development/plan.md)
- [开发任务索引](docs/development/README.md)
- [开发任务模板](docs/development/TEMPLATE.md)
- [检索实验与执行约定](docs/requirements/retrieval-lab.md)
- [数据集调研记录](docs/datasets.md)

## 当前阶段说明

实验算法与数据分析阶段已形成可复用成果。SRS v0.2 已于 2026-11-21 冻结，SDS v0.1 已于 2026-11-21 通过设计评审；本文和需求文档描述的是目标要求，不代表软件功能已经实现。技术选型、模块边界、任务与结果机制见 SDS，剩余版本锁定和实现验证项见待处理事项。
