# TextMind

TextMind 是一个在本机运行的文本检索实验台。它把 BM25、Dense Retrieval 和 Hybrid Retrieval（RRF）实验封装成 Web 界面，适合浏览预设结果、重新运行单个变体，以及比较同一方案版本下的多次结果。

TextMind 不依赖云服务。默认只监听本机，也可以由主机主动开启可信局域网访问。软件不会自动下载模型或数据集，也不会在应用内删除运行记录。

## 当前版本与支持范围

- 支持 Windows 11 与 macOS。
- 已验证 Python 3.11。
- 浏览器支持基线：**Safari 17、Edge 120**。
- 发布包自带 SciFact 数据和 17 条预置结果，不需要额外模型即可查看。
- 重新运行 BM25 只需要 SciFact；Dense 和 Hybrid 需要用户自行准备本地 Embedding 模型。
- 不随包分发 DuRetrieval 或 Embedding 模型。

## 快速开始

先在项目根目录安装依赖。依赖只写入项目内的 `.venv`，不会安装到系统 Python。

### macOS

```bash
# 当前 python3 不是 3.11 时，用 --python 指定 Python 3.11 解释器
python3 scripts/install.py --python /path/to/python3.11

.venv/bin/python scripts/check_env.py
.venv/bin/python scripts/start.py

# 也可以在访达中双击 start.command
```

### Windows

```bat
py -3.11 scripts\install.py

.venv\Scripts\python.exe scripts\check_env.py
.venv\Scripts\python.exe scripts\start.py

:: 也可以双击 start.bat
```

如果 Windows 上 pip 长时间停在 `Using pip`，且没有网络或磁盘活动，可先执行：

```bat
set PIP_NO_CACHE_DIR=1
py -3.11 scripts\install.py
```

启动后在浏览器打开：

```text
http://127.0.0.1:8765/
```

服务默认只监听回环地址，不会自动打开浏览器。启动前会检查 Python 和核心依赖；依赖缺失或端口被占用时会阻止启动并在终端显示原因。

## 使用流程

### 1. 浏览实验方案

“实验中心”提供三套预设方案：

- SciFact 英文检索方法与模型对比。
- DuRetrieval 中文检索方法对比。
- DuRetrieval 加权 RRF 参数探索。

每套方案包含多个固定变体。进入方案详情可以查看算法、数据集划分、模型、查询前缀、RRF 权重、资源状态和预置结果。

### 2. 查看预置结果

历史页按来源展示三类记录：

| 来源 | 目录 | 说明 |
| --- | --- | --- |
| 预置结果 | `results/presets/` | 软件包随附的参考结果 |
| 本机运行 | `results/runs/` | 用户在本机重新运行产生的结果 |
| 导入记录 | `results/imports/` | 从单份 JSON 导入的结果 |

RRF 等权结果只作为参照，不计入普通历史；可以从对应的加权 RRF 方案详情打开。

### 3. 运行单个变体

在方案详情中选择一个可运行变体，点击“运行”。每次任务只执行一个变体，不会隐式运行同方案的其他变体。

同一时间只有一个实验运行，其他任务按提交顺序等待。关闭或刷新浏览器不会取消已经提交的任务。

### 4. 观察和取消任务

“运行监控”显示当前任务、等待队列、阶段、已完成阶段和运行时间。取消运行中的任务时，任务先进入取消中，并在安全的阶段边界停止；“取消中”可能持续到当前模型加载或批量编码结束。

等待中的任务可以立即取消，且不会生成运行记录。

### 5. 查看和比较结果

结果页显示：

- 方案、版本、变体、运行 ID 和状态。
- P@K、R@K、MRR、NDCG@K 等聚合指标。
- 数据集加载、模型加载、缓存、编码、检索与融合等分阶段耗时。
- 图表、配置和运行环境。

结果页不展示查询级检索明细。历史页只能比较方案 ID 和版本均相同的记录；比较图按所选记录即时生成，不写回原 JSON。

## 准备数据与模型

### SciFact

SciFact 随软件提供，位于 `data/raw/scifact/`，包含：

```text
corpus.jsonl
queries.jsonl
qrels/test.tsv
```

来源、署名和 CC-BY-SA-4.0 许可证见 [NOTICE.md](NOTICE.md) 与 [SciFact 来源说明](resources/scifact/SOURCE.md)。

### DuRetrieval

DuRetrieval 不随软件提供。用户自行获取 dev 数据后，在“设置”中填写目录。目录需要包含：

```text
corpus.jsonl
queries.jsonl
qrels_dev.tsv
```

获取入口、固定 commit 和许可信息见 [NOTICE.md](NOTICE.md) 与 [数据集说明](docs/datasets.md)。

### Embedding 模型

所有模型放在同一个模型根目录下，子目录名必须与下表一致：

```text
<model_root>/
  Qwen3-Embedding-0.6B/
  bge-m3/
  all-MiniLM-L6-v2/
  bge-small-zh-v1.5/
```

| 模型 | revision | 来源 |
| --- | --- | --- |
| Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | [Hugging Face](https://huggingface.co/BAAI/bge-m3) |
| sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| BAAI/bge-small-zh-v1.5 | `7999e1d3359715c523056ef9478215996d62a620` | [Hugging Face](https://huggingface.co/BAAI/bge-small-zh-v1.5) |

应用只从本地目录加载模型并启用离线文件模式，**不会自动下载**模型或数据集，也不会自动扫描磁盘。缺少某个模型时，只禁用依赖该模型的变体，BM25 和预置结果仍可使用。

资源配置和实际设备状态可在 `/settings` 页面查看。

## 局域网访问

默认只允许本机访问。主机可以在 `/settings` 的“局域网访问”中开启第二个监听器：

1. 设置 LAN 端口，默认 `8766`。
2. 点击“开启局域网”。
3. 在其他设备打开页面显示的 `http://<主机 IPv4>:<端口>/`。
4. 输入主机设置页显示的 6 位数字临时访问码。

局域网客户端可以查看方案、监控、结果和历史，提交或取消实验任务，并确认系统提示。局域网客户端不能修改主机设置、导入 JSON 或开关局域网。

访问码只保存在内存中；关闭局域网或重启服务后失效。6 位数字码应配合登录限流使用，并且只适用于可信私人局域网。局域网使用 HTTP 明文传输，请勿暴露到公网。

## 结果文件管理

每个运行记录位于 `results/<来源>/<run_id>/`：

```text
run.json
charts/
  metrics.png
  timing.png
```

- `run.json` 是运行记录的唯一事实源。
- JSON 使用相对路径引用图表。
- 移动整个记录目录可以保留图表；只移动 JSON 会导致图表不可用。
- 应用内没有删除功能。请通过文件管理器管理记录。
- 预置结果被删除后不会自动恢复。

完整说明见 [结果目录说明](results/README.md)。

## 导入 JSON

在“历史记录”中选择单份 `.json` 文件导入，最大 5 MiB。导入时：

- JSON 会复制到 `results/imports/<run_id>/`，原始内容不变。
- 原始实验时间和环境信息保持不变。
- 缺失图表不会阻止导入，结果页会逐项标记不可用。
- 重复 Run ID、不兼容格式版本或路径越界会被拒绝。
- 未知方案或变体可以只读查看，但不能参与比较。

## 常见问题

### 启动时提示环境检查失败

重新执行安装脚本，然后运行：

```bash
.venv/bin/python scripts/check_env.py
```

Windows 对应命令为：

```bat
.venv\Scripts\python.exe scripts\check_env.py
```

### 端口被占用

默认本机端口是 `8765`。关闭占用端口的程序，或指定其他端口：

```bash
.venv/bin/python scripts/start.py --port 8780
```

### Dense 或 Hybrid 显示“不可运行”

在 `/settings` 检查模型根目录、模型子目录名、权重文件、tokenizer 文件和计算设备。显式选择不可用的 CUDA/MPS 时不会静默回退到 CPU。

### DuRetrieval 变体不可运行

确认 `duretrieval_root` 指向包含 `corpus.jsonl`、`queries.jsonl` 和 `qrels_dev.tsv` 的目录。只缺少 DuRetrieval 不会影响 SciFact BM25 和预置结果。

### 局域网设备打不开页面

确认功能已开启、端口未被防火墙阻止、设备与主机在同一局域网，并使用设置页显示的具体 IPv4 地址。HTTP 连接在地址栏可能显示“不安全”，这是明文局域网的预期提示。

## 发布包

发布 ZIP 自带 SciFact、来源与许可证、17 条预置结果、图表、配置、Schema 和运行代码。用户解压后仍需按上面的步骤创建 `.venv`。

发布包不包含 Embedding 模型、DuRetrieval、本机运行记录或应用数据。

## 文档与许可

- [产品范围](docs/project-scope.md)
- [软件需求规格说明书](docs/requirements/SRS.md)
- [软件设计说明书](docs/design/SDS.md)
- [数据集说明](docs/datasets.md)
- [检索实验约定](docs/requirements/retrieval-lab.md)
- [结果目录说明](results/README.md)
- [第三方资源与许可](NOTICE.md)
- [SciFact 来源与许可](resources/scifact/SOURCE.md)

本软件及其第三方资源的许可信息以 [NOTICE.md](NOTICE.md) 和随附许可证文件为准。

## 当前版本

当前版本 `v0.1.1` 已完成 Windows 11 实机，以及 Safari 26.6、Edge 158 人工验收。Chrome/Firefox 不在支持范围内。
