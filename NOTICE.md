# 第三方资源与许可声明

本文件汇总 TextMind 使用的第三方数据集、模型与许可信息。具体数据准备步骤见根
`README.md` 与 `docs/datasets.md`。

## SciFact

- 用途：英文检索评测（test 划分，BM25 / Dense / Hybrid）。
- 来源：BEIR 数据集集合中的 SciFact；论文 Wadden et al., EMNLP 2020。
- 获取：<https://github.com/beir-cellar/beir>、<https://huggingface.co/datasets/BeIR/scifact>
- 许可证：CC-BY-SA-4.0（全文见 `resources/scifact/LICENSE-CC-BY-SA-4.0.txt`）。
- 分发方式：本应用随软件提供 `data/raw/scifact/` 中的 SciFact 数据（`corpus.jsonl`、`queries.jsonl`、`qrels/test.tsv`），并按 CC-BY-SA-4.0 分发。
- 本应用只读取原始文件、不修改内容；`results/presets/` 中的预置结果为派生评测结果。

## DuRetrieval（C-MTEB）

- 用途：中文检索评测（dev 划分）。
- 来源：C-MTEB 在 Hugging Face 上整理的 DuRetrieval 评测版本。
- 获取入口：<https://huggingface.co/datasets/C-MTEB/DuRetrieval>（Corpus / Queries）、<https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels>（Qrels）。
- 固定 commit：Corpus / Queries `a1a333e290fe30b10f3f56498e3a0d911a693ced`；Qrels `497b7bd1bbb25cb3757ff34d95a8be50a3de2279`（dev split）。
- 本应用使用：`corpus.jsonl`、`queries.jsonl`、`qrels_dev.tsv`。
- 许可证核验：官方 `baidu/DuReader` 仓库声明 Apache License 2.0（Copyright 2017 Baidu.com），授权核验完成于 2026-11-20；Hugging Face 的 C-MTEB 镜像数据卡未标注 license，以官方仓库许可为准。
- 数据不随包分发，应用不会自动下载；完整说明见 `docs/datasets.md`。

## Embedding 模型

以下模型文件不随包分发，用户需按来源链接自行下载到统一的模型根目录：

| 模型 | 目录名 | revision | 来源 | 仓库许可 |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3-Embedding-0.6B | Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 | <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> | Apache-2.0 |
| BAAI/bge-m3 | bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 | <https://huggingface.co/BAAI/bge-m3> | MIT |
| sentence-transformers/all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 | <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2> | Apache-2.0 |
| BAAI/bge-small-zh-v1.5 | bge-small-zh-v1.5 | 7999e1d3359715c523056ef9478215996d62a620 | <https://huggingface.co/BAAI/bge-small-zh-v1.5> | MIT |

- 许可证以各模型仓库页面为准；下载后请保留模型仓库自带的 LICENSE/模型卡。
- 本应用只从 `<模型根目录>/<目录名>` 加载本地文件，并启用本地文件模式，不会按 Hub 名称联网下载。

## Python 依赖

运行时依赖及其版本锁定见 `requirements.lock`。
