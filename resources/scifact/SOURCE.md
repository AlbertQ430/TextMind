# SciFact 数据来源

本应用使用 SciFact 数据集（BEIR 版本）作为英文检索评测数据，并随软件提供
`data/raw/scifact/` 中的原始文件。

## 1. 来源

- 数据集：SciFact（BEIR 数据集集合中的 SciFact）。
- 论文：David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi. *Fact or Fiction: Verifying Scientific Claims*. EMNLP 2020.
- 获取入口：BEIR 官方数据仓库 <https://github.com/beir-cellar/beir> 及 Hugging Face `BeIR/scifact`。
- 上游数据页面：<https://huggingface.co/datasets/BeIR/scifact>

## 2. 本应用实际使用的文件

```text
data/raw/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

- `corpus.jsonl`：文档集合（`_id`、`title`、`text`）。
- `queries.jsonl`：查询集合。
- `qrels/test.tsv`：test 划分的相关性标注。

应用只读取上述文件、不修改原始数据；数据随软件按 CC-BY-SA-4.0 分发。

## 3. 许可证与署名

SciFact 数据按 **Creative Commons Attribution-ShareAlike 4.0 International（CC-BY-SA-4.0）** 发布。
许可证全文见同目录 `LICENSE-CC-BY-SA-4.0.txt`，署名与免责声明见 `NOTICE.md`。

## 4. 分发与使用边界

- 数据随软件提供，并按 CC-BY-SA-4.0 分发；应用只读取原始文件，不修改其内容。
- 预置结果构建前会校验本文件、`NOTICE.md`、许可证与 SciFact 原始数据文件存在且非空。
- 派生评测结果（`results/presets/`）保留原始指标与环境信息，不修改原始数据。
