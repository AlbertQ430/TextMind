# SciFact 数据署名与许可声明

本应用内置的 SciFact 评测数据（`data/raw/scifact/`）来自 BEIR 数据集集合，按
**Creative Commons Attribution-ShareAlike 4.0 International（CC-BY-SA-4.0）** 提供。

## 署名

- 数据集：SciFact
- 论文：Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020
- 上游入口：<https://github.com/beir-cellar/beir>，<https://huggingface.co/datasets/BeIR/scifact>
- 本应用使用文件：`corpus.jsonl`、`queries.jsonl`、`qrels/test.tsv`

## 许可证

> This work is licensed under the Creative Commons
> Attribution-ShareAlike 4.0 International License.
> To view a copy of this license, visit
> <https://creativecommons.org/licenses/by-sa/4.0/> or see the bundled
> `LICENSE-CC-BY-SA-4.0.txt`.

## 免责声明

SciFact 数据按“现状”提供，不附带任何明示或暗示的担保。原始数据版权归原作者所有；
本应用仅按许可证要求署名并保留许可证全文。评测口径、预置结果与本应用代码的许可
以项目根目录 `LICENSE`（如存在）和 `NOTICE.md` 为准。

## 修改说明

本应用不修改原始数据文件；预置结果 `results/presets/` 是在原始数据上运行评测得到的
派生结果，包含原始指标与环境信息。
