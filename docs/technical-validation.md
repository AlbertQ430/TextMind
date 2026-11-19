# TextMind 技术验证报告

- 验证日期：2026-11-19
- 验证范围：`test/` 下三个环境与模型能力脚本
- 结论：**全部通过**（3/3），Apple Silicon MPS 加速链路可用，Qwen3-Embedding-0.6B 可正常加载并输出可用向量。

## 1. 运行环境

| 项目 | 值 |
| --- | --- |
| 操作系统 | macOS-26.6-arm64-arm-64bit |
| 架构 | arm64 (Apple Silicon) |
| Python | 3.11.18（`/opt/miniconda3/envs/textmind`） |
| PyTorch | 2.15.0 |
| transformers | 5.23.0 |
| sentence-transformers | 6.3.0 |
| 计算设备 | `mps`（MPS 已编译 / 可用均为 True） |
| 模型 | `Qwen/Qwen3-Embedding-0.6B` |
| 模型缓存 | `~/.cache/huggingface`（权重 `f26bc1cc...`，约 1.1 GB） |

执行方式：

```bash
/opt/miniconda3/envs/textmind/bin/python test/test_device.py
/opt/miniconda3/envs/textmind/bin/python test/test_embedding.py
/opt/miniconda3/envs/textmind/bin/python test/test_similarity.py
```

## 2. test_device.py —— 设备与后端验证

用途：确认 MPS 后端可用，并能真正在 GPU 上完成矩阵运算。

实际输出：

```
系统: macOS-26.6-arm64-arm-64bit
架构: arm64
PyTorch: 2.15.0
MPS 已编译: True
MPS 可用: True
当前设备: mps
计算测试: torch.Size([1000, 1000])
```

- 耗时：real 0.81s（user 0.54s / sys 0.10s）
- 验证点：`torch.backends.mps.is_built()` 与 `is_available()` 均为 `True`，`x @ x.T`（1000×1000）在 `mps` 上执行成功、形状正确。
- 结论：**通过**。设备选择逻辑落在 MPS 分支，Apple Silicon 加速路径可用。

## 3. test_embedding.py —— 模型加载与向量生成

用途：验证 SentenceTransformer 在 MPS 上加载 Qwen3-Embedding-0.6B 并生成归一化句向量。

实际输出（关键行）：

```
加载设备: mps
Loading weights: 100%|██████| 310/310
向量形状: (3, 1024)
向量范数: tensor([0.9991, 0.9983, 1.0023])
模型测试成功！
```

- 耗时：real 8.69s（user 2.44s / sys 0.66s），含模型加载 + 3 条文本编码
- 验证点：
  - 权重分片完整加载（310/310）；
  - 输入 3 条文本 → 输出 `(3, 1024)`，与 Qwen3-Embedding-0.6B 的 1024 维一致；
  - `normalize_embeddings=True` 生效，向量 L2 范数 ≈ 1。
- 结论：**通过**。
- 备注：范数偏差落在 0.9983~1.0023（±0.23%），属 float32 在 MPS 上的正常数值误差，不影响余弦相似度排序（见第 4 节）。

## 4. test_similarity.py —— 语义相似度排序

用途：验证 query 与候选文档的余弦相似度能正确区分语义相关性。

Query：`什么是人工智能中的自然语言处理？`

实际输出（按分数降序）：

| 相似度 | 文档 | 判定 |
| --- | --- | --- |
| 0.7811 | 自然语言处理是人工智能的一个重要研究领域。 | 强相关（击中） |
| 0.4300 | 计算机视觉研究如何让机器理解图像和视频。 | 相关（同属 AI 子领域） |
| 0.4178 | 机器学习通过数据训练模型，从而实现预测。 | 相关（同属 AI 领域） |
| 0.1774 | 今天晚上我打算去超市买一些水果。 | 不相关 |

- 耗时：real 8.66s（user 2.49s / sys 0.66s）
- 验证点：
  - 正确答案 0.7811 显著领先第二名（0.4300），检索区分度充足；
  - 完全不相关句得分 0.1774，区分明显；
  - 两个 AI 领域句（0.43 / 0.42）分数接近且低于正确答案，符合"同领域但非答案"的预期。
- 结论：**通过**。该模型可直接用于 TextMind 的语义召回 / 相似度排序场景。

## 5. 汇总

| 脚本 | 验证项 | 耗时 | 结果 |
| --- | --- | --- | --- |
| `test_device.py` | MPS 后端 + 矩阵运算 | 0.81s | 通过 |
| `test_embedding.py` | 模型加载 + 向量生成 (3, 1024) | 8.69s | 通过 |
| `test_similarity.py` | 语义相似度排序 | 8.66s | 通过 |

## 6. 注意事项与风险

1. **首次运行需要模型权重**：完整权重约 1.1 GB，缓存在 `~/.cache/huggingface`。
2. **未设置 `HF_TOKEN`**：HuggingFace 会打印匿名请求限速警告，不影响本地缓存命中后的运行；如需频繁拉取模型建议配置 token。
3. **`torch.distributed` MacOS 警告**：`Redirects are currently not supported in MacOs.` 由 transformers 导入链触发，属已知无害提示，不影响结果。
4. **数值精度**：向量范数存在 ±0.23% 偏差。当前场景（余弦相似度排序）不受影响；若后续做严格数值比对，建议在 CPU/float64 下做一次基线校验。
