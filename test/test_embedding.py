"""Smoke test：Qwen3-Embedding-0.6B 加载与向量生成。

独立运行：``python test/test_embedding.py``

本模块在导入时不加载模型、不访问网络、不执行计算，
因此可以安全地被 ``python -m unittest discover`` 导入。
"""

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("加载设备:", device)
    print("正在加载模型...")

    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [
        "人工智能正在改变自然语言处理技术。",
        "自然语言处理是人工智能的重要研究方向。",
        "今天的天气非常晴朗。",
    ]

    print("正在生成文本向量...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=2,
        show_progress_bar=True,
    )

    print("向量形状:", embeddings.shape)
    print("向量范数:", torch.tensor(embeddings).norm(dim=1))
    print("模型测试成功！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
