"""Smoke test：query 与候选文档的余弦相似度排序。

独立运行：``python test/test_similarity.py``

本模块在导入时不加载模型、不访问网络、不执行计算，
因此可以安全地被 ``python -m unittest discover`` 导入。
"""

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    import torch

    #使用 MPS 可用性检测，必要时回退 CPU。
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)

    query = "什么是人工智能中的自然语言处理？"

    documents = [
        "自然语言处理是人工智能的一个重要研究领域。",
        "机器学习通过数据训练模型，从而实现预测。",
        "今天晚上我打算去超市买一些水果。",
        "计算机视觉研究如何让机器理解图像和视频。",
    ]

    q = model.encode(query, normalize_embeddings=True)
    d = model.encode(documents, normalize_embeddings=True)

    scores = cos_sim(q, d)[0]

    results = sorted(
        zip(documents, scores.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    for text, score in results:
        print(f"{score:.4f} | {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
