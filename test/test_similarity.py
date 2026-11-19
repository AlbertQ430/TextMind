from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="mps"
)

query = "什么是人工智能中的自然语言处理？"

documents = [
    "自然语言处理是人工智能的一个重要研究领域。",
    "机器学习通过数据训练模型，从而实现预测。",
    "今天晚上我打算去超市买一些水果。",
    "计算机视觉研究如何让机器理解图像和视频。"
]

q = model.encode(
    query,
    normalize_embeddings=True
)

d = model.encode(
    documents,
    normalize_embeddings=True
)

scores = cos_sim(q, d)[0]

results = sorted(
    zip(documents, scores.tolist()),
    key=lambda x: x[1],
    reverse=True
)

for text, score in results:
    print(f"{score:.4f} | {text}")
