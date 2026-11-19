import torch
from sentence_transformers import SentenceTransformer

model_name = "Qwen/Qwen3-Embedding-0.6B"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("加载设备:", device)
print("正在加载模型...")

model = SentenceTransformer(
    model_name,
    device=device
)

texts = [
    "人工智能正在改变自然语言处理技术。",
    "自然语言处理是人工智能的重要研究方向。",
    "今天的天气非常晴朗。"
]

print("正在生成文本向量...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=2,
    show_progress_bar=True
)

print("向量形状:", embeddings.shape)
print("向量范数:", torch.tensor(embeddings).norm(dim=1))
print("模型测试成功！")
