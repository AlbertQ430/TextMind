"""检索器统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class Retriever(ABC):
    """检索器抽象基类。

    约定：
    - ``fit`` 负责构建索引，返回自身，便于链式调用。
    - ``search`` 返回按分数降序排列的 ``(doc_id, score)`` 列表。
    """

    @abstractmethod
    def fit(self, documents) -> "Retriever":
        """基于文档集合构建索引。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """检索 Top-K 文档。"""
