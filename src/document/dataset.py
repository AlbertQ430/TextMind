"""数据集模型与加载器。

当前只实现 SciFact（BEIR / HuggingFace 格式）的加载。内部统一为
Corpus / Queries / Qrels 三部分，所有 ID 一律按字符串处理。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Document:
    """Corpus 中的一篇文档。"""

    doc_id: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        """检索实际使用的文本：标题 + 正文。"""
        parts = [part for part in (self.title, self.text) if part]
        return " ".join(parts)


@dataclass(frozen=True)
class Query:
    """一条检索查询。"""

    query_id: str
    text: str


@dataclass
class RetrievalDataset:
    """Corpus + Queries + Qrels 的统一容器。"""

    documents: List[Document] = field(default_factory=list)
    queries: List[Query] = field(default_factory=list)
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)
    name: str = ""

    def doc_ids(self) -> List[str]:
        return [doc.doc_id for doc in self.documents]

    def query_ids(self) -> List[str]:
        return [query.query_id for query in self.queries]

    def validate(self) -> None:
        """校验 ID 唯一性和 Qrels 引用完整性，不通过时抛 ValueError。"""
        seen_docs = set()
        for doc in self.documents:
            if doc.doc_id in seen_docs:
                raise ValueError(f"重复的文档 ID: {doc.doc_id}")
            seen_docs.add(doc.doc_id)

        seen_queries = set()
        for query in self.queries:
            if query.query_id in seen_queries:
                raise ValueError(f"重复的查询 ID: {query.query_id}")
            seen_queries.add(query.query_id)

        missing_queries = set(self.qrels) - seen_queries
        if missing_queries:
            sample = sorted(missing_queries)[:5]
            raise ValueError(f"Qrels 中存在未定义的查询 ID: {sample}")

        missing_docs = set()
        for query_id, rels in self.qrels.items():
            for doc_id in rels:
                if doc_id not in seen_docs:
                    missing_docs.add(doc_id)
        if missing_docs:
            sample = sorted(missing_docs)[:5]
            raise ValueError(f"Qrels 中存在未收录的文档 ID: {sample}")


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """读取 TREC 风格 qrels（query-id / corpus-id / score，tab 分隔）。

    同一 (query_id, doc_id) 出现多次时抛出 ValueError，避免静默覆盖。
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if "query-id" not in header:
            raise ValueError(f"Qrels 文件缺少表头: {path}")
        for lineno, line in enumerate(handle, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{lineno} 格式错误：需要 3 列（query-id/corpus-id/score），"
                    f"实际 {len(parts)} 列"
                )
            query_id, doc_id, score = parts
            rels = qrels.setdefault(query_id, {})
            if doc_id in rels:
                raise ValueError(
                    f"{path}:{lineno} 重复标注：query {query_id} - document {doc_id}"
                )
            rels[doc_id] = int(score)
    return qrels


def load_scifact(
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
    name: str = "scifact",
) -> RetrievalDataset:
    """加载 SciFact 数据集并校验引用完整性。"""
    corpus_path = Path(corpus_path)
    queries_path = Path(queries_path)
    qrels_path = Path(qrels_path)

    documents = [
        Document(
            doc_id=str(record["_id"]),
            title=str(record.get("title") or ""),
            text=str(record.get("text") or ""),
        )
        for record in _load_jsonl(corpus_path)
    ]
    queries = [
        Query(query_id=str(record["_id"]), text=str(record["text"]))
        for record in _load_jsonl(queries_path)
    ]

    dataset = RetrievalDataset(
        documents=documents,
        queries=queries,
        qrels=_load_qrels(qrels_path),
        name=name,
    )
    dataset.validate()
    return dataset
