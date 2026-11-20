#!/usr/bin/env python3
"""中文检索结果错误分析（DuRetrieval）。

对 BM25 / Dense / Hybrid 的 query 级结果做三类诊断：

- ``hit_at_1``：Top-1 命中相关文档；
- ``ranked_low``：Top-10 内有相关文档，但 Top-1 不是；
- ``miss_top_k``：Top-10 内没有相关文档（召回失败）。

同时对比 Hybrid 与 Dense 的 Top-1 命中差异，抽取典型失败案例，
输出 ``data/evaluation/error_analysis.json`` 与 ``docs/error-analysis.md``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from document.dataset import load_duretrieval  # noqa: E402
from retrieval.bm25 import BM25Retriever, jieba_tokenize  # noqa: E402
from retrieval.dense import DenseRetriever  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "duretrieval"
RESULT_DIR = PROJECT_ROOT / "data" / "evaluation"
DOCS_DIR = PROJECT_ROOT / "docs"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_PREFIX = "为这个句子生成表示以用于检索相关文章："

Ranked = List[Tuple[str, float]]


def diagnose(ranked: Ranked, relevant: set, k: int = 10) -> str:
    """把单条 query 的结果分为三类之一。"""
    if not ranked:
        return "empty"
    if ranked[0][0] in relevant:
        return "hit_at_1"
    if relevant & {doc_id for doc_id, _ in ranked[:k]}:
        return "ranked_low"
    return "miss_top_k"


def snippet(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def collect_ranked(retriever, queries: Sequence) -> Dict[str, Ranked]:
    return {query.query_id: retriever.search(query.text, top_k=100) for query in queries}


def collect_ranked_dense(dense: DenseRetriever, texts: Sequence[str], query_ids: Sequence[str]) -> Dict[str, Ranked]:
    vectors = dense.encode_queries(texts)
    return {
        query_id: dense.search_vector(vector, top_k=100)
        for query_id, vector in zip(query_ids, vectors)
    }


def collect_ranked_hybrid(hybrid: HybridRetriever, texts: Sequence[str], query_ids: Sequence[str]) -> Dict[str, Ranked]:
    fused = hybrid.search_batch(texts, top_k=100)
    return dict(zip(query_ids, fused))


def summarize(ranked_by_query: Dict[str, Ranked], qrels: Dict[str, Dict[str, int]], k: int = 10) -> dict:
    counts = {"hit_at_1": 0, "ranked_low": 0, "miss_top_k": 0, "empty": 0}
    for query_id, ranked in ranked_by_query.items():
        relevant = {doc_id for doc_id, score in qrels[query_id].items() if score > 0}
        counts[diagnose(ranked, relevant, k)] += 1
    total = sum(counts.values()) or 1
    return {
        "counts": counts,
        "ratios": {key: round(value / total, 4) for key, value in counts.items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DuRetrieval 检索结果错误分析")
    parser.add_argument("--limit", type=int, default=None, help="只分析前 N 条 query（调试用）")
    parser.add_argument("--output-json", default=str(RESULT_DIR / "error_analysis.json"))
    parser.add_argument("--output-md", default=str(DOCS_DIR / "error-analysis.md"))
    args = parser.parse_args(argv)

    dataset = load_duretrieval(
        DATA_DIR / "corpus.jsonl", DATA_DIR / "queries.jsonl", DATA_DIR / "qrels_dev.tsv"
    )
    queries = [q for q in dataset.queries if q.query_id in dataset.qrels]
    if args.limit is not None:
        queries = queries[: args.limit]

    from sentence_transformers import SentenceTransformer

    bm25 = BM25Retriever(k1=1.2, b=0.75, tokenizer=jieba_tokenize).fit(dataset.documents)
    model = SentenceTransformer(DEFAULT_MODEL, device="mps")
    dense = DenseRetriever(
        model,
        cache_dir=CACHE_DIR,
        cache_key=DEFAULT_MODEL,
        model_name=DEFAULT_MODEL,
    ).fit(dataset.documents)
    hybrid = HybridRetriever(
        [bm25, dense],
        rrf_k=60,
        query_transforms=[None, lambda query: DEFAULT_PREFIX + query],
    )

    query_ids = [query.query_id for query in queries]
    raw_texts = [query.text for query in queries]
    prefixed = [DEFAULT_PREFIX + text for text in raw_texts]

    bm25_ranked = collect_ranked(bm25, queries)
    dense_ranked = collect_ranked_dense(dense, prefixed, query_ids)
    hybrid_ranked = collect_ranked_hybrid(hybrid, raw_texts, query_ids)

    qrels = {query_id: dataset.qrels[query_id] for query_id in query_ids}
    methods = {
        "bm25_jieba": summarize(bm25_ranked, qrels),
        "dense_bge_small_zh": summarize(dense_ranked, qrels),
        "hybrid_bm25_dense": summarize(hybrid_ranked, qrels),
    }

    # Hybrid 相对 Dense 的 Top-1 命中变化
    gains, losses = [], []
    for query in queries:
        relevant = {doc_id for doc_id, score in qrels[query.query_id].items() if score > 0}
        dense_hit = diagnose(dense_ranked[query.query_id], relevant) == "hit_at_1"
        hybrid_hit = diagnose(hybrid_ranked[query.query_id], relevant) == "hit_at_1"
        if not dense_hit and hybrid_hit:
            gains.append(query.query_id)
        elif dense_hit and not hybrid_hit:
            losses.append(query.query_id)

    # 典型失败案例：Dense 的 ranked_low / miss_top_k
    doc_text = {doc.doc_id: doc.text for doc in dataset.documents}
    query_text = {query.query_id: query.text for query in queries}
    cases = {"ranked_low": [], "miss_top_k": []}
    for query in queries:
        relevant = {doc_id for doc_id, score in qrels[query.query_id].items() if score > 0}
        diag = diagnose(dense_ranked[query.query_id], relevant)
        if diag in cases and len(cases[diag]) < 3:
            top1 = dense_ranked[query.query_id][0][0] if dense_ranked[query.query_id] else None
            rel_docs = sorted(relevant)[:2]
            cases[diag].append(
                {
                    "query_id": query.query_id,
                    "query": query_text[query.query_id],
                    "relevant_docs": [
                        {"doc_id": doc_id, "snippet": snippet(doc_text.get(doc_id, ""))}
                        for doc_id in rel_docs
                    ],
                    "dense_top1": {"doc_id": top1, "snippet": snippet(doc_text.get(top1, ""))},
                }
            )

    report = {
        "dataset": "duretrieval",
        "num_queries": len(queries),
        "k": 10,
        "methods": methods,
        "hybrid_vs_dense": {
            "gains_top1": len(gains),
            "losses_top1": len(losses),
            "gain_queries": gains[:10],
            "loss_queries": losses[:10],
        },
        "cases": cases,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = build_markdown(report)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")

    print(f"分析完成：{len(queries)} 条 query")
    for name, stats in methods.items():
        print(f"  {name}: {stats['counts']}")
    print(f"Hybrid vs Dense Top-1：收益 {len(gains)} / 损失 {len(losses)}")
    print(f"已写入: {output_json}")
    print(f"已写入: {output_md}")
    return 0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown(report: dict) -> str:
    lines = [
        "# DuRetrieval 检索结果错误分析",
        "",
        f"分析 query 数：{report['num_queries']}，诊断窗口 K = {report['k']}。",
        "诊断口径：Top-1 命中记为 `hit_at_1`；Top-10 内有相关但 Top-1 不是记为 `ranked_low`；",
        "Top-10 内无相关记为 `miss_top_k`。",
        "",
        "## 1. 总体命中分布",
        "",
        "| 方法 | Top-1 命中 | Top-1 未命中但 Top-10 内 | Top-10 召回失败 |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in report["methods"].items():
        ratios = stats["ratios"]
        lines.append(
            f"| {name} | {_pct(ratios['hit_at_1'])} | {_pct(ratios['ranked_low'])} | "
            f"{_pct(ratios['miss_top_k'])} |"
        )
    bm25 = report["methods"]["bm25_jieba"]["ratios"]
    dense = report["methods"]["dense_bge_small_zh"]["ratios"]
    hybrid = report["methods"]["hybrid_bm25_dense"]["ratios"]
    gains = report["hybrid_vs_dense"]["gains_top1"]
    losses = report["hybrid_vs_dense"]["losses_top1"]
    lines += [
        "",
        "## 2. Hybrid 相对 Dense 的 Top-1 变化",
        "",
        f"- Dense 未命中而 Hybrid 命中：{gains} 条；",
        f"- Dense 命中而 Hybrid 未命中：{losses} 条。",
        "",
        "等权 RRF 融合对 BM25 与 Dense 的排名做证据叠加，收益与损失同时存在。",
        "",
        "## 3. 典型失败案例（Dense）",
        "",
    ]
    for diag, title in (("ranked_low", "Top-10 召回但排序偏低"), ("miss_top_k", "Top-10 召回失败")):
        lines.append(f"### {title}")
        lines.append("")
        for case in report["cases"][diag]:
            lines.append(f"- **Query** `{case['query_id']}`：{case['query']}")
            for doc in case["relevant_docs"]:
                lines.append(f"  - 相关文档 `{doc['doc_id']}`：{doc['snippet']}")
            lines.append(f"  - Dense Top-1 `{case['dense_top1']['doc_id']}`：{case['dense_top1']['snippet']}")
        lines.append("")
    lines += [
        "## 4. 结论",
        "",
        f"1. Dense 的 Top-1 命中率 {_pct(dense['hit_at_1'])}，显著高于 BM25"
        f"（{_pct(bm25['hit_at_1'])}）；Dense 的 Top-10 召回失败率仅 {_pct(dense['miss_top_k'])}，"
        "说明中文语义模型已基本解决召回问题，剩余损失主要在排序。",
        f"2. `ranked_low` 案例（Dense {_pct(dense['ranked_low'])}）多为语义近似但主题不同的"
        "同类文档（例如拆机类问题混淆不同型号），属于排序校准问题。",
        f"3. `miss_top_k` 案例（Dense {_pct(dense['miss_top_k'])}）集中在术语歧义、领域漂移"
        "（游戏术语、口语化短 query）以及短口语 query 与网页长文之间的语义鸿沟。",
        f"4. Hybrid 相对 Dense 的 Top-1 收益 {gains} 条、损失 {losses} 条"
        f"（净变化 {gains - losses:+d} 条）；等权 RRF 在 BM25 明显弱于 Dense 时不占优，"
        "后续可尝试加权 RRF 或按数据集调低 BM25 权重。",
        "",
        f"> 本报告由 `src/scripts/error_analysis.py` 生成，样例按 query 顺序抽取，可复现。"
        f"完整统计见 `data/evaluation/error_analysis.json`。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
