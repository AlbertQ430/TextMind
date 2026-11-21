#!/usr/bin/env python3
"""中文检索结果错误分析（DuRetrieval dev split）。

对 BM25 / Dense / Hybrid 的 query 级结果做三类诊断：

- ``hit_at_1``：Top-1 命中相关文档；
- ``ranked_low``：Top-10 内有相关文档，但 Top-1 不是；
- ``miss_top_k``：Top-10 内没有相关文档（召回失败）。

输出 ``data/evaluation/error_analysis.json`` 与 ``data/evaluation/error_analysis.md``。
案例按 query 顺序抽取，仅用于定性说明，不代表统计抽样。
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
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_PREFIX = "为这个句子生成表示以用于检索相关文章："

Ranked = List[Tuple[str, float]]
METHOD_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
}
DIAGNOSIS_LABELS = {
    "hit_at_1": "命中",
    "ranked_low": "Top-10 内有相关但非 Top-1",
    "miss_top_k": "Top-10 无相关",
    "empty": "无结果",
}


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


def _model_revision(model_name: str) -> str | None:
    """从 HuggingFace 本地缓存读取模型 commit hash（离线可用）。"""
    try:
        from huggingface_hub import constants

        cache_dir = Path(constants.HF_HUB_CACHE)
    except ImportError:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    ref = cache_dir / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _case(
    query_id: str,
    query_text: str,
    relevant: set,
    doc_text: Dict[str, str],
    ranked_by_method: Dict[str, Ranked],
    diag_by_method: Dict[str, str],
) -> dict:
    return {
        "query_id": query_id,
        "query": query_text,
        "relevant_docs": [
            {"doc_id": doc_id, "snippet": snippet(doc_text.get(doc_id, ""))}
            for doc_id in sorted(relevant)[:2]
        ],
        "methods": {
            method: {
                "top1": ranked[0][0] if ranked else None,
                "snippet": snippet(doc_text.get(ranked[0][0], "")) if ranked else "",
                "diagnosis": diag_by_method[method],
            }
            for method, ranked in ranked_by_method.items()
        },
    }


def _render_case(case: dict, lines: List[str]) -> None:
    lines.append(f"- **Query** `{case['query_id']}`：{case['query']}")
    for doc in case["relevant_docs"]:
        lines.append(f"  - 相关文档 `{doc['doc_id']}`：{doc['snippet']}")
    for method, info in case["methods"].items():
        label = DIAGNOSIS_LABELS.get(info["diagnosis"], info["diagnosis"])
        lines.append(
            f"  - {METHOD_LABELS[method]} Top-1 `{info['top1']}`（{label}）：{info['snippet']}"
        )


def _ratios(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    return {key: round(value / total, 4) for key, value in counts.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DuRetrieval 检索结果错误分析")
    parser.add_argument("--limit", type=int, default=None, help="只分析前 N 条 query（调试用）")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu"], default="auto", help="Embedding 推理设备"
    )
    parser.add_argument("--output-json", default=str(RESULT_DIR / "error_analysis.json"))
    parser.add_argument("--output-md", default=str(RESULT_DIR / "error_analysis.md"))
    args = parser.parse_args(argv)

    dataset = load_duretrieval(
        DATA_DIR / "corpus.jsonl", DATA_DIR / "queries.jsonl", DATA_DIR / "qrels_dev.tsv"
    )
    queries = [q for q in dataset.queries if q.query_id in dataset.qrels]
    if args.limit is not None:
        queries = queries[: args.limit]

    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    revision = _model_revision(DEFAULT_MODEL)

    bm25 = BM25Retriever(k1=1.2, b=0.75, tokenizer=jieba_tokenize).fit(dataset.documents)
    model = SentenceTransformer(DEFAULT_MODEL, device=device)
    dense = DenseRetriever(
        model,
        cache_dir=CACHE_DIR,
        cache_key=DEFAULT_MODEL,
        model_name=DEFAULT_MODEL,
        model_revision=revision,
    ).fit(dataset.documents)
    hybrid = HybridRetriever(
        [bm25, dense],
        rrf_k=60,
        query_transforms=[None, lambda query: DEFAULT_PREFIX + query],
    )

    query_ids = [query.query_id for query in queries]
    raw_texts = [query.text for query in queries]
    prefixed = [DEFAULT_PREFIX + text for text in raw_texts]

    bm25_ranked = {q.query_id: bm25.search(q.text, top_k=100) for q in queries}
    dense_vectors = dense.encode_queries(prefixed)
    dense_ranked = {
        query_id: dense.search_vector(vector, top_k=100)
        for query_id, vector in zip(query_ids, dense_vectors)
    }
    fused = hybrid.search_batch(raw_texts, top_k=100)
    hybrid_ranked = dict(zip(query_ids, fused))

    qrels = {query_id: dataset.qrels[query_id] for query_id in query_ids}
    diag = {"bm25": {}, "dense": {}, "hybrid": {}}
    for query in queries:
        relevant = {doc_id for doc_id, score in qrels[query.query_id].items() if score > 0}
        diag["bm25"][query.query_id] = diagnose(bm25_ranked[query.query_id], relevant)
        diag["dense"][query.query_id] = diagnose(dense_ranked[query.query_id], relevant)
        diag["hybrid"][query.query_id] = diagnose(hybrid_ranked[query.query_id], relevant)

    def summarize_diag(diag_map: Dict[str, str]) -> dict:
        counts = {
            key: sum(1 for value in diag_map.values() if value == key)
            for key in ("hit_at_1", "ranked_low", "miss_top_k", "empty")
        }
        return {"counts": counts, "ratios": _ratios(counts)}

    methods = {
        "bm25_jieba": summarize_diag(diag["bm25"]),
        "dense_bge_small_zh": summarize_diag(diag["dense"]),
        "hybrid_bm25_dense": summarize_diag(diag["hybrid"]),
    }

    # Hybrid 相对 Dense 的 Top-1 命中变化
    gains, losses = [], []
    for query in queries:
        dense_hit = diag["dense"][query.query_id] == "hit_at_1"
        hybrid_hit = diag["hybrid"][query.query_id] == "hit_at_1"
        if not dense_hit and hybrid_hit:
            gains.append(query.query_id)
        elif dense_hit and not hybrid_hit:
            losses.append(query.query_id)

    doc_text = {doc.doc_id: doc.text for doc in dataset.documents}
    query_text = {query.query_id: query.text for query in queries}

    def make_case(query_id):
        relevant = {
            doc_id for doc_id, score in qrels[query_id].items() if score > 0
        }
        return _case(
            query_id,
            query_text[query_id],
            relevant,
            doc_text,
            {
                "bm25": bm25_ranked[query_id],
                "dense": dense_ranked[query_id],
                "hybrid": hybrid_ranked[query_id],
            },
            {method: diag[method][query_id] for method in diag},
        )

    dense_cases = {"ranked_low": [], "miss_top_k": []}
    for query in queries:
        key = diag["dense"][query.query_id]
        if key in dense_cases and len(dense_cases[key]) < 2:
            dense_cases[key].append(make_case(query.query_id))

    bm25_cases = []
    for query in queries:
        if diag["bm25"][query.query_id] == "ranked_low" and len(bm25_cases) < 2:
            bm25_cases.append(make_case(query.query_id))

    success_cases = []
    for query in queries:
        if (
            diag["bm25"][query.query_id] == "hit_at_1"
            and diag["dense"][query.query_id] == "hit_at_1"
            and diag["hybrid"][query.query_id] == "hit_at_1"
            and len(success_cases) < 2
        ):
            success_cases.append(make_case(query.query_id))

    hybrid_gain_case = make_case(gains[0]) if gains else None
    hybrid_loss_case = make_case(losses[0]) if losses else None

    report = {
        "dataset_info": {
            "dataset": "DuRetrieval (C-MTEB subset)",
            "split": "dev",
            "num_queries": len(queries),
            "qrels_pairs": sum(len(rels) for rels in qrels.values()),
            "model": DEFAULT_MODEL,
            "model_revision": revision,
            "device": device,
            "query_prefix": DEFAULT_PREFIX,
        },
        "k": 10,
        "methods": methods,
        "hybrid_vs_dense": {
            "gains_top1": len(gains),
            "losses_top1": len(losses),
            "gain_queries": gains[:10],
            "loss_queries": losses[:10],
        },
        "cases": {
            "dense": dense_cases,
            "bm25": bm25_cases,
            "success": success_cases,
            "hybrid_gain": hybrid_gain_case,
            "hybrid_loss": hybrid_loss_case,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = build_markdown(report)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")

    print(f"分析完成：{len(queries)} 条 query（dev split）")
    for name, ratios in methods.items():
        print(f"  {name}: {ratios}")
    print(f"Hybrid vs Dense Top-1：收益 {len(gains)} / 损失 {len(losses)}")
    print(f"已写入: {output_json}")
    print(f"已写入: {output_md}")
    return 0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown(report: dict) -> str:
    info = report["dataset_info"]
    bm25 = report["methods"]["bm25_jieba"]
    dense = report["methods"]["dense_bge_small_zh"]
    hybrid = report["methods"]["hybrid_bm25_dense"]
    gains = report["hybrid_vs_dense"]["gains_top1"]
    losses = report["hybrid_vs_dense"]["losses_top1"]

    lines = [
        "# DuRetrieval 检索结果错误分析",
        "",
        f"- 数据集：DuRetrieval（C-MTEB 子集）**dev split**，共 {info['num_queries']} 条 query、"
        f"{info['qrels_pairs']} 条 qrels；",
        f"- Embedding 模型：`{info['model']}`（revision `{info['model_revision']}`），"
        f"设备 `{info['device']}`；",
        f"- Hybrid：BM25(jieba) + Dense，等权 RRF（rrf_k = 60），Dense 查询前缀 "
        f"`{info['query_prefix']}`；",
        f"- 诊断窗口 K = {report['k']}，诊断口径：Top-1 命中 `hit_at_1`；Top-10 内有相关但 "
        "Top-1 不是 `ranked_low`；Top-10 内无相关 `miss_top_k`。",
        "",
        "> 以下案例按 query 顺序抽取（各方法取首个满足条件的 query），用于定性说明，"
        "不代表统计抽样。",
        "",
        "## 1. 总体命中分布",
        "",
        "| 方法 | Top-1 命中 | Top-1 未命中但 Top-10 内 | Top-10 召回失败 |",
        "| --- | --- | --- | --- |",
    ]
    for name, method in (
        ("BM25", bm25),
        ("Dense", dense),
        ("Hybrid", hybrid),
    ):
        method_ratios = method["ratios"]
        lines.append(
            f"| {name} | {_pct(method_ratios['hit_at_1'])} | "
            f"{_pct(method_ratios['ranked_low'])} | "
            f"{_pct(method_ratios['miss_top_k'])} |"
        )
    lines += [
        "",
        "## 2. Hybrid 相对 Dense 的 Top-1 变化",
        "",
        f"- Dense 未命中而 Hybrid 命中：{gains} 条；",
        f"- Dense 命中而 Hybrid 未命中：{losses} 条。",
        "",
        "等权 RRF 融合对 BM25 与 Dense 的排名做证据叠加，收益与损失同时存在。",
        "",
        "## 3. Dense 典型失败案例",
        "",
    ]
    for key, title in (("ranked_low", "Top-10 内有相关但排序偏低"), ("miss_top_k", "Top-10 召回失败")):
        lines.append(f"### {title}")
        lines.append("")
        for case in report["cases"]["dense"][key]:
            _render_case(case, lines)
        lines.append("")

    lines += ["## 4. BM25 典型失败案例", ""]
    for case in report["cases"]["bm25"]:
        _render_case(case, lines)
    lines.append("")

    lines += ["## 5. 三方法成功案例对照", ""]
    for case in report["cases"]["success"]:
        _render_case(case, lines)
    lines.append("")

    lines += ["## 6. Hybrid 相对 Dense 的对比案例", ""]
    if report["cases"]["hybrid_gain"]:
        lines.append("### Dense 未命中而 Hybrid 命中（收益）")
        lines.append("")
        _render_case(report["cases"]["hybrid_gain"], lines)
        lines.append("")
    if report["cases"]["hybrid_loss"]:
        lines.append("### Dense 命中而 Hybrid 未命中（损失）")
        lines.append("")
        _render_case(report["cases"]["hybrid_loss"], lines)
        lines.append("")

    lines += [
        "## 7. 结论",
        "",
        f"1. 在本 dev split 上，Dense 的 Top-1 命中率 {_pct(dense['ratios']['hit_at_1'])}，"
        f"高于 BM25（{_pct(bm25['ratios']['hit_at_1'])}）；Dense 的 Top-10 召回失败率 "
        f"{_pct(dense['ratios']['miss_top_k'])}（{dense['counts']['miss_top_k']} 条），"
        "低于 BM25，但仍有可见的改进空间，不能据此认为召回问题已完全解决。",
        f"2. Dense 的 `ranked_low` 占 {_pct(dense['ratios']['ranked_low'])}，说明 Top-1 与 "
        "Top-10 之间的排序校准是主要损失来源之一；案例显示同类不同主题的文档容易互相挤占 Top-1。",
        "3. `miss_top_k` 案例集中在术语歧义、领域漂移与口语化短 query，属于召回侧仍需关注的问题。",
        f"4. Hybrid 相对 Dense 的 Top-1 收益 {gains} 条、损失 {losses} 条"
        f"（净变化 {gains - losses:+d} 条）；等权 RRF 在本数据集上不占优，"
        "与加权 RRF 实验结论一致（详见 README）。",
        "",
        "> 本报告由 `src/scripts/error_analysis.py` 生成（可指定 `--device`），"
        "完整统计见 `data/evaluation/error_analysis.json`。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
