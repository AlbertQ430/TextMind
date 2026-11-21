"""单变体实验执行器。

统一完成：数据加载、检索器构建、模型加载、检索、指标评价和分阶段计时。
不负责 Run ID、任务状态、结果目录写入、图表或来源字段；这些由后续任务组装。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

from document.dataset import Query, RetrievalDataset, load_duretrieval, load_scifact
from evaluation.metrics import evaluate
from experiment.environment import collect_environment, resolve_device
from experiment.plans import ModelSpec, PlanDefinition, RunVariant
from experiment.stages import (
    STAGE_BUILD_INDEX,
    STAGE_ENCODE_QUERIES,
    STAGE_EVALUATE,
    STAGE_LOAD_DATASET,
    STAGE_LOAD_MODEL,
    STAGE_PREPARE,
    STAGE_RETRIEVE,
    StageEvent,
    not_applicable_stages,
)
from experiment.timing import DurationRecorder
from paths import DATA_DIR
from records.schema import StageProgress
from resources.settings import Settings
from retrieval.bm25 import BM25Retriever, jieba_tokenize, simple_tokenize
from retrieval.dense import DenseRetriever, Encoder
from retrieval.hybrid import HybridRetriever


class CancellationToken(Protocol):
    """阶段边界取消检查。"""

    def raise_if_cancelled(self, stage: str) -> None: ...


class NeverCancel:
    """默认不取消实现，供 CLI 和测试使用。"""

    def raise_if_cancelled(self, stage: str) -> None:
        return None


class RunCancelled(Exception):
    """在阶段边界检测到取消。"""

    def __init__(self, stage: str, completed: tuple[str, ...] = ()) -> None:
        self.stage = stage
        self.completed = completed
        super().__init__(f"运行在阶段 {stage} 被取消")


@dataclass(frozen=True)
class ExecutionResult:
    """单变体执行结果；不包含 Run ID、状态、图表或来源信息。"""

    dataset: dict[str, object]
    variant_config: dict[str, object]
    metrics: dict[str, float]
    num_queries: int
    num_skipped: int
    timing: dict[str, float | None]
    environment: dict[str, str | None]
    stage_progress: StageProgress


@dataclass
class _Built:
    """一次执行中构建的检索器组合。"""

    retriever: object
    dense: DenseRetriever | None = None
    bm25: BM25Retriever | None = None
    hybrid: HybridRetriever | None = None


def _tokenizer(name: str) -> Callable[[str], Sequence[str]]:
    return jieba_tokenize if name == "jieba" else simple_tokenize


def default_dataset_loader(
    plan: PlanDefinition, settings: Settings | None = None
) -> RetrievalDataset:
    """按方案数据集名称加载本地数据集；不自动下载或创建文件。"""
    name = plan.dataset.name
    if name == "scifact":
        base = DATA_DIR / "raw" / "scifact"
        return load_scifact(
            base / "corpus.jsonl",
            base / "queries.jsonl",
            base / "qrels" / "test.tsv",
            name=name,
        )
    if name == "duretrieval":
        root = getattr(settings, "duretrieval_root", None)
        if not root:
            raise ValueError(
                "未配置 DuRetrieval 数据目录，请在设置页填写后重试"
            )
        base = Path(root)
        return load_duretrieval(
            base / "corpus.jsonl",
            base / "queries.jsonl",
            base / "qrels_dev.tsv",
            name=name,
        )
    raise ValueError(f"未知数据集: {name}")


def default_model_loader(
    model: ModelSpec, device: str, model_root: str | None = None
) -> Encoder:
    """只从 `<model_root>/<directory_name>` 本地目录加载模型。"""
    if not model_root:
        raise ValueError("未配置模型根目录，请在设置页填写后重试")
    from sentence_transformers import SentenceTransformer

    directory = Path(model_root) / model.directory_name
    return SentenceTransformer(
        str(directory), device=device, local_files_only=True
    )


class ExperimentExecutor:
    """单变体执行器；通过注入模型/数据集/设备适配器支持无模型测试。"""

    def __init__(
        self,
        *,
        model_loader: Callable[[ModelSpec, str], Encoder] | None = None,
        dataset_loader: Callable[[PlanDefinition], RetrievalDataset] | None = None,
        device_resolver: Callable[[], str] | None = None,
        cache_dir: Path | None = None,
        show_progress: bool = False,
        settings_provider: Callable[[], Settings] | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._dataset_loader = dataset_loader
        self._device_resolver = device_resolver
        self._settings_provider = settings_provider or Settings
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._show_progress = show_progress

    # ------------------------------------------------------------------ 公共接口
    def execute(
        self,
        plan: PlanDefinition,
        variant: RunVariant,
        report_stage: Callable[[StageEvent], None],
        cancellation: CancellationToken | None = None,
    ) -> ExecutionResult:
        cancellation = cancellation or NeverCancel()
        method = variant.method
        top_k = plan.evaluation.top_k
        k_values = plan.evaluation.k_values
        not_applicable = not_applicable_stages(method)
        recorder = DurationRecorder()
        completed: list[str] = []
        current: str | None = None

        def emit(stage: str, event: str) -> None:
            report_stage(StageEvent(stage, event, datetime.now(timezone.utc)))

        def begin(stage: str) -> None:
            nonlocal current
            current = stage
            cancellation.raise_if_cancelled(stage)
            emit(stage, "started")

        def end(stage: str) -> None:
            nonlocal current
            cancellation.raise_if_cancelled(stage)
            emit(stage, "completed")
            completed.append(stage)
            current = None

        try:
            begin(STAGE_PREPARE)
            settings = self._settings_provider()
            device = (
                None if method == "bm25" else self._resolve_device(settings)
            )
            end(STAGE_PREPARE)

            begin(STAGE_LOAD_DATASET)
            started = time.perf_counter()
            dataset = self._load_dataset(plan, settings)
            recorder.record(
                "dataset_load_seconds", time.perf_counter() - started
            )
            end(STAGE_LOAD_DATASET)

            encoder = None
            if method == "bm25":
                emit(STAGE_LOAD_MODEL, "skipped")
            else:
                begin(STAGE_LOAD_MODEL)
                started = time.perf_counter()
                encoder = self._load_model(
                    variant.model, device, settings
                )
                recorder.record(
                    "model_load_seconds", time.perf_counter() - started
                )
                end(STAGE_LOAD_MODEL)

            begin(STAGE_BUILD_INDEX)
            built = self._build(variant, encoder)
            started = time.perf_counter()
            built.retriever.fit(dataset.documents)
            recorder.record(
                "index_build_seconds", time.perf_counter() - started
            )
            if built.dense is not None:
                recorder.record(
                    "doc_cache_load_seconds", built.dense.timing["doc_cache_load"]
                )
                recorder.record(
                    "corpus_encode_seconds", built.dense.timing["corpus_encode"]
                )
            end(STAGE_BUILD_INDEX)

            eval_queries = [
                query for query in dataset.queries if query.query_id in dataset.qrels
            ]
            if not eval_queries:
                raise ValueError("没有可评价的查询（queries 与 qrels 无交集）")

            vectors = None
            if method == "bm25":
                emit(STAGE_ENCODE_QUERIES, "skipped")
            else:
                begin(STAGE_ENCODE_QUERIES)
                texts = [variant.query_prefix + query.text for query in eval_queries]
                vectors = built.dense.encode_queries(texts)
                recorder.record(
                    "query_cache_load_seconds",
                    built.dense.timing["query_cache_load"],
                )
                recorder.record(
                    "query_encode_seconds", built.dense.timing["query_encode"]
                )
                end(STAGE_ENCODE_QUERIES)

            begin(STAGE_RETRIEVE)
            retrieved = self._retrieve(
                method, variant, built, eval_queries, vectors, top_k, recorder
            )
            end(STAGE_RETRIEVE)

            begin(STAGE_EVALUATE)
            subset_qrels = {
                query.query_id: dataset.qrels[query.query_id]
                for query in eval_queries
            }
            started = time.perf_counter()
            evaluation = evaluate(retrieved, subset_qrels, ks=k_values)
            recorder.record(
                "evaluation_seconds", time.perf_counter() - started
            )
            end(STAGE_EVALUATE)

            return ExecutionResult(
                dataset={
                    "name": dataset.name or plan.dataset.name,
                    "split": plan.dataset.split,
                    "corpus_size": len(dataset.documents),
                    "query_count": len(eval_queries),
                    "qrels_pairs": sum(
                        len(subset_qrels[query.query_id]) for query in eval_queries
                    ),
                },
                variant_config=self._variant_config(plan, variant),
                metrics=evaluation["metrics"],
                num_queries=evaluation["num_queries"],
                num_skipped=evaluation["num_skipped"],
                timing=recorder.snapshot(),
                environment=collect_environment(device),
                stage_progress=StageProgress(
                    completed=tuple(completed),
                    current=None,
                    not_applicable=not_applicable,
                ),
            )
        except RunCancelled as exc:
            raise RunCancelled(exc.stage, completed=tuple(completed)) from None

    # ------------------------------------------------------------------ 内部实现
    def _resolve_device(self, settings: Settings) -> str:
        if self._device_resolver is not None:
            return self._device_resolver()
        return resolve_device(settings.device)

    def _load_dataset(
        self, plan: PlanDefinition, settings: Settings
    ) -> RetrievalDataset:
        if self._dataset_loader is not None:
            return self._dataset_loader(plan)
        return default_dataset_loader(plan, settings)

    def _load_model(
        self, model: ModelSpec, device: str, settings: Settings
    ) -> Encoder:
        if self._model_loader is not None:
            return self._model_loader(model, device)
        return default_model_loader(model, device, settings.model_root)

    def _build(self, variant: RunVariant, encoder: Encoder | None) -> _Built:
        method = variant.method
        if method == "bm25":
            bm25 = BM25Retriever(
                k1=variant.k1,
                b=variant.b,
                tokenizer=_tokenizer(variant.tokenizer),
            )
            return _Built(retriever=bm25, bm25=bm25)

        assert encoder is not None and variant.model is not None
        dense = DenseRetriever(
            encoder,
            batch_size=variant.batch_size,
            cache_dir=self._cache_dir,
            cache_key=variant.model.name,
            model_name=variant.model.name,
            model_revision=variant.model.revision,
            show_progress=self._show_progress,
        )
        if method == "dense":
            return _Built(retriever=dense, dense=dense)

        bm25 = BM25Retriever(
            k1=variant.k1,
            b=variant.b,
            tokenizer=_tokenizer(variant.tokenizer),
        )
        hybrid = HybridRetriever(
            [bm25, dense],
            rrf_k=variant.rrf_k,
            query_transforms=[None, lambda query: variant.query_prefix + query],
            weights=list(variant.rrf_weights),
        )
        return _Built(retriever=hybrid, dense=dense, bm25=bm25, hybrid=hybrid)

    def _retrieve(
        self,
        method: str,
        variant: RunVariant,
        built: _Built,
        eval_queries: Sequence[Query],
        vectors,
        top_k: int,
        recorder: DurationRecorder,
    ) -> dict[str, list[str]]:
        retrieved: dict[str, list[str]] = {}
        times: list[float] = []

        if method == "bm25":
            assert built.bm25 is not None
            for query in eval_queries:
                started = time.perf_counter()
                results = built.bm25.search(query.text, top_k=top_k)
                elapsed = time.perf_counter() - started
                times.append(elapsed)
                retrieved[query.query_id] = [doc_id for doc_id, _ in results]
            recorder.record("bm25_search_total_seconds", sum(times))
            self._record_retrieval_times(recorder, times)
            return retrieved

        if method == "dense":
            assert built.dense is not None
            for query, vector in zip(eval_queries, vectors):
                started = time.perf_counter()
                results = built.dense.search_vector(vector, top_k=top_k)
                elapsed = time.perf_counter() - started
                times.append(elapsed)
                retrieved[query.query_id] = [doc_id for doc_id, _ in results]
            recorder.record("dense_search_total_seconds", sum(times))
            self._record_retrieval_times(recorder, times)
            return retrieved

        assert built.hybrid is not None and built.bm25 is not None
        assert built.dense is not None
        depth = max(top_k, 100)  # 与 HybridRetriever 默认 candidate_depth 一致
        bm25_total = 0.0
        dense_total = 0.0
        rrf_total = 0.0
        for query, vector in zip(eval_queries, vectors):
            started = time.perf_counter()
            bm25_results = built.bm25.search(query.text, top_k=depth)
            bm25_elapsed = time.perf_counter() - started

            started = time.perf_counter()
            dense_results = built.dense.search_vector(vector, top_k=depth)
            dense_elapsed = time.perf_counter() - started

            started = time.perf_counter()
            fused = HybridRetriever.rrf_fuse(
                [bm25_results, dense_results],
                built.hybrid.rrf_k,
                top_k,
                built.hybrid.weights,
            )
            rrf_elapsed = time.perf_counter() - started

            bm25_total += bm25_elapsed
            dense_total += dense_elapsed
            rrf_total += rrf_elapsed
            times.append(bm25_elapsed + dense_elapsed + rrf_elapsed)
            retrieved[query.query_id] = [doc_id for doc_id, _ in fused]

        recorder.record("bm25_search_total_seconds", bm25_total)
        recorder.record("dense_search_total_seconds", dense_total)
        recorder.record("rrf_fusion_seconds", rrf_total)
        self._record_retrieval_times(recorder, times)
        return retrieved

    @staticmethod
    def _record_retrieval_times(
        recorder: DurationRecorder, times: Sequence[float]
    ) -> None:
        if not times:
            recorder.record("retrieval_total_seconds", 0.0)
            recorder.record("retrieval_mean_seconds", 0.0)
            recorder.record("retrieval_median_seconds", 0.0)
            return
        recorder.record("retrieval_total_seconds", sum(times))
        recorder.record("retrieval_mean_seconds", statistics.fmean(times))
        recorder.record("retrieval_median_seconds", statistics.median(times))

    @staticmethod
    def _variant_config(
        plan: PlanDefinition, variant: RunVariant
    ) -> dict[str, object]:
        config: dict[str, object] = {
            "method": variant.method,
            "top_k": plan.evaluation.top_k,
            "k_values": list(plan.evaluation.k_values),
        }
        if variant.method in ("bm25", "hybrid"):
            config.update(
                {"k1": variant.k1, "b": variant.b, "tokenizer": variant.tokenizer}
            )
        if variant.method in ("dense", "hybrid"):
            config.update(
                {
                    "model": variant.model.name,
                    "model_revision": variant.model.revision,
                    "batch_size": variant.batch_size,
                    "normalize_embeddings": variant.normalize_embeddings,
                    "query_prefix": variant.query_prefix,
                }
            )
        if variant.method == "hybrid":
            config.update(
                {
                    "rrf_k": variant.rrf_k,
                    "rrf_weights": list(variant.rrf_weights),
                }
            )
        return config
