"""按数据集、模型和设备检查资源可用性。

检查只读文件系统元数据与少量 JSON（分片索引），不加载模型权重，
不自动创建、下载或移动任何数据。原因文本不包含主机名、用户名或 IP。
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable

from experiment.environment import resolve_device
from experiment.plans import (
    PlanCatalog,
    PlanDefinition,
    ReferenceVariant,
    RunVariant,
)
from paths import DATA_DIR, PROJECT_ROOT

SCIFACT_DIR: Path = DATA_DIR / "raw" / "scifact"
SCIFACT_DATASET_FILES: tuple[str, ...] = (
    "corpus.jsonl",
    "queries.jsonl",
    "qrels/test.tsv",
)
DURETRIEVAL_DATASET_FILES: tuple[str, ...] = (
    "corpus.jsonl",
    "queries.jsonl",
    "qrels_dev.tsv",
)

WEIGHT_FILES: tuple[str, ...] = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
INDEX_FILES: tuple[str, ...] = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES: tuple[str, ...] = ("vocab.txt", "vocab.json", "merges.txt", "spiece.model")

DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("jinja2", "jinja2"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("pydantic", "pydantic"),
    ("torch", "torch"),
)

DEVICE_REASONS: dict[str, str] = {
    "cuda": "CUDA 设备不可用",
    "mps": "MPS 设备不可用",
}


class ResourceUnavailableError(Exception):
    """提交时资源不足；不创建 Pending 行。"""

    def __init__(self, message: str, missing: list[str] | None = None) -> None:
        self.message = message
        self.missing = list(missing or [])
        super().__init__(message)


@dataclass(frozen=True)
class ResourceState:
    state: str
    summary: str
    missing: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "summary": self.summary,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class DeviceState:
    configured: str
    effective: str
    available: bool
    summary: str

    def as_dict(self) -> dict:
        return {
            "configured": self.configured,
            "effective": self.effective,
            "available": self.available,
            "summary": self.summary,
        }


def _device_available(device: str) -> bool:
    if device in ("auto", "cpu"):
        return True
    try:
        import torch
    except ImportError:  # pragma: no cover - 无 torch 的降级路径
        return False
    if device == "mps":
        return bool(torch.backends.mps.is_available())
    if device == "cuda":
        return bool(torch.cuda.is_available())
    return False


def _file_reason(path: Path) -> str | None:
    """返回文件不可用原因；可用返回 None。"""
    try:
        if not path.exists():
            return "缺失"
        if not path.is_file():
            return "不是常规文件"
        if path.stat().st_size <= 0:
            return "为空"
        with path.open("rb") as handle:
            handle.read(1)
    except OSError:
        return "不可读"
    return None


def _directory_reason(path: Path) -> str | None:
    try:
        if not path.exists():
            return "缺失"
        if not path.is_dir():
            return "不是目录"
        # 目录可列出即可视为可读。
        next(path.iterdir(), None)
    except OSError:
        return "不可读"
    return None


class ResourceChecker:
    """资源状态计算；设置通过 provider 每次读取最新值。"""

    def __init__(
        self,
        catalog: PlanCatalog,
        settings_provider: Callable[[], object],
        *,
        device_probe: Callable[[str], bool] | None = None,
        scifact_dir: Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._settings = settings_provider
        self._device_probe = device_probe or _device_available
        self._scifact_dir = Path(scifact_dir) if scifact_dir else SCIFACT_DIR

    # ------------------------------------------------------------------ 数据集
    def dataset_state(self, name: str, settings=None) -> ResourceState:
        settings = settings or self._settings()
        if name == "scifact":
            root = self._scifact_dir
            files = SCIFACT_DATASET_FILES
            split = "test"
        elif name == "duretrieval":
            root = getattr(settings, "duretrieval_root", None)
            files = DURETRIEVAL_DATASET_FILES
            split = "dev"
            if root is None:
                return ResourceState(
                    "not_configured",
                    "未配置 DuRetrieval 数据目录（dev 划分）",
                    ("duretrieval_root",),
                )
        else:
            return ResourceState(
                "missing", f"未知数据集 {name}", (f"dataset:{name}",)
            )

        root_path = Path(root)
        missing: list[str] = []
        invalid = False
        for relative in files:
            reason = _file_reason(root_path / relative)
            if reason is None:
                continue
            identifier = f"{name}:{relative}:{reason}"
            missing.append(identifier)
            if reason not in ("缺失", "不是常规文件"):
                invalid = True
        if not missing:
            return ResourceState(
                "available", f"{name} 数据可用（{split} 划分）"
            )
        state = "invalid" if invalid else "missing"
        return ResourceState(
            state,
            f"{name} 数据不可用（{split} 划分）",
            tuple(missing),
        )

    # ------------------------------------------------------------------ 模型
    def model_state(self, model, settings=None) -> ResourceState:
        settings = settings or self._settings()
        root = getattr(settings, "model_root", None)
        if root is None:
            return ResourceState(
                "not_configured", "未配置模型根目录", ("model_root",)
            )
        directory = Path(root) / model.directory_name
        directory_reason = _directory_reason(directory)
        if directory_reason is not None:
            return ResourceState(
                "missing" if directory_reason == "缺失" else "invalid",
                f"模型目录不可用: {model.directory_name}",
                (f"model:{model.directory_name}:{directory_reason}",),
            )

        missing: list[str] = []
        invalid = False

        def record(relative: str) -> None:
            nonlocal invalid
            reason = _file_reason(directory / relative)
            if reason is None:
                return
            missing.append(f"model:{model.directory_name}:{relative}:{reason}")
            if reason not in ("缺失", "不是常规文件"):
                invalid = True

        record("config.json")
        weight_candidates: list[str] = []
        for name in WEIGHT_FILES:
            candidate = directory / name
            reason = _file_reason(candidate)
            if reason is None:
                weight_candidates.append(name)
                continue
            if candidate.exists():
                missing.append(
                    f"model:{model.directory_name}:{name}:{reason}"
                )
                invalid = True
        if not weight_candidates:
            missing.append(f"model:{model.directory_name}:weights:缺失")

        for index_name in INDEX_FILES:
            index_path = directory / index_name
            if not index_path.is_file():
                continue
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing.append(
                    f"model:{model.directory_name}:{index_name}:不可读"
                )
                invalid = True
                continue
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                missing.append(
                    f"model:{model.directory_name}:{index_name}:缺少 weight_map"
                )
                invalid = True
                continue
            for shard in sorted(set(weight_map.values())):
                reason = _file_reason(directory / str(shard))
                if reason is not None:
                    missing.append(
                        f"model:{model.directory_name}:{shard}:{reason}"
                    )
                    if reason not in ("缺失", "不是常规文件"):
                        invalid = True

        tokenizer_ok = _file_reason(directory / "tokenizer.json") is None
        if not tokenizer_ok:
            config_ok = _file_reason(directory / "tokenizer_config.json") is None
            vocab_ok = any(
                _file_reason(directory / name) is None
                for name in TOKENIZER_FILES
            )
            if not (config_ok and vocab_ok):
                missing.append(
                    f"model:{model.directory_name}:tokenizer:缺失"
                )

        if not missing:
            return ResourceState(
                "available", f"模型 {model.directory_name} 可用"
            )
        state = "invalid" if invalid else "missing"
        return ResourceState(
            state,
            f"模型 {model.directory_name} 不可用",
            tuple(sorted(set(missing))),
        )

    def unique_models(self) -> list:
        models: dict[str, object] = {}
        for plan in self._catalog.plans:
            for variant in plan.variants:
                if isinstance(variant, RunVariant) and variant.model is not None:
                    models.setdefault(variant.model.directory_name, variant.model)
        return list(models.values())

    # ------------------------------------------------------------------ 设备
    def device_state(self, settings=None) -> DeviceState:
        settings = settings or self._settings()
        configured = getattr(settings, "device", "auto")
        if configured == "auto":
            effective = resolve_device("auto")
            return DeviceState(
                configured=configured,
                effective=effective,
                available=True,
                summary=f"自动选择，实际生效 {effective}",
            )
        available = True if configured == "cpu" else bool(
            self._device_probe(configured)
        )
        if available:
            summary = f"显式设置 {configured}，设备可用"
        else:
            summary = (
                f"显式设置 {configured}，但{DEVICE_REASONS.get(configured, '设备不可用')}"
            )
        return DeviceState(
            configured=configured,
            effective=configured,
            available=available,
            summary=summary,
        )

    # ------------------------------------------------------------------ 变体
    def variant_state(self, plan: PlanDefinition, variant) -> dict:
        if isinstance(variant, ReferenceVariant):
            return {
                "state": "not_applicable",
                "summary": "参照结果，无需运行资源",
                "missing": [],
            }
        settings = self._settings()
        dataset = self.dataset_state(plan.dataset.name, settings)
        if dataset.state != "available":
            return {
                "state": "unavailable",
                "summary": f"{dataset.summary}；{variant.name} 不可运行",
                "missing": list(dataset.missing),
            }
        if variant.method == "bm25":
            return {"state": "available", "summary": "资源可用", "missing": []}

        device = self.device_state(settings)
        if not device.available:
            return {
                "state": "unavailable",
                "summary": f"{device.summary}；{variant.name} 不可运行",
                "missing": [f"device:{device.configured}"],
            }
        assert variant.model is not None
        model = self.model_state(variant.model, settings)
        if model.state != "available":
            return {
                "state": "unavailable",
                "summary": f"{model.summary}；{variant.name} 不可运行",
                "missing": list(model.missing),
            }
        return {"state": "available", "summary": "资源可用", "missing": []}

    def require_available(self, plan: PlanDefinition, variant) -> None:
        status = self.variant_state(plan, variant)
        if status["state"] != "available":
            raise ResourceUnavailableError(
                f"{variant.name} 当前不可运行", status["missing"]
            )

    def plan_runnable_count(self, plan: PlanDefinition) -> int:
        return sum(
            1
            for variant in plan.variants
            if isinstance(variant, RunVariant)
            and self.variant_state(plan, variant)["state"] == "available"
        )

    # ------------------------------------------------------------------ 快照
    def snapshot(self) -> dict:
        settings = self._settings()
        datasets = [
            {"id": name, **self.dataset_state(name, settings).as_dict()}
            for name in ("scifact", "duretrieval")
        ]
        models = [
            {
                "id": model.directory_name,
                "name": model.name,
                "revision": model.revision,
                **self.model_state(model, settings).as_dict(),
            }
            for model in self.unique_models()
        ]
        return {
            "datasets": datasets,
            "models": models,
            "device": self.device_state(settings).as_dict(),
        }


def environment_summary() -> dict:
    """返回脱敏的环境与已安装依赖版本，不加载 torch 模型。"""
    dependencies: dict[str, str | None] = {}
    for label, distribution in DEPENDENCIES:
        try:
            dependencies[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependencies[label] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
