"""方案目录模型、Schema 校验与加载接口。

`config/plans.json` 是唯一的运行时方案来源；本模块只读取配置和
`schemas/plan_catalog.schema.json`，不加载数据集、模型或预置结果文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from paths import CONFIG_DIR, SCHEMAS_DIR

CATALOG_SCHEMA_PATH: Path = SCHEMAS_DIR / "plan_catalog.schema.json"
DEFAULT_CATALOG_PATH: Path = CONFIG_DIR / "plans.json"

PLAN_VERSION = "v1"

REQUIRED_PLAN_IDS: tuple[str, ...] = (
    "scifact-comparison",
    "duretrieval-comparison",
    "duretrieval-rrf-weight-sweep",
)

REQUIRED_VARIANT_IDS: frozenset[str] = frozenset(
    {
        "scifact-bm25",
        "scifact-dense-qwen3-base",
        "scifact-dense-qwen3-instruct",
        "scifact-dense-bge-m3-base",
        "scifact-dense-bge-m3-prefix",
        "scifact-dense-minilm",
        "scifact-hybrid-qwen3-base",
        "scifact-hybrid-qwen3-instruct",
        "scifact-hybrid-qwen3-instruct-w0.5",
        "duretrieval-bm25",
        "duretrieval-dense-bge-small-base",
        "duretrieval-dense-bge-small-prefix",
        "duretrieval-hybrid-bge-small-prefix-equal",
        "duretrieval-hybrid-bge-small-prefix-w0.5",
        "duretrieval-hybrid-bge-small-prefix-w0.3",
        "duretrieval-hybrid-bge-small-prefix-w0.2",
        "duretrieval-hybrid-bge-small-prefix-w0.1",
    }
)

# 每个方案允许的运行变体集合；数量相同但成员互换也必须拒绝。
EXPECTED_PLAN_VARIANTS: dict[str, frozenset[str]] = {
    "scifact-comparison": frozenset(
        {
            "scifact-bm25",
            "scifact-dense-qwen3-base",
            "scifact-dense-qwen3-instruct",
            "scifact-dense-bge-m3-base",
            "scifact-dense-bge-m3-prefix",
            "scifact-dense-minilm",
            "scifact-hybrid-qwen3-base",
            "scifact-hybrid-qwen3-instruct",
            "scifact-hybrid-qwen3-instruct-w0.5",
        }
    ),
    "duretrieval-comparison": frozenset(
        {
            "duretrieval-bm25",
            "duretrieval-dense-bge-small-base",
            "duretrieval-dense-bge-small-prefix",
            "duretrieval-hybrid-bge-small-prefix-equal",
        }
    ),
    "duretrieval-rrf-weight-sweep": frozenset(
        {
            "duretrieval-hybrid-bge-small-prefix-w0.5",
            "duretrieval-hybrid-bge-small-prefix-w0.3",
            "duretrieval-hybrid-bge-small-prefix-w0.2",
            "duretrieval-hybrid-bge-small-prefix-w0.1",
        }
    ),
}

# 每个方案允许的参照项集合；参照项只能出现在加权 RRF 方案中。
EXPECTED_REFERENCE_VARIANTS: dict[str, frozenset[str]] = {
    "scifact-comparison": frozenset(),
    "duretrieval-comparison": frozenset(),
    "duretrieval-rrf-weight-sweep": frozenset(
        {"duretrieval-hybrid-bge-small-prefix-equal"}
    ),
}

IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
PRESET_RUN_ID_PATTERN = r"^preset-[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
PRESET_SOURCE_PATTERN = r"^data/evaluation/[^/]+\.json$"
PLAN_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

_BM25_FIELDS = ("k1", "b", "tokenizer")
_DENSE_FIELDS = ("model", "query_prefix", "batch_size", "normalize_embeddings")
_HYBRID_FIELDS = ("rrf_k", "rrf_weights")


class PlanCatalogError(Exception):
    """方案目录读取或校验失败；消息中包含字段路径。"""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = list(errors or [])
        super().__init__(message)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DatasetSpec(_FrozenModel):
    name: str = Field(min_length=1)
    language: str = Field(min_length=1)
    split: str = Field(min_length=1)


class EvaluationSpec(_FrozenModel):
    top_k: int = Field(ge=1)
    k_values: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_k_values(self) -> "EvaluationSpec":
        if any(k <= 0 for k in self.k_values):
            raise ValueError(f"k_values 必须全为正整数，实际 {list(self.k_values)}")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError(f"k_values 不能重复，实际 {list(self.k_values)}")
        if self.top_k < max(self.k_values):
            raise ValueError(
                f"top_k ({self.top_k}) 必须 >= 最大评价 K ({max(self.k_values)})"
            )
        return self


class ModelSpec(_FrozenModel):
    name: str = Field(min_length=1)
    directory_name: str = Field(min_length=1)
    revision: str | None = Field(min_length=1)


class RunVariant(_FrozenModel):
    variant_id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    kind: Literal["run"] = "run"
    method: Literal["bm25", "dense", "hybrid"]
    preset_run_id: str = Field(pattern=PRESET_RUN_ID_PATTERN)
    preset_source: str = Field(pattern=PRESET_SOURCE_PATTERN)

    # BM25 / Hybrid 子路
    k1: float | None = Field(default=None, ge=0)
    b: float | None = Field(default=None, ge=0, le=1)
    tokenizer: Literal["simple", "jieba"] | None = None

    # Dense / Hybrid 子路
    model: ModelSpec | None = None
    query_prefix: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    normalize_embeddings: bool | None = None

    # Hybrid 专有
    rrf_k: int | None = Field(default=None, ge=1)
    rrf_weights: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _validate_method_fields(self) -> "RunVariant":
        errors: list[str] = []

        def require(fields: tuple[str, ...]) -> None:
            missing = [name for name in fields if getattr(self, name) is None]
            if missing:
                errors.append(
                    f"{self.variant_id}: method={self.method} 缺少必需字段 {missing}"
                )

        def forbid(fields: tuple[str, ...]) -> None:
            present = [name for name in fields if getattr(self, name) is not None]
            if present:
                errors.append(
                    f"{self.variant_id}: method={self.method} 不允许字段 {present}"
                )

        if self.method == "bm25":
            require(_BM25_FIELDS)
            forbid(_DENSE_FIELDS + _HYBRID_FIELDS)
        elif self.method == "dense":
            require(_DENSE_FIELDS)
            forbid(_BM25_FIELDS + _HYBRID_FIELDS)
        else:
            require(_BM25_FIELDS + _DENSE_FIELDS + _HYBRID_FIELDS)

        if self.rrf_weights is not None:
            if any(weight < 0 for weight in self.rrf_weights):
                errors.append(
                    f"{self.variant_id}: rrf_weights 必须非负，实际 {list(self.rrf_weights)}"
                )
            if all(weight == 0 for weight in self.rrf_weights):
                errors.append(f"{self.variant_id}: rrf_weights 不能全为 0")

        if errors:
            raise ValueError("; ".join(errors))
        return self


class ReferenceVariant(_FrozenModel):
    variant_id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    kind: Literal["reference"] = "reference"
    source_plan_id: str = Field(min_length=1)
    source_plan_version: str = Field(min_length=1)
    source_variant_id: str = Field(pattern=IDENTIFIER_PATTERN)


VariantDefinition = Annotated[
    Union[RunVariant, ReferenceVariant],
    Field(discriminator="kind"),
]


class PlanDefinition(_FrozenModel):
    plan_id: str = Field(pattern=PLAN_ID_PATTERN)
    plan_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    dataset: DatasetSpec
    evaluation: EvaluationSpec
    variants: tuple[VariantDefinition, ...] = Field(min_length=1)


class PlanCatalog(_FrozenModel):
    schema_version: Literal[1]
    plans: tuple[PlanDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_catalog(self) -> "PlanCatalog":
        errors: list[str] = []
        plan_keys: set[tuple[str, str]] = set()
        run_variants: dict[tuple[str, str, str], RunVariant] = {}
        preset_ids: dict[str, str] = {}
        preset_sources: dict[str, str] = {}
        all_variant_ids: set[str] = set()

        for plan_index, plan in enumerate(self.plans):
            plan_key = (plan.plan_id, plan.plan_version)
            if plan_key in plan_keys:
                errors.append(
                    f"plans[{plan_index}]: 方案 ID/版本重复 {plan.plan_id}/{plan.plan_version}"
                )
            plan_keys.add(plan_key)

            seen_in_plan: set[str] = set()
            for variant_index, variant in enumerate(plan.variants):
                path = f"plans[{plan_index}].variants[{variant_index}]"
                if variant.variant_id in seen_in_plan:
                    errors.append(
                        f"{path}.variant_id: 方案内重复变体 ID {variant.variant_id}"
                    )
                seen_in_plan.add(variant.variant_id)

                if isinstance(variant, ReferenceVariant):
                    continue

                all_variant_ids.add(variant.variant_id)
                run_variants[(*plan_key, variant.variant_id)] = variant

                expected_preset_id = f"preset-{variant.variant_id}"
                if variant.preset_run_id != expected_preset_id:
                    errors.append(
                        f"{path}.preset_run_id: 必须为 {expected_preset_id}"
                    )
                if variant.preset_run_id in preset_ids:
                    errors.append(
                        f"{path}.preset_run_id: 重复 {variant.preset_run_id}"
                        f"（已被 {preset_ids[variant.preset_run_id]} 使用）"
                    )
                preset_ids[variant.preset_run_id] = path

                if variant.preset_source in preset_sources:
                    errors.append(
                        f"{path}.preset_source: 重复 {variant.preset_source}"
                        f"（已被 {preset_sources[variant.preset_source]} 使用）"
                    )
                preset_sources[variant.preset_source] = path

            run_ids = {
                variant.variant_id
                for variant in plan.variants
                if isinstance(variant, RunVariant)
            }
            reference_ids = {
                variant.variant_id
                for variant in plan.variants
                if isinstance(variant, ReferenceVariant)
            }
            expected_run_ids = EXPECTED_PLAN_VARIANTS.get(plan.plan_id)
            if expected_run_ids is not None and run_ids != expected_run_ids:
                errors.append(
                    f"plans[{plan_index}]: 方案 {plan.plan_id} 的运行变体集合不符："
                    f"缺少 {sorted(expected_run_ids - run_ids)}，"
                    f"多出 {sorted(run_ids - expected_run_ids)}"
                )
            expected_reference_ids = EXPECTED_REFERENCE_VARIANTS.get(
                plan.plan_id, frozenset()
            )
            if reference_ids != expected_reference_ids:
                errors.append(
                    f"plans[{plan_index}]: 方案 {plan.plan_id} 的参照项集合不符："
                    f"期望 {sorted(expected_reference_ids)}，实际 {sorted(reference_ids)}"
                )

        expected_plans = {(plan_id, PLAN_VERSION) for plan_id in REQUIRED_PLAN_IDS}
        missing_plans = sorted(expected_plans - plan_keys)
        extra_plans = sorted(plan_keys - expected_plans)
        if missing_plans:
            errors.append(f"plans: 缺少必需方案 {missing_plans}")
        if extra_plans:
            errors.append(f"plans: 包含未定义方案 {extra_plans}")

        missing_variants = sorted(REQUIRED_VARIANT_IDS - all_variant_ids)
        extra_variants = sorted(all_variant_ids - REQUIRED_VARIANT_IDS)
        if missing_variants:
            errors.append(f"plans: 缺少必需变体 {missing_variants}")
        if extra_variants:
            errors.append(f"plans: 包含未定义变体 {extra_variants}")

        for plan_index, plan in enumerate(self.plans):
            for variant_index, variant in enumerate(plan.variants):
                if not isinstance(variant, ReferenceVariant):
                    continue
                path = f"plans[{plan_index}].variants[{variant_index}]"
                source_key = (
                    variant.source_plan_id,
                    variant.source_plan_version,
                    variant.source_variant_id,
                )
                if source_key not in run_variants:
                    errors.append(
                        f"{path}: 参照来源不存在 {variant.source_plan_id}/"
                        f"{variant.source_plan_version}/{variant.source_variant_id}"
                    )
                if variant.variant_id != variant.source_variant_id:
                    errors.append(
                        f"{path}.variant_id: 参照项必须复用来源变体 ID"
                        f" {variant.source_variant_id}"
                    )

        if errors:
            raise ValueError("\n".join(errors))
        return self


def _schema_path() -> Path:
    return CATALOG_SCHEMA_PATH


_schema_cache: dict | None = None


def _catalog_schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = json.loads(_schema_path().read_text(encoding="utf-8"))
    return _schema_cache


def _json_path(error) -> str:
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _format_pydantic_error(error: ValidationError) -> str:
    lines: list[str] = []
    for item in error.errors():
        loc = "$"
        for part in item["loc"]:
            loc += f"[{part}]" if isinstance(part, int) else f".{part}"
        lines.append(f"{loc}: {item['msg']}")
    return "\n".join(lines)


def _variant_branch(instance) -> int | None:
    """oneOf 中与变体 kind/method 对应的分支下标。"""
    if not isinstance(instance, dict):
        return None
    if instance.get("kind") == "reference":
        return 3
    return {"bm25": 0, "dense": 1, "hybrid": 2}.get(instance.get("method"))


def _error_branch(error) -> int | None:
    path = list(error.schema_path)
    for index, part in enumerate(path):
        if part == "oneOf" and index + 1 < len(path) and isinstance(path[index + 1], int):
            return path[index + 1]
    # $ref 解析后的子错误以 oneOf 分支下标开头。
    if path and isinstance(path[0], int):
        return path[0]
    return None


def _flatten_schema_errors(errors) -> list:
    """展开 oneOf/anyOf 的笼统错误，保留方法对应分支的最具体子错误。"""
    flattened: list = []
    for error in errors:
        if error.validator in ("oneOf", "anyOf") and error.context:
            branch = _variant_branch(error.instance)
            candidates = [
                child for child in error.context if _error_branch(child) == branch
            ] if branch is not None else []
            if not candidates:
                candidates = list(error.context)
            # 缺字段时优先报告 required；其余情况选最具体的子错误。
            required_errors = [
                child for child in candidates if child.validator == "required"
            ]
            chosen = required_errors[0] if required_errors else best_match(candidates)
            if chosen is not None:
                flattened.append(chosen)
                continue
        flattened.append(error)
    return flattened


def _validate_against_schema(data: dict, source: Path) -> None:
    validator = Draft202012Validator(_catalog_schema())
    errors = sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path))
    if errors:
        messages = [
            f"{_json_path(error)}: {error.message}"
            for error in _flatten_schema_errors(errors)
        ]
        raise PlanCatalogError(
            f"方案目录 {source} 不符合 plan_catalog.schema.json：\n"
            + "\n".join(messages),
            messages,
        )


def load_plan_catalog(path: Path | str | None = None) -> PlanCatalog:
    """读取并校验方案目录，返回不可变的方案定义。

    配置无效时抛出 `PlanCatalogError`，错误消息包含字段路径。
    """
    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    try:
        raw = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanCatalogError(f"无法读取方案目录 {catalog_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlanCatalogError(
            f"方案目录不是合法 JSON：{catalog_path}:{exc.lineno}:{exc.colno} {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise PlanCatalogError(f"方案目录 {catalog_path} 顶层必须是 JSON 对象")

    _validate_against_schema(data, catalog_path)

    try:
        return PlanCatalog.model_validate(data)
    except ValidationError as exc:
        raise PlanCatalogError(
            f"方案目录 {catalog_path} 校验失败：\n" + _format_pydantic_error(exc)
        ) from exc
