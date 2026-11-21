"""实验方案、执行器、阶段、计时与环境信息。"""

from experiment.environment import collect_environment, resolve_device
from experiment.executor import (
    CancellationToken,
    ExecutionResult,
    ExperimentExecutor,
    NeverCancel,
    RunCancelled,
    default_dataset_loader,
    default_model_loader,
)
from experiment.plans import (
    DEFAULT_CATALOG_PATH,
    EXPECTED_PLAN_VARIANTS,
    EXPECTED_REFERENCE_VARIANTS,
    REQUIRED_PLAN_IDS,
    REQUIRED_VARIANT_IDS,
    PlanCatalog,
    PlanCatalogError,
    PlanDefinition,
    ReferenceVariant,
    RunVariant,
    load_plan_catalog,
)
from experiment.stages import (
    STAGE_EVENTS,
    STAGE_LABELS,
    STAGES,
    StageEvent,
    applicable_stages,
    not_applicable_stages,
    stage_is_applicable,
)
from experiment.timing import (
    TIMING_FIELDS,
    DurationRecorder,
    hybrid_subtotal,
    hybrid_total_matches,
)

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "EXPECTED_PLAN_VARIANTS",
    "EXPECTED_REFERENCE_VARIANTS",
    "REQUIRED_PLAN_IDS",
    "REQUIRED_VARIANT_IDS",
    "PlanCatalog",
    "PlanCatalogError",
    "PlanDefinition",
    "ReferenceVariant",
    "RunVariant",
    "load_plan_catalog",
    "STAGE_EVENTS",
    "STAGE_LABELS",
    "STAGES",
    "StageEvent",
    "applicable_stages",
    "not_applicable_stages",
    "stage_is_applicable",
    "TIMING_FIELDS",
    "DurationRecorder",
    "hybrid_subtotal",
    "hybrid_total_matches",
    "collect_environment",
    "resolve_device",
    "CancellationToken",
    "ExecutionResult",
    "ExperimentExecutor",
    "NeverCancel",
    "RunCancelled",
    "default_dataset_loader",
    "default_model_loader",
]
