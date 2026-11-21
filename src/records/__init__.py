"""运行记录模型与 Schema。

结果目录、图表与发布器位于 `records.repository`、`records.charts`、
`records.publisher`，按需导入，避免与 `experiment` 形成循环依赖。
"""

from records.schema import (
    FORMAT_VERSION,
    RUN_RECORD_SCHEMA_PATH,
    RunRecord,
    RunRecordError,
    parse_run_record,
    validate_run_record_schema,
)

__all__ = [
    "FORMAT_VERSION",
    "RUN_RECORD_SCHEMA_PATH",
    "RunRecord",
    "RunRecordError",
    "parse_run_record",
    "validate_run_record_schema",
]
