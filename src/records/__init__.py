"""运行记录模型、Schema 与解析接口。"""

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
