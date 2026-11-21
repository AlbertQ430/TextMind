"""项目路径与目录边界。

所有持久化目录都由本模块给出绝对路径，避免各模块自行拼接相对路径。
`ensure_app_dirs()` 只创建缺失目录，不删除也不清空已有目录。
"""

from pathlib import Path

# src/ 的上一级即项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
APP_DATA_DIR: Path = DATA_DIR / "app"
TASKS_DB_PATH: Path = APP_DATA_DIR / "textmind.db"
INSTANCE_LOCK_PATH: Path = APP_DATA_DIR / "textmind.lock"

CONFIG_DIR: Path = PROJECT_ROOT / "config"
SCHEMAS_DIR: Path = PROJECT_ROOT / "schemas"

WEB_DIR: Path = PROJECT_ROOT / "src" / "web"
TEMPLATES_DIR: Path = WEB_DIR / "templates"
STATIC_DIR: Path = WEB_DIR / "static"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
PRESETS_DIR: Path = RESULTS_DIR / "presets"
RUNS_DIR: Path = RESULTS_DIR / "runs"
IMPORTS_DIR: Path = RESULTS_DIR / "imports"

LOGS_DIR: Path = PROJECT_ROOT / "logs"

# 应用启动时必须存在的目录，顺序与其在磁盘上的层级无关。
APP_DIRS: tuple[Path, ...] = (
    APP_DATA_DIR,
    RESULTS_DIR,
    PRESETS_DIR,
    RUNS_DIR,
    IMPORTS_DIR,
    LOGS_DIR,
)


def ensure_app_dirs() -> None:
    """幂等创建应用运行目录。

    已存在的目录保持原样，目录内的文件不会被删除或覆盖。
    创建失败（权限不足、同名文件冲突等）时抛出 OSError，由调用方阻止启动。
    """
    for directory in APP_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
