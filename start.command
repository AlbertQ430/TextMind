#!/bin/sh
# TextMind macOS 启动入口：从脚本所在目录定位项目根目录，使用项目内 .venv。
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
cd "$SCRIPT_DIR" || exit 1

if [ ! -x ".venv/bin/python" ]; then
  echo "[错误] 未找到 .venv/bin/python。" >&2
  echo "请先安装 Python 3.11 依赖：python3 scripts/install.py --python /path/to/python3.11" >&2
  exit 1
fi

exec .venv/bin/python scripts/start.py "$@"
