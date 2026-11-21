"""TextMind Web 层。

包内模块以顶层模块方式导入（`uvicorn web.app:create_app --app-dir src`），
因此跨模块引用使用 `paths` 等顶层模块名，而不是 `src.paths`。
"""
