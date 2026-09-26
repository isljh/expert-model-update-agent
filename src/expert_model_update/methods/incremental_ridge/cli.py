"""JSON stdin/file entry point for incremental ridge tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_model_update.methods.incremental_ridge.core import (
    METHOD_ID,
    METHOD_VERSION,
    SCHEMA_VERSION,
    inspect_state,
    update_incremental_ridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查或更新解析增量岭回归状态。")
    parser.add_argument("action", choices=("inspect", "update"))
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--stdin", action="store_true", dest="read_stdin")
    args = parser.parse_args()
    if args.read_stdin == (args.input is not None):
        parser.error("必须且只能选择一个输入来源：文件路径或 --stdin。")
    return args


def failure(action: str, code: str, message: str) -> dict[str, Any]:
    return {
        "report_type": "IncrementalRidgeToolReport",
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "action": action,
        "status": "failed",
        "issues": [{"code": code, "path": "$", "message": message}],
        "state_sha256": None,
        "state": None,
        "summary": None,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        if args.read_stdin:
            payload = json.load(sys.stdin)
        else:
            with args.input.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        report = failure(args.action, "INVALID_INPUT", f"无法读取 JSON 输入：{exc}")
        print(json.dumps(report, ensure_ascii=False))
        return 2

    try:
        report = inspect_state(payload) if args.action == "inspect" else update_incremental_ridge(payload)
    except Exception as exc:  # Keep the process boundary explicit for the harness.
        report = failure(args.action, "EXECUTION_ERROR", f"解析增量岭回归工具执行失败：{type(exc).__name__}: {exc}")
        print(json.dumps(report, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] in {"passed", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
