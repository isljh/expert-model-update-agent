"""Run a read-only availability preflight for assets declared by TaskInput."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


REPORT_TYPE = "AssetPreflightReport"
REPORT_SCHEMA_VERSION = "0.1"

CheckStatus = Literal["passed", "failed", "skipped"]
ExpectedKind = Literal["directory", "file", "file_or_directory"]


@dataclass(frozen=True)
class AssetCheck:
    field: str
    supplied_path: str | None
    expected_kind: ExpectedKind
    status: CheckStatus
    resolved_path: str | None
    reason_code: str | None
    message: str


class PreflightInputError(ValueError):
    """Raised when the caller did not provide the minimum TaskInput identity/assets."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _task_identity(task_input: Any) -> tuple[str, int, dict[str, Any]]:
    if not isinstance(task_input, dict):
        raise PreflightInputError("INVALID_TASK_INPUT", "TaskInput 应为 JSON 对象。")

    task_id = _nonempty_string(task_input.get("task_id"))
    if task_id is None:
        raise PreflightInputError("INVALID_TASK_ID", "TaskInput 缺少非空 task_id。")

    version = task_input.get("task_input_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PreflightInputError(
            "INVALID_TASK_INPUT_VERSION",
            "TaskInput 的 task_input_version 应为不小于 1 的整数。",
        )

    assets = task_input.get("assets")
    if not isinstance(assets, dict):
        raise PreflightInputError("INVALID_ASSETS", "TaskInput 缺少 assets 对象。")

    return task_id, version, assets


def asset_input_digest(task_id: str, version: int, assets: dict[str, Any]) -> str:
    """Bind a report to asset declarations without including confirmation state."""
    canonical = json.dumps(
        {
            "task_id": task_id,
            "task_input_version": version,
            "assets": assets,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _nested_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def _failed_check(
    field: str,
    supplied_path: str | None,
    expected_kind: ExpectedKind,
    reason_code: str,
    message: str,
    *,
    resolved_path: str | None = None,
) -> AssetCheck:
    return AssetCheck(
        field=field,
        supplied_path=supplied_path,
        expected_kind=expected_kind,
        status="failed",
        resolved_path=resolved_path,
        reason_code=reason_code,
        message=message,
    )


def _resolve_candidate(location: str, base_directory: Path) -> Path:
    candidate = Path(location).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return candidate


def _check_readable(path: Path, kind: ExpectedKind) -> None:
    if kind == "directory":
        with os.scandir(path):
            return
    if kind == "file":
        with path.open("rb"):
            return
    raise ValueError(f"不支持的可读性检查类型：{kind}")


def check_path(
    field: str,
    supplied_value: Any,
    expected_kind: ExpectedKind,
    base_directory: Path,
) -> AssetCheck:
    supplied_path = _nonempty_string(supplied_value)
    if supplied_path is None:
        return _failed_check(
            field,
            None,
            expected_kind,
            "missing_value",
            "没有提供资产路径。",
        )

    try:
        candidate = _resolve_candidate(supplied_path, base_directory)
        metadata = candidate.stat()
    except FileNotFoundError:
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "path_not_found",
            "路径不存在。",
        )
    except PermissionError:
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "permission_denied",
            "没有权限访问该路径。",
        )
    except OSError as exc:
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "path_error",
            f"无法检查路径：{exc}",
        )

    actual_kind: ExpectedKind | Literal["unsupported"]
    if stat.S_ISDIR(metadata.st_mode):
        actual_kind = "directory"
    elif stat.S_ISREG(metadata.st_mode):
        actual_kind = "file"
    else:
        actual_kind = "unsupported"

    try:
        resolved_path = str(candidate.resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "resolution_error",
            f"无法解析规范路径：{exc}",
        )

    if actual_kind == "unsupported":
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "unsupported_path_type",
            "路径不是普通文件或目录。",
            resolved_path=resolved_path,
        )
    if expected_kind == "directory" and actual_kind != "directory":
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "not_a_directory",
            "路径存在，但不是目录。",
            resolved_path=resolved_path,
        )
    if expected_kind == "file" and actual_kind != "file":
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "not_a_file",
            "路径存在，但不是普通文件。",
            resolved_path=resolved_path,
        )

    concrete_kind: ExpectedKind = actual_kind
    try:
        _check_readable(candidate, concrete_kind)
    except PermissionError:
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "permission_denied",
            "路径存在，但当前进程无法读取。",
            resolved_path=resolved_path,
        )
    except OSError as exc:
        return _failed_check(
            field,
            supplied_path,
            expected_kind,
            "read_error",
            f"路径存在，但读取检查失败：{exc}",
            resolved_path=resolved_path,
        )

    return AssetCheck(
        field=field,
        supplied_path=supplied_path,
        expected_kind=expected_kind,
        status="passed",
        resolved_path=resolved_path,
        reason_code=None,
        message=f"路径存在，是{('目录' if actual_kind == 'directory' else '普通文件')}，当前可读。",
    )


def preflight_task_assets(
    task_input: Any,
    *,
    base_directory: Path | None = None,
) -> dict[str, Any]:
    task_id, version, assets = _task_identity(task_input)
    base = (base_directory or Path.cwd()).resolve(strict=True)

    base_model = _nested_object(assets, "base_model")
    new_data = _nested_object(assets, "new_data")

    checks = [
        check_path(
            "assets.base_model.model_file",
            base_model.get("model_file"),
            "file",
            base,
        ),
        check_path(
            "assets.base_model.config",
            base_model.get("config"),
            "file",
            base,
        ),
        check_path(
            "assets.base_model.checkpoint",
            base_model.get("checkpoint"),
            "file",
            base,
        ),
        check_path(
            "assets.new_data.location",
            new_data.get("location"),
            "file_or_directory",
            base,
        ),
    ]

    overall_status = (
        "failed" if any(check.status == "failed" for check in checks) else "passed"
    )
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_id": task_id,
        "task_input_version": version,
        "asset_input_digest": asset_input_digest(task_id, version, assets),
        "base_directory": str(base),
        "overall_status": overall_status,
        "checks": [asdict(check) for check in checks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 TaskInput 声明资产的存在性、类型和可读性。")
    parser.add_argument("input", nargs="?", type=Path, help="TaskInput JSON 文件路径")
    parser.add_argument(
        "--stdin",
        action="store_true",
        dest="read_stdin",
        help="从标准输入读取 TaskInput JSON；用于 Harness 工具适配。",
    )
    parser.add_argument(
        "--base-directory",
        type=Path,
        default=None,
        help="解析相对资产路径的基准目录；默认使用当前工作目录。",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="输出人类可读文本或机器可读 JSON。",
    )
    args = parser.parse_args()
    if args.read_stdin == (args.input is not None):
        parser.error("必须且只能选择一个输入来源：文件路径或 --stdin。")
    return args


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_id": None,
        "task_input_version": None,
        "asset_input_digest": None,
        "base_directory": str(Path.cwd()),
        "overall_status": "error",
        "checks": [],
        "execution_error": True,
        "error": {"code": code, "message": message},
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        if args.read_stdin:
            task_input = json.load(sys.stdin)
        else:
            with args.input.open("r", encoding="utf-8") as stream:
                task_input = json.load(stream)
        report = preflight_task_assets(
            task_input,
            base_directory=args.base_directory,
        )
    except FileNotFoundError as exc:
        report = _error_payload("INPUT_NOT_FOUND", f"找不到输入或基准目录：{exc}")
    except json.JSONDecodeError as exc:
        report = _error_payload("INVALID_JSON", f"无法解析 TaskInput JSON：{exc}")
    except PreflightInputError as exc:
        report = _error_payload(exc.code, str(exc))
    except OSError as exc:
        report = _error_payload("PREFLIGHT_ERROR", f"资产预检执行失败：{exc}")

    if args.output == "json":
        print(json.dumps(report, ensure_ascii=False))
    elif report.get("execution_error"):
        error = report["error"]
        print(f"资产预检执行失败 [{error['code']}]：{error['message']}", file=sys.stderr)
    else:
        print(
            "资产预检"
            + ("通过" if report["overall_status"] == "passed" else "失败")
            + f"：task_id={report['task_id']}, version={report['task_input_version']}"
        )
        for check in report["checks"]:
            print(f"- {check['field']}: {check['status']} — {check['message']}")

    if report.get("execution_error"):
        return 2
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
